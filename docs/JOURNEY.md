# The journey - how this solution was built, step by step

> 🇪🇬 النسخة العربية: [JOURNEY.ar.md](JOURNEY.ar.md)

This is the engineering log behind the result: what was tried, why each tool was picked, what failed, what was replaced and
what it taught us. Nothing here is hidden behind the final numbers - the goal is that a reviewer can see *how* the system
was reasoned out, not only what it outputs.

**Constraints**: 72 hours, two short confidential videos, no labelled data for the target roads, own laptop without a GPU
(heavy work ran on free Kaggle Tesla T4 x2 sessions). Videos never leave the SSD and a private dataset.

---

## 0. The two clips and why they are a good test

| | clipA | clipB |
|---|---|---|
| Scene | paved city street, dusk / night, headlights, many cars | unpaved rural road, daylight, grey sky |
| Size / fps | 496x368, 36 fps, 59.4 s | 848x478, 25 fps (duplicated frames), 15 s |
| What is hard | low light, low resolution, glare, thin cracks, manholes that look like holes | dozens of water-filled potholes of every size, mud colour = ground colour |
| What the generic detector expects | asphalt, daylight (RDD2022) | a paved road - it has never seen a dirt track |

The two clips pull the system in opposite directions, so a single threshold or model could never be right for both. That is
why the final design is *several small specialised models + temporal reasoning* instead of one big detector.

---

## 1. Decisions and the reasoning behind them

| Decision | Why | Alternatives we kept as challengers |
|---|---|---|
| **YOLO26s fine-tuned on RDD2022** as the base defect detector | fast (177 FPS TensorRT FP16 on a T4), NMS-free head -> exports cleanly to TensorRT/Jetson, RDD2022 is the largest public road-damage set with the 4 classes the task names | RF-DETR-S (prepared in notebook 02, not run - no time), YOLO11 |
| **Several small models instead of one** (road mask, surface, open-vocabulary, furniture) | each failure mode of the base detector (off-road false alarms, unpaved road, manholes, markings) is a *different* problem; a 20 MB detector cannot learn them without new labels | one large multimodal model (too slow for 25 FPS, hard to deploy on Jetson) |
| **Cityscapes SegFormer-B0 road mask** | removes the most common false positives (cars, kerbs, sky) for ~17 ms per frame; mask refreshed every 3 frames | geometric ROI (breaks on turns) |
| **ADE20K SegFormer-B0 surface segmentation** | gives paved / unpaved / water / sidewalk per frame -> "is the road paved, is there a sidewalk" findings, puddle detection | colour heuristics (mud vs asphalt fails at night) |
| **OWLv2 open-vocabulary** for manhole / drain grate | RDD2022 has no manhole class; a text prompt avoids labelling work | train a new class (no labelled data) |
| **Mapillary Mask2Former** for lane paint, crosswalks, signs, lights | the only public segmenter that knows those classes; runs every 4th frame at short side 576 | CLIP on crops (turned out to be noise on tiny crops) |
| **BoT-SORT + own track-merge pass** | the requirement is *report each defect once*; BoT-SORT has camera-motion compensation (the camera is moving), the merge pass joins fragments and drops one-frame blips | ByteTrack (similar in the sweep, slightly fewer tracks) |
| **Monocular visual odometry** for speed / route / map | no GPS or IMU in the clips; Lucas-Kanade on ground features + a flat-ground model gives approximate speed and a 2-D route with defect positions | none available without extra sensors |
| **Causal ("past-only") temporal reasoning** | a deployed system cannot see the future; stability must come from *previous* frames only (see section 4) | centred smoothing windows (look ahead -> not deployable) |

---

## 2. Timeline

### Day 1 - Wed 14/10 - set-up
* Read the brief, chose the architecture above, built the `roadscan` package.
* Videos uploaded to a **private** Kaggle dataset (chunked, rebuilt in the notebook) - code to a second dataset.
* Notebook 01 trains YOLO26s on RDD2022 (Kaggle GPU).

### Day 2 - Thu 15/10 - v1 results, review, v2 pipeline
* Training: ~76 epochs, RDD2022 val mAP50 **0.676** (pothole AP50 0.571).
* **v1 inference**: clipA 21 defects, clipB 28 (all potholes), ~25-28 FPS for detect+road+track.
* **Ablation sweep** (see `docs/images/fig_ablation.png`): confidence 0.35 and image size 960 lose real defects, tiled 480 finds more
  but is 2x slower, the road filter costs ~40 % of the time and changed nothing on these two clips (kept - it matters elsewhere),
  disabling track merge produces duplicates.
* **Review of v1** by eye: water-filled potholes (clipB) and manholes (clipA) were missed; no notion of paved/unpaved/sidewalk;
  nothing about speed or position. -> v2 pipeline: `surface.py`, `openvocab.py`, `odometry.py`, `mapping.py`, `vehicles.py`,
  new classes 4 and 5.
* **Benchmark** (T4): PyTorch FP16 78.6 FPS, TensorRT FP16 177.3 FPS (5.6 ms/frame), ONNX 8 FPS *because of a CPU fallback*
  (missing `libcublasLt` in the onnxruntime-gpu build - an environment issue, not a model issue).
* **Fine-tune v2** (27 epochs, lower LR, RDD only): mAP50 0.666 vs 0.676 - validation splits differ, so no clear winner;
  kept both, v1 is the safe default, v2 detects more on clipA.

### Day 3 - Fri 16/10 - road furniture + stability + polishing
* New ask: **markings, signs, traffic lights** (flag when damaged / faded) and **stable decisions** (no flicker).
* Built `furniture.py` and `temporal.py` (section 3-4).
* Ran on Kaggle (v6, 21.6 min), reviewed frame by frame, found and fixed the issues of section 5 (v7).
* Wrote this documentation and packaged the repository.

---

## 3. Road furniture (markings, signs, lights)

*Question*: with no labelled "damaged sign" data, how do we say anything honest?
*Answer*: measure condition cues that exist in the pixels, vote them over time, and call the result a **possible problem**,
never a certified defect.

| Item | Cue | Why it is robust enough |
|---|---|---|
| Lane marking / crosswalk | Weber contrast of the brightest quarter of the paint mask vs the surrounding asphalt, blended with the segmenter's own confidence, compared with a *running 85th-percentile reference of the same video* (+ an absolute floor) | the reference adapts to lighting; dashed lines are drawn by the segmenter as continuous strokes, so only the brightest quarter of the mask is judged |
| Traffic light | HSV lamp state (red / yellow / green / off) voted over time; flagged only after >= 8 *dark* observations and never having been lit | the back of a signal looks like a dead one, hence the strict rule |
| Sign | lean of the pole (PCA) or the face rectangle; CLIP zero-shot only for crops >= 40 px | CLIP on 10x24 px crops is pure noise (it said "hidden behind vegetation" for everything) - we measured that and disabled it for small crops |

Result on clipA: 33 signs, 62 traffic lights, 2 crosswalks inventoried; markings visible in 95 % of frames, **0 worn
stretches**, 1 traffic light flagged (see `results/clipA/road_furniture.csv`). The honest reading: the clip simply does not
contain much damaged furniture, and the false-positive risk is higher than the false-negative risk for such heuristics.

---

## 4. Stability - "the model must not flicker"

The request: *a defect seen in frame 1 and gone in frame 2 must not blink; speed must not go 40 -> 0 -> 40; use the previous
frames like an attention window but never future frames.*

Implemented in `roadscan/temporal.py` (24 unit tests in `tests/`):

| Mechanism | Used for |
|---|---|
| `TrackScore` / `TrackMemory` - EMA score + hysteresis (on 0.35 / off 0.20), min 2 hits, 3-frame coast | which tracks are *confirmed* and drawn |
| `TemporalPrior` | previous confirmed tracks **feed back into the next detection step**: confidence blend (weight 0.35, capped +0.15) and class vote inside the crack group - the "attention to the past" |
| `DisplaySmoother` | one box per defect, EMA-smoothed coordinates, coasting for 8 frames after a dropout, label shows best confidence so far |
| `SpeedFilter` | causal trimmed mean (1.6 s) -> EMA -> acceleration/deceleration limiter (2.5 / 4.5 m/s^2) with a start-up guard |
| `DwellState`, `hysteresis_class` | paved/unpaved/sidewalk change only after the new state persisted 1.5 s |
| `DecayedVotes` | class and lamp-state votes with exponential memory |

Trade-off (measured): state changes are reported ~1-2 s late. For a deployed system that is the price of stability.

---

## 5. Problems met and how each was fixed (chronological)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | RDD2022 download 403 | host refused direct download | figshare + range requests |
| 2 | Videos not found in Kaggle | dataset mount path differs, chunked upload | glob + rebuild chunks in the notebook |
| 3 | ONNX export failed | `/kaggle/input` read-only | copy weights to `/kaggle/working` |
| 4 | Missed water-filled potholes / manholes | classes not in RDD2022 | segmentation + puddle blobs, OWLv2; `absorb_water_overlaps` |
| 5 | "GPU session cap" | ghost "Cancelling" session | one GPU notebook at a time |
| 6 | Map heading drifted 67 deg on a straight road | yaw from noisy ground residuals | yaw from far-field features near the horizon |
| 7 | Inference used an OLD code copy | fine-tune notebook output contained a copy of the code | glob restricted to `/kaggle/input/datasets/**/roadscan/__init__.py` |
| 8 | Speed alternated 4.8 / 9.6 km/h on clipB | the clip has **duplicated frames** | trimmed mean instead of median (median was 20 % low) |
| 9 | CLIP "sign hidden" everywhere | tiny crops | only crops >= 40 px; ignore "hidden" |
| 10 | Marking contrast tiny on dashed lines | segmenter draws continuous strokes | judge the brightest quarter of the mask |
| 11 | **v6 review**: speed 140 km/h in the first seconds of clipA, mean 29 instead of ~25 km/h | first optical-flow sample was junk and seeded the filter | 6-sample warm-up before any output; longer window; test added |
| 12 | **v6 review**: speed still saw-tooth 40 -> 10 -> 40 | dark road = weak features = noisy raw speed; limiter alone only bounds slope | window 1.6 s, trim 30 %, EMA 0.2, limits 2.5 / 4.5 m/s^2 (total variation 10.5 -> 4.6 km/h per s on the same data) |
| 13 | **v6 review**: the same `#id` drawn on two boxes, boxes blink | several raw tracks merged into one report id, drawn raw | one box per id, EMA + 8-frame coast (`DisplaySmoother`) |
| 14 | **v6 review**: confusing colours (teal water, wide green strips, orange ground) | overlays had no key and overlapped | legend; water tint removed (the pothole box already says "water"); marking paint overlay only when *worn*; unpaved tint lighter |
| 15 | **v6 review**: dark manhole covers flagged as potholes at night | the pothole detector fires on dark round shapes | pothole tracks sitting on a manhole/drain track are absorbed into it |
| 16 | Paved/night recall felt low | RDD2022 is daylight; clipA is 496x368 at night | optional CLAHE ensemble on dark frames (+44 % raw detections in a 250-frame test; precision not measurable without labels) |

---

## 6. What we would do next

* Hand-label ~100 frames per clip -> real precision/recall instead of counts.
* Fine-tune on night / low-resolution city data and on water-filled potholes.
* Replace heuristics (sign damage, lamp state) by a small classifier trained on a few hundred crops.
* Metric scale for odometry from lane width instead of an assumed camera height.
* TensorRT FP16 end-to-end on a Jetson (see `docs/JETSON.md`): detector 5.6 ms, road mask every N frames, SAM only on best frames.
