# Running the pipeline on an NVIDIA Jetson

Target: Jetson Orin Nano / Orin NX / AGX Orin on JetPack 6.x (TensorRT 10.x). Facts below come from the Ultralytics
Jetson and DeepStream guides; the numbers marked *reference* are Ultralytics' YOLO26**n** TensorRT latencies at 640 px
(model only, no pre/post-processing). Our YOLO26**s** is larger, so expect it to be slower than these. **Nothing here has been
measured by us on a Jetson**; the report quotes our own numbers only for the Kaggle T4.

| Device (reference, YOLO26n, 640) | FP32 | FP16 | INT8 |
|---|---|---|---|
| AGX Orin 64GB | 4.18 ms | 2.62 ms | 2.30 ms |
| Orin NX 16GB | 7.01 ms | 4.13 ms | 3.49 ms |
| Orin Nano Super | 7.53 ms | 4.57 ms | 3.80 ms |

## 1. Install (on the Jetson)

- JetPack 6.1+: `pip install ultralytics`, then the NVIDIA-built `torch`/`torchvision` wheels for JetPack and a matching
  `onnxruntime-gpu` wheel (a stock PyPI PyTorch has no CUDA on Jetson). Or use the Ultralytics Jetson docker image.
- `pip install -r requirements.txt` for the rest (`opencv-python-headless` is fine; use the JetPack OpenCV if you want
  GStreamer/NVDEC capture).
- Max performance: `sudo nvpmodel -m 0 && sudo jetson_clocks`; watch temperature/utilisation with `jtop`.

## 2. Build the TensorRT engine **on the device** (engines are device- and TensorRT-version-specific)

```bash
# FP16 (first choice: negligible accuracy change, ~1.5-1.7x faster than FP32 per the reference table)
yolo export model=yolo26s_rdd.pt format=engine half=True imgsz=640 device=0
# INT8: needs ~500-1000 representative frames (include night, wet road, low-res clips) for calibration;
# re-run the validation split afterwards and keep INT8 only if per-class AP does not drop noticeably.
yolo export model=yolo26s_rdd.pt format=engine int8=True imgsz=640 data=data.yaml device=0
```

Then point the pipeline at the engine: `--weights yolo26s_rdd.engine`. `scripts/benchmark.py` measures PyTorch vs ONNX
vs TensorRT on real video frames and writes the JSON to quote.

## 3. Make the *whole* pipeline fit, not only the detector

| Stage | Cost on a Jetson | What to do |
|---|---|---|
| Video decode | CPU if OpenCV default | use GStreamer + NVDEC (`nvv4l2decoder`) for capture |
| Defect detector | dominant GPU cost | TensorRT FP16/INT8 engine; keep `imgsz=640`; tiling only if accuracy needs it |
| Road-surface mask (SegFormer-B0) | second GPU model | run every 3-5 frames (`road.every_n`) at 512-640 px width, TensorRT FP16 (or drop it and use a fixed trapezoid ROI) |
| Tracker (BoT-SORT) | CPU, light | keep `gmc_method: sparseOptFlow`; it works on a down-scaled frame |
| SAM 2 masks | too heavy for real time | run only on each defect's best frame, after the track is closed, or switch to MobileSAM; disable for pure real-time mode |
| Scene detector (bonus) | extra GPU model | share the batch with the defect model, or run every 2nd frame (`stride`) |

Practical settings: `--set stride=2` (process every 2nd frame; the tracker is tuned in processed frames),
`--set detect.half=true`, headless run (`render.enabled=false`) and write only the CSV/JSON/crops in the field.

## 4. DeepStream alternative

Export ONNX with `nms=False`, build the engine through `nvinfer` (FP16 via `network-mode=2`), and use `nvtracker` for
tracking. The Ultralytics DeepStream guide covers detection only; the track-merge logic in `roadscan/tracking.py` would
then run as a small Python probe on the tracker output. DeepStream gives the best multi-stream throughput, but the Python
loop above is simpler to maintain and is what we benchmark.

## 5. Validate on the device

1. Run `scripts/benchmark.py` (latency/FPS per format) and `scripts/run_video.py` on the two reference videos.
2. Compare `defects.csv` against the Kaggle run: the number of unique defects should match within a small margin; a large
   change means the INT8 calibration or a pre-processing difference is hurting.
3. Log `jtop` power/thermal state: sustained FPS on a hot, power-limited Jetson can be lower than a cold benchmark.
