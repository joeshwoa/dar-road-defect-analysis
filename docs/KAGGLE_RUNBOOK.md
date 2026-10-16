# Kaggle runbook

Deadline: **Sat 17/10/2026, 14:28 Cairo time.** Plan for the first training run to start tonight.

## 0. One-time preparation (10 minutes)

1. **Zip the code.** In the folder that contains `road-defect-analyzer/`:
   `zip -r roadscan-code.zip road-defect-analyzer -x "*/__pycache__/*" "*/outputs/*"`
   (a ready-made `roadscan-code.zip` is also provided next to this folder).
2. **Upload it as a private dataset.** kaggle.com -> Datasets -> New Dataset -> drop `roadscan-code.zip`
   -> name `roadscan-code` -> Visibility **Private** -> Create. Kaggle unzips it automatically.
3. **(Recommended) extra pothole data for water-filled potholes.** Video 2 is full of water-filled potholes, which RDD2022
   barely contains. Download *"An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications"*
   (Mendeley Data, DOI 10.17632/tp95cdvgm8.1, file `Potholes.zip`, Pascal-VOC XML included) and upload it as a private
   dataset named `water-filled-potholes`. The Mendeley page does not state a licence: say so in the README.
   Then set `EXTRA_DATASETS = ["/kaggle/input/water-filled-potholes"]` in notebooks 01 and 02.
   The converter prints every label name it sees, and maps anything containing "pothole" to the pothole class.
4. **Videos.** Create a *separate* private dataset `dar-videos` with the two `.mp4` files only when you reach
   step 3 below (inference). They are confidential: keep the dataset private, never attach it to a public notebook,
   delete it after you have submitted.

## 1. Train the main detector (notebook 01) - start this first

1. Kaggle -> Code -> New Notebook -> File -> **Import Notebook** -> `notebooks/01_train_yolo26_kaggle.ipynb`.
2. Right panel -> **Add Input** -> `roadscan-code` (and `water-filled-potholes` if you made it).
3. Settings: **Accelerator = GPU T4 x2**, **Internet = On**, Persistence = *Files only*.
4. Run the first two cells interactively once (setup + data) to catch typos, then **Save Version -> Save & Run All (Commit)**.
   A committed run continues in the background even if you close the browser/laptop.
5. Expected: ~30-45 min download + conversion, then up to 6 h of training (`--hours 6` caps it; Ultralytics fits the
   learning-rate schedule to that budget). The result is `weights/yolo26s_rdd.pt` in the notebook's **Output** tab
   plus `runs/yolo26s_rdd/val_summary.json` (precision, recall, mAP50, mAP50-95, per-class AP).

## 2. Train the challenger (notebook 02) - after 01 finishes (or in parallel on another account/session)

Same steps with `02_train_rfdetr_kaggle.ipynb`. Look at the time of the **first epoch** in the log and change `EPOCHS`
so the run fits in ~6 h. Output: `weights/rfdetr_small_rdd.pth`.

## 3. Run on the videos (notebook 03)

1. Upload the two videos as the private dataset `dar-videos`.
2. Import `notebooks/03_infer_videos_kaggle.ipynb`. **Add Input**: `roadscan-code`, `dar-videos`, and the *output of
   notebook 01* (Add Input -> Notebook Output Files -> your notebook 01). GPU on, Internet on.
3. Run all cells. You get, per video: `annotated.mp4`, `defects.csv`, `defects.json`, `images/` (one annotated best
   frame per defect, SAM 2 mask overlay), `crops/`, `run_info.json` (FPS, hardware, per-stage timing).
   The last cell zips everything to `submission_outputs.zip`.
4. The FPS to quote is `processing_fps` in `run_info.json`, with the GPU name from the same file
   (Kaggle T4). The TensorRT cell writes `benchmark_t4.json` with PyTorch/ONNX/TensorRT numbers.

## 4. Look at the results, then iterate (this is where the accuracy comes from)

* Open `images/` and `annotated.mp4`. Note false positives (manholes, patches, shadows, lane markings, puddles that are
  not potholes) and misses. These go straight into the README's "errors" section.
* Tune with `--set key=value` and re-run, e.g. `--set detect.imgsz=960`, `--set detect.tile_size=480`,
  `--set road.min_overlap=0.4`, `--set report.min_max_conf=0.5`, `--set track.tracker=bytetrack`.
* `scripts/sweep.py` runs the standard variants in one go and writes `sweep_summary.csv`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `wget` for a country fails or the zip is truncated | re-run the cell (it resumes with `-c`); or drop Norway from `COUNTRIES` (9.9 GB) |
| "No space left on device" | lower `MAX_PER_COUNTRY`, or drop Norway |
| CUDA out of memory | `--batch 16` |
| `ModuleNotFoundError: roadscan` | the `roadscan-code` dataset is not attached, or was zipped without the inner folder |
| session killed before the end | use *Save & Run All (Commit)*; re-run with `--hours` smaller |
| TensorRT export fails on Kaggle | the benchmark script keeps going and records the error; PyTorch/ONNX numbers are still written |

## Confidentiality checklist

- [ ] `dar-videos` is **Private**
- [ ] notebooks that read it are **Private**
- [ ] outputs that contain frames from the videos are only shared with the company
- [ ] after submitting: delete the `dar-videos` dataset and any notebook version that contains its frames
