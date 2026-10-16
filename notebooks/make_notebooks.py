#!/usr/bin/env python3
"""Generate the three Kaggle notebooks (.ipynb) from the cell lists below.

    python notebooks/make_notebooks.py          # writes notebooks/*.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP = code('''
# --- locate the code dataset, copy it to /kaggle/working and install pinned deps ---
import glob, io, os, shutil, subprocess, sys
REPO = "/kaggle/working/road-defect-analyzer"
if not os.path.exists(REPO):
    # only the real dataset (never a copy of the code inside another notebook's output)
    hits = (glob.glob("/kaggle/input/datasets/**/roadscan/__init__.py", recursive=True)
            or glob.glob("/kaggle/input/**/roadscan/__init__.py", recursive=True))
    assert hits, "Add the 'roadscan-code' dataset via 'Add Input' (see docs/KAGGLE_RUNBOOK.md)"
    shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), REPO)
os.chdir(REPO)
!pip install -q ultralytics==8.4.177 lapx "transformers>=4.44" matplotlib
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')

PREP = code('''
# --- download RDD2022 (CC BY 4.0, official figshare release) and convert it to one YOLO dataset ---
# The figshare file is a zip of per-country zips stored uncompressed (13.3 GB, 10.6 GB of it Norway), so we fetch
# only the wanted countries with HTTP range requests (~2.6 GB) and keep just their labelled *train* folders.
import subprocess, struct, zipfile, shutil, requests
RAW, OUT = "/kaggle/temp/rdd_raw", "/kaggle/temp/yolo_rdd"
ZIP_URL = "https://ndownloader.figshare.com/files/38030910"      # RDD2022_released_through_CRDDC2022.zip
COUNTRIES = ["Japan", "India", "Czech", "United_States", "China_MotorBike", "China_Drone"]   # Norway skipped: 10.6 GB
EXTRA_DATASETS = []   # e.g. ["/kaggle/input/water-filled-potholes"]  (Pascal-VOC XML + images)
MAX_PER_COUNTRY = {max_per}

os.makedirs(RAW, exist_ok=True)
free_gb = shutil.disk_usage("/kaggle/temp").free / 1e9
print(f"free space in /kaggle/temp: {free_gb:.0f} GB")
assert free_gb > 15, "not enough scratch disk"

def fresh_url():   # figshare redirects to a presigned URL that expires in ~10 s, so ask again for every request
    return requests.get(ZIP_URL, allow_redirects=False, timeout=60).headers["Location"]
def get_range(a, b):
    r = requests.get(fresh_url(), headers={"Range": f"bytes={a}-{b}"}, timeout=300)
    r.raise_for_status(); return r
total = int(get_range(0, 0).headers["Content-Range"].split("/")[1])
print("remote zip size:", total)

class RangeFile(io.RawIOBase):          # lets zipfile read the central directory without downloading the file
    def __init__(self): self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else total + off
        return self.pos
    def readinto(self, buf):
        if not len(buf) or self.pos >= total: return 0
        data = get_range(self.pos, min(total - 1, self.pos + len(buf) - 1)).content
        buf[:len(data)] = data; self.pos += len(data); return len(data)

outer = zipfile.ZipFile(io.BufferedReader(RangeFile(), buffer_size=4 << 20))
members = {os.path.basename(i.filename): i for i in outer.infolist()}
print("members in the outer zip:", {k: round(v.file_size / 1e6) for k, v in members.items()}, "MB")

RDD_ROOT = f"{RAW}/RDD2022"
for c in COUNTRIES:
    info = members[f"{c}.zip"]
    assert info.compress_type == 0, "expected an uncompressed (stored) member"
    hdr = get_range(info.header_offset, info.header_offset + 29).content      # local file header
    start = info.header_offset + 30 + struct.unpack("<HH", hdr[26:30])[0] + struct.unpack("<HH", hdr[26:30])[1]
    cz = f"{RAW}/{c}.zip"
    # -f: fail on HTTP errors (the first run saved an error page as a .zip); each retry re-resolves the redirect
    for attempt in range(5):
        r = subprocess.run(["curl", "-fsSL", "-r", f"{start}-{start + info.file_size - 1}", "-o", cz, ZIP_URL])
        if r.returncode == 0 and os.path.getsize(cz) == info.file_size and zipfile.is_zipfile(cz):
            break
        print(f"  retry {attempt + 1} for {c}")
    else:
        raise RuntimeError(f"could not download {c}")
    with zipfile.ZipFile(cz) as zf:
        zf.extractall(RDD_ROOT, [n for n in zf.namelist() if n.startswith(f"{c}/train/")])
    os.remove(cz)
    print(c, "ready")

n_xml = sum(1 for _ in glob.iglob(f"{RDD_ROOT}/*/train/annotations/xmls/*.xml"))
print("annotation files found:", n_xml)
assert n_xml > 1000, "no RDD2022 annotations found - stopping before wasting a training run"

extra = " ".join(f"--extra extra{i}={p}" for i, p in enumerate(EXTRA_DATASETS))
subprocess.run(f"python scripts/prepare_data.py --rdd-root {RDD_ROOT} --out {OUT} {extra} "
               f"--max-per-country {MAX_PER_COUNTRY} --max-side 1024 --workers 4", shell=True, check=True)
shutil.rmtree(RAW, ignore_errors=True)   # free disk
n_train = len(glob.glob(f"{OUT}/train/images/*.jpg")); n_val = len(glob.glob(f"{OUT}/valid/images/*.jpg"))
print("train images:", n_train, "| valid images:", n_val)
assert n_train > 1000 and n_val > 100, "dataset conversion produced (almost) no images - stopping"
''')




def build_train_yolo() -> dict:
    return notebook([
        md('''
# 01 - Train the road-defect detector (YOLO26s) on Kaggle
**Settings (right panel):** Accelerator = GPU T4 x2 (one is used), Internet = On, Persistence = Files only.
Run it with **Save Version -> Save & Run All (Commit)** so it keeps going in the background (the session is not tied to your browser).
Outputs: `/kaggle/working/weights/yolo26s_rdd.pt`, training curves, `runs/yolo26s_rdd/val_summary.json`.
'''),
        SETUP,
        code("".join(PREP["source"]).replace("{max_per}", "5000")),
        code('''
# --- train: YOLO26s, COCO-pretrained -> RDD2022. `--hours 6` makes Ultralytics fit its schedule to the time budget ---
!python scripts/train_yolo.py --data {OUT}/data.yaml --model yolo26s.pt --imgsz 640 --batch 32 --device 0 \\
    --hours 6 --patience 15 --project /kaggle/working/runs --name yolo26s_rdd \\
    --copy-best-to /kaggle/working/weights/yolo26s_rdd.pt
'''),
        code('''
import json
print(json.dumps(json.load(open("/kaggle/working/runs/yolo26s_rdd/val_summary.json")), indent=2))
from IPython.display import Image, display
for f in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
    for p in (f"/kaggle/working/runs/yolo26s_rdd/{f}", f"/kaggle/working/runs/yolo26s_rdd/final_val/{f}"):
        if os.path.exists(p):
            display(Image(p)); break
'''),
    ])


def build_train_rfdetr() -> dict:
    prep = "".join(PREP["source"]).replace("{max_per}", "3000")
    return notebook([
        md('''
# 02 - Train the challenger detector (RF-DETR-S) on Kaggle
Same data as notebook 01 (fewer images per country because transformer training is slower).
**Check the time of the first epoch**, then adjust `EPOCHS` so the run fits in ~6-7 hours.
Outputs: `/kaggle/working/weights/rfdetr_small_rdd.pth`.
'''),
        SETUP,
        code("!pip install -q 'rfdetr[train,loggers]>=1.11'"),
        code(prep),
        code('''
EPOCHS = 15
!python scripts/train_rfdetr.py --data {OUT} --output /kaggle/working/rfdetr_run --variant small --resolution 640 \\
    --epochs {EPOCHS} --batch 8 --accum 2 --copy-best-to /kaggle/working/weights/rfdetr_small_rdd.pth
'''),
    ])


def build_infer() -> dict:
    return notebook([
        md('''
# 03 - Run the pipeline on the videos (Kaggle GPU) and collect the deliverables
**Add inputs:** (1) `roadscan-code`, (2) your **private** videos dataset, (3) the weights - either the output of notebook 01
or a dataset containing `yolo26s_rdd.pt`.  **Internet = On** (SegFormer / SAM 2 weights download).
Keep the videos dataset private and delete it from Kaggle after you have submitted.
'''),
        SETUP,
        code('''
import shlex
assert os.path.exists(REPO + "/roadscan/furniture.py"), "an OLD version of the roadscan-code dataset is attached - re-add it"
# Large files were split into byte-exact chunks (name.mp4.part0, part1, ...) to pass the upload limit.
# Rebuild them into normal .mp4 files; the result is byte-identical to the originals.
os.makedirs("/kaggle/working/videos", exist_ok=True)
groups = {}
for part in glob.glob("/kaggle/input/**/*.mp4.part*", recursive=True):
    groups.setdefault(part.rsplit(".part", 1)[0], []).append(part)
for base, parts in groups.items():
    with open("/kaggle/working/videos/" + os.path.basename(base), "wb") as out:
        for part in sorted(parts, key=lambda q: int(q.rsplit(".part", 1)[1])):
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)
    print("rebuilt", os.path.basename(base), "from", len(parts), "chunks")
videos = sorted(glob.glob("/kaggle/input/**/*.mp4", recursive=True)
                + glob.glob("/kaggle/working/videos/*.mp4"))
# prefer the continued-training weights (notebook 04) when present
WEIGHTS = ((glob.glob("/kaggle/input/**/yolo26s_rdd_v2.pt", recursive=True)
            or glob.glob("/kaggle/input/**/yolo26s_rdd.pt", recursive=True)) or [None])[0]
print("videos :", *videos, sep="\\n  ")
print("weights:", WEIGHTS)
assert videos and WEIGHTS
V = " ".join(shlex.quote(v) for v in videos)
'''),
        code('''
# --- main run: defects + road surface (paved/unpaved/sidewalk) + water-filled potholes + manholes/drains
#     + lane markings / crosswalks / signs / traffic lights (condition) + vehicles + route map.
#     All temporal smoothing is causal (past frames only). Higher input size and a lower confidence floor favour recall (the tracker + report
#     filter then decide what survives). Camera height is only used for the schematic map (1.4 m dashcam, 1.0 m low camera).
for v in videos:
    cam = 1.0 if "(1)" in os.path.basename(v) else 1.4
    !python scripts/run_video.py {shlex.quote(v)} --weights {WEIGHTS} --sam --scene --openvocab --furniture \\
        --set detect.imgsz=800 --set detect.conf=0.15 --cam-height {cam} --out /kaggle/working/outputs
'''),
        code('''
import json, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for d in sorted(glob.glob("/kaggle/working/outputs/*/")):
    info = json.load(open(d + "run_info.json"))
    print("\\n==", info["video"])
    print("FPS (detect+road+track):", info["timing"]["processing_fps"], "| all models:", info["timing"].get("processing_fps_all_models"),
          "| end-to-end pass 1:", info["timing"]["end_to_end_fps_pass1"])
    print("road surface:", json.dumps(info.get("road_surface", {}).get("findings") if info.get("road_surface") else None))
    print("odometry:", info.get("odometry"))
    print("temporal:", info.get("temporal"))
    print("furniture:", json.dumps((info.get("furniture") or {}).get("marking", {}).get("mean_visibility")), "inventory items:",
          (info.get("furniture") or {}).get("inventory_items"))
    print("hardware:", info["hardware"]["gpus"], "|", info["hardware"]["cpu"])
    print("tracks:", info["tracking"], "| road filter:", info["road_filter"])
    print(pd.read_csv(d + "defects.csv")[["track_id","defect_type","first_seen_timecode","max_confidence","water_filled","surface_at_defect","map_x_m","map_y_m"]].head(60))
'''),
        md("### Optional: ablations (counts + speed for each variant) and a TensorRT benchmark"),
        code('''
RUN_SWEEP = False      # ablations were already produced in an earlier run; set True to repeat them (~20+ min)
if RUN_SWEEP:
    !python scripts/sweep.py {V} --weights {WEIGHTS} --out /kaggle/working/outputs_sweep
'''),
        code('''
shutil.copy(WEIGHTS, "/kaggle/working/bench_weights.pt")      # export writes next to the weights: /kaggle/input is read-only
!python scripts/benchmark.py {shlex.quote(videos[0])} --weights /kaggle/working/bench_weights.pt --formats pt onnx engine --out /kaggle/working/benchmark_t4.json
'''),
        md("### Optional: RF-DETR challenger (needs `rfdetr_small_rdd.pth` from notebook 02)"),
        code('''
RF = (glob.glob("/kaggle/input/**/rfdetr_small_rdd.pth", recursive=True) or [None])[0]
if RF:
    !pip install -q 'rfdetr[train,loggers]>=1.11'
    !python scripts/run_video.py {V} --backend rfdetr --weights {RF} --out /kaggle/working/outputs_rfdetr
'''),
        code('''
shutil.make_archive("/kaggle/working/submission_outputs", "zip", "/kaggle/working/outputs")
print(os.path.getsize("/kaggle/working/submission_outputs.zip") / 1e6, "MB")
'''),
    ])


def build_finetune() -> dict:
    prep = "".join(PREP["source"]).replace("{max_per}", "1300")
    prep = prep.replace('EXTRA_DATASETS = []', 'EXTRA_DATASETS = EXTRA_FROM_KAGGLE')
    extras = code('''
# --- optional extra pothole datasets (public Kaggle sets; each is skipped if it cannot be fetched) ---
# Water-filled / wet-road potholes are under-represented in RDD2022, so extra pothole photos help recall.
EXTRA_FROM_KAGGLE = []
CANDIDATES = ["sovitrath/road-pothole-images-for-pothole-detection", "chitholian/annotated-potholes-dataset"]
try:
    import kagglehub
except ImportError:
    !pip install -q kagglehub
    import kagglehub
for slug in CANDIDATES:
    try:
        d = kagglehub.dataset_download(slug)
        n_xml = len(glob.glob(d + "/**/*.xml", recursive=True))
        print(slug, "->", d, "| xml files:", n_xml)
        if n_xml > 20:
            EXTRA_FROM_KAGGLE.append(d)
    except Exception as e:
        print("skip", slug, "->", repr(e)[:200])
print("extra datasets used:", EXTRA_FROM_KAGGLE)
''')
    return notebook([
        md('''
# 04 - CONTINUE training the road-defect detector (fine-tune from the existing weights, ~1.5 h cap)
**Add inputs:** `roadscan-code` and the output of notebook 01 (contains `weights/yolo26s_rdd.pt`). Internet = On, GPU T4.
Does NOT train from scratch: starts from `yolo26s_rdd.pt`, lower learning rate, slightly higher resolution, extra pothole photos.
Output: `/kaggle/working/weights/yolo26s_rdd_v2.pt`.
'''),
        SETUP,
        extras,
        code(prep),
        code('''
import json
from ultralytics import YOLO
START = [p for p in glob.glob("/kaggle/input/**/yolo26s_rdd.pt", recursive=True)][0]
print("starting from", START)
HOURS = 1.4
model = YOLO(START)
model.train(data=f"{OUT}/data.yaml", imgsz=800, epochs=12, time=HOURS, batch=16, device=0, workers=4,
            project="/kaggle/working/runs", name="yolo26s_rdd_v2", exist_ok=True, patience=6,
            optimizer="AdamW", lr0=0.0004, lrf=0.1, warmup_epochs=0.5, cos_lr=True, close_mosaic=3, amp=True,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.6, degrees=4.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.0,
            mosaic=1.0, mixup=0.05, plots=True, save_period=2)
best = "/kaggle/working/runs/yolo26s_rdd_v2/weights/best.pt"
os.makedirs("/kaggle/working/weights", exist_ok=True)
shutil.copy(best, "/kaggle/working/weights/yolo26s_rdd_v2.pt")
m = YOLO(best)
mt = m.val(data=f"{OUT}/data.yaml", imgsz=800, device=0, plots=True, verbose=False,
           project="/kaggle/working/runs/yolo26s_rdd_v2", name="final_val", exist_ok=True)
per = {m.names[int(c)]: {"AP50": float(mt.box.ap50[j]), "AP50-95": float(mt.box.ap[j])} for j, c in enumerate(mt.box.ap_class_index)}
summary = {"precision": float(mt.box.mp), "recall": float(mt.box.mr), "mAP50": float(mt.box.map50),
           "mAP50-95": float(mt.box.map), "per_class": per}
json.dump(summary, open("/kaggle/working/weights/val_summary_v2.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
'''),
        code('''
from IPython.display import Image, display
for f in ("results.png", "confusion_matrix_normalized.png"):
    for p in (f"/kaggle/working/runs/yolo26s_rdd_v2/{f}", f"/kaggle/working/runs/yolo26s_rdd_v2/final_val/{f}"):
        if os.path.exists(p):
            display(Image(p)); break
'''),
    ])


if __name__ == "__main__":
    for name, nb in (("01_train_yolo26_kaggle.ipynb", build_train_yolo()),
                     ("02_train_rfdetr_kaggle.ipynb", build_train_rfdetr()),
                     ("03_infer_videos_kaggle.ipynb", build_infer()),
                     ("04_finetune_yolo26_kaggle.ipynb", build_finetune())):
        (HERE / name).write_text(json.dumps(nb, indent=1))
        print("wrote", name)
