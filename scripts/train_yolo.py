#!/usr/bin/env python3
"""Fine-tune YOLO26 (or YOLO11 / YOLOv8) on the prepared road-defect dataset.

Kaggle (1x T4):   python scripts/train_yolo.py --data /kaggle/temp/yolo_rdd/data.yaml --project /kaggle/working/runs
Resume after a session cut:  add --resume (uses <project>/<name>/weights/last.pt)

Augmentation notes (targets are low-res, compressed, often night footage):
  hsv_v / hsv_s raised for lighting robustness, scale 0.5 for size variety, no vertical flip.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26s.pt", help="COCO-pretrained checkpoint (yolo26s.pt, yolo11s.pt, ...)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hours", type=float, default=0.0, help="wall-clock cap in hours (overrides epochs); 0 = off")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="'0' (one T4) or '0,1' (DDP on both T4s)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolo26s_rdd")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--copy-best-to", default="", help="copy best.pt here when finished")
    a = ap.parse_args()

    from ultralytics import YOLO

    run_dir = Path(a.project) / a.name
    if a.resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        kw = dict(
            data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, workers=a.workers,
            project=a.project, name=a.name, patience=a.patience, cos_lr=True, close_mosaic=8, amp=True,
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.6, degrees=3.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.0,
            mosaic=1.0, mixup=0.0, plots=True, save_period=5, exist_ok=True,
        )
        if a.hours > 0:
            kw["time"] = a.hours
        model.train(**kw)

    best = run_dir / "weights" / "best.pt"
    print("best weights:", best)
    m = YOLO(str(best))
    metrics = m.val(data=a.data, imgsz=a.imgsz, device=a.device.split(",")[0], plots=True, verbose=False,
                    project=str(run_dir), name="final_val", exist_ok=True)
    # Only classes that actually occur in the validation split have a real AP (maps[] would repeat the mean)
    per_class = {m.names[i]: "no validation instances" for i in range(len(m.names))}
    for j, ci in enumerate(metrics.box.ap_class_index):
        per_class[m.names[int(ci)]] = {"AP50": float(metrics.box.ap50[j]), "AP50-95": float(metrics.box.ap[j])}
    summary = {
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map), "per_class": per_class,
    }
    (run_dir / "val_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.copy_best_to:
        Path(a.copy_best_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, a.copy_best_to)
        print("copied to", a.copy_best_to)


if __name__ == "__main__":
    main()
