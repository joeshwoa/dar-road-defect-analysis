#!/usr/bin/env python3
"""Fine-tune RF-DETR (Apache-2.0) as the challenger detector.

It reads the same YOLO-format dataset prepared by scripts/prepare_data.py
(OUT/train/{images,labels}, OUT/valid/{images,labels}, OUT/data.yaml).

Kaggle (1x T4):  python scripts/train_rfdetr.py --data /kaggle/temp/yolo_rdd --output /kaggle/working/rfdetr_run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir containing data.yaml")
    ap.add_argument("--output", default="rfdetr_run")
    ap.add_argument("--variant", default="small", choices=["nano", "small", "medium", "base"])
    ap.add_argument("--resolution", type=int, default=640, help="must be divisible by 32 for the small/medium models")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--copy-best-to", default="")
    a = ap.parse_args()

    import rfdetr

    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
           "base": rfdetr.RFDETRBase}[a.variant]
    model = cls(resolution=a.resolution)
    kw = dict(
        dataset_dir=a.data, dataset_file="yolo", output_dir=a.output, epochs=a.epochs, batch_size=a.batch,
        grad_accum_steps=a.accum, lr=a.lr, num_workers=a.workers, early_stopping=True,
        early_stopping_patience=a.patience,
    )
    if a.resume:
        kw["resume"] = a.resume
    model.train(**kw)

    out = Path(a.output)
    cands = [out / n for n in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth", "checkpoint_best_regular.pth")]
    best = next((c for c in cands if c.exists()), None)
    print("best checkpoint:", best)
    if best and a.copy_best_to:
        Path(a.copy_best_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, a.copy_best_to)
        print("copied to", a.copy_best_to)


if __name__ == "__main__":
    main()
