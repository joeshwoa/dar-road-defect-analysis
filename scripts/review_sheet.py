#!/usr/bin/env python3
"""Make contact sheets of every reported defect so a human can label them quickly.

    python scripts/review_sheet.py outputs/<video_name>          # writes review_sheet_*.jpg + review.csv

Fill the `verdict` column of review.csv with one of:  tp (real defect) | fp (not a defect) | dup (same defect as
another ID) | unsure. Then run scripts/score_review.py to get precision / duplicate rate per class.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell", type=int, default=360, help="cell width in px")
    a = ap.parse_args()

    run = Path(a.run_dir)
    df = pd.read_csv(run / "defects.csv")
    cw = a.cell
    ch = int(cw * 0.75)
    per_page = a.cols * a.rows
    for p0 in range(0, len(df), per_page):
        page = np.full((a.rows * (ch + 26), a.cols * cw, 3), 30, np.uint8)
        for k, (_, r) in enumerate(df.iloc[p0:p0 + per_page].iterrows()):
            img = cv2.imread(str(run / r["image"])) if isinstance(r["image"], str) and r["image"] else None
            if img is None:
                continue
            h, w = img.shape[:2]
            s = min(cw / w, ch / h)
            img = cv2.resize(img, (int(w * s), int(h * s)))
            y, x = (k // a.cols) * (ch + 26), (k % a.cols) * cw
            page[y + 26:y + 26 + img.shape[0], x:x + img.shape[1]] = img
            cap = f"#{int(r['track_id'])} {r['defect_type']} {r['max_confidence']:.2f} @ {r['first_seen_timecode']}"
            cv2.putText(page, cap, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(run / f"review_sheet_{p0 // per_page + 1:02d}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 90])
    tmpl = df[["track_id", "defect_type", "max_confidence", "first_seen_timecode"]].copy()
    tmpl["verdict"] = ""
    tmpl["notes"] = ""
    out = run / "review.csv"
    if not out.exists():
        tmpl.to_csv(out, index=False)
    print(f"wrote {-(-len(df) // per_page)} sheet(s) and {out}")


if __name__ == "__main__":
    main()
