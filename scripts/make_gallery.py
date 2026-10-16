"""Frame galleries from the annotated videos (needs ffmpeg).

    python scripts/make_gallery.py --results results --prev results_history/v5 --out docs/images

Writes: gallery_clipA.jpg, gallery_clipB.jpg (4 evenly spaced frames each), before_after_clipA.jpg / _clipB.jpg
(previous run on the left, current on the right) and top_crops.jpg (the highest-confidence crops of every class).
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def grab(video: Path, t: float) -> Image.Image | None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "f.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(out)],
                       check=False)
        return Image.open(out).convert("RGB") if out.exists() else None


def duration(video: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def caption(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, img.height - 18, img.width, img.height], fill=(0, 0, 0))
    d.text((6, img.height - 15), text, fill=(255, 255, 255))
    return img


def grid(ims: list[Image.Image], cols: int, width: int = 1500) -> Image.Image:
    cw = width // cols
    ims = [im.resize((cw, int(im.height * cw / im.width))) for im in ims]
    rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
    H = sum(max(i.height for i in r) for r in rows)
    sheet = Image.new("RGB", (cw * cols, H), (255, 255, 255))
    y = 0
    for r in rows:
        for j, im in enumerate(r):
            sheet.paste(im, (j * cw, y))
        y += max(i.height for i in r)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--prev", default=None)
    ap.add_argument("--out", default="docs/images")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    prev = Path(a.prev) if a.prev else None
    out.mkdir(parents=True, exist_ok=True)
    for clip in ("clipA", "clipB"):
        v = res / clip / "annotated.mp4"
        if not v.exists():
            continue
        dur = duration(v)
        ts = [dur * f for f in (0.15, 0.38, 0.62, 0.85)]
        ims = [caption(im, f"{clip}  t={t:.1f}s") for t in ts if (im := grab(v, t)) is not None]
        if ims:
            grid(ims, 2).save(out / f"gallery_{clip}.jpg", quality=90)
        if prev is not None and (prev / clip / "annotated.mp4").exists():
            pairs = []
            for t in ts[1:3]:
                l, r = grab(prev / clip / "annotated.mp4", t), grab(v, t)
                if l is not None and r is not None:
                    pairs += [caption(l, "before"), caption(r, "after")]
            if pairs:
                grid(pairs, 2).save(out / f"before_after_{clip}.jpg", quality=90)
    # top crops per class
    crops = []
    for clip in ("clipA", "clipB"):
        d = res / clip / "defects.csv"
        if not d.exists():
            continue
        df = pd.read_csv(d)
        for t, g in df.groupby("defect_type"):
            for _, row in g.sort_values("max_confidence", ascending=False).head(2).iterrows():
                p = res / clip / str(row["crop"])
                if p.exists():
                    im = Image.open(p).convert("RGB")
                    im = im.resize((220, int(im.height * 220 / im.width))) if im.width else im
                    crops.append(caption(im.crop((0, 0, 220, min(im.height, 220))), f"{t} {row['max_confidence']:.2f}"))
    if crops:
        grid(crops[:24], 6, width=1320).save(out / "top_crops.jpg", quality=90)
    print("gallery written to", out)


if __name__ == "__main__":
    main()
