#!/usr/bin/env python3
"""Convert RDD2022 (Pascal-VOC XML) and optional extra pothole sets to one YOLO dataset.

Output layout (works for Ultralytics AND RF-DETR's YOLO loader):
    OUT/data.yaml
    OUT/train/images, OUT/train/labels
    OUT/valid/images, OUT/valid/labels

Design choices
  * cap images per country so Norway/Japan do not swamp the mix (domain balance -> better generalisation)
  * long side resized to --max-side: the target videos are low-resolution, and it saves disk/time
  * validation split is grouped by consecutive image number (neighbouring frames stay together),
    which reduces train/val leakage. RDD2022 has no sequence metadata, so this is a best effort.
  * a few label-free images are kept as negatives to teach the model what *not* to fire on
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadscan.classes import DEFECT_NAMES, RDD_CODE_TO_ID, normalise_name_to_id

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- discovery
def index_images(root: Path) -> dict[tuple[str, str], Path]:
    """(source, stem) -> image path. `source` = first folder below root (e.g. Japan)."""
    idx: dict[tuple[str, str], Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            src = p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else root.name
            key = (src, p.stem)
            if key not in idx or "train" in p.parts:   # prefer train copies if a stem repeats
                idx[key] = p
    return idx


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        objs.append((name, *[float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]))
    return w, h, objs


def to_class(name: str) -> int | None:
    if name in RDD_CODE_TO_ID:
        return RDD_CODE_TO_ID[name]
    return normalise_name_to_id(name)


def group_key(source: str, stem: str) -> str:
    m = re.search(r"(\d+)$", stem)
    return f"{source}_{int(m.group(1)) // 25}" if m else f"{source}_{stem}"


def is_val(group: str, frac: float, seed: int) -> bool:
    h = int(hashlib.md5(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) < frac * 10_000


# ----------------------------------------------------------------------------- worker
def convert_one(job):
    img_path, xml_path, out_img, out_lbl, max_side, name_prefix, label_tag = job
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    _xw, _xh, objs = parse_voc(xml_path)
    xw, xh = w, h                       # boxes are in the pixel space of the image we actually read
    lines, counts = [], Counter()
    skipped = Counter()
    for name, x1, y1, x2, y2 in objs:
        cid = to_class(name)
        if cid is None:
            skipped[name] += 1
            continue
        x1, x2 = max(0.0, min(x1, xw)), max(0.0, min(x2, xw))
        y1, y2 = max(0.0, min(y1, xh)), max(0.0, min(y2, xh))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{cid} {(x1 + x2) / 2 / xw:.6f} {(y1 + y2) / 2 / xh:.6f} {(x2 - x1) / xw:.6f} {(y2 - y1) / xh:.6f}")
        counts[cid] += 1
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
    return label_tag, dict(counts), dict(skipped)


# ----------------------------------------------------------------------------- main
def collect(root: Path, tag_prefix: str):
    """Pair every XML with its image: same (source, stem) first, else a unique stem match anywhere under root.

    RDD2022 keeps images and XMLs in sibling folders of one country; other datasets (e.g. the Mendeley
    potholes set) use separate IMG/ and XML/ folders, hence the stem fallback.
    """
    imgs = index_images(root)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for (_src, stem), p in imgs.items():
        by_stem[stem].append(p)
    items = []
    for xml in root.rglob("*.xml"):
        rel = xml.relative_to(root).parts
        src = rel[0] if len(rel) > 1 else root.name
        img = imgs.get((src, xml.stem))
        if img is None and len(by_stem.get(xml.stem, [])) == 1:
            img = by_stem[xml.stem][0]
            rp = img.relative_to(root).parts
            src = "all"                      # one logical source for IMG/XML-style layouts
        if img is None:
            continue
        items.append((src, xml, img))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rdd-root", required=True, help="folder containing the RDD2022 country folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=DIR",
                    help="extra VOC-format dataset (e.g. mendeley=/kaggle/input/potholes). Repeatable.")
    ap.add_argument("--max-per-country", type=int, default=5000)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--neg-ratio", type=float, default=0.08, help="label-free images kept, as a share of positives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images per source")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out)
    sources: list[tuple[str, Path]] = [("rdd", Path(a.rdd_root))]
    for e in a.extra:
        n, d = e.split("=", 1)
        sources.append((n, Path(d)))

    jobs, per_source = [], defaultdict(int)
    for tag, root in sources:
        items = collect(root, tag)
        by_src = defaultdict(list)
        for src, xml, img in items:
            by_src[src].append((xml, img))
        for src, lst in sorted(by_src.items()):
            lst.sort(key=lambda t: t[1].stem)
            rng.shuffle(lst)
            cap = a.limit or (a.max_per_country if tag == "rdd" else 10 ** 9)
            # keep positives first, then a bounded share of negatives
            pos, neg = [], []
            for xml, img in lst:
                try:
                    _, _, objs = parse_voc(xml)
                except ET.ParseError:
                    continue
                (pos if any(to_class(o[0]) is not None for o in objs) else neg).append((xml, img))
            take = pos[:cap] + neg[: min(len(neg), int(a.neg_ratio * min(len(pos), cap)))]
            for xml, img in take:
                split = "valid" if is_val(group_key(f"{tag}_{src}", img.stem), a.val_frac, a.seed) else "train"
                name = f"{tag}_{src}_{img.stem}"
                jobs.append((img, xml, out / split / "images" / f"{name}.jpg",
                             out / split / "labels" / f"{name}.txt", a.max_side, name, (split, f"{tag}/{src}")))
                per_source[f"{tag}/{src}"] += 1
            print(f"{tag}/{src}: {len(pos)} positive, {len(neg)} negative available -> using {len(take)}")

    if not jobs:
        sys.exit("ERROR: no image/annotation pairs found under the given roots - check the download/extraction "
                 "(expected <root>/<Country>/train/{images,annotations/xmls}).")
    print(f"converting {len(jobs)} images with {a.workers} workers ...")
    stats = defaultdict(Counter)       # (split) -> class counts
    imgs_split = Counter()
    skipped_names = Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(convert_one, jobs, chunksize=32), start=1):
            if res is None:
                continue
            (split, _src), counts, skipped = res
            imgs_split[split] += 1
            for c, n in counts.items():
                stats[split][c] += n
            skipped_names.update(skipped)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}")

    (out).mkdir(parents=True, exist_ok=True)
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(DEFECT_NAMES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\nnames:\n{names_yaml}\n")

    print("\n== dataset summary ==")
    for split in ("train", "valid"):
        print(f"{split}: {imgs_split[split]} images; boxes per class: "
              + ", ".join(f"{DEFECT_NAMES[c]}={stats[split][c]}" for c in range(len(DEFECT_NAMES))))
    if skipped_names:
        print("ignored labels (not mapped to a defect class):", dict(skipped_names.most_common(15)))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
