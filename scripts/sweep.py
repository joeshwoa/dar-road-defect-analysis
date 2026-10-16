#!/usr/bin/env python3
"""Run several pipeline variants on the same video(s) and print one comparison table.

There is no ground truth for the target videos, so the table reports *counts and speed*; use
review_sheet.py + score_review.py to turn a manual look at the exported crops into precision numbers.

    python scripts/sweep.py video1.mp4 video2.mp4 --weights best.pt --out outputs/sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadscan.classes import DEFECT_NAMES
from roadscan.config import Config
from roadscan.pipeline import Analyzer

VARIANTS = {
    "base_640":            {},
    "imgsz_960":           {"detect.imgsz": 960},
    "tiled_480":           {"detect.tile_size": 480},
    "no_road_filter":      {"road.enabled": False},
    "bytetrack":           {"track.tracker": "bytetrack"},
    "no_track_merge":      {"merge.enabled": False},
    "conf_0.35":           {"detect.conf": 0.35, "report.min_max_conf": 0.5},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--backend", default="ultralytics")
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), help=f"subset of {list(VARIANTS)}")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    base = Config().override({"detect.weights": a.weights, "detect.backend": a.backend, "detect.device": a.device,
                              "render.enabled": False, "sam.enabled": False,
                              "merge.compatible_classes": [[0, 1, 2]]})
    rows = []
    for name in a.variants:
        cfg = base.override(VARIANTS[name])
        an = Analyzer(cfg, class_names=DEFECT_NAMES if a.backend == "rfdetr" else None, log=lambda *_: None)
        for v in a.videos:
            res = an.run(v, Path(a.out) / name)
            recs = res["records"]
            row = {"variant": name, "video": Path(v).name[:28], "unique_defects": len(recs),
                   "fps_model_stages": res["run_info"]["timing"]["processing_fps"],
                   "raw_tracks": res["run_info"]["tracking"]["raw_tracks"]}
            for cname in DEFECT_NAMES:
                row[cname] = sum(r["defect_type"] == cname for r in recs)
            rows.append(row)
            print(row)
        del an
    df = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "sweep_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
