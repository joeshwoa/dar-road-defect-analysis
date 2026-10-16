#!/usr/bin/env python3
"""Analyse one or more videos.

Examples
    python scripts/run_video.py videos/*.mp4 --weights runs/yolo26s/weights/best.pt --out outputs
    python scripts/run_video.py clip.mp4 --weights best.pt --sam --set detect.imgsz=960 --set road.enabled=false
    python scripts/run_video.py clip.mp4 --backend rfdetr --weights rfdetr_best.pth --names longitudinal_crack transverse_crack alligator_crack pothole
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadscan.classes import DEFECT_NAMES
from roadscan.config import Config, parse_overrides
from roadscan.pipeline import Analyzer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video file(s)")
    ap.add_argument("--out", default="outputs", help="output root directory")
    ap.add_argument("--config", help="YAML config (see configs/default.yaml)")
    ap.add_argument("--weights", help="detector weights (.pt for Ultralytics, .pth for RF-DETR)")
    ap.add_argument("--backend", choices=["ultralytics", "rfdetr"], help="detector backend")
    ap.add_argument("--names", nargs="+", help="class names (needed for RF-DETR if not stored in the checkpoint)")
    ap.add_argument("--device", help="auto | cpu | 0 | mps")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--conf", type=float)
    ap.add_argument("--tile", type=int, help="tile size in px for sliced inference (0 = off)")
    ap.add_argument("--stride", type=int, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-road-mask", action="store_true", help="disable the road-surface filter")
    ap.add_argument("--sam", action="store_true", help="add SAM 2 masks on each defect's best frame")
    ap.add_argument("--scene", action="store_true", help="bonus: also track vehicles/pedestrians/signals (COCO model)")
    ap.add_argument("--no-surface", action="store_true", help="disable the paved/unpaved/sidewalk/water analysis")
    ap.add_argument("--no-odometry", action="store_true", help="disable the visual-odometry route map")
    ap.add_argument("--openvocab", action="store_true", help="also detect manholes / drain grates (OWLv2, slower)")
    ap.add_argument("--furniture", action="store_true",
                    help="also assess lane markings, crosswalks, signs and traffic lights (Mapillary Mask2Former + CLIP)")
    ap.add_argument("--no-temporal", action="store_true", help="disable the causal temporal stabilisation (ablation)")
    ap.add_argument("--cam-height", type=float, help="assumed camera height in metres for the route map (default 1.4)")
    ap.add_argument("--hfov", type=float, help="assumed horizontal field of view in degrees (default 75)")
    ap.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="override any config key, e.g. track.tracker=bytetrack")
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config) if a.config else Config()
    ov = parse_overrides(a.set)
    for key, val in {"detect.weights": a.weights, "detect.backend": a.backend, "detect.device": a.device,
                     "detect.imgsz": a.imgsz, "detect.conf": a.conf, "detect.tile_size": a.tile,
                     "stride": a.stride, "max_frames": a.max_frames}.items():
        if val is not None:
            ov[key] = val
    if a.no_road_mask:
        ov["road.enabled"] = False
    if a.sam:
        ov["sam.enabled"] = True
    if a.scene:
        ov["scene.enabled"] = True
    if a.no_surface:
        ov["surface.enabled"] = False
    if a.no_odometry:
        ov["odometry.enabled"] = False
    if a.openvocab:
        ov["openvocab.enabled"] = True
    if a.furniture:
        ov["furniture.enabled"] = True
    if a.no_temporal:
        ov["temporal.enabled"] = False
        ov["surface.causal"] = False
    if a.cam_height is not None:
        ov["odometry.cam_height_m"] = a.cam_height
    if a.hfov is not None:
        ov["odometry.hfov_deg"] = a.hfov
    if a.no_video:
        ov["render.enabled"] = False
    cfg = cfg.override(ov)

    names = a.names or (DEFECT_NAMES if cfg.detect.backend == "rfdetr" else None)
    an = Analyzer(cfg, class_names=names)
    summary = {}
    for v in a.videos:
        res = an.run(v, a.out)
        summary[Path(v).name] = {
            "unique_defects": len(res["records"]),
            "processing_fps": res["run_info"]["timing"]["processing_fps"],
            "end_to_end_fps_pass1": res["run_info"]["timing"]["end_to_end_fps_pass1"],
            "out_dir": res["out_dir"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
