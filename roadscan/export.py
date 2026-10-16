"""CSV / JSON exports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .tracking import FinalTrack
from .video_io import timecode

def _jdefault(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


CSV_COLUMNS = [
    "track_id", "defect_type", "first_seen_s", "first_seen_timecode", "max_confidence", "image",
    "crop", "first_seen_frame", "last_seen_s", "n_frames_detected", "mean_confidence",
    "best_frame", "best_frame_s", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "box_area_px",
    "mask_area_px", "mask_area_pct_of_frame", "road_overlap", "raw_track_ids",
    "water_filled", "surface_at_defect", "sidewalk_at_defect", "map_x_m", "map_y_m",
    "route_distance_m", "range_ahead_m",
]


def track_record(t: FinalTrack, image: str, crop: str, frame_area: int,
                 mask_area_px: int | None) -> dict[str, Any]:
    b = t.best
    box_area = float((b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]))
    return {
        "track_id": t.track_id,
        "defect_type": t.cls_name,
        "first_seen_s": round(t.first_pts, 3),
        "first_seen_timecode": timecode(t.first_pts),
        "max_confidence": round(t.max_conf, 4),
        "image": image,
        "crop": crop,
        "first_seen_frame": t.first_frame,
        "last_seen_s": round(t.last_pts, 3),
        "n_frames_detected": t.n_obs,
        "mean_confidence": round(t.mean_conf, 4),
        "best_frame": b.frame_idx,
        "best_frame_s": round(b.pts, 3),
        "bbox_x1": round(float(b.xyxy[0]), 1), "bbox_y1": round(float(b.xyxy[1]), 1),
        "bbox_x2": round(float(b.xyxy[2]), 1), "bbox_y2": round(float(b.xyxy[3]), 1),
        "box_area_px": round(box_area, 1),
        "mask_area_px": mask_area_px if mask_area_px is not None else "",
        "mask_area_pct_of_frame": round(100.0 * mask_area_px / frame_area, 3) if mask_area_px is not None else "",
        "road_overlap": "" if np.isnan(b.road_overlap) else round(float(b.road_overlap), 3),
        "raw_track_ids": ";".join(map(str, t.raw_ids)),
        "water_filled": bool(t.water_filled) or t.cls_name == "water_filled_pothole",
    }


def write_outputs(out_dir: Path, records: list[dict[str, Any]], run_info: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    df.to_csv(out_dir / "defects.csv", index=False)
    summary = {
        "total_unique_defects": len(records),
        "by_type": df["defect_type"].value_counts().to_dict() if len(df) else {},
    }
    with open(out_dir / "defects.json", "w", encoding="utf-8") as f:
        json.dump({"video": run_info.get("video"), "summary": summary, "defects": records}, f, indent=2, default=_jdefault)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=_jdefault)


def write_scene(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "road_users.csv", index=False)
    summary = {"total": len(rows), "by_class": df["class"].value_counts().to_dict() if len(df) else {}}
    with open(out_dir / "road_users.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "road_users": rows}, f, indent=2, default=_jdefault)


def write_surface(out_dir: Path, segments: list[dict[str, Any]], summary: dict[str, Any], odometry: dict | None) -> None:
    pd.DataFrame(segments).to_csv(out_dir / "road_segments.csv", index=False)
    with open(out_dir / "road_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "odometry": odometry, "segments": segments}, f, indent=2, default=_jdefault)


def write_furniture(out_dir: Path, inventory: list[dict[str, Any]], marking_summary: dict[str, Any],
                    marking_hist: list[dict[str, Any]]) -> None:
    """road_furniture.csv/json (every sign / light / crosswalk looked at + condition), marking_visibility.csv, crops."""
    import cv2

    fdir = out_dir / "furniture"
    fdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in inventory:
        crop = r.pop("_crop", None)
        name = f"{r['kind']}_{r['item_id']:03d}.jpg"
        if crop is not None and crop.size:
            cv2.imwrite(str(fdir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            r["crop"] = f"furniture/{name}"
        else:
            r["crop"] = ""
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "road_furniture.csv", index=False)
    pd.DataFrame(marking_hist).to_csv(out_dir / "marking_visibility.csv", index=False)
    with open(out_dir / "road_furniture.json", "w", encoding="utf-8") as f:
        json.dump({"marking": marking_summary, "items": rows}, f, indent=2, default=_jdefault)
