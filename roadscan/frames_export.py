"""Per-frame export: every video frame gets its own record (what was seen, where, how the road looks).

Outputs (inside <clip>/frames/):
  frames.jsonl        one JSON object per frame (nested: defects, road users, road furniture, road state, ego motion)
  frame_summary.csv   one row per frame (counts + road state + speed) - open it in Excel
  detections.csv      one row per detection per frame (frame, time, kind, track id, class, confidence, box)
  img/frame_NNNNN.jpg the annotated frame itself (optional, render.save_frames = all)
All values are what the pipeline *knew at that frame* (past-only logic), exactly what is drawn on annotated.mp4.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

SUMMARY_COLS = [
    "frame", "t_s", "timecode", "image", "surface", "sidewalk", "lane_marking_state", "lane_marking_vis",
    "lane_paint_missing", "speed_kmh_est", "ego_x_m", "ego_y_m", "route_distance_m",
    "n_defects", "defect_ids", "defect_types", "unique_defects_so_far",
    "n_road_users", "n_signs", "n_traffic_lights", "n_crosswalks", "n_flagged_furniture",
    "detector_ran_this_frame", "furniture_ran_this_frame",
]
DET_COLS = ["frame", "t_s", "kind", "track_id", "class", "label", "confidence", "flagged",
            "x1", "y1", "x2", "y2"]


def timecode(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def _box(b) -> list[float]:
    return [round(float(v), 1) for v in b]


class FrameLog:
    def __init__(self, out_dir: Path, save_images: bool = True, jpeg_quality: int = 75):
        self.dir = Path(out_dir) / "frames"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.q = int(jpeg_quality)
        if save_images:
            (self.dir / "img").mkdir(exist_ok=True)
        self._jl = open(self.dir / "frames.jsonl", "w", encoding="utf-8")
        self._sf = open(self.dir / "frame_summary.csv", "w", newline="", encoding="utf-8")
        self._df = open(self.dir / "detections.csv", "w", newline="", encoding="utf-8")
        self._sw = csv.DictWriter(self._sf, fieldnames=SUMMARY_COLS)
        self._dw = csv.DictWriter(self._df, fieldnames=DET_COLS)
        self._sw.writeheader()
        self._dw.writeheader()
        self.n = 0
        self.n_det = 0

    def add(self, idx: int, t: float, canvas, *, surface: str, sidewalk: str, marking: dict, paint_missing: bool,
            speed_kmh: float | None, pose, defects: list[dict], road_users: list[dict], furniture: list[dict],
            unique_so_far: int, detector_ran: bool, furniture_ran: bool) -> None:
        img_rel = ""
        if self.save_images and canvas is not None:
            img_rel = f"img/frame_{idx:05d}.jpg"
            cv2.imwrite(str(self.dir / img_rel), canvas, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        tc = timecode(t)
        rec: dict[str, Any] = {
            "frame": idx, "t_s": round(t, 3), "timecode": tc, "image": img_rel,
            "road": {"surface": surface, "sidewalk": sidewalk,
                     "lane_marking_state": marking.get("state", "not visible") if marking.get("present") else "not visible",
                     "lane_marking_vis": round(float(marking["vis"]), 3) if marking.get("present") and "vis" in marking else None,
                     "lane_paint_missing": bool(paint_missing)},
            "ego": {"speed_kmh_est": None if speed_kmh is None else round(float(speed_kmh), 1),
                    "x_m": None if pose is None else round(float(pose.x), 2),
                    "y_m": None if pose is None else round(float(pose.y), 2),
                    "route_distance_m": None if pose is None else round(float(pose.dist_m), 2)},
            "unique_defects_so_far": int(unique_so_far),
            "defects": defects, "road_users": road_users, "road_furniture": furniture,
            "detector_ran_this_frame": bool(detector_ran), "furniture_ran_this_frame": bool(furniture_ran),
        }
        self._jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kinds = [f.get("kind", "") for f in furniture]
        self._sw.writerow({
            "frame": idx, "t_s": rec["t_s"], "timecode": tc, "image": img_rel, "surface": surface, "sidewalk": sidewalk,
            "lane_marking_state": rec["road"]["lane_marking_state"], "lane_marking_vis": rec["road"]["lane_marking_vis"],
            "lane_paint_missing": int(bool(paint_missing)), "speed_kmh_est": rec["ego"]["speed_kmh_est"],
            "ego_x_m": rec["ego"]["x_m"], "ego_y_m": rec["ego"]["y_m"], "route_distance_m": rec["ego"]["route_distance_m"],
            "n_defects": len(defects), "defect_ids": ";".join(str(d["track_id"]) for d in defects),
            "defect_types": ";".join(sorted({d["class"] for d in defects})), "unique_defects_so_far": int(unique_so_far),
            "n_road_users": len(road_users), "n_signs": kinds.count("sign"), "n_traffic_lights": kinds.count("light"),
            "n_crosswalks": kinds.count("crosswalk"), "n_flagged_furniture": sum(1 for f in furniture if f.get("flagged")),
            "detector_ran_this_frame": int(bool(detector_ran)), "furniture_ran_this_frame": int(bool(furniture_ran)),
        })
        for kind, items in (("defect", defects), ("road_user", road_users), ("road_furniture", furniture)):
            for it in items:
                x1, y1, x2, y2 = it["box"]
                self._dw.writerow({
                    "frame": idx, "t_s": rec["t_s"], "kind": kind if kind != "road_furniture" else it.get("kind", "furniture"),
                    "track_id": it.get("track_id", ""), "class": it.get("class", it.get("kind", "")),
                    "label": it.get("label", ""), "confidence": it.get("confidence", ""),
                    "flagged": int(bool(it.get("flagged", False))), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                self.n_det += 1
        self.n += 1

    def close(self) -> dict[str, Any]:
        for f in (self._jl, self._sf, self._df):
            f.close()
        return {"frames": self.n, "detection_rows": self.n_det, "images": bool(self.save_images),
                "files": ["frames/frames.jsonl", "frames/frame_summary.csv", "frames/detections.csv"]
                         + (["frames/img/frame_NNNNN.jpg"] if self.save_images else [])}
