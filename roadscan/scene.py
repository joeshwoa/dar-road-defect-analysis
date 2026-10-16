"""Scene context (bonus): vehicles, pedestrians, traffic lights and stop signs.

A second, COCO-pretrained detector is tracked with the same BoT-SORT wrapper. For every tracked road user
we export class, first/last time, confidence and two cheap motion cues derived from the box:
  * size_trend  - slope of log(box height) over time: >0 approaching, <0 receding (relative to the ego camera)
  * lane_hint   - horizontal position of the box centre: left / centre / right of the frame
No camera calibration is assumed, so no metric speed or distance is reported.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .config import Config, DetectCfg, ReportCfg
from .detectors import UltralyticsDetector
from .tracking import FinalTrack, Obs, finalize_tracks
from .video_io import timecode

ROAD_USER_IDS = {0, 1, 2, 3, 5, 7}


def build_scene_detector(cfg: Config) -> UltralyticsDetector:
    sc = cfg.scene
    dcfg = DetectCfg(backend="ultralytics", weights=sc.weights, imgsz=sc.imgsz, conf=sc.conf,
                     device=cfg.detect.device, half=cfg.detect.half, classes=list(sc.classes))
    return UltralyticsDetector(dcfg)


def scene_finalize(raw: dict[int, list[Obs]], names: dict[int, str], cfg: Config):
    c = copy.deepcopy(cfg)
    c.report = ReportCfg(min_observations=cfg.scene.min_observations, min_max_conf=cfg.scene.min_max_conf)
    c.merge.compatible_classes = cfg.scene.compatible_classes
    return finalize_tracks(raw, names, c)


def _motion_cues(t: FinalTrack, frame_w: int) -> tuple[str, float | str, str]:
    times = np.array([o.pts for o in t.obs])
    heights = np.array([max(o.xyxy[3] - o.xyxy[1], 1.0) for o in t.obs])
    cx = np.mean([(o.xyxy[0] + o.xyxy[2]) / 2 for o in t.obs]) / max(frame_w, 1)
    lane = "left" if cx < 0.40 else ("right" if cx > 0.60 else "centre")
    if len(t.obs) < 8 or times[-1] - times[0] < 0.5:
        return "unknown", "", lane
    slope = float(np.polyfit(times, np.log(heights), 1)[0])
    trend = "approaching" if slope > 0.15 else ("receding" if slope < -0.15 else "steady")
    return trend, round(slope, 3), lane


def scene_records(final: list[FinalTrack], frame_w: int) -> list[dict[str, Any]]:
    rows = []
    for t in final:
        trend, slope, lane = _motion_cues(t, frame_w)
        rows.append({
            "track_id": t.track_id, "class": t.cls_name, "first_seen_s": round(t.first_pts, 3),
            "first_seen_timecode": timecode(t.first_pts), "last_seen_s": round(t.last_pts, 3),
            "visible_s": round(t.last_pts - t.first_pts, 3), "n_frames_detected": t.n_obs,
            "max_confidence": round(t.max_conf, 4), "size_trend": trend, "log_height_slope_per_s": slope,
            "lane_hint": lane,
            "mean_box_height_px": round(float(np.mean([o.xyxy[3] - o.xyxy[1] for o in t.obs])), 1),
        })
    return rows
