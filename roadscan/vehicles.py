"""Bonus: per-vehicle log (type, colour, side, closest distance, approach rate, whether it was passed, map position).

Distances come from the same flat-ground model as the route map (vehicle box bottom edge -> ground range), so
they are rough. No licence plates or other identifying details are read or stored.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .tracking import FinalTrack
from .video_io import timecode

VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def colour_name(bgr_pixels: np.ndarray) -> str:
    """Rough colour name from the median HSV of a crop (BGR uint8, N x 3)."""
    import cv2

    if bgr_pixels.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = [float(np.median(hsv[:, i])) for i in range(3)]
    if v < 55:
        return "black"
    if s < 45:
        return "white" if v > 175 else ("silver/grey" if v > 95 else "dark grey")
    if h < 8 or h >= 170:
        return "red"
    if h < 20:
        return "brown" if v < 130 else "orange"
    if h < 35:
        return "yellow/beige" if s < 120 else "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple/pink"


def vehicle_rows(final: list[FinalTrack], ego, width: int, height: int,
                 colours: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for t in final:
        if t.cls_name not in VEHICLE_NAMES:
            continue
        obs = t.obs
        times = np.array([o.pts for o in obs])
        cx = np.array([(o.xyxy[0] + o.xyxy[2]) / 2 for o in obs]) / width
        hh = np.array([max(o.xyxy[3] - o.xyxy[1], 1.0) for o in obs])
        y2 = np.array([o.xyxy[3] for o in obs])
        ranges, wx, wy = [], [], []
        for o in obs:
            w = ego.world_of_box(o.frame_idx, o.xyxy) if ego is not None else None
            ranges.append(w[2] if w else np.nan)
            wx.append(w[0] if w else np.nan)
            wy.append(w[1] if w else np.nan)
        ranges, wx, wy = np.array(ranges), np.array(wx), np.array(wy)
        good = np.isfinite(ranges)
        side = "left" if np.mean(cx) < 0.40 else ("right" if np.mean(cx) > 0.60 else "ahead")
        slope = float(np.polyfit(times, np.log(hh), 1)[0]) if len(obs) >= 8 and times[-1] - times[0] > 0.5 else 0.0
        trend = "approaching" if slope > 0.15 else ("receding" if slope < -0.15 else "steady")
        closest = float(np.nanmin(ranges)) if good.any() else float("nan")
        first_d = float(ranges[good][0]) if good.any() else float("nan")
        rate = ""
        if good.sum() >= 6 and times[good][-1] - times[good][0] > 0.5:
            rate = round(float(-np.polyfit(times[good], ranges[good], 1)[0]) * 3.6, 1)   # + = closing in
        exits_side = bool(cx[-1] < 0.20 or cx[-1] > 0.80)
        passed = bool(trend == "approaching" and side in ("left", "right") and (exits_side or y2[-1] > 0.85 * height))
        k = int(np.nanargmin(ranges)) if good.any() else len(obs) - 1
        rows.append({
            "vehicle_id": t.track_id, "type": t.cls_name, "colour_approx": colours.get(t.track_id, "unknown"),
            "first_seen_s": round(t.first_pts, 3), "first_seen_timecode": timecode(t.first_pts),
            "last_seen_s": round(t.last_pts, 3), "visible_s": round(t.last_pts - t.first_pts, 3),
            "side_in_frame": side, "motion_vs_ego": trend, "passed_by_ego": "yes" if passed else "no",
            "first_distance_m_approx": round(first_d, 1) if np.isfinite(first_d) else "",
            "closest_distance_m_approx": round(closest, 1) if np.isfinite(closest) else "",
            "closing_speed_kmh_approx": rate,
            "closest_time_s": round(float(times[k]), 2),
            "map_x_m": round(float(wx[k]), 1) if np.isfinite(wx[k]) else "",
            "map_y_m": round(float(wy[k]), 1) if np.isfinite(wy[k]) else "",
            "max_confidence": round(t.max_conf, 3), "n_frames_detected": t.n_obs,
            "_frame_closest": obs[k].frame_idx,
        })
    return rows
