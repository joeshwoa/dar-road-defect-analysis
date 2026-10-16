"""Schematic 2-D map of the route with the defects / road furniture / vehicles placed on it.

The route comes from monocular visual odometry (odometry.py); positions are therefore approximate.
Orientation: the initial driving direction points up ("north"); units are metres.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

# marker style per defect type: (matplotlib marker, colour, label)
STYLE = {
    "pothole": ("o", "#d62728", "pothole"),
    "water_filled_pothole": ("o", "#1f77b4", "water-filled pothole"),
    "longitudinal_crack": ("_", "#ff7f0e", "longitudinal crack"),
    "transverse_crack": ("|", "#ff7f0e", "transverse crack"),
    "alligator_crack": ("^", "#e377c2", "alligator crack"),
    "manhole_or_drain": ("s", "#222222", "manhole / drain"),
    "vehicle_passed": ("D", "#7f7f7f", "vehicle passed"),
}
SURFACE_COLOUR = {"paved": "#2ca02c", "mixed": "#bcbd22", "unpaved": "#8c564b", "unknown": "#999999"}


def write_route_csv(path: Path, ego, speed_kmh: np.ndarray, surface_at=None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x_m", "y_m", "heading_deg", "speed_kmh_approx", "distance_m", "surface"])
        for p, v in zip(ego.poses, speed_kmh):
            w.writerow([p.frame, round(p.t, 3), round(p.x, 2), round(p.y, 2), round(np.degrees(p.heading), 1),
                        round(float(v), 1), round(p.dist_m, 1), surface_at(p.t) if surface_at else ""])


def write_points_csv(path: Path, points: Sequence[dict[str, Any]]) -> None:
    cols = ["kind", "id", "type", "time_s", "x_m", "y_m", "distance_along_route_m", "range_ahead_m"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in points:
            w.writerow({k: p.get(k, "") for k in cols})


def render_map_png(path: Path, route: np.ndarray, route_surface: Sequence[str], points: Sequence[dict[str, Any]],
                   title: str, footnote: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 9), dpi=130)
    if len(route) >= 2:
        for i in range(len(route) - 1):
            ax.plot(route[i:i + 2, 0], route[i:i + 2, 1], color=SURFACE_COLOUR.get(route_surface[i], "#999999"),
                    lw=5, solid_capstyle="round", alpha=0.55, zorder=1)
        ax.plot(route[:, 0], route[:, 1], color="#333333", lw=0.8, zorder=2)
        ax.scatter(*route[0], marker="*", s=220, c="#2ca02c", edgecolor="k", zorder=5, label="start")
        ax.scatter(*route[-1], marker="X", s=140, c="#d62728", edgecolor="k", zorder=5, label="end")
    seen = set()
    for p in points:
        kind = p["type"] if p["kind"] == "defect" else "vehicle_passed"
        m, c, lab = STYLE.get(kind, ("o", "#555555", kind))
        ax.scatter(p["x_m"], p["y_m"], marker=m, s=46, c=c, edgecolor="k" if m not in "_|" else c,
                   linewidths=0.6, zorder=4, label=lab if lab not in seen else None)
        seen.add(lab)
    handles, labels = ax.get_legend_handles_labels()
    for s_name, col in SURFACE_COLOUR.items():
        if s_name in set(route_surface):
            handles.append(plt.Line2D([0], [0], color=col, lw=5, alpha=0.55))
            labels.append(f"{s_name} road")
    ax.legend(handles, labels, loc="best", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("metres (approx.)")
    ax.set_ylabel("metres (approx.) - start heading is up")
    ax.set_title(title, fontsize=11)
    fig.text(0.5, 0.012, footnote, ha="center", fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


class MiniMap:
    """Small live map drawn into a corner of the annotated video."""

    def __init__(self, ego, markers: Sequence[dict[str, Any]], size: int = 190, margin: int = 8):
        self.size, self.margin = size, margin
        self.frames = np.array([p.frame for p in ego.poses])
        self.xy = ego.route()
        self.markers = sorted(markers, key=lambda m: m["frame"])
        pts = self.xy if len(self.xy) else np.zeros((1, 2))
        allp = np.vstack([pts] + ([np.array([[m["x_m"], m["y_m"]] for m in markers])] if markers else []))
        lo, hi = allp.min(0), allp.max(0)
        span = float(max((hi - lo).max(), 20.0))
        centre = (lo + hi) / 2
        self.lo = centre - span / 2 * 1.12
        self.span = span * 1.12

    def _px(self, xy) -> tuple[int, int]:
        u = (xy[0] - self.lo[0]) / self.span
        v = 1.0 - (xy[1] - self.lo[1]) / self.span
        return int(8 + u * (self.size - 16)), int(8 + v * (self.size - 16))

    def draw(self, canvas: np.ndarray, frame_idx: int) -> None:
        if len(self.xy) < 2:
            return
        H, W = canvas.shape[:2]
        s = self.size
        x0, y0 = W - s - self.margin, H - s - self.margin
        tile = (canvas[y0:y0 + s, x0:x0 + s] * 0.30).astype(np.uint8)
        k = int(np.searchsorted(self.frames, frame_idx, side="right"))
        k = max(2, min(k, len(self.xy)))
        pts = np.array([self._px(p) for p in self.xy[:k]], np.int32)
        full = np.array([self._px(p) for p in self.xy], np.int32)
        cv2.polylines(tile, [full], False, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.polylines(tile, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
        for m in self.markers:
            if m["frame"] > frame_idx:
                break
            cv2.circle(tile, self._px((m["x_m"], m["y_m"])), 3, m["color"], -1, cv2.LINE_AA)
        cv2.circle(tile, tuple(pts[-1]), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(tile, "route map (approx.)", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (s - 1, s - 1), (200, 200, 200), 1)
        canvas[y0:y0 + s, x0:x0 + s] = tile
