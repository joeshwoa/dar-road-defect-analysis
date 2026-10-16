"""Monocular visual odometry for a 2-D route map.

Assumptions (all approximate - the map is a schematic, not a survey):
  * forward-looking camera at a fixed height `cam_height_m` above a flat ground plane, no roll,
  * horizontal field of view `hfov_deg` (focal length f = W / (2 tan(hfov/2))),
  * horizon row = median top of the road in the segmentation (or a fixed fraction).

For a ground pixel at image row v (below the horizon row y0) the forward distance is  Z = f * h / (v - y0).
Tracking ground features between consecutive frames gives the distance driven  d = Z_prev - Z_now  (median over
features), and the rotation-corrected horizontal residual gives the yaw change. Integrating (d, yaw) yields a
trajectory in metres; any image point on the ground can then be placed on that map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import OdometryCfg, TemporalCfg
from .temporal import SpeedFilter


@dataclass
class Pose:
    frame: int
    t: float
    x: float          # metres, east (arbitrary origin = start)
    y: float          # metres, north (initial driving direction)
    heading: float    # radians, CCW from +y
    speed_ms: float   # causal-filtered speed when the temporal filter is on, else the raw per-frame value
    dist_m: float     # distance along the route
    raw_speed_ms: float = 0.0


class EgoMotion:
    def __init__(self, cfg: OdometryCfg, width: int, height: int, fps: float = 25.0,
                 tcfg: TemporalCfg | None = None):
        self.cfg = cfg
        self._sf = (SpeedFilter(fps, tcfg.speed_median_s, tcfg.speed_ema, tcfg.max_accel_ms2, tcfg.max_decel_ms2)
                    if (tcfg is not None and tcfg.enabled) else None)
        self.causal = self._sf is not None
        self.W, self.H = width, height
        self.f = width / (2.0 * math.tan(math.radians(cfg.hfov_deg) / 2.0))
        self.cx, self.cy = width / 2.0, height / 2.0
        self.scale = min(1.0, 480.0 / width)                  # track at reduced resolution for speed
        self._prev: np.ndarray | None = None
        self._prev_t = 0.0
        self.poses: list[Pose] = []
        self._x = self._y = self._th = self._dist = 0.0
        self._horizon = cfg.horizon_frac * height if cfg.horizon_frac > 0 else 0.45 * height
        self._d_hist: list[float] = []
        self.n_ok = 0
        self.n_total = 0

    def set_horizon(self, y0: float) -> None:
        if self.cfg.horizon_frac <= 0 and np.isfinite(y0):
            self._horizon = float(np.clip(y0, 0.03 * self.H, 0.7 * self.H))

    # ---------------------------------------------------------------- geometry
    def ground_xz(self, u: float, v: float) -> tuple[float, float] | None:
        """Image point on the ground -> (lateral X to the right, forward Z) in metres in the camera frame."""
        dv = v - self._horizon
        if dv < 0.04 * self.H:
            return None
        z = self.f * self.cfg.cam_height_m / dv
        if z > 250:
            return None
        return float((u - self.cx) * z / self.f), float(z)

    def pose_at(self, frame: int) -> Pose | None:
        if not self.poses:
            return None
        frames = [p.frame for p in self.poses]
        i = int(np.searchsorted(frames, frame))
        i = min(max(i, 0), len(self.poses) - 1)
        if i > 0 and abs(frames[i - 1] - frame) < abs(frames[i] - frame):
            i -= 1
        return self.poses[i]

    def to_world(self, pose: Pose, xz: tuple[float, float]) -> tuple[float, float]:
        X, Z = xz
        fx, fy = -math.sin(pose.heading), math.cos(pose.heading)      # forward
        rx, ry = math.cos(pose.heading), math.sin(pose.heading)       # right
        return pose.x + Z * fx + X * rx, pose.y + Z * fy + X * ry

    def world_of_box(self, frame: int, xyxy) -> tuple[float, float, float] | None:
        """(x, y, distance_ahead_m) of the bottom-centre of a box, placed with the pose of `frame`."""
        pose = self.pose_at(frame)
        if pose is None:
            return None
        g = self.ground_xz((xyxy[0] + xyxy[2]) / 2.0, float(xyxy[3]))
        if g is None:
            return None
        x, y = self.to_world(pose, g)
        return float(x), float(y), float(g[1])

    # ---------------------------------------------------------------- update
    def update(self, frame: int, t: float, img: np.ndarray, exclude: list | None = None) -> Pose:
        self.n_total += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.scale < 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        d, dth = 0.0, 0.0
        valid = False
        dt = max(t - self._prev_t, 1e-3) if self._prev is not None else 1.0 / 25.0
        if self._prev is not None:
            res = self._step(self._prev, gray, exclude or [])
            if res is not None:
                d, dth = res
                valid = True
                self.n_ok += 1
                self._d_hist.append(d)
            elif self._d_hist:                                   # keep going at the recent speed
                d = float(np.median(self._d_hist[-8:]))
        self._prev, self._prev_t = gray, t
        # robustify against single-frame spikes (bumps): clamp to 3x recent median
        if len(self._d_hist) >= 5:
            med = float(np.median(self._d_hist[-15:]))
            d = float(np.clip(d, 0.0, max(3.0 * med, 0.05)))
        raw_v = d / dt
        if self._sf is not None and self._prev_t is not None and self.n_total > 1:
            # causal speed: median -> EMA -> accel/decel limiter; distance is integrated from the filtered speed so
            # a one-frame flow failure cannot make the car "stop" on the map either
            v_kmh = self._sf.update(t, raw_v * 3.6, valid=valid or len(self._d_hist) > 0)
            d = v_kmh / 3.6 * dt
            speed = v_kmh / 3.6
        else:
            speed = raw_v
        self._th += dth
        self._x += d * (-math.sin(self._th))
        self._y += d * math.cos(self._th)
        self._dist += d
        p = Pose(frame, t, self._x, self._y, self._th, speed, self._dist, raw_v)
        self.poses.append(p)
        return p

    def _step(self, a: np.ndarray, b: np.ndarray, exclude) -> tuple[float, float] | None:
        s = self.scale
        h, w = a.shape
        y0 = self._horizon * s
        mask = np.zeros_like(a)
        top = int(max(y0 + 0.10 * h / 1.0, 0))
        mask[top:] = 255                                          # ground region only
        for (x1, y1, x2, y2) in exclude:                          # moving vehicles / people
            mask[max(int(y1 * s) - 4, 0):int(y2 * s) + 4, max(int(x1 * s) - 4, 0):int(x2 * s) + 4] = 0
        pts = cv2.goodFeaturesToTrack(a, maxCorners=220, qualityLevel=0.01, minDistance=7, mask=mask)
        if pts is None or len(pts) < 12:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, winSize=(21, 21), maxLevel=3)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=2)[:, 0] < 1.0)
        if ok.sum() < 10:
            return None
        p1, p2 = pts[ok, 0] / s, nxt[ok, 0] / s                   # back to full-resolution pixels
        v1, v2 = p1[:, 1] - self._horizon, p2[:, 1] - self._horizon
        good = (v1 > 0.08 * self.H) & (v2 > 0.08 * self.H)
        if good.sum() < 8:
            return None
        p1, p2, v1, v2 = p1[good], p2[good], v1[good], v2[good]
        hgt = self.cfg.cam_height_m
        z1, z2 = self.f * hgt / v1, self.f * hgt / v2
        dz = z1 - z2                                              # forward motion (positive when driving ahead)
        keep = np.abs(dz - np.median(dz)) < 2.5 * (1.4826 * np.median(np.abs(dz - np.median(dz))) + 1e-3)
        if keep.sum() < 6:
            return None
        d = float(np.median(dz[keep]))
        # yaw: horizontal residual after removing the forward-motion expansion (u2-cx = (u1-cx) * z1/z2)
        ratio = z1[keep] / np.maximum(z2[keep], 1e-3)
        pred_u2 = self.cx + (p1[keep, 0] - self.cx) * ratio
        resid = p2[keep, 0] - pred_u2
        dth_ground = float(np.median(resid)) / self.f             # biased by any error in the horizon / scale
        dth = self._far_yaw(a, b)
        if dth is None:                                           # no far-field texture: use a damped ground estimate
            dth = 0.25 * dth_ground
        return max(d, 0.0), float(np.clip(dth, -0.08, 0.08))

    def _far_yaw(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """Yaw from far-field features (near/above the horizon): translation barely moves them, rotation moves all
        depths by the same f*dtheta, so their median horizontal flow is a clean yaw measurement."""
        s = self.scale
        h, w = a.shape
        y0 = self._horizon * s
        lo, hi = int(max(y0 - 0.22 * h, 0)), int(min(y0 + 0.06 * h, h))
        if hi - lo < 8:
            return None
        mask = np.zeros_like(a)
        mask[lo:hi] = 255
        pts = cv2.goodFeaturesToTrack(a, maxCorners=120, qualityLevel=0.01, minDistance=7, mask=mask)
        if pts is None or len(pts) < 8:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, winSize=(21, 21), maxLevel=3)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=2)[:, 0] < 1.0)
        if ok.sum() < 8:
            return None
        du = (nxt[ok, 0, 0] - pts[ok, 0, 0]) / s
        return float(np.median(du)) / self.f

    # ---------------------------------------------------------------- results
    def smoothed_speed_kmh(self, win: int = 9) -> np.ndarray:
        v = np.array([p.speed_ms for p in self.poses], float) * 3.6
        if self.causal:                     # already causal-filtered online: no centred window (that would peek ahead)
            return v
        if len(v) < win:
            return v
        pad = win // 2
        vp = np.pad(v, pad, mode="edge")
        return np.array([np.median(vp[i:i + win]) for i in range(len(v))])

    def route(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.poses], float).reshape(-1, 2)

    def stats(self) -> dict:
        v = self.smoothed_speed_kmh() if self.poses else np.zeros(1)
        return {
            "assumed_camera_height_m": self.cfg.cam_height_m,
            "assumed_hfov_deg": self.cfg.hfov_deg,
            "horizon_row_px": round(float(self._horizon), 1),
            "route_length_m_approx": round(float(self._dist), 1),
            "mean_speed_kmh_approx": round(float(np.mean(v)), 1),
            "max_speed_kmh_approx": round(float(np.max(v)), 1),
            "frames_with_valid_motion": self.n_ok,
            "frames_total": self.n_total,
            "net_heading_change_deg": round(math.degrees(self._th), 1),
            "note": "monocular, scale from an assumed camera height: treat distances/speeds as rough estimates",
        }
