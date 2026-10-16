"""Road-surface analysis: paved vs unpaved, sidewalk presence, standing water.

An ADE20K SegFormer labels every pixel; classes are grouped into

    1 paved      road
    2 unpaved    earth / dirt track / sand / path
    3 water      water / river / lake          (puddles on the road)
    4 sidewalk   sidewalk
    5 vegetation grass / field

Per processed frame we measure, inside a region of interest below the horizon:
    unpaved_ratio  = share of the drivable surface that is unpaved (water counts as unpaved when the road is)
    sidewalk share on the left and on the right of the frame
and smooth them over time into *segments*  (surface = paved | unpaved | mixed, sidewalk = none | left | right | both).

Standing water on an unpaved / damaged road is turned into `water_filled_pothole` candidates (the pothole detector
trained on RDD2022 has very few water-filled examples, and those are the potholes it misses most often).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

from .classes import WATER_POTHOLE_ID
from .config import SurfaceCfg
from .detectors import Dets, resolve_device
from .temporal import DwellState, hysteresis_class

PAVED, UNPAVED, WATER, SIDEWALK, VEG = 1, 2, 3, 4, 5
_GROUPS = {PAVED: [6], UNPAVED: [13, 46, 52, 91], WATER: [21, 60, 128], SIDEWALK: [11], VEG: [9, 29, 17]}
LUT = np.zeros(256, np.uint8)
for _g, _ids in _GROUPS.items():
    for _i in _ids:
        LUT[_i] = _g

# overlay colours (BGR)
TINT = {UNPAVED: (0, 140, 255), WATER: (255, 190, 0), SIDEWALK: (60, 200, 60)}


class SurfaceAnalyzer:
    def __init__(self, cfg: SurfaceCfg, device: str = "auto"):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.cfg = cfg
        dev = resolve_device(device)
        self.torch_device = torch.device("cpu" if dev == "cpu" else ("mps" if dev == "mps" else f"cuda:{dev}"))
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(cfg.model_id).to(self.torch_device).eval()
        self.half = self.torch_device.type == "cuda"
        if self.half:
            self.model.half()
        self.group: np.ndarray | None = None       # uint8 (H, W) of the latest analysed frame
        self.records: list[dict[str, Any]] = []
        self._horizons: list[float] = []
        self._last_frame_idx = -10 ** 9
        self._H = self._W = 0
        self.last: dict[str, Any] = {}

    # ---------------------------------------------------------------- segmentation
    def _segment(self, img: np.ndarray) -> np.ndarray:
        import torch

        h, w = img.shape[:2]
        sw = self.cfg.input_width
        sh = max(64, int(round(h * sw / w / 32)) * 32)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA if sw < w else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pix = self.processor(images=rgb, return_tensors="pt", do_resize=False)["pixel_values"].to(self.torch_device)
        if self.half:
            pix = pix.half()
        with torch.inference_mode():
            lg = self.model(pixel_values=pix).logits.float()
            lg = torch.nn.functional.interpolate(lg, size=(sh, sw), mode="bilinear", align_corners=False)
            ade = lg.argmax(1)[0].cpu().numpy().astype(np.uint8)
        grp = LUT[ade]
        return cv2.resize(grp, (w, h), interpolation=cv2.INTER_NEAREST)

    def reset(self) -> None:
        """Forget everything about the previous video (models stay loaded)."""
        self.group = None
        self.records = []
        self._horizons = []
        self._last_frame_idx = -10 ** 9
        self.last = {}

    def horizon_y(self, h: int | None = None) -> float:
        h = h or self._H
        if self._horizons:
            return float(np.median(self._horizons))
        return 0.42 * h

    def roi_top(self) -> int:
        return int(self.horizon_y() + self.cfg.roi_top_margin * self._H)

    # ---------------------------------------------------------------- per frame
    def update(self, frame_idx: int, pts: float, img: np.ndarray) -> bool:
        """Analyse this frame if it is due. Returns True when a new analysis was produced."""
        if frame_idx - self._last_frame_idx < self.cfg.every_n and self.group is not None:
            return False
        self._last_frame_idx = frame_idx
        h, w = img.shape[:2]
        self._H, self._W = h, w
        g = self._segment(img)
        self.group = g

        road_like = (g == PAVED) | (g == UNPAVED) | (g == WATER)
        share = road_like.mean(axis=1)
        rows = np.nonzero(share > 0.30)[0]
        if len(rows) and len(self._horizons) < 400:
            self._horizons.append(float(rows[0]))
        top = self.roi_top()
        roi = g[top:]
        n = max(roi.size, 1)
        paved = float((roi == PAVED).sum()) / n
        unpaved = float((roi == UNPAVED).sum()) / n
        water = float((roi == WATER).sum()) / n
        half = w // 2
        left, right = roi[:, :half], roi[:, half:]
        sw_l = float((left == SIDEWALK).sum()) / max(left.size, 1)
        sw_r = float((right == SIDEWALK).sum()) / max(right.size, 1)
        drivable = paved + unpaved + water
        if drivable < 0.03:
            ratio, state = float("nan"), "unknown"
        else:
            wet_to_unpaved = water if unpaved >= paved else 0.0
            ratio = (unpaved + wet_to_unpaved) / drivable
            state = "unpaved" if ratio >= self.cfg.unpaved_thr else ("paved" if ratio <= self.cfg.paved_thr else "mixed")
        rec = {"frame": frame_idx, "t": pts, "paved": paved, "unpaved": unpaved, "water": water,
               "sidewalk_left": sw_l, "sidewalk_right": sw_r, "unpaved_ratio": ratio, "state": state,
               "horizon_y": self.horizon_y()}
        self.records.append(rec)
        self.last = rec
        return True

    # ---------------------------------------------------------------- water -> defect candidates
    def water_dets(self, img: np.ndarray) -> Dets:
        cfg = self.cfg
        if not cfg.water_detect or self.group is None or not self.last:
            return Dets.empty()
        if not (self.last["unpaved_ratio"] >= 0.25):       # puddles on good pavement are not potholes
            return Dets.empty()
        g = self.group
        h, w = g.shape
        top = self.roi_top()
        mask = np.zeros((h, w), np.uint8)
        mask[top:] = (g[top:] == WATER).astype(np.uint8)
        seg_mask = mask.copy()
        if cfg.bright_puddles:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bright = ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 55) & (g != VEG)).astype(np.uint8)
            bright[:top + int(0.03 * h)] = 0
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = np.maximum(mask, bright)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_area = float(h * w)
        boxes, confs = [], []
        near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        vch = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2], (0, 0), 2)
        for i in range(1, n_lab):
            x, y, bw, bh, area = stats[i]
            if area < cfg.water_min_area_frac * frame_area or bw < 5 or bh < 3:
                continue
            comp = (lab == i).astype(np.uint8)
            if area > 0.012 * frame_area:                    # wet mud + several puddles merged: split on the bright water surfaces
                parts = _bright_cores(comp, vch) or _split_blob(comp)
            else:
                parts = [comp]
            for part in parts:
                a = int(part.sum())
                if a < cfg.water_min_area_frac * frame_area or a > cfg.water_max_area_frac * frame_area:
                    continue
                ys, xs = np.nonzero(part)
                x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                if (x2 - x1) / max(y2 - y1, 1) > 14:
                    continue
                ring = cv2.dilate(part, near) - part
                ring_g = g[ring > 0]
                if ring_g.size == 0:
                    continue
                surround = np.isin(ring_g, [PAVED, UNPAVED, SIDEWALK, WATER]).mean()
                if surround < 0.55:                         # water beside the road (pond, river), not on it
                    continue
                seg_share = float(seg_mask[part > 0].mean())
                hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
                solidity = a / max(cv2.contourArea(hull), 1.0)
                conf = 0.38 + 0.30 * seg_share + 0.15 * min(1.0, a / 1500.0) + 0.12 * min(1.0, solidity)
                boxes.append([x1, y1, x2, y2])
                confs.append(min(conf, 0.92))
        if not boxes:
            return Dets.empty()
        # drop boxes that sit (mostly) inside a larger one - over-split puddles
        order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        keep: list[int] = []
        for i in order:
            a = boxes[i]
            aa = max((a[2] - a[0]) * (a[3] - a[1]), 1)
            dup = False
            for j in keep:
                b = boxes[j]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw > 0 and ih > 0 and iw * ih / aa > 0.7:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        boxes = [boxes[i] for i in keep]
        confs = [confs[i] for i in keep]
        return Dets(np.asarray(boxes, np.float32), np.asarray(confs, np.float32),
                    np.full(len(boxes), float(WATER_POTHOLE_ID), np.float32))

    # ---------------------------------------------------------------- overlay
    def overlay(self, canvas: np.ndarray, group: np.ndarray | None = None, alpha: float = 0.30,
                water: bool = True) -> None:
        """Colour unpaved ground (orange), standing water (blue) and sidewalk (green). Paved road stays clear."""
        g = group if group is not None else self.group
        if g is None or not self.cfg.tint:
            return
        if g.shape[:2] != canvas.shape[:2]:
            g = cv2.resize(g, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
        top = max(int(self.horizon_y(canvas.shape[0])) - 4, 0)
        region = canvas[top:]
        gr = g[top:]
        for grp, col in TINT.items():
            if grp == WATER and not water:
                continue
            m = gr == grp
            if m.any():
                region[m] = (region[m] * (1 - alpha) + np.array(col, np.float32) * alpha).astype(np.uint8)

    # ---------------------------------------------------------------- aggregation
    def finalize(self, fps: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Smooth the per-frame measurements into road segments + a route-level summary."""
        recs = [r for r in self.records]
        if not recs:
            return [], {"surface_overall": "unknown", "findings": ["road surface analysis produced no data"]}
        t = np.array([r["t"] for r in recs])
        ratio = np.array([r["unpaved_ratio"] for r in recs], float)
        swl = np.array([r["sidewalk_left"] for r in recs], float)
        swr = np.array([r["sidewalk_right"] for r in recs], float)
        paved = np.array([r["paved"] for r in recs], float)
        unp = np.array([r["unpaved"] for r in recs], float)
        wat = np.array([r["water"] for r in recs], float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
        win = max(3, int(round(self.cfg.smooth_s / max(dt, 1e-3))) | 1)

        def roll(x, med=False):
            x = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x)
            if self.cfg.causal:                       # trailing window: sample i sees only samples <= i
                out = np.empty_like(x)
                for i in range(len(x)):
                    seg = x[max(0, i - win + 1):i + 1]
                    out[i] = np.median(seg) if med else seg.mean()
                return out
            pad = win // 2
            xp = np.pad(x, pad, mode="edge")
            out = np.empty_like(x)
            for i in range(len(x)):
                seg = xp[i:i + win]
                out[i] = np.median(seg) if med else seg.mean()
            return out

        ratio_s = roll(ratio, med=True)
        swl_s, swr_s = roll(swl), roll(swr)
        thr = self.cfg.sidewalk_min_frac
        if self.cfg.causal:
            # hysteresis band around the thresholds + minimum dwell: a state flips only when the new value has led
            # for dwell_s, so paved<->mixed or sidewalk on<->off cannot chatter, and nothing looks into the future
            st_s, st_l, st_r = DwellState(self.cfg.dwell_s), DwellState(self.cfg.dwell_s), DwellState(self.cfg.dwell_s)
            cur = None
            surf_l, left_l, right_l = [], [], []
            for i in range(len(t)):
                cand = hysteresis_class(float(ratio_s[i]), st_s.state, self.cfg.paved_thr, self.cfg.unpaved_thr, 0.05)
                surf_l.append(st_s.update(float(t[i]), cand))
                lc = (swl_s[i] >= thr) if st_l.state != True else (swl_s[i] >= 0.6 * thr)      # noqa: E712
                rc = (swr_s[i] >= thr) if st_r.state != True else (swr_s[i] >= 0.6 * thr)      # noqa: E712
                left_l.append(bool(st_l.update(float(t[i]), bool(lc))))
                right_l.append(bool(st_r.update(float(t[i]), bool(rc))))
            surf = np.array(surf_l)
            left, right = np.array(left_l), np.array(right_l)
        else:
            surf = np.where(ratio_s >= self.cfg.unpaved_thr, "unpaved",
                            np.where(ratio_s <= self.cfg.paved_thr, "paved", "mixed"))
            left, right = swl_s >= thr, swr_s >= thr
        sidewalk = np.where(left & right, "both", np.where(left, "left", np.where(right, "right", "none")))
        keys = list(zip(surf, sidewalk))

        # run-length encode, then absorb runs shorter than min_segment_s into their longer neighbour
        runs: list[list[Any]] = []
        for i, k in enumerate(keys):
            if runs and runs[-1][0] == k:
                runs[-1][2] = i
            else:
                runs.append([k, i, i])
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for j, (k, a, b) in enumerate(runs):
                if (t[b] - t[a] + dt) < self.cfg.min_segment_s:
                    nb = j - 1 if j > 0 else j + 1
                    if j > 0 and j + 1 < len(runs):
                        nb = j - 1 if (runs[j - 1][2] - runs[j - 1][1]) >= (runs[j + 1][2] - runs[j + 1][1]) else j + 1
                    runs[nb][1] = min(runs[nb][1], a)
                    runs[nb][2] = max(runs[nb][2], b)
                    runs.pop(j)
                    changed = True
                    break
            # merge neighbours that now share a key
            k2: list[list[Any]] = []
            for r in runs:
                if k2 and k2[-1][0] == r[0]:
                    k2[-1][2] = r[2]
                else:
                    k2.append(r)
            runs = k2

        segments = []
        for n, (k, a, b) in enumerate(runs, start=1):
            sl = slice(a, b + 1)
            segments.append({
                "segment_id": n,
                "start_s": round(float(t[a]), 2), "end_s": round(float(t[b] + dt), 2),
                "duration_s": round(float(t[b] - t[a] + dt), 2),
                "surface": k[0], "sidewalk": k[1],
                "unpaved_pct": round(100 * float(np.nanmean(ratio[sl])), 1) if np.isfinite(ratio[sl]).any() else "",
                "paved_pct_of_roi": round(100 * float(paved[sl].mean()), 1),
                "unpaved_pct_of_roi": round(100 * float(unp[sl].mean()), 1),
                "standing_water_pct_of_roi": round(100 * float(wat[sl].mean()), 2),
                "sidewalk_left_pct": round(100 * float(swl[sl].mean()), 2),
                "sidewalk_right_pct": round(100 * float(swr[sl].mean()), 2),
            })

        total = float(t[-1] - t[0] + dt)
        share = Counter()
        for s in segments:
            share[s["surface"]] += s["duration_s"]
        sw_share = Counter()
        for s in segments:
            sw_share[s["sidewalk"]] += s["duration_s"]
        pct = lambda c, k: round(100 * c.get(k, 0.0) / max(total, 1e-6), 1)
        overall = max(("paved", "unpaved", "mixed"), key=lambda k: share.get(k, 0.0))
        no_sw = pct(sw_share, "none")
        findings = []
        if pct(share, "unpaved") + pct(share, "mixed") >= 50:
            findings.append(f"Road is predominantly UNPAVED / not properly surfaced "
                            f"({pct(share, 'unpaved')}% unpaved, {pct(share, 'mixed')}% mixed).")
        else:
            findings.append(f"Road is predominantly paved ({pct(share, 'paved')}% of the route).")
        if no_sw >= 50:
            findings.append(f"NO sidewalk along {no_sw}% of the route - missing pedestrian infrastructure.")
        else:
            findings.append(f"Sidewalk present on at least one side for {round(100 - no_sw, 1)}% of the route.")
        if float(np.mean(wat)) > 0.01:
            findings.append(f"Standing water covers on average {round(100 * float(np.mean(wat)), 1)}% of the road area.")
        summary = {
            "duration_s": round(total, 2),
            "surface_overall": overall,
            "surface_time_share_pct": {k: pct(share, k) for k in ("paved", "mixed", "unpaved")},
            "sidewalk_time_share_pct": {k: pct(sw_share, k) for k in ("none", "left", "right", "both")},
            "sidewalk_present_pct": round(100 - no_sw, 1),
            "mean_unpaved_ratio": round(float(np.nanmean(ratio)), 3),
            "mean_standing_water_pct_of_roi": round(100 * float(np.mean(wat)), 2),
            "horizon_y_px": round(self.horizon_y(), 1),
            "findings": findings,
            "n_segments": len(segments),
            "method": f"{self.cfg.model_id} (ADE20K), region of interest = below horizon, "
                      + (f"causal (past-only) smoothing {self.cfg.smooth_s}s + hysteresis, state change dwell {self.cfg.dwell_s}s"
                         if self.cfg.causal else f"centred smoothing {self.cfg.smooth_s}s"),
        }
        return segments, summary


def _bright_cores(comp: np.ndarray, vch: np.ndarray) -> list[np.ndarray]:
    """Individual puddles inside a merged water/wet-mud blob: puddle surfaces reflect the sky and are brighter."""
    vals = vch[comp > 0]
    if vals.size < 50:
        return []
    thr = max(float(np.percentile(vals, 60)), float(np.median(vals)) + 8.0)
    cores = ((comp > 0) & (vch > thr)).astype(np.uint8)
    cores = cv2.morphologyEx(cores, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cores = cv2.morphologyEx(cores, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    parts = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 60:
            continue
        part = cv2.dilate((lab == i).astype(np.uint8), k) & comp
        parts.append(part)
    return parts


def _split_blob(comp: np.ndarray) -> list[np.ndarray]:
    """Split a big merged water blob into separate puddles (distance-transform peaks + watershed)."""
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    if dist.max() < 4:
        return [comp]
    cores = (dist > 0.55 * dist.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(cores)
    if n <= 2:
        return [comp]
    m = np.zeros(comp.shape, np.int32)                       # 0 = unknown (to be flooded)
    m[markers > 0] = markers[markers > 0] + 1                # puddle cores: labels 2..n
    m[comp == 0] = 1                                         # background
    img = cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    m = cv2.watershed(img, m)
    parts = []
    for lab in range(2, n + 1):
        p = ((m == lab) & (comp > 0)).astype(np.uint8)
        if p.sum() > 0:
            parts.append(p)
    return parts or [comp]
