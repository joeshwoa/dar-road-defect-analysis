"""Road furniture: lane markings, crosswalks, traffic signs, traffic lights - and whether they are in good condition.

Locating things: a Mapillary-Vistas Mask2Former (65 classes incl. "Lane Marking - General", "Lane Marking - Crosswalk",
"Traffic Sign (Front)", "Traffic Light", "Pole") runs every N frames.  Judging condition has NO labelled training data
here, so it uses cheap, explainable cues instead of a trained "damage" classifier, and each cue is smoothed with the
PAST only (roadscan.temporal) so a verdict never flickers:

  lane markings   visibility = 0.5 * Weber contrast of the paint against the asphalt around it
                             + 0.5 * the segmenter's own confidence on the marking pixels
                  both are compared with a causal running 85th percentile of the same video (what "good paint" looks
                  like on THIS road) plus an absolute floor.  A stretch is reported as `lane_marking_worn` when its
                  smoothed visibility stays low for >= dwell seconds.
  crosswalks      same visibility score, per tracked crosswalk  ->  `crosswalk_faded`
  traffic signs   lean of the pole (PCA) or of a rectangular face off vertical + zero-shot CLIP on the crop
                  ("bent / broken / faded / hidden" vs "intact")  ->  `road_sign_damaged`
  traffic lights  lamp colour state by HSV (red / yellow / green / off) voted over the past; a lamp that never lights
                  for a long time, or a leaning housing  ->  `traffic_light_fault`

Everything above is heuristic and reported as "possible"; the intact items are listed too (road_furniture.csv) so a
reviewer can see what was looked at and judged fine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .classes import CROSSWALK_FADED_ID, MARKING_WORN_ID, SIGN_DAMAGED_ID, SIGNAL_FAULT_ID
from .config import FurnitureCfg
from .detectors import resolve_device
from .temporal import EMA, DecayedVotes, DwellState

# Mapillary-Vistas label names (matched case-insensitively against model.config.id2label)
LABEL_GROUPS = {
    "marking": ["lane marking - general"],
    "crosswalk": ["lane marking - crosswalk"],
    "sign": ["traffic sign (front)"],
    "light": ["traffic light"],
    "pole": ["pole", "traffic sign frame", "utility pole"],
    "road": ["road", "service lane", "bike lane", "crosswalk - plain", "parking"],
}
KINDS = ["sign", "light", "crosswalk"]
KIND_CLASS = {"sign": SIGN_DAMAGED_ID, "light": SIGNAL_FAULT_ID, "crosswalk": CROSSWALK_FADED_ID}
CLIP_PROMPTS = [
    "a photo of an intact, clean road sign",
    "a photo of a bent, broken or fallen road sign",
    "a photo of a heavily faded, rusty or vandalised road sign",
    "a photo of a road sign hidden behind vegetation or other objects",
]


# =========================================================================== pure analysis helpers (unit-tested)
def components(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    """Connected components of a boolean mask -> [{box, area, mask(crop)}]."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        out.append({"box": np.array([x, y, x + w, y + h], np.float32), "area": int(a),
                    "mask": (lab[y:y + h, x:x + w] == i)})
    return out


def light_state(crop_bgr: np.ndarray) -> tuple[str, int]:
    """Lit lamp colour of a traffic-light crop: ('red'|'yellow'|'green'|'off'|'unknown', lit pixel count)."""
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 8:
        return "unknown", 0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s >= 80) & (v >= 150)
    red = lit & ((h <= 10) | (h >= 165))
    yellow = lit & (h >= 15) & (h <= 35)
    green = lit & (h >= 40) & (h <= 95)
    counts = {"red": int(red.sum()), "yellow": int(yellow.sum()), "green": int(green.sum())}
    need = max(3, int(0.006 * crop_bgr.shape[0] * crop_bgr.shape[1]))
    best = max(counts, key=counts.get)
    if counts[best] >= need:
        return best, counts[best]
    return "off", 0


def lean_from_mask(mask: np.ndarray, min_len: int = 25, min_elong: float = 4.0) -> float | None:
    """Lean (deg from vertical, 0..90) of a tall thin component (a pole) by PCA; None if it is not pole-like."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    pts -= pts.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))            # ascending eigenvalues
    elong = float(np.sqrt(vals[1] / max(vals[0], 1e-3)))
    length = 3.4 * float(np.sqrt(vals[1]))                # ~ full extent along the principal axis
    if elong < min_elong or length < min_len:
        return None
    vx, vy = vecs[:, 1]
    return float(math.degrees(math.atan2(abs(vx), abs(vy))))     # 0 = vertical


def lean_from_face(mask: np.ndarray, min_fill: float = 0.80) -> float | None:
    """Tilt of a rectangular sign face (deg off the nearest axis) - None for round / triangular faces."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 150:
        return None
    (_, _), (rw, rh), ang = cv2.minAreaRect(c)
    if rw < 6 or rh < 6 or cv2.contourArea(c) / max(rw * rh, 1) < min_fill:
        return None
    dev = abs(ang) % 90
    return float(min(dev, 90 - dev))


def marking_stats(gray: np.ndarray, marking: np.ndarray, road: np.ndarray, score: np.ndarray | None,
                  zone: tuple[int, int], min_px: int = 60) -> dict[str, float] | None:
    """Paint contrast / segmenter confidence of the lane-marking pixels inside rows [zone) of the frame.

    Returns None when there is too little paint to judge (a road without markings is reported separately)."""
    y0, y1 = zone
    if y1 - y0 < 8:
        return None
    m = marking[y0:y1]
    if int(m.sum()) < min_px:
        return None
    g = gray[y0:y1].astype(np.float32)
    k3, k9 = np.ones((3, 3), np.uint8), np.ones((9, 9), np.uint8)
    m8 = m.astype(np.uint8)
    core = cv2.erode(m8, k3).astype(bool)
    if core.sum() < 20:
        core = m
    ring = (cv2.dilate(m8, k9).astype(bool)) & ~(cv2.dilate(m8, k3).astype(bool)) & road[y0:y1]
    if ring.sum() < 80:
        return None
    # the segmenter draws dashed lines as continuous strokes, so judge the PAINT: the brightest quarter of the mask
    vals = np.sort(g[core])
    lm = float(vals[-max(int(0.25 * len(vals)), 10):].mean())
    lr = float(np.median(g[ring]))
    weber = (lm - lr) / max(lr, 20.0)
    seg = float(np.mean(score[y0:y1][m])) if score is not None else 0.6
    ys, xs = np.nonzero(m)
    box = np.array([xs.min(), ys.min() + y0, xs.max() + 1, ys.max() + 1 + y0], np.float32)
    return {"weber": float(weber), "seg": seg, "px": int(m.sum()), "box": box}


class RunningQuantile:
    """Causal reference level: q-th percentile of everything seen so far (bounded memory)."""

    def __init__(self, q: float = 85.0, default: float = 0.5, min_n: int = 10, maxlen: int = 400):
        self.q, self.default, self.min_n, self.maxlen = q, default, min_n, maxlen
        self.buf: list[float] = []

    def update(self, x: float) -> None:
        self.buf.append(float(x))
        if len(self.buf) > self.maxlen:
            self.buf.pop(0)

    def ref(self) -> float:
        if len(self.buf) < self.min_n:
            return self.default
        return float(np.percentile(self.buf, self.q))


def visibility(weber: float, seg: float, ref_w: float, ref_s: float) -> float:
    cw = float(np.clip(weber / max(ref_w, 0.05), 0.0, 1.0))
    cs = float(np.clip(seg / max(ref_s, 0.2), 0.0, 1.0))
    return 0.5 * cw + 0.5 * cs


# =========================================================================== tiny greedy tracker
class SimpleTracker:
    """Greedy IoU / centre-distance association for boxes that are only observed every few frames."""

    def __init__(self, max_age: int = 4, iou_thr: float = 0.10, dist_factor: float = 1.2):
        self.max_age, self.iou_thr, self.dist_factor = max_age, iou_thr, dist_factor
        self.tracks: dict[int, dict[str, Any]] = {}
        self._next = 1

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(x2 - x1, 0) * max(y2 - y1, 0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, boxes: list[np.ndarray], kinds: list[str]) -> list[int]:
        ids = [-1] * len(boxes)
        used: set[int] = set()
        cand = []
        for i, (b, k) in enumerate(zip(boxes, kinds)):
            cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            for tid, t in self.tracks.items():
                if t["kind"] != k:
                    continue
                pb = t["box"] + 0.0
                pc = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
                side = max(b[2] - b[0], b[3] - b[1], pb[2] - pb[0], pb[3] - pb[1], 1.0)
                dist = float(np.linalg.norm(cb - pc)) / side
                iou = self._iou(b, pb)
                if iou >= self.iou_thr or dist <= self.dist_factor:
                    cand.append((1.0 - iou + 0.3 * dist, i, tid))
        for _c, i, tid in sorted(cand):
            if ids[i] != -1 or tid in used:
                continue
            ids[i] = tid
            used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] == -1:
                ids[i] = self._next
                self._next += 1
            self.tracks[ids[i]] = {"box": np.asarray(b, np.float32), "kind": kinds[i], "age": 0}
        for tid in list(self.tracks):
            if tid not in ids:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        return ids


# =========================================================================== item / marking state
@dataclass
class Item:
    kind: str
    first_frame: int
    first_pts: float
    last_frame: int = 0
    last_pts: float = 0.0
    n_obs: int = 0
    ema_bad: EMA = field(default_factory=lambda: EMA(0.35))
    state: DwellState = field(default_factory=lambda: DwellState(1.2))
    peak_bad: float = 0.0
    best_area: float = 0.0
    best_crop: np.ndarray | None = None
    best_frame: int = 0
    best_pts: float = 0.0
    best_box: np.ndarray | None = None
    lean: list[float] = field(default_factory=list)
    clip_bad: EMA = field(default_factory=lambda: EMA(0.4))
    light_votes: DecayedVotes = field(default_factory=lambda: DecayedVotes(0.92, 0.2))
    lit_obs: int = 0
    dark_obs: int = 0
    vis: EMA = field(default_factory=lambda: EMA(0.4))
    weber_med: list[float] = field(default_factory=list)
    flagged_obs: int = 0
    flagged_since: float | None = None


@dataclass
class FrameResult:
    items: list[tuple[int, np.ndarray, str, str, bool, float]]      # (id, box, kind, label, flagged, bad_ema)
    issue_obs: list[tuple[int, np.ndarray, float, int]]            # (raw_id, box, conf, class_id) for the defect list
    marking: dict[str, Any] | None
    marking_mask_half: np.ndarray | None


# =========================================================================== analyzer
class FurnitureAnalyzer:
    def __init__(self, cfg: FurnitureCfg, device: str = "auto", log=print):
        import torch
        from transformers import Mask2FormerForUniversalSegmentation

        self.cfg, self.log = cfg, log
        dev = resolve_device(device)
        self.torch = torch
        self.dev = torch.device("cpu" if dev == "cpu" else ("mps" if dev == "mps" else f"cuda:{dev}"))
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(cfg.model_id).to(self.dev).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ids = {g: [i for i, n in id2label.items() if n in names] for g, names in LABEL_GROUPS.items()}
        for g, v in self.ids.items():
            if not v and g in ("marking", "sign", "light"):
                raise RuntimeError(f"model {cfg.model_id} has no label for group '{g}'")
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.clip = None
        if cfg.clip_enabled:
            try:
                from transformers import CLIPModel, CLIPProcessor

                self.clip = CLIPModel.from_pretrained(cfg.clip_id).to(self.dev).eval()
                self.clip_proc = CLIPProcessor.from_pretrained(cfg.clip_id)
            except Exception as e:                                           # CLIP is optional
                self.log(f"  (furniture) CLIP unavailable, sign damage uses lean only: {e}")
        self._lut = {g: np.isin(np.arange(256), v) for g, v in self.ids.items()}
        self.reset(0, 0, 25.0)

    # ------------------------------------------------------------------ state
    def reset(self, width: int, height: int, fps: float) -> None:
        self.W, self.H, self.fps = width, height, fps
        self.tracker = SimpleTracker(max_age=self.cfg.track_max_age)
        self.items: dict[int, Item] = {}
        self.ref_w = RunningQuantile(85.0, 0.30)
        self.ref_s = RunningQuantile(85.0, 0.55)
        self.vis_ema = EMA(0.35)
        self.mark_state = DwellState(self.cfg.marking_dwell_s)
        self.mark_hist: list[dict[str, Any]] = []         # per update: t, vis, weber, seg, state, dist
        self.n_updates = 0
        self.n_mark_absent = 0
        self.episode = 0
        self._ep_open = False
        self.episodes: list[dict[str, Any]] = []
        self._last_frame = -10 ** 9

    # ------------------------------------------------------------------ segmentation
    def _segment(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (label map uint8, score map float32), both at frame resolution."""
        torch = self.torch
        h, w = img.shape[:2]
        scale = self.cfg.short_side / min(h, w)
        sh, sw = int(round(h * scale / 32)) * 32, int(round(w * scale / 32)) * 32
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        x = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
        pix = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.dev)
        with torch.inference_mode():
            if self.dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model(pixel_values=pix)
            else:
                out = self.model(pixel_values=pix)
            cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
            msk = out.masks_queries_logits.float().sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", cls, msk)
            seg = torch.nn.functional.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
            score, lab = seg.max(1)
        return lab[0].cpu().numpy().astype(np.uint8), score[0].cpu().numpy().astype(np.float32)

    def _group(self, lab: np.ndarray, g: str) -> np.ndarray:
        return self._lut[g][lab]

    # ------------------------------------------------------------------ zone of the road used to judge paint
    def _zone(self, horizon_y: float, ego) -> tuple[int, int]:
        H = self.H
        if ego is not None and getattr(ego, "f", None):
            hh = ego.cfg.cam_height_m
            far = horizon_y + ego.f * hh / self.cfg.marking_far_m
            near = horizon_y + ego.f * hh / self.cfg.marking_near_m
        else:
            far, near = horizon_y + 0.30 * (H - horizon_y), H
        y0 = int(np.clip(far, 0.30 * H, 0.9 * H))
        y1 = int(np.clip(near, y0 + 8, 0.97 * H))
        return y0, y1

    # ------------------------------------------------------------------ update
    def due(self, frame_idx: int) -> bool:
        return frame_idx - self._last_frame >= self.cfg.every_n

    def update(self, frame_idx: int, pts: float, img: np.ndarray, horizon_y: float, ego=None,
               route_dist_m: float | None = None) -> FrameResult | None:
        if not self.due(frame_idx):
            return None
        self._last_frame = frame_idx
        self.n_updates += 1
        cfg = self.cfg
        lab, score = self._segment(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        road = self._group(lab, "road") | self._group(lab, "marking") | self._group(lab, "crosswalk")
        issue_obs: list[tuple[int, np.ndarray, float, int]] = []

        # ---- lane markings (stretch level)
        marking = self._group(lab, "marking")
        mres = None
        zone = self._zone(horizon_y, ego)
        st = marking_stats(gray, marking, road, score, zone)
        mk: dict[str, Any] | None = None
        worn_now = False
        if st is None:
            self.n_mark_absent += 1
            mk = {"present": False, "state": self.mark_state.state or "unknown", "vis": self.vis_ema.value}
        else:
            self.ref_w.update(st["weber"])
            self.ref_s.update(st["seg"])
            vis = visibility(st["weber"], st["seg"], self.ref_w.ref(), self.ref_s.ref())
            v = self.vis_ema.update(vis)
            abs_bad = st["weber"] < cfg.marking_abs_weber
            cand = "worn" if (v < cfg.marking_worn_thr or (abs_bad and v < 0.7)) else (
                "ok" if v >= cfg.marking_ok_thr else (self.mark_state.state or "ok"))
            state = self.mark_state.update(pts, cand)
            worn_now = state == "worn"
            mk = {"present": True, "state": state, "vis": float(v), "weber": st["weber"], "seg": st["seg"],
                  "box": st["box"]}
            self.mark_hist.append({"t": pts, "frame": frame_idx, "vis": float(v), "weber": st["weber"], "seg": st["seg"],
                                   "state": state, "dist_m": route_dist_m if route_dist_m is not None else ""})
            if worn_now:
                if not self._ep_open:
                    self.episode += 1
                    self._ep_open = True
                    self.episodes.append({"id": self.episode, "start_s": pts, "start_dist": route_dist_m,
                                          "min_vis": float(v)})
                ep = self.episodes[-1]
                ep["end_s"], ep["end_dist"], ep["min_vis"] = pts, route_dist_m, min(ep["min_vis"], float(v))
                issue_obs.append((4_000_000 + self.episode, st["box"], float(np.clip(1.0 - v, 0.0, 1.0)),
                                  MARKING_WORN_ID))
            else:
                self._ep_open = False

        # ---- tracked items: signs, lights, crosswalks
        boxes, kinds, meta = [], [], []
        for kind in KINDS:
            mask = self._group(lab, "crosswalk" if kind == "crosswalk" else kind)
            for comp in components(mask, cfg.min_area_px[kind]):
                x1, y1, x2, y2 = [int(v) for v in comp["box"]]
                if y2 <= 0.05 * self.H and kind != "light":
                    continue
                if kind == "crosswalk" and y1 < horizon_y:
                    continue
                sc = float(np.mean(score[y1:y2, x1:x2][comp["mask"]]))
                boxes.append(comp["box"])
                kinds.append(kind)
                meta.append((comp, sc))
        ids = self.tracker.update(boxes, kinds)
        pole_mask = self._group(lab, "pole")
        out_items = []
        for tid, box, kind, (comp, sc) in zip(ids, boxes, kinds, meta):
            it = self.items.get(tid)
            if it is None:
                it = self.items[tid] = Item(kind, frame_idx, pts)
            it.last_frame, it.last_pts = frame_idx, pts
            it.n_obs += 1
            x1, y1, x2, y2 = [int(v) for v in box]
            area = float((x2 - x1) * (y2 - y1))
            crop = img[max(y1, 0):y2, max(x1, 0):x2]
            bad, label = self._judge(it, kind, box, comp, crop, pole_mask, img, ego, horizon_y)
            e = it.ema_bad.update(bad)
            it.peak_bad = max(it.peak_bad, float(e))
            on_thr, off_thr = cfg.bad_on[kind], cfg.bad_off[kind]
            cand = "bad" if e >= on_thr else ("ok" if e <= off_thr else (it.state.state or "ok"))
            state = it.state.update(pts, cand)
            flagged = state == "bad" and it.n_obs >= cfg.min_obs[kind]
            if area > it.best_area and crop.size:
                it.best_area, it.best_crop = area, crop.copy()
                it.best_frame, it.best_pts, it.best_box = frame_idx, pts, np.asarray(box, np.float32).copy()
            if flagged:
                it.flagged_obs += 1
                it.flagged_since = it.flagged_since if it.flagged_since is not None else pts
                issue_obs.append((5_000_000 + tid, np.asarray(box, np.float32), float(e), KIND_CLASS[kind]))
            out_items.append((tid, np.asarray(box, np.float32), kind, label, flagged, float(e)))
        half = cv2.resize(marking.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        return FrameResult(out_items, issue_obs, mk, half)

    # ------------------------------------------------------------------ condition cues per kind
    def _judge(self, it: Item, kind: str, box, comp, crop, pole_mask, img, ego, horizon_y) -> tuple[float, str]:
        cfg = self.cfg
        x1, y1, x2, y2 = [int(v) for v in box]
        if kind == "light":
            name, _n = light_state(crop)
            big = (y2 - y1) >= cfg.light_min_h_px
            if name in ("red", "yellow", "green"):
                it.lit_obs += 1
                it.light_votes.add({"red": 0, "yellow": 1, "green": 2}[name])
            elif name == "off" and big:
                it.dark_obs += 1
            lean = lean_from_face(comp["mask"])
            if lean is not None:
                it.lean.append(lean)
            seen = it.lit_obs + it.dark_obs
            dark_frac = it.dark_obs / seen if seen else 0.0
            # "dark" counts only after the lamp has been in view long enough and never once showed a lit lamp
            dark_bad = 1.0 if (it.lit_obs == 0 and it.dark_obs >= cfg.light_dark_obs) else 0.0
            lean_bad = float(np.clip((float(np.median(it.lean[-5:])) - 12.0) / 15.0, 0, 1)) if it.lean else 0.0
            lbl = ["red", "yellow", "green"][it.light_votes.label] if it.light_votes.label is not None else (
                "off?" if dark_frac > 0.5 else "?")
            return max(dark_bad, lean_bad), f"light:{lbl}"
        if kind == "sign":
            lean = lean_from_face(comp["mask"])
            # pole directly below / through the sign: lean of the connected pole component
            pm = pole_mask[y1:min(int(y2 + 1.2 * (y2 - y1) + 40), self.H), max(x1 - 8, 0):min(x2 + 8, self.W)]
            pole_lean = None
            if pm.size:
                comps = components(pm, 40)
                if comps:
                    pc = max(comps, key=lambda c: c["box"][3] - c["box"][1])
                    pole_lean = lean_from_mask(pc["mask"])
            val = pole_lean if pole_lean is not None else lean
            if val is not None:
                it.lean.append(val)
            lean_bad = float(np.clip((float(np.median(it.lean[-5:])) - cfg.lean_ok_deg) / 15.0, 0, 1)) if it.lean else 0.0
            cb = self._clip_bad(crop)
            clip_bad = 0.0
            if cb is not None:
                clip_bad = float(it.clip_bad.update(cb))
            elif it.clip_bad.value is not None:
                clip_bad = float(it.clip_bad.value)
            clip_score = float(np.clip((clip_bad - cfg.clip_bad_floor) / (1.0 - cfg.clip_bad_floor), 0, 1))
            return max(lean_bad, clip_score), "sign"
        # crosswalk: paint visibility of the crosswalk stripes relative to this video's good paint
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
        if gray is None or min(gray.shape) < 10:
            return 0.0, "crosswalk"
        cm = comp["mask"]
        ring_src = np.pad(cm, 12)
        k = np.ones((9, 9), np.uint8)
        ring = cv2.dilate(ring_src.astype(np.uint8), k).astype(bool) & ~ring_src
        ring = ring[12:-12, 12:-12]
        if cm.sum() < 40 or ring.sum() < 40:
            return 0.0, "crosswalk"
        g = gray.astype(np.float32)
        w = (float(g[cm].mean()) - float(np.median(g[ring]))) / max(float(np.median(g[ring])), 20.0)
        # per-crosswalk visibility vs the video's reference paint contrast
        vis = float(np.clip(w / max(self.ref_w.ref(), 0.05), 0.0, 1.0))
        it.weber_med.append(w)
        v = it.vis.update(vis)
        return float(1.0 - v), f"crosswalk vis {v:.2f}"

    def _clip_bad(self, crop: np.ndarray) -> float | None:
        if self.clip is None or crop is None or crop.size == 0 or min(crop.shape[:2]) < self.cfg.clip_min_px:
            return None
        torch = self.torch
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inp = self.clip_proc(text=CLIP_PROMPTS, images=pil, return_tensors="pt", padding=True).to(self.dev)
        with torch.inference_mode():
            p = self.clip(**inp).logits_per_image.float().softmax(-1)[0].cpu().numpy()
        return float(p[1] + p[2])           # bent/broken + faded; 'hidden' is ignored (CLIP over-predicts it on sign crops)

    # ------------------------------------------------------------------ results
    def inventory(self) -> list[dict[str, Any]]:
        rows = []
        n = 0
        for tid, it in sorted(self.items.items(), key=lambda kv: kv[1].first_pts):
            if it.n_obs < self.cfg.inventory_min_obs:
                continue
            n += 1
            lean = float(np.median(it.lean)) if it.lean else ""
            row = {"item_id": n, "kind": it.kind, "first_seen_s": round(it.first_pts, 3),
                   "last_seen_s": round(it.last_pts, 3), "n_observations": it.n_obs,
                   "max_box_area_px": int(it.best_area),
                   "condition": "possible_problem" if it.flagged_obs >= self.cfg.min_obs[it.kind] else "ok",
                   "peak_bad_score": round(it.peak_bad, 3),
                   "lean_deg_median": round(lean, 1) if lean != "" else "",
                   "lamp_state": (["red", "yellow", "green"][it.light_votes.label]
                                  if it.kind == "light" and it.light_votes.label is not None else ""),
                   "lamp_lit_obs": it.lit_obs if it.kind == "light" else "",
                   "lamp_dark_obs": it.dark_obs if it.kind == "light" else "",
                   "sign_clip_bad": round(float(it.clip_bad.value), 3) if (it.kind == "sign" and it.clip_bad.value is not None) else "",
                   "paint_visibility": round(float(it.vis.value), 3) if (it.kind == "crosswalk" and it.vis.value is not None) else "",
                   "_crop": it.best_crop}
            rows.append(row)
        return rows

    def marking_summary(self) -> dict[str, Any]:
        h = self.mark_hist
        total = self.n_updates
        out: dict[str, Any] = {
            "updates": total, "updates_with_visible_markings": len(h),
            "markings_visible_pct": round(100.0 * len(h) / max(total, 1), 1),
            "episodes_worn": len([e for e in self.episodes if "end_s" in e]),
            "episodes": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()} for e in self.episodes],
        }
        if h:
            vis = np.array([x["vis"] for x in h])
            wb = np.array([x["weber"] for x in h])
            out.update({"mean_visibility": round(float(vis.mean()), 3), "min_visibility": round(float(vis.min()), 3),
                        "median_paint_contrast_weber": round(float(np.median(wb)), 3),
                        "reference_contrast_p85": round(self.ref_w.ref(), 3),
                        "worn_time_pct": round(100.0 * float(np.mean([x["state"] == "worn" for x in h])), 1)})
        out["method"] = ("paint contrast + segmenter confidence vs a causal running reference of this video; "
                         "heuristic, not a calibrated pavement-marking retroreflectivity measurement")
        return out
