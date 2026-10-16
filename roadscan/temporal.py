"""Causal temporal stabilisation: every decision uses the PAST only (no look-ahead, no centred windows).

The idea is the one from sequence models, restricted to a left-only context window: a new measurement is never
trusted on its own, it is blended with what the previous frames said, with exponentially decaying weights, and a
category only flips when the new evidence is clearly stronger and has lasted a minimum time (hysteresis + dwell).

Building blocks (pure numpy / python, unit-tested):
    EMA              exponential moving average, NaN-safe
    CausalMedian     median of the last K samples
    HoldLimiter      rate limiter (max rise / max fall per second) for a physical quantity
    SpeedFilter      median -> EMA -> accel/decel limiter; holds the last value when the measurement is invalid
    DwellState       categorical state with hysteresis margin and minimum dwell time before it may change
    DecayedVotes     class votes with exponential forgetting and a switch margin (stable label per track)
    TrackScore       per-track confidence memory with ON/OFF hysteresis and a short coasting period
    TemporalPrior    feeds the previous frame's confirmed tracks back into the NEXT detection step
                     (confidence blend + class-vote relabel), the "attention over the past" of the detector
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------- scalar filters
class EMA:
    def __init__(self, alpha: float, init: float | None = None):
        self.alpha = float(alpha)
        self.value = init

    def update(self, x: float | None) -> float | None:
        if x is None or not math.isfinite(x):
            return self.value
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x
        return self.value


class CausalMedian:
    def __init__(self, k: int):
        self.buf: deque[float] = deque(maxlen=max(int(k), 1))

    def update(self, x: float | None) -> float | None:
        if x is not None and math.isfinite(x):
            self.buf.append(float(x))
        return float(np.median(self.buf)) if self.buf else None


class CausalTrimmedMean:
    """Mean of the last K samples after dropping the `trim` share of lowest and highest ones.

    Unlike the median it conserves the average of quantised / alternating signals (a video with duplicated frames
    gives per-frame speeds like 4.8, 9.6, 4.8, 9.6 whose median is 20 % too low) while still rejecting a handful of
    outliers (flow lost for a few frames -> 0, a bump -> spike)."""

    def __init__(self, k: int, trim: float = 0.2):
        self.buf: deque[float] = deque(maxlen=max(int(k), 1))
        self.trim = float(trim)

    def update(self, x: float | None) -> float | None:
        if x is not None and math.isfinite(x):
            self.buf.append(float(x))
        if not self.buf:
            return None
        v = np.sort(np.asarray(self.buf, float))
        c = int(math.floor(self.trim * len(v)))
        v = v[c:len(v) - c] if len(v) - 2 * c >= 1 else v
        return float(v.mean())


class HoldLimiter:
    """Limit how fast a value may rise / fall per second (physical plausibility)."""

    def __init__(self, max_rise_per_s: float, max_fall_per_s: float):
        self.rise, self.fall = float(max_rise_per_s), float(max_fall_per_s)
        self.value: float | None = None
        self._t: float | None = None

    def update(self, t: float, x: float | None) -> float | None:
        if x is None or not math.isfinite(x):
            self._t = t if self._t is None else self._t
            return self.value
        if self.value is None or self._t is None:
            self.value, self._t = float(x), t
            return self.value
        dt = max(t - self._t, 1e-3)
        self._t = t
        lo, hi = self.value - self.fall * dt, self.value + self.rise * dt
        self.value = float(min(max(x, lo), hi))
        return self.value


class SpeedFilter:
    """Ego speed: causal trimmed mean over ~`median_s` -> EMA -> accel/decel limiter. Invalid samples are held.

    A momentary 40 -> 0 -> 40 km/h dip (flow lost on a bump, a vehicle filling the view) cannot pass: the trimmed mean
    drops up to 20 % outliers of the window, a longer dip is only partly followed, and the limiter caps braking at
    `max_decel_ms2` (a real stop still comes through, with a delay of about one window).
    """

    def __init__(self, fps: float, median_s: float = 0.6, ema_alpha: float = 0.35,
                 max_accel_ms2: float = 3.5, max_decel_ms2: float = 6.0, max_kmh: float = 140.0):
        self.med = CausalTrimmedMean(max(int(round(median_s * fps)), 8), 0.3)
        self.ema = EMA(ema_alpha)
        self.lim = HoldLimiter(max_accel_ms2 * 3.6, max_decel_ms2 * 3.6)     # km/h per second
        self.max_kmh = max_kmh
        self.last: float = 0.0
        self.n_valid = 0
        self.warmup = 6              # robust start: no output until a few samples agree (first flow frames are junk)

    def update(self, t: float, speed_kmh: float | None, valid: bool = True) -> float:
        x = None
        if valid and speed_kmh is not None and math.isfinite(speed_kmh):
            x = float(min(max(speed_kmh, 0.0), self.max_kmh))
        m = self.med.update(x)
        if x is not None:
            self.n_valid += 1
        if self.n_valid < self.warmup:      # start-up: a single bogus first sample (e.g. 140 km/h) must not seed the filter
            self.last = 0.0
            return self.last
        e = self.ema.update(m)
        v = self.lim.update(t, e)
        self.last = 0.0 if v is None else float(v)
        return self.last


# --------------------------------------------------------------------------- categorical state
class DwellState:
    """Categorical state that changes only when another category has led by `margin` for `dwell_s` seconds."""

    def __init__(self, dwell_s: float = 1.5):
        self.dwell_s = float(dwell_s)
        self.state: Hashable | None = None
        self._cand: Hashable | None = None
        self._since: float = 0.0

    def update(self, t: float, candidate: Hashable) -> Hashable:
        if self.state is None:
            self.state = candidate
            return self.state
        if candidate == self.state:
            self._cand = None
            return self.state
        if candidate != self._cand:
            self._cand, self._since = candidate, t
        if t - self._since >= self.dwell_s:
            self.state, self._cand = candidate, None
        return self.state


def hysteresis_class(ratio: float, current: str | None, lo: float, hi: float, margin: float = 0.05) -> str:
    """paved / mixed / unpaved from a ratio with a hysteresis band of +-margin around the two thresholds."""
    if not math.isfinite(ratio):
        return current or "mixed"
    if current == "unpaved":
        hi = hi - margin
    elif current == "paved":
        lo = lo + margin
    elif current == "mixed":
        lo, hi = lo - margin, hi + margin
    return "unpaved" if ratio >= hi else ("paved" if ratio <= lo else "mixed")


class DecayedVotes:
    """Exponentially forgetting class votes; the label flips only if a rival beats it by `margin` (share)."""

    def __init__(self, decay: float = 0.9, margin: float = 0.15):
        self.decay, self.margin = decay, margin
        self.w: dict[int, float] = {}
        self.label: int | None = None

    def add(self, cls: int, weight: float = 1.0) -> int:
        for k in self.w:
            self.w[k] *= self.decay
        self.w[int(cls)] = self.w.get(int(cls), 0.0) + float(weight)
        tot = sum(self.w.values()) or 1.0
        best = max(self.w, key=self.w.get)
        if self.label is None or self.label not in self.w:
            self.label = best
        elif best != self.label and (self.w[best] - self.w[self.label]) / tot >= self.margin:
            self.label = best
        return self.label

    def share(self, cls: int) -> float:
        tot = sum(self.w.values()) or 1.0
        return self.w.get(int(cls), 0.0) / tot


# --------------------------------------------------------------------------- per-track memory
@dataclass
class TrackScore:
    """Confidence memory of one track: EMA of its score, hit counter, ON/OFF hysteresis, coasting.

    on_thr/off_thr: the track becomes "confirmed" once the EMA reaches on_thr after `min_hits` hits and stays
    confirmed until the EMA falls below off_thr (< on_thr). A missed frame decays the EMA (miss_decay) and the
    last box is still drawn for `coast` frames, so a one-frame detector dropout does not make a box blink.
    """

    alpha: float = 0.4
    on_thr: float = 0.35
    off_thr: float = 0.20
    min_hits: int = 2
    miss_decay: float = 0.85
    coast: int = 6
    ema: float = 0.0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    ever_confirmed: bool = False
    last_box: Optional[np.ndarray] = None
    last_frame: int = -1
    cls_votes: DecayedVotes = field(default_factory=DecayedVotes)

    def hit(self, frame: int, conf: float, box: np.ndarray, cls: int) -> None:
        self.ema = conf if self.hits == 0 else (1 - self.alpha) * self.ema + self.alpha * conf
        self.hits += 1
        self.misses = 0
        self.last_box, self.last_frame = np.asarray(box, np.float32).copy(), frame
        self.cls_votes.add(cls, conf)
        self._refresh()

    def miss(self) -> None:
        self.misses += 1
        self.ema *= self.miss_decay
        self._refresh()

    def _refresh(self) -> None:
        if not self.confirmed and self.hits >= self.min_hits and self.ema >= self.on_thr:
            self.confirmed = True
            self.ever_confirmed = True
        elif self.confirmed and self.ema < self.off_thr:
            self.confirmed = False

    @property
    def visible(self) -> bool:
        """Draw it: confirmed, and either seen now or lost for less than `coast` frames."""
        return self.confirmed and self.misses <= self.coast


class TrackMemory:
    """TrackScore for every raw tracker id, updated once per frame with the ids seen in that frame."""

    def __init__(self, **kw):
        self.kw = kw
        self.tracks: dict[int, TrackScore] = {}

    def update(self, frame: int, seen: Iterable[tuple[int, float, np.ndarray, int]]) -> dict[int, TrackScore]:
        seen_ids = set()
        for tid, conf, box, cls in seen:
            ts = self.tracks.setdefault(tid, TrackScore(**self.kw))
            ts.hit(frame, float(conf), box, int(cls))
            seen_ids.add(tid)
        for tid, ts in self.tracks.items():
            if tid not in seen_ids:
                ts.miss()
        return self.tracks

    def visible_boxes(self) -> list[tuple[int, np.ndarray, float, int, int]]:
        """[(tid, box, ema_conf, cls_label, misses)] of tracks that should be drawn right now."""
        out = []
        for tid, ts in self.tracks.items():
            if ts.visible and ts.last_box is not None:
                out.append((tid, ts.last_box, ts.ema, ts.cls_votes.label if ts.cls_votes.label is not None else -1,
                            ts.misses))
        return out


# --------------------------------------------------------------------------- feedback into detection
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class TemporalPrior:
    """Feed the previous frames back into the next detection step (past only).

    For every new detection that overlaps (IoU >= iou_thr after a coarse camera-motion shift) a box that was
    tracked and confirmed in the previous frame:
      * its confidence is blended with that track's EMA:  c' = (1-w)*c + w*ema,  capped at c + max_boost, so a weak
        but consistent defect is not dropped for one bad frame, and an isolated strong blip is not over-trusted;
      * its class is replaced by the track's voted class when both belong to the same compatibility group
        (crack sub-types flicker between frames; potholes are never relabelled as cracks).
    Detections with no support keep their raw confidence (the tracker's new-track threshold still guards them).
    """

    def __init__(self, memory: TrackMemory, weight: float = 0.35, max_boost: float = 0.15, iou_thr: float = 0.30,
                 compatible: list[list[int]] | None = None, max_age: int = 2):
        self.mem, self.w, self.max_boost, self.iou_thr = memory, weight, max_boost, iou_thr
        self.compatible = compatible or []
        self.max_age = max_age
        self.n_boosted = 0
        self.n_relabelled = 0

    def _same_group(self, a: int, b: int) -> bool:
        return a == b or any(a in g and b in g for g in self.compatible)

    def adjust(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, frame: int
               ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (new_conf, new_cls) for the detections; inputs are not modified."""
        conf2, cls2 = conf.copy(), cls.copy()
        prev = [(ts.last_box, ts.ema, ts.cls_votes.label) for ts in self.mem.tracks.values()
                if ts.confirmed and ts.last_box is not None and frame - ts.last_frame <= self.max_age + 1]
        if not prev:
            return conf2, cls2
        for i in range(len(xyxy)):
            best, bi = 0.0, None
            for j, (b, _e, _l) in enumerate(prev):
                v = _iou(xyxy[i], b)
                if v > best:
                    best, bi = v, j
            if bi is None or best < self.iou_thr:
                continue
            _b, ema, lab = prev[bi]
            blended = (1 - self.w) * float(conf[i]) + self.w * float(ema)
            if blended > conf2[i]:
                conf2[i] = min(blended, float(conf[i]) + self.max_boost, 0.95)
                self.n_boosted += 1
            if lab is not None and int(lab) != int(cls[i]) and self._same_group(int(lab), int(cls[i])):
                cls2[i] = float(lab)
                self.n_relabelled += 1
        return conf2, cls2


class DisplaySmoother:
    """Past-only calming of what is DRAWN: one box per defect, EMA-smoothed coordinates, a box that coasts a few
    frames (constant damped velocity) after a detection dropout, and a confidence label that never goes down
    (best value seen so far). Nothing here looks at future frames."""

    def __init__(self, alpha: float = 0.5, hold: int = 8, vel_decay: float = 0.85):
        self.alpha, self.hold, self.vel_decay = alpha, hold, vel_decay
        self.state: dict[int, dict] = {}

    def update(self, frame: int, obs: dict) -> list[tuple[int, np.ndarray, float]]:
        """obs: final_id -> (box xyxy, conf) seen this frame. Returns [(final_id, box, conf)] to draw."""
        for fid, (box, conf) in obs.items():
            box = np.asarray(box, np.float64)
            st = self.state.get(fid)
            if st is None or frame - st["last"] > self.hold:
                self.state[fid] = {"box": box.copy(), "vel": np.zeros(4), "last": frame, "cmax": float(conf)}
                continue
            new = st["box"] * (1 - self.alpha) + box * self.alpha
            dt = max(frame - st["last"], 1)
            st["vel"] = 0.6 * st["vel"] + 0.4 * (new - st["box"]) / dt
            st["box"], st["last"], st["cmax"] = new, frame, max(st["cmax"], float(conf))
        out = []
        for fid, st in self.state.items():
            gap = frame - st["last"]
            if gap == 0:
                out.append((fid, st["box"].copy(), st["cmax"]))
            elif gap <= self.hold:
                k = sum(self.vel_decay ** i for i in range(1, gap + 1))
                out.append((fid, st["box"] + st["vel"] * k, st["cmax"]))
        return out

    def last_box(self, fid: int):
        st = self.state.get(fid)
        return None if st is None else st["box"]
