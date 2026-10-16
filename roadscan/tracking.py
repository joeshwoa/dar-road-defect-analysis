"""Tracking + 'report each defect once' logic.

1. `DefectTracker` wraps the Ultralytics BoT-SORT / ByteTrack implementations as a standalone
   tracker (we own detection and filtering, the tracker only associates).
   BoT-SORT adds global-motion compensation, which matters because the camera is moving.
2. `finalize_tracks` post-processes raw tracks: it merges fragmented tracks and duplicate tracks,
   votes the defect type, drops flicker, and returns one record per physical defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .config import Config, MergeCfg, ReportCfg, TrackCfg
from .detectors import Dets


# --------------------------------------------------------------------------- data
@dataclass
class Obs:
    frame_idx: int
    pts: float
    xyxy: np.ndarray        # raw detection box (4,)
    conf: float
    cls: int
    road_overlap: float = float("nan")


@dataclass
class FinalTrack:
    track_id: int
    raw_ids: list[int]
    cls: int
    cls_name: str
    first_frame: int
    first_pts: float
    last_frame: int
    last_pts: float
    n_obs: int
    max_conf: float
    mean_conf: float
    best: Obs
    obs: list[Obs] = field(default_factory=list, repr=False)
    water_filled: bool = False          # a pothole whose box overlaps a detected puddle


# --------------------------------------------------------------------------- tracker wrapper
class DefectTracker:
    def __init__(self, cfg: TrackCfg):
        from ultralytics.trackers import BOTSORT, BYTETracker
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml

        name = {"botsort": "botsort.yaml", "bytetrack": "bytetrack.yaml"}[cfg.tracker]
        base = dict(YAML.load(check_yaml(name)))
        for k in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh"):
            base[k] = getattr(cfg, k)
        if cfg.tracker == "botsort":
            base["gmc_method"] = cfg.gmc_method
            base["with_reid"] = cfg.with_reid
        args = SimpleNamespace(**base)
        self.tracker = BOTSORT(args) if cfg.tracker == "botsort" else BYTETracker(args)

    def update(self, dets: Dets, frame: np.ndarray) -> np.ndarray:
        """Returns rows [x1, y1, x2, y2, track_id, score, cls, det_idx] (det_idx indexes `dets`)."""
        out = self.tracker.update(dets, frame)
        return np.zeros((0, 8), np.float32) if out is None or len(out) == 0 else np.asarray(out)


# --------------------------------------------------------------------------- geometry helpers
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _cwh(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=np.float64)


def _from_cwh(v: np.ndarray) -> np.ndarray:
    w, h = max(v[2], 1.0), max(v[3], 1.0)
    return np.array([v[0] - w / 2, v[1] - h / 2, v[0] + w / 2, v[1] + h / 2])


def _extrapolate(obs: list[Obs], target_frame: int, k: int = 5) -> np.ndarray:
    """Linear extrapolation of the track's last `k` boxes to `target_frame` (clamped)."""
    tail = obs[-k:]
    last = _cwh(tail[-1].xyxy)
    if len(tail) < 2 or tail[-1].frame_idx == tail[0].frame_idx:
        return tail[-1].xyxy.astype(np.float64)
    dt = tail[-1].frame_idx - tail[0].frame_idx
    vel = (last - _cwh(tail[0].xyxy)) / dt
    gap = target_frame - tail[-1].frame_idx
    pred = last + vel * gap
    pred[2:] = np.clip(pred[2:], 0.5 * last[2:], 2.0 * last[2:])  # don't let size run away
    return _from_cwh(pred)


# --------------------------------------------------------------------------- merging
class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _compatible(c1: int, c2: int, groups: list[list[int]]) -> bool:
    if c1 == c2:
        return True
    return any(c1 in g and c2 in g for g in groups)


def _dominant_cls(obs: list[Obs]) -> int:
    score: dict[int, float] = {}
    for o in obs:
        score[o.cls] = score.get(o.cls, 0.0) + o.conf
    return max(score, key=score.get)


def merge_raw_tracks(raw: dict[int, list[Obs]], cfg: MergeCfg) -> dict[int, int]:
    """Return mapping raw_id -> merged group id (smallest raw id in the group)."""
    ids = sorted(raw)
    uf = _UF(ids)
    if not cfg.enabled or len(ids) < 2:
        return {i: uf.find(i) for i in ids}

    info = {}
    for i in ids:
        o = sorted(raw[i], key=lambda x: x.frame_idx)
        raw[i] = o
        info[i] = {"first": o[0].frame_idx, "last": o[-1].frame_idx, "cls": _dominant_cls(o),
                   "by_frame": {x.frame_idx: x.xyxy for x in o}}

    # (a) duplicates: tracks that coexist on the same physical defect
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            ia, ib = info[a], info[b]
            if not _compatible(ia["cls"], ib["cls"], cfg.compatible_classes):
                continue
            shared = set(ia["by_frame"]) & set(ib["by_frame"])
            if len(shared) < 2:
                continue
            good = sum(_iou(ia["by_frame"][f], ib["by_frame"][f]) >= cfg.dup_iou for f in shared)
            if good / len(shared) >= cfg.dup_frac:
                uf.union(a, b)

    # (b) fragmentation: A ends, B starts shortly after, B is where A should be
    cands = []
    for a in ids:
        ia = info[a]
        for b in ids:
            if a == b:
                continue
            ib = info[b]
            gap = ib["first"] - ia["last"]
            if gap <= 0 or gap > cfg.max_gap_frames:
                continue
            if not _compatible(ia["cls"], ib["cls"], cfg.compatible_classes):
                continue
            first_b = ib["by_frame"][ib["first"]]
            best = None
            for pred in (_extrapolate(raw[a], ib["first"]), raw[a][-1].xyxy.astype(np.float64)):
                iou = _iou(pred, first_b)
                pc, bc = _cwh(pred), _cwh(first_b)
                scale = max(pc[2], pc[3], bc[2], bc[3], 1.0)
                dist = float(np.hypot(pc[0] - bc[0], pc[1] - bc[1]) / scale)
                ratio = max(pc[2] * pc[3], 1.0) / max(bc[2] * bc[3], 1.0)
                ratio = max(ratio, 1.0 / ratio)
                if ratio > cfg.size_ratio:
                    continue
                if iou >= cfg.iou_thr or dist <= cfg.dist_factor:
                    cost = (1.0 - iou) + 0.5 * dist + 0.01 * gap
                    best = cost if best is None else min(best, cost)
            if best is not None:
                cands.append((best, a, b))
    used_pred, used_succ = set(), set()
    for cost, a, b in sorted(cands):
        if a in used_pred or b in used_succ:
            continue
        used_pred.add(a)
        used_succ.add(b)
        uf.union(a, b)
    return {i: uf.find(i) for i in ids}


# --------------------------------------------------------------------------- water / pothole de-duplication
def absorb_water_overlaps(cands: list[FinalTrack], water_cls: int, pothole_cls: int,
                          min_frac: float = 0.35) -> list[FinalTrack]:
    """A puddle that sits on a detected pothole is the same physical defect: keep the pothole (flag it as
    water-filled) and drop the puddle track. Puddles without a pothole detection stay as their own class."""
    water = [t for t in cands if t.cls == water_cls]
    holes = [t for t in cands if t.cls == pothole_cls]
    drop: set[int] = set()
    for w in water:
        wb = {o.frame_idx: o.xyxy for o in w.obs}
        for p in holes:
            pb = {o.frame_idx: o.xyxy for o in p.obs}
            shared = set(wb) & set(pb)
            if not shared:
                # extrapolate by nearest frame (extras are sampled on a different cadence)
                continue
            hit = 0
            for f in shared:
                a, b = wb[f], pb[f]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                inside = a[0] <= cx <= a[2] and a[1] <= cy <= a[3]
                a_area = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
                b_area = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
                hit += int((inside and a_area <= 5.0 * b_area) or _iou(a, b) >= 0.25)
            if hit / len(shared) >= min_frac:
                p.water_filled = True
                drop.add(id(w))
    return [t for t in cands if id(t) not in drop]


def absorb_into_manholes(cands: list[FinalTrack], manhole_cls: int, pothole_cls: int,
                         min_frac: float = 0.35) -> list[FinalTrack]:
    """A dark round cover that the pothole detector also fires on (low light) is a manhole, not a pothole:
    drop the pothole track when it sits on a manhole / drain track (open-vocabulary detector)."""
    mans = [t for t in cands if t.cls == manhole_cls]
    drop: set[int] = set()
    for p in [t for t in cands if t.cls == pothole_cls]:
        pb = {o.frame_idx: o.xyxy for o in p.obs}
        for m in mans:
            mb = {o.frame_idx: o.xyxy for o in m.obs}
            frames = sorted(set(pb) & set(mb))
            if not frames and mb and pb:       # extras are sampled on another cadence: compare nearest frames
                mf = np.array(sorted(mb))
                frames = [f for f in pb if np.abs(mf - f).min() <= 3]
                mb = {f: mb[int(mf[np.abs(mf - f).argmin()])] for f in frames}
            if not frames:
                continue
            hit = 0
            for f in frames:
                a, b = mb[f], pb[f]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                inside = a[0] <= cx <= a[2] and a[1] <= cy <= a[3]
                hit += int(inside or _iou(a, b) >= 0.30)
            if hit / len(frames) >= min_frac:
                drop.add(id(p))
                break
    return [t for t in cands if id(t) not in drop]


# --------------------------------------------------------------------------- finalize
def finalize_tracks(raw: dict[int, list[Obs]], names: dict[int, str], cfg: Config
                    ) -> tuple[list[FinalTrack], dict[int, int], dict]:
    """Merge, vote, filter, renumber.

    Returns (final_tracks sorted by first_seen, raw_id -> final track_id for REPORTED tracks, stats).
    """
    raw = {k: v for k, v in raw.items() if v}
    group_of = merge_raw_tracks(raw, cfg.merge)
    groups: dict[int, list[int]] = {}
    for rid, g in group_of.items():
        groups.setdefault(g, []).append(rid)

    candidates: list[FinalTrack] = []
    for g, rids in groups.items():
        obs = sorted((o for r in rids for o in raw[r]), key=lambda o: o.frame_idx)
        cls = _dominant_cls(obs)
        best = max(obs, key=lambda o: (o.conf, (o.xyxy[2] - o.xyxy[0]) * (o.xyxy[3] - o.xyxy[1])))
        candidates.append(FinalTrack(
            track_id=-1, raw_ids=sorted(rids), cls=cls, cls_name=names.get(cls, str(cls)),
            first_frame=obs[0].frame_idx, first_pts=obs[0].pts, last_frame=obs[-1].frame_idx,
            last_pts=obs[-1].pts, n_obs=len({o.frame_idx for o in obs}), max_conf=best.conf,
            mean_conf=float(np.mean([o.conf for o in obs])), best=best, obs=obs))

    from .classes import MANHOLE_ID, POTHOLE_ID, WATER_POTHOLE_ID
    candidates = absorb_water_overlaps(candidates, WATER_POTHOLE_ID, POTHOLE_ID)
    candidates = absorb_into_manholes(candidates, MANHOLE_ID, POTHOLE_ID)

    rep = cfg.report
    kept = [t for t in candidates
            if t.n_obs >= int(rep.class_min_obs.get(t.cls, rep.min_observations))
            and t.max_conf >= float(rep.class_min_conf.get(t.cls, rep.min_max_conf))]
    kept.sort(key=lambda t: (t.first_pts, t.first_frame))
    raw_to_final: dict[int, int] = {}
    for i, t in enumerate(kept, start=1):
        t.track_id = i
        for r in t.raw_ids:
            raw_to_final[r] = i
    stats = {
        "raw_tracks": len(raw),
        "after_merge": len(candidates),
        "reported": len(kept),
        "dropped_short_or_low_conf": len(candidates) - len(kept),
    }
    return kept, raw_to_final, stats
