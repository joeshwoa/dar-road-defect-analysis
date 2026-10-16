import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadscan.config import Config, parse_overrides
from roadscan.tracking import Obs, finalize_tracks
from roadscan.video_io import timecode

NAMES = {0: "longitudinal_crack", 1: "transverse_crack", 2: "alligator_crack", 3: "pothole"}


def approaching_defect(start, n, x0=300, y0=200, cls=3, conf=0.7, grow=2.0, fps=25.0):
    """A defect that drifts down and grows as the camera approaches."""
    obs = []
    for k in range(n):
        f = start + k
        w = 30 + grow * (f - 0)
        h = 20 + grow * 0.7 * (f - 0)
        cx, cy = x0 + 0.5 * f, y0 + 3.0 * f
        obs.append(Obs(f, f / fps, np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), conf, cls))
    return obs


def test_fragmented_track_is_merged():
    full = approaching_defect(0, 40)
    raw = {1: full[:18], 2: full[24:]}           # tracker lost it for 6 frames and re-issued an ID
    final, mapping, stats = finalize_tracks(raw, NAMES, Config())
    assert len(final) == 1 and sorted(final[0].raw_ids) == [1, 2]
    assert mapping[1] == mapping[2] == 1
    assert final[0].first_frame == 0


def test_different_defects_stay_separate():
    a = approaching_defect(0, 30, x0=100)
    b = approaching_defect(0, 30, x0=600, y0=100)
    final, _, _ = finalize_tracks({1: a, 2: b}, NAMES, Config())
    assert len(final) == 2


def test_duplicate_coexisting_tracks_merge():
    a = approaching_defect(0, 30)
    b = [Obs(o.frame_idx, o.pts, o.xyxy + 2.0, o.conf * 0.8, o.cls) for o in a]
    final, _, _ = finalize_tracks({1: a, 2: b}, NAMES, Config())
    assert len(final) == 1
    assert abs(final[0].max_conf - 0.7) < 1e-6        # keeps the highest confidence


def test_flicker_is_not_reported():
    flick = approaching_defect(0, 2, conf=0.9)          # only two frames -> below min_observations
    weak = approaching_defect(0, 20, x0=500, conf=0.2)   # never reaches report confidence
    final, mapping, stats = finalize_tracks({1: flick, 2: weak}, NAMES, Config())
    assert final == [] and mapping == {}
    assert stats["dropped_short_or_low_conf"] == 2


def test_potholes_and_cracks_do_not_merge():
    a = approaching_defect(0, 20, cls=3)
    b = approaching_defect(0, 20, cls=0)
    cfg = Config.from_dict({"merge": {"compatible_classes": [[0, 1, 2]]}})
    final, _, _ = finalize_tracks({1: a, 2: b}, NAMES, cfg)
    assert len(final) == 2


def test_class_vote_and_first_seen_ordering():
    late = approaching_defect(30, 20, x0=100)
    early = approaching_defect(0, 20, x0=500)
    final, _, _ = finalize_tracks({7: late, 3: early}, NAMES, Config())
    assert [t.track_id for t in final] == [1, 2]
    assert final[0].first_frame == 0 and final[1].first_frame == 30


def test_config_overrides_and_timecode():
    cfg = Config().override(parse_overrides(["detect.conf=0.35", "road.enabled=false", "stride=2"]))
    assert cfg.detect.conf == 0.35 and cfg.road.enabled is False and cfg.stride == 2
    assert timecode(3661.5) == "01:01:01.500"
