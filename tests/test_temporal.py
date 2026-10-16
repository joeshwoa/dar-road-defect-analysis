import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadscan.temporal import (DecayedVotes, DwellState, SpeedFilter, TemporalPrior, TrackMemory, TrackScore,
                               hysteresis_class)


def test_speed_filter_ignores_single_frame_dropout():
    fps = 25.0
    f = SpeedFilter(fps)
    out = []
    for i in range(200):
        t = i / fps
        raw = 40.0
        if 80 <= i < 83:               # flow lost for 3 frames -> 0 km/h
            raw = 0.0
        out.append(f.update(t, raw))
    assert min(out[100:]) > 35.0, "a 3-frame dropout must not reach the output"
    assert out[-1] > 36.0


def test_speed_filter_follows_real_braking_but_rate_limited():
    fps = 25.0
    f = SpeedFilter(fps, max_decel_ms2=6.0)
    out = []
    for i in range(300):
        t = i / fps
        raw = 40.0 if i < 100 else 0.0            # genuinely stops (hard step)
        out.append(f.update(t, raw))
    assert out[-1] < 2.0                          # it does converge to 0
    # never falls faster than 6 m/s^2 = 21.6 km/h per second
    d = -np.diff(out) * fps
    assert d.max() <= 21.6 + 1e-6


def test_speed_filter_holds_when_invalid():
    f = SpeedFilter(25.0)
    for i in range(50):
        f.update(i / 25, 30.0)
    v = f.update(2.0, None, valid=False)
    assert 25 < v < 35


def test_dwell_state_needs_persistence():
    s = DwellState(dwell_s=1.0)
    s.update(0.0, "paved")
    assert s.update(0.2, "unpaved") == "paved"
    assert s.update(0.5, "paved") == "paved"          # candidate vanished -> reset
    assert s.update(0.6, "unpaved") == "paved"
    assert s.update(1.7, "unpaved") == "unpaved"


def test_hysteresis_class_does_not_chatter():
    cur, seq = "mixed", []
    for r in [0.55, 0.62, 0.58, 0.61, 0.57, 0.63, 0.59]:     # hovering around the 0.60 threshold
        cur = hysteresis_class(r, cur, lo=0.25, hi=0.60, margin=0.05)
        seq.append(cur)
    assert len(set(seq)) <= 2 and seq.count("unpaved") in (0, len(seq) - seq.index("unpaved") if "unpaved" in seq else 0)
    flips = sum(a != b for a, b in zip(seq, seq[1:]))
    assert flips <= 1


def test_decayed_votes_stable_label():
    v = DecayedVotes(decay=0.9, margin=0.15)
    labels = [v.add(c) for c in [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]]
    assert set(labels) == {0}
    labels = [v.add(1) for _ in range(30)]
    assert labels[-1] == 1


def test_track_score_hysteresis_and_coast():
    ts = TrackScore(alpha=0.5, on_thr=0.4, off_thr=0.2, min_hits=2, coast=3)
    box = np.array([0, 0, 10, 10.0])
    ts.hit(0, 0.6, box, 3)
    assert not ts.confirmed                           # needs min_hits
    ts.hit(1, 0.6, box, 3)
    assert ts.confirmed and ts.visible
    ts.miss()
    assert ts.visible                                 # one dropout: still drawn
    for _ in range(5):
        ts.miss()
    assert not ts.visible


def test_temporal_prior_boosts_and_relabels_only_when_supported():
    mem = TrackMemory(alpha=0.5, on_thr=0.3, min_hits=1)
    box = np.array([100, 100, 160, 140.0])
    for f in range(5):
        mem.update(f, [(1, 0.8, box, 0)])             # confirmed longitudinal crack, strong
    pr = TemporalPrior(mem, weight=0.4, max_boost=0.2, compatible=[[0, 1, 2]])
    xyxy = np.array([[102, 101, 162, 141], [400, 300, 440, 330]], np.float32)
    conf = np.array([0.30, 0.30], np.float32)
    cls = np.array([1, 1], np.float32)
    c2, k2 = pr.adjust(xyxy, conf, cls, frame=5)
    assert c2[0] > 0.40 and c2[0] <= 0.5 + 1e-6       # blended and capped
    assert k2[0] == 0                                  # relabelled within the crack group
    assert c2[1] == 0.30 and k2[1] == 1                # unsupported detection is untouched
    # a pothole is never relabelled to a crack
    c3, k3 = pr.adjust(xyxy[:1], conf[:1], np.array([3], np.float32), frame=5)
    assert k3[0] == 3


def test_trimmed_mean_keeps_average_of_alternating_signal_and_rejects_dropout():
    from roadscan.temporal import CausalTrimmedMean, CausalMedian
    tm, md = CausalTrimmedMean(15, 0.2), CausalMedian(15)
    for x in [4.8, 9.6] * 20:
        a, b = tm.update(x), md.update(x)
    assert abs(a - 7.2) < 0.5                 # duplicated frames: mean is the truth ...
    for x in [0.0, 0.0, 0.0]:                 # ... and a 3-sample dropout is trimmed away
        a = tm.update(x)
    assert a > 6.0


def test_display_smoother_coasts_then_drops_and_confidence_never_falls():
    from roadscan.temporal import DisplaySmoother
    sm = DisplaySmoother(alpha=0.5, hold=3)
    box = np.array([100, 100, 140, 130.0])
    out = sm.update(0, {1: (box, 0.6)})
    assert len(out) == 1
    out = sm.update(1, {1: (box + [4, 2, 4, 2], 0.4)})
    assert out[0][2] == 0.6                           # label keeps the best confidence so far
    for f in (2, 3, 4):                               # detector drops out: still drawn for `hold` frames
        out = sm.update(f, {})
        assert len(out) == 1
    assert sm.update(5, {}) == []                     # ... then it disappears
    out = sm.update(6, {1: (box, 0.5)})               # a late re-detection starts a fresh state
    assert len(out) == 1 and abs(out[0][1][0] - 100) < 1e-6


def test_speed_filter_ignores_bogus_first_samples():
    f = SpeedFilter(25.0)
    out = [f.update(i / 25, 140.0 if i == 1 else 30.0) for i in range(120)]
    assert max(out) < 40.0, "one junk start-up sample must not seed the filter"
    assert abs(out[-1] - 30.0) < 3.0
