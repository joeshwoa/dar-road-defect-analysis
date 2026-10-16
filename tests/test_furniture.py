import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadscan.furniture import (RunningQuantile, SimpleTracker, components, lean_from_face, lean_from_mask,
                                light_state, marking_stats, visibility)


def _lamp(color_bgr, on=True):
    img = np.full((60, 24, 3), 25, np.uint8)
    if on:
        cv2.circle(img, (12, 14), 6, color_bgr, -1)
    return img


def test_light_state_colours_and_off():
    assert light_state(_lamp((0, 0, 255)))[0] == "red"
    assert light_state(_lamp((0, 255, 0)))[0] == "green"
    assert light_state(_lamp((0, 220, 255)))[0] == "yellow"
    assert light_state(_lamp((0, 0, 255), on=False))[0] == "off"
    assert light_state(np.zeros((4, 4, 3), np.uint8))[0] == "unknown"


def _pole(angle_deg, h=120, w=5):
    m = np.zeros((h + 40, h + 40), np.uint8)
    cx, cy = (h + 40) // 2, (h + 40) // 2
    dx, dy = np.sin(np.radians(angle_deg)) * h / 2, np.cos(np.radians(angle_deg)) * h / 2
    cv2.line(m, (int(cx - dx), int(cy + dy)), (int(cx + dx), int(cy - dy)), 1, w)
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(bool)


def test_pole_lean():
    assert lean_from_mask(_pole(0)) < 3
    assert abs(lean_from_mask(_pole(20)) - 20) < 4
    assert lean_from_mask(np.ones((20, 20), bool)) is None          # a blob is not a pole


def test_sign_face_tilt():
    m = np.zeros((100, 100), np.uint8)
    box = cv2.boxPoints(((50, 50), (40, 30), 18)).astype(np.int32)
    cv2.fillPoly(m, [box], 1)
    assert abs(lean_from_face(m.astype(bool)) - 18) < 3
    m2 = np.zeros((100, 100), np.uint8)
    cv2.circle(m2, (50, 50), 25, 1, -1)
    assert lean_from_face(m2.astype(bool)) is None                  # round sign: no tilt estimate


def _road(paint_level):
    g = np.full((200, 300), 90, np.uint8)
    road = np.ones((200, 300), bool)
    mk = np.zeros((200, 300), bool)
    for x in (80, 150, 220):
        mk[100:180, x:x + 6] = True
    g[mk] = paint_level
    return g, road, mk


def test_marking_contrast_orders_good_and_faded():
    gg, road, mk = _road(220)
    gf, _, _ = _road(105)
    good = marking_stats(gg, mk, road, np.full(mk.shape, 0.8, np.float32), (90, 190))
    faded = marking_stats(gf, mk, road, np.full(mk.shape, 0.4, np.float32), (90, 190))
    assert good["weber"] > 1.0 and faded["weber"] < 0.25
    v_good = visibility(good["weber"], good["seg"], 0.9, 0.8)
    v_faded = visibility(faded["weber"], faded["seg"], 0.9, 0.8)
    assert v_good > 0.9 and v_faded < 0.45
    assert marking_stats(gg, np.zeros_like(mk), road, None, (90, 190)) is None


def test_running_quantile_is_causal_reference():
    q = RunningQuantile(85, default=0.3, min_n=5)
    assert q.ref() == 0.3
    for x in [0.9, 0.8, 1.0, 0.7, 0.85, 0.9]:
        q.update(x)
    assert 0.85 < q.ref() <= 1.0


def test_simple_tracker_keeps_ids_across_updates():
    t = SimpleTracker(max_age=2)
    a = np.array([100, 100, 130, 160], np.float32)
    ids1 = t.update([a], ["sign"])
    ids2 = t.update([a + [6, 3, 10, 6]], ["sign"])        # approaching: moved and grew
    assert ids1 == ids2
    ids3 = t.update([np.array([400, 50, 420, 90], np.float32)], ["sign"])
    assert ids3[0] != ids1[0]
    ids4 = t.update([a], ["light"])                       # same place but a different kind -> new id
    assert ids4[0] not in (ids1[0], ids3[0])


def test_components_min_area():
    m = np.zeros((50, 50), bool)
    m[5:15, 5:15] = True
    m[30:32, 30:32] = True
    comps = components(m, 20)
    assert len(comps) == 1 and comps[0]["area"] == 100
