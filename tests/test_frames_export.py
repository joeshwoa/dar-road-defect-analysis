import csv, json
from types import SimpleNamespace
import numpy as np
from roadscan.frames_export import FrameLog, timecode


def test_frame_export(tmp_path):
    fl = FrameLog(tmp_path, save_images=True)
    img = np.zeros((40, 60, 3), np.uint8)
    pose = SimpleNamespace(x=1.2, y=3.4, dist_m=5.0)
    for i in range(3):
        fl.add(i, i / 25, img, surface="paved", sidewalk="left", marking={"present": True, "state": "ok", "vis": 0.8},
               paint_missing=False, speed_kmh=30.0, pose=pose,
               defects=[{"track_id": 1, "class": "pothole", "confidence": 0.5, "box": [1, 2, 3, 4]}] if i else [],
               road_users=[{"track_id": 7, "class": "car", "confidence": 0.9, "box": [5, 6, 7, 8]}],
               furniture=[{"track_id": 2, "kind": "sign", "label": "sign", "flagged": i == 2, "box": [0, 0, 5, 5]}],
               unique_so_far=1 if i else 0, detector_ran=True, furniture_ran=i % 2 == 0)
    info = fl.close()
    assert info["frames"] == 3 and info["detection_rows"] == 8
    rows = list(csv.DictReader(open(tmp_path / "frames/frame_summary.csv")))
    assert len(rows) == 3 and rows[2]["n_flagged_furniture"] == "1" and rows[1]["defect_ids"] == "1"
    recs = [json.loads(l) for l in open(tmp_path / "frames/frames.jsonl")]
    assert recs[1]["defects"][0]["class"] == "pothole" and recs[0]["road"]["lane_marking_state"] == "ok"
    assert (tmp_path / "frames/img/frame_00002.jpg").exists()
    assert timecode(61.5) == "00:01:01.500"


def test_minimap_is_causal():
    from roadscan.mapping import MiniMap
    pose = lambda i, x, y, d: SimpleNamespace(frame=i, x=x, y=y, dist_m=d)

    class Ego:
        def __init__(self, n):
            self.poses = [pose(i, 0.0, 1.0 * i, 1.0 * i) for i in range(n)]
        def route(self):
            return np.array([[p.x, p.y] for p in self.poses])

    short, long_ = Ego(60), Ego(400)       # same past, very different future
    canvas = np.full((200, 300, 3), 120, np.uint8)
    a, b = canvas.copy(), canvas.copy()
    MiniMap(short, []).draw(a, 50)
    MiniMap(long_, []).draw(b, 50)
    assert (a == b).all()                  # the future (and the total route length) never changes the picture


def test_rate_scaling():
    from roadscan.config import Config
    from roadscan.pipeline import Analyzer
    an = Analyzer.__new__(Analyzer)
    an.cfg = Config()
    an.cfg.target_fps = 10.0
    info = SimpleNamespace(fps_avg=36.0, fps_nominal=36.0)
    r = an._set_rate(info)
    assert r["stride"] == 4 and abs(r["analysis_fps"] - 9.0) < 1e-6
    assert an.cfg.furniture.every_n == 4 * an._base_cfg.furniture.every_n
    assert an.cfg.track.track_buffer == round(an._base_cfg.track.track_buffer / 4)
    an.cfg.target_fps = 0.0
    assert an._set_rate(info)["stride"] == 1 and an.cfg.furniture.every_n == an._base_cfg.furniture.every_n


def test_rate_caps_and_observation_rescale():
    from roadscan.config import Config
    from roadscan.pipeline import Analyzer
    an = Analyzer.__new__(Analyzer)
    an.cfg = Config()
    an.cfg.target_fps = 18.0
    r = an._set_rate(SimpleNamespace(fps_avg=36.015, fps_nominal=36.0))
    assert r["stride"] == 2 and r["analysis_fps"] <= 18.01                 # never faster than the cap
    r = an._set_rate(SimpleNamespace(fps_avg=25.0, fps_nominal=25.0))
    assert r["stride"] == 2 and r["analysis_fps"] <= 18.0                  # 25 FPS source, 18 FPS cap -> 12.5 FPS
    assert an.cfg.report.min_observations == 2                             # 3 frames at full rate -> 2 at half rate
    assert all(v >= 2 for v in an.cfg.report.class_min_obs.values())
    an.cfg.target_fps = 0.0
    an._set_rate(SimpleNamespace(fps_avg=25.0, fps_nominal=25.0))
    assert an.cfg.report.min_observations == an._base_cfg.report.min_observations
