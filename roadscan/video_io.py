"""Video reading with true presentation timestamps (works for variable-frame-rate files)."""
from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps_nominal: float      # container-reported
    fps_avg: float          # frames / duration (honest for VFR)
    frame_count: int
    duration_s: float
    is_vfr: bool


@dataclass
class Frame:
    idx: int                # index among *decoded* frames (0-based)
    pts: float              # presentation time in seconds
    image: np.ndarray       # BGR


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = n / fps if fps > 0 and n > 0 else 0.0
    # ffprobe gives a more reliable duration / avg rate when available
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=avg_frame_rate,r_frame_rate,nb_frames,duration", "-of", "default=nw=1", str(path)],
                capture_output=True, text=True, timeout=20).stdout
            kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
            if kv.get("duration") not in (None, "N/A") and float(kv["duration"]) > 0:
                dur = float(kv["duration"])
            if kv.get("nb_frames") not in (None, "N/A") and int(kv["nb_frames"]) > 0:
                n = int(kv["nb_frames"])
        except Exception:
            pass
    fps_avg = n / dur if dur > 0 and n > 0 else fps
    is_vfr = fps > 0 and abs(fps_avg - fps) / fps > 0.02
    return VideoInfo(str(path), w, h, fps, fps_avg, n, dur, is_vfr)


def iter_frames(path: str | Path, stride: int = 1, max_frames: int = 0) -> Iterator[Frame]:
    """Yield frames with presentation timestamps.

    `idx` counts every decoded frame (so it stays aligned between passes even when
    `stride` > 1); only every `stride`-th frame is yielded.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    idx = -1
    yielded = 0
    last_pts = -1.0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # sanity: fall back to idx/fps if the backend reports nothing useful
            if (pts <= 0.0 and idx > 0) or pts < last_pts:
                pts = idx / nominal if nominal > 0 else float(idx)
            last_pts = pts
            if idx % stride:
                continue
            yield Frame(idx, pts, img)
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
    finally:
        cap.release()


def timecode(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
