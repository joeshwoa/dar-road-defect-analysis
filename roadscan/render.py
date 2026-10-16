"""Overlay drawing and H.264 encoding."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .classes import color_for
from .config import RenderCfg


def draw_box(img: np.ndarray, xyxy: Sequence[float], label: str, color, cfg: RenderCfg) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, cfg.thickness)
    if not label:
        return
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)
    ty = y1 - 4 if y1 - th - bl - 4 >= 0 else y2 + th + 4
    ty = min(ty, img.shape[0] - 2)
    cv2.rectangle(img, (x1, ty - th - bl), (x1 + tw + 4, ty + bl), color, -1)
    cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def fill_box(img: np.ndarray, xyxy: Sequence[float], color, alpha: float = 0.28, ellipse: bool = True) -> None:
    """Translucent fill of a defect box (ellipse for potholes / puddles) so damaged road areas stand out."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    m = np.zeros(roi.shape[:2], np.uint8)
    if ellipse:
        cv2.ellipse(m, ((x2 - x1) // 2, (y2 - y1) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
    else:
        m[:] = 255
    sel = m > 0
    roi[sel] = (roi[sel] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.40) -> np.ndarray:
    out = img.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 1, cv2.LINE_AA)
    return out


def draw_hud(img: np.ndarray, lines: list[str], cfg: RenderCfg, alert: set[int] | None = None) -> None:
    pad, lh = 6, int(18 * max(cfg.font_scale / 0.5, 0.8))
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad
    sub = img[0:h, 0:w]
    img[0:h, 0:w] = (sub * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (pad, lh * (i + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    (90, 90, 255) if (alert and i in alert) else (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(img: np.ndarray, entries: list[tuple[tuple[int, int, int], str]], cfg: RenderCfg) -> None:
    """Colour key in the lower-left corner: only the colours that are actually used in this video."""
    if not entries:
        return
    fs = max(cfg.font_scale * 0.8, 0.36)
    lh = int(16 * max(fs / 0.4, 1.0))
    wmax = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for _, t in entries) + 26
    h = lh * len(entries) + 8
    H = img.shape[0]
    y0 = H - h - 6
    sub = img[y0:y0 + h, 4:4 + wmax]
    img[y0:y0 + h, 4:4 + wmax] = (sub * 0.40).astype(np.uint8)
    for i, (col, text) in enumerate(entries):
        y = y0 + 4 + lh * i
        cv2.rectangle(img, (9, y + 3), (19, y + 13), col, -1)
        cv2.putText(img, text, (24, y + 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)


class VideoSink:
    """Writes mp4v with OpenCV, then (optionally) re-encodes to H.264/yuv420p with ffmpeg."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], cfg: RenderCfg):
        self.final = Path(path)
        self.cfg = cfg
        self.use_ffmpeg = bool(cfg.encode_h264 and shutil.which("ffmpeg"))
        self.tmp = self.final.with_suffix(".tmp.mp4") if self.use_ffmpeg else self.final
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.w = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.w.isOpened():
            raise IOError(f"cannot open video writer: {self.tmp}")

    def write(self, img: np.ndarray) -> None:
        self.w.write(img)

    def close(self) -> None:
        self.w.release()
        if self.use_ffmpeg:
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(self.tmp), "-c:v", "libx264", "-preset", "medium",
                   "-crf", str(self.cfg.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.final)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                self.tmp.unlink(missing_ok=True)
            else:  # keep the mp4v file if libx264 is unavailable
                self.tmp.replace(self.final)
