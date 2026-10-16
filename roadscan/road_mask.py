"""Road-surface segmentation (SegFormer / Cityscapes) used as a soft false-positive filter.

Cityscapes label 0 = road. The mask is refreshed every N frames and reused in between.
If the segmenter finds (almost) no road, we assume it failed on this scene (night, dirt road,
heavy damage) and **do not filter** that frame instead of deleting every detection.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import RoadCfg
from .detectors import Dets, resolve_device

ROAD_ID = 0  # Cityscapes train id for "road"


class RoadSegmenter:
    def __init__(self, cfg: RoadCfg, device: str = "auto"):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.cfg = cfg
        dev = resolve_device(device)
        self.torch_device = torch.device("cpu" if dev == "cpu" else ("mps" if dev == "mps" else f"cuda:{dev}"))
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(cfg.model_id).to(self.torch_device).eval()
        self.use_half = self.torch_device.type == "cuda"
        if self.use_half:
            self.model.half()
        self._mask: np.ndarray | None = None
        self._frames_since = 10 ** 9
        self.frames_total = 0
        self.frames_failed = 0
        self.dets_in = 0
        self.dets_dropped = 0

    # ------------------------------------------------------------------ mask
    def _segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        h, w = frame_bgr.shape[:2]
        scale = self.cfg.input_width / w
        small = cv2.resize(frame_bgr, (self.cfg.input_width, max(32, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt", do_resize=False)
        pix = inputs["pixel_values"].to(self.torch_device)
        if self.use_half:
            pix = pix.half()
        with torch.inference_mode():
            logits = self.model(pixel_values=pix).logits  # (1, C, h/4, w/4)
            logits = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
            road = (logits.argmax(1)[0] == ROAD_ID).cpu().numpy().astype(np.uint8)
        if self.cfg.dilate_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.dilate_px, self.cfg.dilate_px))
            road = cv2.dilate(road, k)
        return road.astype(bool)

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Current road mask (bool HxW) or None when the segmenter judged the scene a failure."""
        self.frames_total += 1
        if self._mask is None or self._frames_since >= self.cfg.every_n:
            self._mask = self._segment(frame_bgr)
            self._frames_since = 0
        self._frames_since += 1
        if self._mask.mean() < self.cfg.min_road_fraction:
            self.frames_failed += 1
            return None
        return self._mask

    # ------------------------------------------------------------------ filter
    def filter(self, dets: Dets, frame_bgr: np.ndarray) -> tuple[Dets, np.ndarray]:
        """Drop detections lying mostly off-road. Returns (kept, road_overlap per kept det)."""
        m = self.mask(frame_bgr)
        n = len(dets)
        self.dets_in += n
        if n == 0:
            return dets, np.zeros((0,), np.float32)
        if m is None:
            return dets, np.full((n,), np.nan, np.float32)
        h, w = m.shape
        ov = np.zeros(n, np.float32)
        for i, (x1, y1, x2, y2) in enumerate(dets.xyxy):
            xa, ya = max(int(x1), 0), max(int(y1), 0)
            xb, yb = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
            if xb <= xa or yb <= ya:
                continue
            ov[i] = m[ya:yb, xa:xb].mean()
        keep = ov >= self.cfg.min_overlap
        self.dets_dropped += int((~keep).sum())
        return dets[keep], ov[keep]

    def stats(self) -> dict:
        return {
            "frames": self.frames_total,
            "frames_mask_failed": self.frames_failed,
            "detections_in": self.dets_in,
            "detections_dropped_off_road": self.dets_dropped,
        }
