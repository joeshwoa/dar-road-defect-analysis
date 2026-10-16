"""SAM 2 box-prompted masks, run once per reported defect on its best frame.

RDD2022 only has boxes, so instead of training a segmenter we ask SAM 2 for the mask inside each
detected box. This yields defect area (px and % of frame) and an overlay for the exported image.
"""
from __future__ import annotations

import numpy as np

from .config import SamCfg
from .detectors import resolve_device


class SamMasker:
    def __init__(self, cfg: SamCfg, device: str = "auto"):
        from ultralytics import SAM

        self.model = SAM(cfg.weights)
        self.device = resolve_device(device)

    def mask_for_box(self, frame_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        box = [float(max(xyxy[0], 0)), float(max(xyxy[1], 0)), float(min(xyxy[2], w - 1)), float(min(xyxy[3], h - 1))]
        try:
            res = self.model.predict(frame_bgr, bboxes=[box], device=self.device, verbose=False)[0]
        except Exception:
            return None
        if res.masks is None or len(res.masks.data) == 0:
            return None
        m = res.masks.data[0].cpu().numpy() > 0.5
        if m.shape != (h, w):
            import cv2

            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return m
