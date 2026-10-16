"""Defect detectors behind one interface: Ultralytics (YOLO26/11/8) and RF-DETR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import DetectCfg


# --------------------------------------------------------------------------- Dets
@dataclass
class Dets:
    """Detections in a tracker-friendly container (numpy only)."""

    xyxy: np.ndarray   # (N, 4) float32, pixels
    conf: np.ndarray   # (N,)  float32
    cls: np.ndarray    # (N,)  float32 (kept float for the Ultralytics tracker)

    @staticmethod
    def empty() -> "Dets":
        return Dets(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32))

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, idx) -> "Dets":  # boolean / index array
        return Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])

    @property
    def xywh(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4), np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1).astype(np.float32)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- low-light enhancement
def clahe_enhance(frame: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Local contrast boost on the luminance channel only (dark asphalt: cracks / holes become visible)."""
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def should_enhance(frame: np.ndarray, mode: str, lum_thr: float) -> bool:
    if mode == "clahe":
        return True
    if mode == "auto":
        h = frame.shape[0]
        return float(frame[int(0.55 * h):].mean()) < lum_thr     # lower (road) half is dark: dusk / night
    return False


def merge_dets(a: "Dets", b: "Dets", iou: float) -> "Dets":
    """Union of two detection sets with class-wise NMS (keeps the more confident duplicate)."""
    if len(b) == 0:
        return a
    if len(a) == 0:
        return b
    import torch
    from torchvision.ops import batched_nms

    xyxy = np.concatenate([a.xyxy, b.xyxy]); conf = np.concatenate([a.conf, b.conf]); cls = np.concatenate([a.cls, b.cls])
    keep = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(conf), torch.from_numpy(cls).long(), min(iou, 0.5)).numpy()
    return Dets(xyxy[keep], conf[keep], cls[keep])


# --------------------------------------------------------------------------- tiling
def _tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def tiled_predict(predict: Callable[[np.ndarray], Dets], frame: np.ndarray, tile: int, overlap: float,
                  nms_iou: float = 0.5) -> Dets:
    """Run `predict` on overlapping tiles + the whole frame, then merge with class-wise NMS."""
    import torch
    from torchvision.ops import batched_nms

    h, w = frame.shape[:2]
    parts = [predict(frame)]
    for y0 in _tile_origins(h, tile, overlap):
        for x0 in _tile_origins(w, tile, overlap):
            if tile >= h and tile >= w:
                continue
            crop = frame[y0:y0 + tile, x0:x0 + tile]
            d = predict(crop)
            if len(d):
                d = Dets(d.xyxy + np.array([x0, y0, x0, y0], np.float32), d.conf, d.cls)
                parts.append(d)
    xyxy = np.concatenate([p.xyxy for p in parts if len(p)] or [np.zeros((0, 4), np.float32)])
    conf = np.concatenate([p.conf for p in parts if len(p)] or [np.zeros((0,), np.float32)])
    cls = np.concatenate([p.cls for p in parts if len(p)] or [np.zeros((0,), np.float32)])
    if len(xyxy) == 0:
        return Dets.empty()
    keep = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(conf), torch.from_numpy(cls).long(), nms_iou).numpy()
    return Dets(xyxy[keep], conf[keep], cls[keep])


# --------------------------------------------------------------------------- detectors
class UltralyticsDetector:
    """YOLO26 / YOLO11 / YOLOv8 (detect). Class names come from the checkpoint."""

    def __init__(self, cfg: DetectCfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = YOLO(cfg.weights)
        self.names: dict[int, str] = dict(self.model.names)
        self.half = bool(cfg.half and self.device not in ("cpu", "mps"))

    def _predict(self, img: np.ndarray) -> Dets:
        extra = {"quantize": 16} if self.half else {}   # 'half' is deprecated in ultralytics >= 8.4
        r = self.model.predict(
            img, imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou, device=self.device,
            max_det=self.cfg.max_det, classes=self.cfg.classes, verbose=False, **extra,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Dets.empty()
        return Dets(b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                    b.cls.cpu().numpy().astype(np.float32))

    def _run(self, frame: np.ndarray) -> Dets:
        if self.cfg.tile_size > 0:
            return tiled_predict(self._predict, frame, self.cfg.tile_size, self.cfg.tile_overlap)
        return self._predict(frame)

    def __call__(self, frame: np.ndarray) -> Dets:
        d = self._run(frame)
        mode = getattr(self.cfg, "enhance", "off")
        if mode == "off" or not should_enhance(frame, mode, self.cfg.enhance_lum_thr):
            return d
        d2 = self._run(clahe_enhance(frame))
        return merge_dets(d, d2, self.cfg.iou)


class RFDETRDetector:
    """RF-DETR (Apache-2.0). Loads fine-tuned weights produced by scripts/train_rfdetr.py."""

    def __init__(self, cfg: DetectCfg, class_names: Optional[list[str]] = None):
        import rfdetr

        variants = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
                    "base": "RFDETRBase", "large": "RFDETRLarge"}
        cls = getattr(rfdetr, variants[cfg.rfdetr_variant])
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        dev = "cpu" if self.device == "cpu" else ("mps" if self.device == "mps" else f"cuda:{self.device.split(':')[-1]}")
        self.model = cls(pretrain_weights=cfg.weights, resolution=cfg.rfdetr_resolution, device=dev)
        names = class_names or getattr(self.model, "class_names", None)
        if isinstance(names, dict):
            self.names = {int(k): str(v) for k, v in names.items()}
        elif names:
            self.names = {i: str(n) for i, n in enumerate(names)}
        else:
            self.names = {}

    def _predict(self, img: np.ndarray) -> Dets:
        rgb = img[:, :, ::-1]
        det = self.model.predict(rgb, threshold=self.cfg.conf)
        if det is None or len(det) == 0:
            return Dets.empty()
        cls = det.class_id.astype(np.float32)
        keep = np.ones(len(cls), bool) if not self.cfg.classes else np.isin(cls, self.cfg.classes)
        return Dets(det.xyxy[keep].astype(np.float32), det.confidence[keep].astype(np.float32), cls[keep])

    def __call__(self, frame: np.ndarray) -> Dets:
        if self.cfg.tile_size > 0:
            return tiled_predict(self._predict, frame, self.cfg.tile_size, self.cfg.tile_overlap)
        return self._predict(frame)


def build_detector(cfg: DetectCfg, class_names: Optional[list[str]] = None):
    if cfg.backend == "ultralytics":
        return UltralyticsDetector(cfg)
    if cfg.backend == "rfdetr":
        return RFDETRDetector(cfg, class_names)
    raise ValueError(f"unknown detector backend: {cfg.backend}")
