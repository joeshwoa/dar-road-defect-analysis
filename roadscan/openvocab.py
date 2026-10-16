"""Open-vocabulary detector (OWLv2) for road furniture that RDD2022 has no class for: manhole covers, drain grates.

It runs on the road part of the frame only (below the horizon) every N frames; the crop is padded to a square by the
processor, so cropping away the sky also raises the effective resolution on small, distant covers.
"""
from __future__ import annotations

import numpy as np

from .config import OpenVocabCfg
from .detectors import Dets, resolve_device


class OpenVocabDetector:
    def __init__(self, cfg: OpenVocabCfg, device: str = "auto"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.cfg = cfg
        dev = resolve_device(device)
        self.torch_device = torch.device("cpu" if dev == "cpu" else ("mps" if dev == "mps" else f"cuda:{dev}"))
        self.processor = Owlv2Processor.from_pretrained(cfg.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(cfg.model_id).to(self.torch_device).eval()
        self.prompts = list(cfg.prompts.keys())
        self.prompt_cls = [int(cfg.prompts[p]) for p in self.prompts]
        self._text_inputs = None

    def __call__(self, img: np.ndarray, top: int) -> Dets:
        import torch
        from PIL import Image
        from torchvision.ops import batched_nms

        h, w = img.shape[:2]
        top = int(np.clip(top, 0, h - 32))
        crop = img[top:, :, ::-1]
        pil = Image.fromarray(np.ascontiguousarray(crop))
        side = max(crop.shape[0], crop.shape[1])
        inputs = self.processor(text=[self.prompts], images=pil, return_tensors="pt").to(self.torch_device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.cfg.conf, target_sizes=torch.tensor([[side, side]]), text_labels=[self.prompts])[0]
        if len(res["scores"]) == 0:
            return Dets.empty()
        boxes = res["boxes"].cpu().numpy().astype(np.float32)
        scores = res["scores"].cpu().numpy().astype(np.float32)
        labels = res["labels"].cpu().numpy().astype(int)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, crop.shape[0])
        boxes[:, [1, 3]] += top
        cls = np.array([self.prompt_cls[i] for i in labels], np.float32)
        bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (bw >= 6) & (bh >= 3) & (bw * bh <= self.cfg.max_area_frac * h * w) & (bw / np.maximum(bh, 1) <= 12)
        if not keep.any():
            return Dets.empty()
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        k = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(cls).long(), 0.4).numpy()
        return Dets(boxes[k], scores[k], cls[k])
