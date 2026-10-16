#!/usr/bin/env python3
"""Detector-only speed benchmark: PyTorch vs ONNX vs TensorRT (FP16/INT8-ready).

Reports mean latency and FPS over real video frames after warm-up. Use it on Kaggle (T4) and on a
Jetson to quote honest numbers.

    python scripts/benchmark.py clip.mp4 --weights best.pt --imgsz 640 --formats pt engine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadscan.hardware import collect_hardware, cuda_sync
from roadscan.video_io import iter_frames


def bench(model, frames, imgsz, device, half, warmup=20) -> dict:
    kw = {"quantize": 16} if half else {}
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, device=device, verbose=False, **kw)
    cuda_sync()
    lat = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, device=device, verbose=False, **kw)
        cuda_sync()
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    return {"mean_ms": round(float(lat.mean()), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "fps": round(1000 / float(lat.mean()), 1), "frames": len(frames)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--formats", nargs="+", default=["pt", "engine"], help="pt onnx engine")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="benchmark.json")
    a = ap.parse_args()

    from ultralytics import YOLO

    frames = [f.image for f in iter_frames(a.video, max_frames=a.frames)]
    results = {"hardware": collect_hardware(), "imgsz": a.imgsz, "results": {}}
    for fmt in a.formats:
        try:
            if fmt == "pt":
                m, path, half = YOLO(a.weights), a.weights, a.device != "cpu"
            else:
                path = YOLO(a.weights).export(format=fmt, imgsz=a.imgsz, half=(fmt == "engine"), device=a.device)
                m, half = YOLO(path), False
            results["results"][fmt + ("_fp16" if fmt == "engine" else "")] = bench(m, frames, a.imgsz, a.device, half)
            print(fmt, results["results"][fmt + ("_fp16" if fmt == "engine" else "")])
        except Exception as e:  # keep going: a missing TensorRT must not kill the run
            results["results"][fmt] = {"error": repr(e)[:300]}
            print(fmt, "FAILED:", repr(e)[:300])
    Path(a.out).write_text(json.dumps(results, indent=2, default=str))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
