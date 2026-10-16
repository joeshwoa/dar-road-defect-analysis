"""Hardware / software fingerprint and per-stage timing (for the FPS report)."""
from __future__ import annotations

import platform
import re
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3)
        if out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"MemTotal:\s+(\d+)\s+kB", f.read())
            if m:
                return round(int(m.group(1)) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def collect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_logical_cores": __import__("os").cpu_count(),
        "ram_gb": _ram_gb(),
        "gpus": [],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1e9, 1)})
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["mps"] = True
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    for mod in ("ultralytics", "rfdetr", "transformers", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class StageTimer:
    """Accumulates wall time per named stage; syncs CUDA so GPU work is attributed correctly."""

    def __init__(self, sync: bool = True):
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.sync = sync

    @contextmanager
    def stage(self, name: str):
        if self.sync:
            cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                cuda_sync()
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def summary(self, n_frames: int) -> dict[str, Any]:
        return {
            k: {"total_s": round(v, 4), "ms_per_frame": round(1000 * v / max(n_frames, 1), 3)}
            for k, v in self.totals.items()
        }
