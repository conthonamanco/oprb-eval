"""Describe the machine and software, so every result can state where it was produced."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "rapidocr",
    "onnxruntime",
    "onnxruntime-gpu",
    "python-doctr",
    "tesserocr",
    "numpy",
    "pillow",
)


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=20, cwd=Path(__file__).parent
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def describe() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    gpus = _command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpus": gpus.splitlines() if gpus else [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "commit": _command("git", "rev-parse", "HEAD"),
        "packages": packages,
    }


def _cuda():
    """torch.cuda if the model already uses it, else None. Never imports torch itself."""
    torch = sys.modules.get("torch")
    return torch.cuda if torch is not None and torch.cuda.is_available() else None


def reset_peak_vram() -> None:
    if cuda := _cuda():
        cuda.reset_peak_memory_stats()


def peak_vram_mb() -> int | None:
    cuda = _cuda()
    return round(cuda.max_memory_allocated() / 1e6) if cuda else None


def release_vram() -> None:
    if cuda := _cuda():
        cuda.empty_cache()
