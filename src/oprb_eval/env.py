"""Describe the machine and software, so every result can state where it was produced."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "rapidocr",
    "python-doctr",
    "tesserocr",
    "pillow",
)


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20, cwd=Path(__file__).parent)
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
