"""One module per model family. Heavy imports happen only when a model is built."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir(name: str) -> Path:
    root = Path(os.environ.get("OPRB_CACHE", Path.home() / ".cache" / "oprb_eval"))
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def files_mb(path: Path, pattern: str) -> float:
    return round(sum(p.stat().st_size for p in path.glob(pattern)) / 1e6, 1)
