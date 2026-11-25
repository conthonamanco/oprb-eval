"""Model interface and registry.

An adapter receives only image paths, so ground truth cannot reach a model.
"""

from __future__ import annotations

import importlib
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    text: str
    output_tokens: int | None = None  # None for OCR engines
    truncated: bool = False


class Model(Protocol):
    def __init__(self, config: dict): ...

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        """One prediction per image, in the same order."""
        ...

    def info(self) -> dict:
        """Device, precision and size of the loaded model."""
        ...


# Adapter name -> "module:class" inside oprb_eval.adapters, imported on first use.
ADAPTERS = {
    "constant": "constant:ConstantModel",
    "tesseract": "tesseract:TesseractModel",
    "ppocr": "ppocr:PPOCRModel",
    "doctr": "doctr_words:DoctrModel",
    "lighton": "vlm:LightOnModel",
    "qwen_vl": "vlm:QwenVLModel",
}


def load_config(path: Path, name: str) -> dict:
    with open(path, "rb") as f:
        configs = tomllib.load(f)
    if name not in configs:
        raise SystemExit(f"Unknown model '{name}' in {path}. Available: {', '.join(configs)}")
    return configs[name]


def build(config: dict) -> Model:
    if config["adapter"] not in ADAPTERS:
        raise SystemExit(f"Unknown adapter '{config['adapter']}'.")
    module, _, cls = ADAPTERS[config["adapter"]].partition(":")
    return getattr(importlib.import_module(f".adapters.{module}", __package__), cls)(config)
