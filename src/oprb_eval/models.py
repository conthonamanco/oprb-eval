"""Model interface and registry.

An adapter receives only image paths, so ground truth cannot reach a model.
"""

from __future__ import annotations

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
    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        """One prediction per image, in the same order."""
        ...


class ConstantModel:
    """Returns a fixed string; used for smoke tests."""

    def __init__(self, config: dict):
        self.text = config["text"]

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        return [Prediction(self.text) for _ in image_paths]


ADAPTERS = {"constant": ConstantModel}


def load_config(path: Path, name: str) -> dict:
    with open(path, "rb") as f:
        configs = tomllib.load(f)
    if name not in configs:
        raise SystemExit(f"Unknown model '{name}' in {path}. Available: {', '.join(configs)}")
    return configs[name]


def build(config: dict) -> Model:
    adapter = config["adapter"]
    if adapter not in ADAPTERS:
        raise SystemExit(f"Adapter '{adapter}' is not implemented yet.")
    return ADAPTERS[adapter](config)
