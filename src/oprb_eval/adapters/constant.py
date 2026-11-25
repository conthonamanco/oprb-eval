"""Returns a fixed string; used for smoke tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..models import Prediction


class ConstantModel:
    def __init__(self, config: dict):
        self.text = config["text"]

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        return [Prediction(self.text) for _ in image_paths]

    def info(self) -> dict:
        return {"device": "cpu"}
