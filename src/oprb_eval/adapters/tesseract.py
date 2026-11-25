"""Tesseract 5 through tesserocr, whose wheels bundle the engine. Runs on the CPU."""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from ..models import Prediction
from . import cache_dir, files_mb


def _traineddata(config: dict) -> Path:
    folder = cache_dir("tessdata_best")
    target = folder / f"{config['lang']}.traineddata"
    if not target.exists():
        url = (
            "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/"
            f"{config['tessdata_tag']}/{config['lang']}.traineddata"
        )
        # Write-then-rename, because several shards may start at the same time.
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size != config["tessdata_bytes"]:
            raise RuntimeError(f"{url}: unexpected size {temporary.stat().st_size}")
        os.replace(temporary, target)
    return folder


class TesseractModel:
    def __init__(self, config: dict):
        import tesserocr

        self.version = tesserocr.tesseract_version().split()[1]
        if int(self.version.split(".")[0]) < 5:
            raise SystemExit(f"Tesseract 5 is required, found {self.version}")
        self.folder = _traineddata(config)
        self.lang = config["lang"]
        self.api = tesserocr.PyTessBaseAPI(
            path=str(self.folder), lang=self.lang, psm=config["psm"], oem=tesserocr.OEM.LSTM_ONLY
        )

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        from PIL import Image

        predictions = []
        for path in image_paths:
            with Image.open(path) as image:
                self.api.SetImage(image.convert("RGB"))
            predictions.append(Prediction(self.api.GetUTF8Text()))
            self.api.Clear()
        return predictions

    def info(self) -> dict:
        return {
            "device": "cpu",
            "version": self.version,
            "weights_mb": files_mb(self.folder, f"{self.lang}.traineddata"),
        }
