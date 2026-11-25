"""PP-OCR through RapidOCR and ONNX Runtime.

RapidOCR publishes the English PP-OCRv5 recogniser for ONNX but not for its torch
engine. The mobile models are small enough for the CPU, which is the default here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from ..models import Prediction
from ..reading_order import to_text
from . import cache_dir, files_mb


class PPOCRModel:
    def __init__(self, config: dict):
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        self.folder = cache_dir("rapidocr")
        version = OCRVersion(config["ocr_version"])
        self.device = config["device"]
        threads = int(os.environ.get("OPRB_THREADS", "-1"))
        self.engine = RapidOCR(
            params={
                "Global.model_root_dir": str(self.folder),
                "Global.log_level": "error",
                "Global.use_cls": False,  # pages are never upside down
                "EngineConfig.onnxruntime.use_cuda": self.device == "cuda",
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": version,
                "Det.lang_type": LangDet.CH,  # the PP-OCRv5 detector is language independent
                "Det.model_type": ModelType(config["det_model"]),
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": version,
                "Rec.lang_type": LangRec(config["rec_lang"]),
                "Rec.model_type": ModelType(config["rec_model"]),
            }
        )

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        predictions = []
        for path in image_paths:
            result = self.engine(str(path))
            # RapidOCR reports "nothing found" with a result that lacks boxes or texts.
            texts, boxes = getattr(result, "txts", None), getattr(result, "boxes", None)
            if texts is None or boxes is None:
                predictions.append(Prediction(""))
                continue
            quads = [[(float(x), float(y)) for x, y in box] for box in boxes]
            predictions.append(Prediction(to_text(list(texts), quads)))
        return predictions

    def info(self) -> dict:
        return {
            "device": self.device,
            "weights_mb": files_mb(self.folder, "*.onnx"),
            "weights": sorted(p.name for p in self.folder.glob("*.onnx")),
        }
