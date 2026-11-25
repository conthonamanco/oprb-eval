"""docTR: word detection followed by word recognition."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..models import Prediction
from ..reading_order import to_text


def page_text(page, width: int, height: int) -> str:
    texts, quads = [], []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                points = [(float(x) * width, float(y) * height) for x, y in word.geometry]
                if len(points) == 2:  # straight box: top-left and bottom-right corners
                    (x0, y0), (x1, y1) = points
                    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                texts.append(word.value)
                quads.append(points)
    return to_text(texts, quads)


class DoctrModel:
    def __init__(self, config: dict):
        import torch
        from doctr.models import ocr_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = ocr_predictor(
            config["det"],
            config["rec"],
            pretrained=True,
            assume_straight_pages=config["assume_straight_pages"],
            # Pages are tilted by at most 5 degrees, never turned by 90 or 180.
            disable_page_orientation=True,
            disable_crop_orientation=True,
            resolve_lines=False,
            resolve_blocks=False,
        ).to(self.device)

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        import numpy as np
        from PIL import Image

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(np.array(image.convert("RGB")))
        pages = self.predictor(images).pages
        return [
            Prediction(page_text(page, image.shape[1], image.shape[0]))
            for image, page in zip(images, pages, strict=True)
        ]

    def info(self) -> dict:
        tensors = [
            *self.predictor.det_predictor.model.parameters(),
            *self.predictor.reco_predictor.model.parameters(),
        ]
        return {
            "device": self.device,
            "parameters": sum(t.numel() for t in tensors),
            "weights_mb": round(sum(t.numel() * t.element_size() for t in tensors) / 1e6, 1),
        }
