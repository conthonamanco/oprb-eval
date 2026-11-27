"""Excerpts for the qualitative section: the same few lines of a page, as each model read them.

The pages come from `analysis.qualitative_picks`, which chose them by fixed rules on scores.
Here every page is cut into windows of a few consecutive ground-truth lines within one
column; each model's output is searched for the span that matches a window best. The window
shown is again chosen by a fixed rule, never by looking at the pages:

- shared_median: the window on which the five models differ most (best and worst side by side);
- every other pick: the window on which the picked model makes the most word errors.

The ground truth is read here, so this runs only after scoring, for the report.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from .data import GTToken, Page, read_gt
from .metrics import edit_distance
from .text import PSEUDO_TOKEN, clean_text, column_count, prediction_text

WINDOW_LINES = 3  # ground-truth lines per excerpt
CROP_MARGIN = 60  # pixels; pages are rotated by up to 5 degrees and GT boxes are not
THUMBNAIL_EDGE = 900
TAIL_WORDS = 60  # words shown from the end of an output that is much longer than the GT


def gt_lines(gt: Sequence[GTToken]) -> list[list[GTToken]]:
    """Ground-truth lines in reading order: one column after the other, top to bottom.

    File order runs along a whole printed line, across both columns of a two-column page, so
    each file-order line is split at the middle of the text area on such pages.
    """
    tokens = [t for t in gt if not PSEUDO_TOKEN.match(t.text) and clean_text(t.text)]
    if not tokens:
        return []
    heights = [t.y1 - t.y0 for t in tokens if t.y1 > t.y0]
    tolerance = 0.5 * (statistics.median(heights) if heights else 1.0)
    rows: list[list[GTToken]] = [[tokens[0]]]
    for previous, token in zip(tokens, tokens[1:]):
        same_row = abs((token.y0 + token.y1) - (previous.y0 + previous.y1)) / 2 <= tolerance
        if same_row and token.x0 >= previous.x0:
            rows[-1].append(token)
        else:
            rows.append([token])
    if column_count(gt) != 2:
        return rows
    middle = (min(t.x0 for t in tokens) + max(t.x1 for t in tokens)) / 2
    columns: list[list[list[GTToken]]] = [[], []]
    for row in rows:
        for side in (0, 1):
            part = [t for t in row if ((t.x0 + t.x1) / 2 >= middle) == bool(side)]
            if part:
                columns[side].append(part)
    return columns[0] + columns[1]


def windows(lines: list[list[GTToken]], size: int = WINDOW_LINES) -> list[list[GTToken]]:
    """Consecutive groups of `size` lines (the last window may be shorter)."""
    return [sum(lines[i : i + size], []) for i in range(0, len(lines), size)]


def words_of(tokens: Sequence[GTToken]) -> list[str]:
    return clean_text(" ".join(t.text for t in tokens)).split()


def best_span(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """(start, end, word errors) of the span of `hypothesis` closest to `reference`.

    Spans start where the hypothesis has one of the reference's first words (or anywhere if
    none matches) and are 80-120 % of the reference length.
    """
    n = len(reference)
    if not hypothesis or not n:
        return 0, 0, n
    anchors = set(reference[:3])
    starts = [i for i, w in enumerate(hypothesis) if w in anchors] or range(len(hypothesis))
    lengths = range(max(1, int(0.8 * n)), int(1.2 * n) + 1)
    best = (0, 0, n)
    for start in starts:
        for length in lengths:
            end = min(start + length, len(hypothesis))
            errors = edit_distance(reference, hypothesis[start:end])
            if errors < best[2]:
                best = (start, end, errors)
            if end == len(hypothesis):
                break
    return best


def excerpt(window: list[GTToken], outputs: dict[str, list[str]]) -> dict:
    reference = words_of(window)
    models = {}
    for model, hypothesis in outputs.items():
        start, end, errors = best_span(reference, hypothesis)
        models[model] = {"text": " ".join(hypothesis[start:end]), "word_errors": errors,
                         "wer": errors / max(len(reference), 1)}
    box = [min(t.x0 for t in window), min(t.y0 for t in window),
           max(t.x1 for t in window), max(t.y1 for t in window)]
    return {"reference": " ".join(reference), "box": box, "models": models}


def choose_window(candidates: list[dict], rule: str, model: str) -> dict:
    """Fixed rules; ties go to the earliest window."""
    if rule == "shared_median":
        def spread(c):
            wers = [m["wer"] for m in c["models"].values()]
            return max(wers) - min(wers)
        return max(candidates, key=spread)
    return max(candidates, key=lambda c: c["models"][model]["word_errors"])


def box_scale(box_max: float, width: int, height: int) -> tuple[float, float]:
    """DocBank stores boxes on a 0-1000 scale per axis (x by width, y by height); scale them
    to the image if that is the case."""
    if box_max <= 1000 < max(width, height):
        return width / 1000, height / 1000
    return 1.0, 1.0


def save_images(page: Page, box: list[float], out: Path, stem: str) -> dict | None:
    """Crop of the excerpt and a page thumbnail with the excerpt outlined. None if the box
    does not fall inside the image."""
    from PIL import Image, ImageDraw

    with Image.open(page.image_path) as image:
        image = image.convert("RGB")
    sx, sy = box_scale(max(box), *image.size)
    x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    crop = (max(0, int(x0 - CROP_MARGIN)), max(0, int(y0 - CROP_MARGIN)),
            min(image.width, int(x1 + CROP_MARGIN)), min(image.height, int(y1 + CROP_MARGIN)))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return None
    scale = [sx, sy]
    image.crop(crop).save(out / f"{stem}_crop.png")
    thumbnail = image.copy()
    ImageDraw.Draw(thumbnail).rectangle(crop, outline=(227, 73, 72), width=6)
    thumbnail.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE))
    thumbnail.save(out / f"{stem}_page.png")
    return {"crop": f"{stem}_crop.png", "page": f"{stem}_page.png", "box_scale": scale,
            "image_size": list(image.size), "crop_box": list(crop)}


def build(picks: list[dict], pages: dict[str, Page], predictions: dict[str, dict[str, dict]],
          out: Path, protocol: str = "normalized", images: bool = True) -> list[dict]:
    """One example per pick. `predictions` is {model: {page key: record}}."""
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for i, pick in enumerate(picks):
        page = pages[pick["key"]]
        gt = read_gt(page.gt_path)
        outputs = {
            model: prediction_text(records[page.key]["text"], protocol).split()
            if records[page.key]["status"] == "ok" else []
            for model, records in predictions.items()
        }
        candidates = [excerpt(w, outputs) for w in windows(gt_lines(gt))]
        example = {
            **pick,
            "category": page.category,
            "occlusion_type": page.occlusion_type,
            "density": page.density,
            "reference_words": len(words_of([t for t in gt if not PSEUDO_TOKEN.match(t.text)])),
            "output_words": {model: len(words) for model, words in outputs.items()},
            "truncated": {model: bool(records[page.key].get("truncated")) for model, records in predictions.items()},
        }
        if candidates:
            example["excerpt"] = choose_window(candidates, pick["rule"], pick["model"])
            if images:
                saved = save_images(page, example["excerpt"]["box"], out, f"{i:02d}_{pick['rule']}")
                if saved:
                    example["images"] = saved
                else:
                    print(f"{page.key}: excerpt box {example['excerpt']['box']} is outside the image, no crop")
        if pick["rule"] in ("most_extra_words", "most_missing_words") and pick["model"] in outputs:
            words = outputs[pick["model"]]
            example["output_tail" if pick["rule"] == "most_extra_words" else "output_head"] = (
                " ".join(words[-TAIL_WORDS:]) if pick["rule"] == "most_extra_words" else " ".join(words[:TAIL_WORDS])
            )
        examples.append(example)
    (out / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return examples


def read_picks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
