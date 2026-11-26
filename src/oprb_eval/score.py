"""Turn saved predictions into per-image and grouped results. Needs no GPU."""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .data import Page, read_gt
from .metrics import score_page
from .text import column_count, prediction_text, reference_text

PROTOCOLS = ("notebook", "normalized", "columns")
RESAMPLES = 1000
GROUPINGS = {
    "overall": (),
    "category": ("category",),
    "layout": ("category", "layout"),
    "type": ("category", "occlusion_type"),
    "density": ("category", "density"),
    "type_density": ("category", "occlusion_type", "density"),
}


def score_pages(
    pages: list[Page], predictions: dict[str, dict], protocol: str
) -> tuple[list[dict], int]:
    """Score `pages`; returns (rows, number of pages skipped for empty ground truth).

    A failed page counts as an empty transcript instead of being dropped, so a model
    cannot gain by crashing on hard pages. `aggregate` also reports the number of failures
    and the means without them, which is what the sample notebook averages.
    """
    rows, empty_gt = [], 0
    for page in pages:
        gt = read_gt(page.gt_path)
        reference = reference_text(gt, protocol)
        if not reference:
            empty_gt += 1
            continue
        record = predictions[page.key]
        ok = record["status"] == "ok"
        hypothesis = prediction_text(record["text"], protocol) if ok else ""
        rows.append(
            {
                "key": page.key,
                "category": page.category,
                "occlusion_type": page.occlusion_type,
                "density": page.density,
                "layout": f"{column_count(gt)}col",
                "status": record["status"],
                "truncated": ok and record["truncated"],
                **asdict(score_page(reference, hypothesis)),
            }
        )
    return rows, empty_gt


def bootstrap(rows: list[dict], metrics: tuple[str, ...]) -> dict[str, float]:
    """95% percentile bootstrap interval of each mean, resampling pages.

    It ignores stratification and the finite size of the test set, so for a sample of the
    test set the interval is, if anything, slightly too wide.
    """
    rng = random.Random(0)
    values = [tuple(row[m] for m in metrics) for row in rows]
    means = [
        [statistics.fmean(column) for column in zip(*rng.choices(values, k=len(values)))]
        for _ in range(RESAMPLES)
    ]
    result = {}
    for name, column in zip(metrics, zip(*means), strict=True):
        ordered = sorted(column)
        result[f"{name}_low"] = ordered[int(0.025 * RESAMPLES)]
        result[f"{name}_high"] = ordered[int(0.975 * RESAMPLES) - 1]
    return result


def aggregate(rows: list[dict], by: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[c] for c in by)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        ref_chars = sum(r["ref_chars"] for r in members)
        ref_words = sum(r["ref_words"] for r in members)
        summary = dict(zip(by, key))
        summary["n"] = len(members)
        summary["n_failed"] = sum(r["status"] != "ok" for r in members)
        summary["n_truncated"] = sum(r["truncated"] for r in members)
        succeeded = [r for r in members if r["status"] == "ok"]
        for metric in ("cer", "wer", "precision", "recall", "f1"):
            summary[f"{metric}_mean"] = statistics.fmean(r[metric] for r in members)
            summary[f"{metric}_ok_mean"] = (
                statistics.fmean(r[metric] for r in succeeded) if succeeded else None
            )
        summary |= bootstrap(members, ("cer", "wer", "f1"))
        summary["cer_median"] = statistics.median(r["cer"] for r in members)
        summary["wer_median"] = statistics.median(r["wer"] for r in members)
        summary["cer_micro"] = sum(r["char_errors"] for r in members) / ref_chars
        summary["wer_micro"] = sum(r["word_errors"] for r in members) / ref_words
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
