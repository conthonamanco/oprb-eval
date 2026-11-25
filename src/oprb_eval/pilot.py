"""Compare precisions and batch sizes of one model on a few training images.

Training pages are used so that no setting is ever chosen by looking at the test set.
No ground truth is involved: every variant is compared with the first one.
"""

from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path

from . import env
from .metrics import score_page
from .models import build
from .text import clean_text


def agreement(reference: list[str], texts: list[str]) -> float | None:
    """Mean token F1 of `texts` against `reference`, over pages the reference could read."""
    scores = [
        score_page(clean_text(r), clean_text(t)).f1
        for r, t in zip(reference, texts, strict=True)
        if clean_text(r)
    ]
    return statistics.fmean(scores) if scores else None


def run_variant(config: dict, images: list[Path], batch_size: int) -> dict:
    env.reset_peak_vram()
    model = build(config)
    model.transcribe(images[:1])  # warm-up
    start = time.perf_counter()
    predictions = []
    for i in range(0, len(images), batch_size):
        predictions += model.transcribe(images[i : i + batch_size])
    seconds = time.perf_counter() - start
    result = {
        "dtype": config.get("dtype"),
        "batch_size": batch_size,
        "seconds_per_page": seconds / len(images),
        "mean_tokens": statistics.fmean(p.output_tokens or 0 for p in predictions),
        "empty": sum(not p.text.strip() for p in predictions),
        "truncated": sum(p.truncated for p in predictions),
        "peak_vram_mb": env.peak_vram_mb(),
        "model_info": model.info(),
        "texts": [p.text for p in predictions],
    }
    del model
    gc.collect()
    env.release_vram()
    return result


def compare(config: dict, images: list[Path], dtypes: list[str], batch_sizes: list[int]) -> list[dict]:
    variants = [config | {"dtype": d} for d in dtypes] if dtypes else [config]
    results: list[dict] = []
    for variant in variants:
        for batch_size in batch_sizes:
            try:
                result = run_variant(variant, images, batch_size)
            except Exception as exc:
                result = {"dtype": variant.get("dtype"), "batch_size": batch_size, "error": repr(exc)}
                gc.collect()
                env.release_vram()
            first = next((r for r in results if "error" not in r), result)
            if "error" not in result:
                result["agreement"] = agreement(first["texts"], result["texts"])
            results.append(result)
            print(summary(result), flush=True)
    return results


def summary(result: dict) -> str:
    head = f"dtype={result['dtype']} batch={result['batch_size']}"
    if "error" in result:
        return f"{head} FAILED {result['error']}"
    same = result["agreement"]
    return (
        f"{head} {result['seconds_per_page']:.2f} s/page tokens={result['mean_tokens']:.0f} "
        f"empty={result['empty']} truncated={result['truncated']} "
        f"vram={result['peak_vram_mb']} MB agreement={'n/a' if same is None else f'{same:.4f}'}"
    )
