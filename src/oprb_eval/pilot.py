"""Compare settings of one model on a few training pages.

Training pages, with training ground truth, are used so that no setting is ever chosen
by looking at the test set.
"""

from __future__ import annotations

import gc
import statistics
import time
import tomllib
from pathlib import Path

from . import env
from .data import Page, read_gt_tokens
from .metrics import score_page
from .models import build
from .text import clean_text, prediction_text, reference_text


def parse_variant(text: str) -> dict:
    """'scale=2,deskew=true,dtype=float16' -> {'scale': 2, 'deskew': True, 'dtype': 'float16'}"""
    values: dict = {}
    for part in filter(None, text.split(",")):
        key, separator, raw = part.partition("=")
        if not separator:
            raise SystemExit(f"Bad variant '{part}', expected key=value.")
        try:
            values[key.strip()] = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            values[key.strip()] = raw.strip()
    return values


def agreement(reference: list[str], texts: list[str]) -> float | None:
    """Mean token F1 of `texts` against `reference`, over pages the reference could read."""
    scores = [
        score_page(clean_text(r), clean_text(t)).f1
        for r, t in zip(reference, texts, strict=True)
        if clean_text(r)
    ]
    return statistics.fmean(scores) if scores else None


def accuracy(pages: list[Page], texts: list[str]) -> dict:
    result = {}
    for protocol in ("notebook", "normalized"):
        scores = []
        for page, text in zip(pages, texts, strict=True):
            reference = reference_text(read_gt_tokens(page.gt_path), protocol)
            if reference:
                scores.append(score_page(reference, prediction_text(text, protocol)))
        if scores:
            result[protocol] = {
                name: statistics.fmean(getattr(s, name) for s in scores)
                for name in ("cer", "wer", "precision", "recall", "f1")
            }
    return result


def run_variant(config: dict, pages: list[Page], batch_size: int) -> dict:
    env.reset_peak_vram()
    images = [p.image_path for p in pages]
    model = build(config)
    model.transcribe(images[:1])  # warm-up
    start = time.perf_counter()
    predictions = []
    for i in range(0, len(images), batch_size):
        predictions += model.transcribe(images[i : i + batch_size])
    seconds = time.perf_counter() - start
    result = {
        "seconds_per_page": seconds / len(images),
        "mean_tokens": statistics.fmean(p.output_tokens or 0 for p in predictions),
        "max_tokens": max(p.output_tokens or 0 for p in predictions),
        "empty": sum(not p.text.strip() for p in predictions),
        "truncated": sum(p.truncated for p in predictions),
        "peak_vram_mb": env.peak_vram_mb(),
        "model_info": model.info(),
        "texts": [p.text for p in predictions],
    }
    del model
    return result


def compare(
    config: dict, pages: list[Page], variants: list[str], batch_sizes: list[int]
) -> list[dict]:
    results: list[dict] = []
    for variant in variants or [""]:
        for batch_size in batch_sizes:
            head = {"variant": variant, "batch_size": batch_size}
            try:
                result = head | run_variant(config | parse_variant(variant), pages, batch_size)
            except Exception as exc:
                result = head | {"error": repr(exc)}
            gc.collect()
            env.release_vram()
            if "error" not in result:
                first = next((r for r in results if "error" not in r), result)
                result["agreement"] = agreement(first["texts"], result["texts"])
                result["accuracy"] = accuracy(pages, result["texts"])
            results.append(result)
            print(summary(result), flush=True)
    return results


def summary(result: dict) -> str:
    head = f"[{result['variant'] or 'config'}] batch={result['batch_size']}"
    if "error" in result:
        return f"{head} FAILED {result['error']}"
    parts = [head, f"{result['seconds_per_page']:.2f} s/page"]
    for protocol, s in result["accuracy"].items():
        parts.append(f"{protocol}: CER={s['cer']:.4f} WER={s['wer']:.4f} F1={s['f1']:.4f}")
    same = result["agreement"]
    parts.append(f"agreement={'n/a' if same is None else f'{same:.4f}'}")
    parts.append(f"tokens={result['mean_tokens']:.0f} (max {result['max_tokens']})")
    parts.append(f"empty={result['empty']} truncated={result['truncated']}")
    parts.append(f"vram={result['peak_vram_mb']} MB")
    return " | ".join(parts)
