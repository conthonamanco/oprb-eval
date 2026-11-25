"""Inference loop: append-only, resumable, shardable, and refuses to mix settings.

Accuracy runs may be batched and spread over GPUs, so they record no per-page time.
Inference time is measured separately by `measure_latency`.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from . import env
from .data import Page
from .models import Model, Prediction

RUN_CONFIG = "run_config.json"
SESSIONS = "sessions.jsonl"
LATENCY = "latency.json"


def fingerprint(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def predictions_file(out_dir: Path, shard: tuple[int, int]) -> Path:
    index, count = shard
    name = "predictions.jsonl" if count == 1 else f"predictions-{index}of{count}.jsonl"
    return out_dir / name


def load_predictions(out_dir: Path) -> dict[str, dict]:
    """Merge all shards. A successful record always wins over a failed one."""
    records: dict[str, dict] = {}
    for path in sorted(out_dir.glob("predictions*.jsonl")):
        # Every complete record ends with "\n", so the last piece is "" or a torn write.
        *lines, _ = path.read_text(encoding="utf-8", errors="replace").split("\n")
        for line in lines:
            record = json.loads(line)
            old = records.get(record["key"])
            if old is None or old["status"] != "ok" or record["status"] == "ok":
                records[record["key"]] = record
    return records


def _drop_torn_line(path: Path) -> None:
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            path.write_bytes(data[: data.rfind(b"\n") + 1])


def _claim_output(out_dir: Path, name: str, config: dict) -> None:
    """Create the output folder, or check that it belongs to the same settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RUN_CONFIG
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint(config):
            raise SystemExit(
                f"{out_dir} holds results from different settings. "
                "Move that folder away or restore the old config."
            )
        return
    info = {"model": name, "fingerprint": fingerprint(config), "config": config, "created": _now()}
    # Write-then-rename, because two shards may start at the same time.
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _transcribe(model: Model, batch: list[Page]) -> list[Prediction | Exception]:
    """Run a batch; if it fails, retry page by page so one bad page costs only itself."""
    try:
        predictions = model.transcribe([p.image_path for p in batch])
        if len(predictions) != len(batch):
            raise RuntimeError(f"{len(predictions)} predictions for {len(batch)} images")
        return list(predictions)
    except Exception as exc:
        if len(batch) == 1:
            return [exc]
    return [result for page in batch for result in _transcribe(model, [page])]


def run(
    build: Callable[[], Model],
    name: str,
    config: dict,
    pages: list[Page],
    out_dir: Path,
    shard: tuple[int, int] = (0, 1),
    batch_size: int = 1,
) -> dict[str, int]:
    """Transcribe every page of this shard that has no successful record yet.

    Failed pages are retried. The model is only loaded when there is work to do.
    """
    _claim_output(out_dir, name, config)
    path = predictions_file(out_dir, shard)
    _drop_torn_line(path)
    index, count = shard
    mine = pages[index::count]
    done = {k for k, r in load_predictions(out_dir).items() if r["status"] == "ok"}
    todo = [p for p in mine if p.key not in done]
    counts = {"skipped": len(mine) - len(todo), "ok": 0, "error": 0}
    if not todo:
        return counts

    model = build()
    session = f"{int(time.time())}-{index}of{count}"
    info = {"session": session, "started": _now(), "batch_size": batch_size, "pages": len(todo)}
    info["model_info"] = model.info()
    _append_json(out_dir / SESSIONS, info | env.describe())

    start = time.perf_counter()
    with path.open("a", encoding="utf-8", newline="\n") as out:
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            for page, result in zip(batch, _transcribe(model, batch), strict=True):
                record: dict = {"key": page.key, "session": session}
                if isinstance(result, Exception):
                    record |= {"status": "error", "error": repr(result)}
                else:
                    record |= {
                        "status": "ok",
                        "output_tokens": result.output_tokens,
                        "truncated": result.truncated,
                        "text": result.text,
                    }
                counts[record["status"]] += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            finished = i + len(batch)
            if (i // batch_size) % 10 == 9 or finished == len(todo):
                rate = finished / (time.perf_counter() - start)
                left = (len(todo) - finished) / rate / 3600
                print(
                    f"{name} [{index}/{count}] {finished}/{len(todo)} errors={counts['error']} "
                    f"{rate * 3600:.0f} pages/h, {left:.1f} h left",
                    flush=True,
                )
    return counts


def measure_latency(
    build: Callable[[], Model],
    name: str,
    config: dict,
    pages: list[Page],
    out_dir: Path,
    warmup: int = 3,
) -> dict:
    """Time one image at a time, in one process, after a few untimed warm-up images."""
    _claim_output(out_dir, name, config)
    env.reset_peak_vram()
    start = time.perf_counter()
    model = build()
    load_seconds = time.perf_counter() - start
    for page in pages[:warmup]:
        _transcribe(model, [page])
    seconds, failed = {}, []
    for page in pages:
        start = time.perf_counter()
        result = _transcribe(model, [page])[0]
        elapsed = time.perf_counter() - start
        if isinstance(result, Exception):
            failed.append(page.key)
        else:
            seconds[page.key] = elapsed
    values = sorted(seconds.values())
    report = {
        "model": name,
        "fingerprint": fingerprint(config),
        "measured": _now(),
        "environment": env.describe(),
        "model_info": model.info(),
        "load_seconds": load_seconds,
        "peak_vram_mb": env.peak_vram_mb(),
        "n": len(values),
        "n_failed": len(failed),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": values[int(0.95 * (len(values) - 1))] if values else None,
        "seconds": seconds,
    }
    (out_dir / LATENCY).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
