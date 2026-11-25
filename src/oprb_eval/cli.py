from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from . import data, env, models, pilot, runner, score
from .text import PSEUDO_TOKEN


def _pages(args) -> list[data.Page]:
    pages = data.collect_pages(args.data)
    data.require_full_test_set(pages)
    if args.sample is not None:
        pages = data.stratified_sample(pages, args.sample, args.seed)
    return pages


def _shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    try:
        shard = int(index), int(count)
    except ValueError:
        raise argparse.ArgumentTypeError("use INDEX/COUNT, for example 0/2") from None
    if not 0 <= shard[0] < shard[1]:
        raise argparse.ArgumentTypeError("INDEX must be between 0 and COUNT - 1")
    return shard


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def cmd_download(args) -> None:
    data.download(args.data)


def cmd_stats(args) -> None:
    pages = _pages(args)
    counts = Counter((p.category, p.occlusion_type, p.density) for p in pages)
    rows = [
        {"category": c, "occlusion_type": t, "density": d, "pages": n}
        for (c, t, d), n in sorted(counts.items())
    ]
    score.write_csv(args.results / "dataset_stats.csv", rows)
    for row in rows:
        print("{category:8} {occlusion_type:18} {density:10} {pages:6}".format(**row))
    for category in data.EXPECTED:
        tokens = [data.read_gt_tokens(p.gt_path) for p in pages if p.category == category]
        sizes = sorted(len(t) for t in tokens)
        pseudo = sum(PSEUDO_TOKEN.match(tok) is not None for t in tokens for tok in t)
        print(
            f"{category}: {len(sizes)} pages, tokens/page mean {statistics.fmean(sizes):.0f}, "
            f"median {statistics.median(sizes):.0f}, p99 {sizes[int(0.99 * (len(sizes) - 1))]}, "
            f"max {sizes[-1]}, empty {sizes.count(0)}, pseudo tokens {pseudo}"
        )


def cmd_pilot(args) -> None:
    config = models.load_config(args.config, args.model)
    pages = data.train_pages(args.data, args.images)
    results = pilot.compare(config, pages, args.variants, args.batch_sizes)
    out = args.results / "pilot" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"environment": env.describe(), "pages": [p.sample_id for p in pages], "results": results}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    best = next((r for r in results if "error" not in r), {"texts": []})
    for page, text in list(zip(pages, best["texts"], strict=False))[:3]:
        print(f"--- {page.sample_id}\n{text[:400]}")
    if all("error" in r for r in results):
        raise SystemExit("Every variant failed.")


def cmd_run(args) -> None:
    config = models.load_config(args.config, args.model)
    counts = runner.run(
        lambda: models.build(config),
        args.model,
        config,
        _pages(args),
        args.results / args.model,
        args.shard,
        args.batch_size,
    )
    print(f"{args.model}: {counts}")


def cmd_latency(args) -> None:
    if args.sample is None:
        raise SystemExit("latency needs --sample, the number of pages per category.")
    config = models.load_config(args.config, args.model)
    report = runner.measure_latency(
        lambda: models.build(config), args.model, config, _pages(args), args.results / args.model
    )
    if not report["n"]:
        raise SystemExit("Every page failed.")
    print(
        f"{args.model}: n={report['n']} failed={report['n_failed']} mean={report['mean']:.3f}s "
        f"median={report['median']:.3f}s p95={report['p95']:.3f}s"
    )


def cmd_score(args) -> None:
    pages = _pages(args)
    out = args.results / args.model
    predictions = runner.load_predictions(out)
    missing = sum(p.key not in predictions for p in pages)
    if missing:
        raise SystemExit(
            f"{missing} of {len(pages)} pages have no prediction. "
            "Finish the run, or score the same --sample that was run."
        )
    # Sample scores go to separate files so they are never mistaken for final ones.
    tag = "full" if args.sample is None else f"sample{args.sample}"
    for protocol in score.PROTOCOLS:
        rows, empty_gt = score.score_pages(pages, predictions, protocol)
        score.write_csv(out / f"per_image_{protocol}_{tag}.csv", rows)
        for name, by in score.GROUPINGS.items():
            score.write_csv(out / f"summary_{protocol}_{tag}_{name}.csv", score.aggregate(rows, by))
        total = score.aggregate(rows, ())[0]
        print(
            f"[{protocol}/{tag}] n={total['n']} failed={total['n_failed']} empty_gt={empty_gt} "
            f"CER={total['cer_mean']:.4f} WER={total['wer_mean']:.4f} "
            f"P={total['precision_mean']:.4f} R={total['recall_mean']:.4f} "
            f"F1={total['f1_mean']:.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oprb_eval")
    commands = parser.add_subparsers(dest="command", required=True)
    functions = {
        "download": cmd_download,
        "stats": cmd_stats,
        "pilot": cmd_pilot,
        "run": cmd_run,
        "latency": cmd_latency,
        "score": cmd_score,
    }
    for name, function in functions.items():
        sub = commands.add_parser(name)
        sub.set_defaults(function=function)
        # Defaults are relative to the working directory: run from the repo root.
        sub.add_argument("--data", type=Path, default=Path("data/oprb_test"))
        if name == "download":
            continue
        sub.add_argument("--results", type=Path, default=Path("results"))
        if name == "pilot":
            sub.add_argument("--model", required=True)
            sub.add_argument("--config", type=Path, default=Path("configs/models.toml"))
            sub.add_argument("--images", type=_positive, default=24)
            sub.add_argument("--variants", nargs="*", default=[], metavar="KEY=VALUE,...")
            sub.add_argument("--batch-sizes", nargs="+", type=_positive, default=[1])
            continue
        sub.add_argument("--sample", type=_positive, help="pages per category; omit for all 9,000")
        sub.add_argument("--seed", type=int, default=42)
        if name == "stats":
            continue
        sub.add_argument("--model", required=True)
        if name in ("run", "latency"):
            sub.add_argument("--config", type=Path, default=Path("configs/models.toml"))
        if name == "run":
            sub.add_argument("--shard", type=_shard, default=(0, 1), metavar="INDEX/COUNT")
            sub.add_argument("--batch-size", type=_positive, default=1)
    args = parser.parse_args(argv)
    args.function(args)
    return 0
