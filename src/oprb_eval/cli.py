from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from . import analysis, data, env, models, pilot, runner, score
from .text import PSEUDO_TOKEN


def _all_pages(args) -> list[data.Page]:
    pages = data.collect_pages(args.data)
    data.require_full_test_set(pages)
    return pages


def _pages(args) -> list[data.Page]:
    pages = _all_pages(args)
    if args.sample is not None:
        pages = data.stratified_sample(pages, args.sample, args.seed)
    return pages


def _tag(args) -> str:
    """Names output files, so results of different page sets never overwrite each other."""
    if args.sample is None:
        return "full"
    return f"sample{args.sample}" + (f"_seed{args.seed}" if args.seed != 42 else "")


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
    tag = _tag(args)
    score.write_csv(args.results / f"dataset_stats_{tag}.csv", rows)
    if args.sample is not None:
        keys = "".join(f"{key}\n" for key in sorted(p.key for p in pages))
        (args.results / f"pages_{tag}.txt").write_text(keys, encoding="utf-8", newline="\n")
    for row in rows:
        print("{category:8} {occlusion_type:18} {density:10} {pages:6}".format(**row))
    for category in data.EXPECTED:
        tokens = [data.read_gt_tokens(p.gt_path) for p in pages if p.category == category]
        if not tokens:
            continue
        sizes = sorted(len(t) for t in tokens)
        pseudo = sum(PSEUDO_TOKEN.match(tok) is not None for t in tokens for tok in t)
        print(
            f"{category}: {len(sizes)} pages, tokens/page mean {statistics.fmean(sizes):.0f}, "
            f"median {statistics.median(sizes):.0f}, p99 {sizes[int(0.99 * (len(sizes) - 1))]}, "
            f"max {sizes[-1]}, empty {sizes.count(0)}, pseudo tokens {pseudo}"
        )


def cmd_pilot(args) -> None:
    config = models.load_config(args.config, args.model)
    pages = data.stratified_sample(_all_pages(args), args.images)
    results = pilot.compare(config, pages, args.variants, args.batch_sizes)
    out = args.results / "pilot" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"environment": env.describe(), "pages": [p.key for p in pages], "results": results}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    best = next((r for r in results if "error" not in r), {"texts": []})
    for page, text in list(zip(pages, best["texts"], strict=False))[:3]:
        print(f"--- {page.key}\n{text[:400]}")
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
        raise SystemExit("latency needs --sample, the number of pages to time.")
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
    tag = _tag(args)
    for protocol in score.PROTOCOLS:
        rows, empty_gt = score.score_pages(pages, predictions, protocol)
        if not rows:
            raise SystemExit("Every page has empty ground truth, so there is nothing to score.")
        score.write_csv(out / f"per_image_{protocol}_{tag}.csv", rows)
        for name, by in score.GROUPINGS.items():
            score.write_csv(out / f"summary_{protocol}_{tag}_{name}.csv", score.aggregate(rows, by))
        total = score.aggregate(rows, ())[0]
        print(
            f"[{protocol}/{tag}] n={total['n']} failed={total['n_failed']} empty_gt={empty_gt} "
            f"CER={total['cer_mean']:.4f} WER={total['wer_mean']:.4f} "
            f"P={total['precision_mean']:.4f} R={total['recall_mean']:.4f} "
            f"F1={total['f1_mean']:.4f} [{total['f1_low']:.4f}, {total['f1_high']:.4f}]"
        )
        if 0 < total["n_failed"] < total["n"]:
            print(
                f"    without failed pages: CER={total['cer_ok_mean']:.4f} "
                f"WER={total['wer_ok_mean']:.4f} F1={total['f1_ok_mean']:.4f}"
            )


def cmd_analyze(args) -> None:
    tag = _tag(args)
    tables = analysis.run(args.results, tag, args.results / "analysis")
    print(f"wrote {len(tables)} tables to {args.results / 'analysis'}")
    if args.figures:
        from . import figures  # needs matplotlib, which the core does not

        figures.run(args.results / "analysis", tag, args.figures)
        print(f"wrote figures to {args.figures}")
    for row in tables["overall"]:
        print(
            "{protocol:10} {model:20} CER={cer_mean:.4f} (rank {rank_cer}) "
            "micro CER={cer_micro:.4f} (rank {rank_cer_micro}) "
            "F1={f1_mean:.4f} (rank {rank_f1})".format(**row)
        )


def cmd_examples(args) -> None:
    from . import qualitative

    tag = _tag(args)
    picks = qualitative.read_picks(args.picks or args.results / "analysis" / f"qualitative_picks_{tag}.csv")
    pages = {p.key: p for p in _pages(args)}
    missing = sorted({p["key"] for p in picks} - pages.keys())
    if missing:
        raise SystemExit(f"{len(missing)} picked pages are not in --sample {args.sample}, e.g. {missing[0]}")
    predictions = {}
    for model in analysis.MODELS:
        records = runner.load_predictions(args.results / model)
        absent = [p["key"] for p in picks if p["key"] not in records]
        if absent:
            raise SystemExit(f"{model}: no prediction for {len(absent)} picked pages; restore its results first.")
        predictions[model] = records
    out = args.out or args.results / "qualitative"
    examples = qualitative.build(picks, pages, predictions, out)
    for e in examples:
        shown = e.get("excerpt", {})
        wers = {m: round(v["wer"], 2) for m, v in shown.get("models", {}).items()}
        print(f"{e['rule']:18} {e['model']:18} {e['key'][:60]:60} WER {wers}")
    print(f"wrote {len(examples)} examples to {out}")


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
        "analyze": cmd_analyze,
        "examples": cmd_examples,
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
        sub.add_argument("--sample", type=_positive, help="total number of random pages; omit for all 9,000")
        sub.add_argument("--seed", type=int, default=42)
        if name == "analyze":
            sub.add_argument("--figures", type=Path, metavar="DIR", help="also draw the figures into DIR")
        if name == "examples":
            sub.add_argument("--picks", type=Path, help="default: results/analysis/qualitative_picks_<tag>.csv")
            sub.add_argument("--out", type=Path, help="default: results/qualitative")
        if name in ("stats", "analyze", "examples"):
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
