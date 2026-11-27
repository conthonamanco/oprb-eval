"""Tables for the report, computed from the per-image scores. Needs no GPU and no ground truth.

Every number in the report comes from these tables. Rules that pick pages (qualitative
examples) are fixed here and depend only on scores, never on looking at the pages.
"""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .score import PROTOCOLS, RESAMPLES, write_csv

MODELS = ("tesseract", "ppocrv5_mobile_en", "doctr_fast_parseq", "lightonocr2_1b", "qwen3_vl_2b")
NUMERIC = ("cer", "wer", "precision", "recall", "f1", "char_errors", "word_errors",
           "ref_chars", "ref_words", "hyp_words")
INTEGER = {"char_errors", "word_errors", "ref_chars", "ref_words", "hyp_words"}
LOWER_IS_BETTER = {"cer": True, "wer": True, "f1": False}


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for name in NUMERIC:
            row[name] = int(row[name]) if name in INTEGER else float(row[name])
        row["truncated"] = row["truncated"] == "True"
    return rows


def load_all(results: Path, tag: str, models=MODELS) -> dict[str, dict[str, list[dict]]]:
    """{protocol: {model: rows}}, every model scored on the same pages in the same order."""
    table: dict[str, dict[str, list[dict]]] = {}
    for protocol in PROTOCOLS:
        table[protocol] = {}
        for model in models:
            rows = load_rows(results / model / f"per_image_{protocol}_{tag}.csv")
            table[protocol][model] = sorted(rows, key=lambda r: r["key"])
        keys = {tuple(r["key"] for r in rows) for rows in table[protocol].values()}
        if len(keys) != 1:
            raise ValueError(f"{protocol}: the models were not scored on the same pages.")
    return table


def resample_indices(n: int, seed: int = 0) -> list[list[int]]:
    """Bootstrap resamples of page indices, shared by every comparison for consistency."""
    rng = random.Random(seed)
    return [rng.choices(range(n), k=n) for _ in range(RESAMPLES)]


def interval(values: list[float], indices: list[list[int]]) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean of `values`."""
    means = sorted(sum(map(values.__getitem__, idx)) / len(idx) for idx in indices)
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def micro_interval(errors: list[int], lengths: list[int], indices: list[list[int]]) -> tuple[float, float]:
    """95% bootstrap interval of a micro rate sum(errors) / sum(lengths), resampling pages."""
    rates = sorted(
        sum(map(errors.__getitem__, idx)) / sum(map(lengths.__getitem__, idx)) for idx in indices
    )
    return rates[int(0.025 * RESAMPLES)], rates[int(0.975 * RESAMPLES) - 1]


def micro_cer(rows: list[dict]) -> float:
    """Character errors over reference characters: long pages weigh more, and a page whose
    reference is a few characters long cannot dominate the result as it can in the mean."""
    return sum(r["char_errors"] for r in rows) / sum(r["ref_chars"] for r in rows)


def difference_interval(a: list[float], b: list[float], seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of mean(a) - mean(b) for two independent groups of pages."""
    rng = random.Random(seed)
    diffs = sorted(
        statistics.fmean(rng.choices(a, k=len(a))) - statistics.fmean(rng.choices(b, k=len(b)))
        for _ in range(RESAMPLES)
    )
    return diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES) - 1]


def ranks(values: dict[str, float], lower_is_better: bool) -> dict[str, int]:
    ordered = sorted(values, key=lambda m: values[m], reverse=not lower_is_better)
    return {model: i + 1 for i, model in enumerate(ordered)}


def overall(table) -> list[dict]:
    """One row per model and protocol, with each model's rank under that protocol."""
    out = []
    for protocol, by_model in table.items():
        means = {
            m: {k: statistics.fmean(r[k] for r in rows) for k in ("cer", "wer", "precision", "recall", "f1")}
            for m, rows in by_model.items()
        }
        rank = {k: ranks({m: means[m][k] for m in means}, LOWER_IS_BETTER[k]) for k in ("cer", "wer", "f1")}
        micro_rank = ranks({m: micro_cer(rows) for m, rows in by_model.items()}, True)
        for model, rows in by_model.items():
            out.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "n": len(rows),
                    "n_truncated": sum(r["truncated"] for r in rows),
                    **{f"{k}_mean": v for k, v in means[model].items()},
                    "cer_median": statistics.median(r["cer"] for r in rows),
                    "cer_micro": micro_cer(rows),
                    "rank_cer_micro": micro_rank[model],
                    "rank_cer": rank["cer"][model],
                    "rank_wer": rank["wer"][model],
                    "rank_f1": rank["f1"][model],
                }
            )
    return out


def paired(table, metrics=("cer", "f1")) -> list[dict]:
    """Every pair of models on the same pages: mean difference A - B and its bootstrap interval.

    Resampling pages and taking the difference on each page removes the variation between
    pages that both models share, so the interval is much narrower than comparing two
    separate intervals.
    """
    out = []
    for protocol, by_model in table.items():
        n = len(next(iter(by_model.values())))
        indices = resample_indices(n)
        for a, b in combinations(by_model, 2):
            for metric in metrics:
                diffs = [ra[metric] - rb[metric] for ra, rb in zip(by_model[a], by_model[b], strict=True)]
                low, high = interval(diffs, indices)
                better = (lambda d: d < 0) if LOWER_IS_BETTER[metric] else (lambda d: d > 0)
                out.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "model_a": a,
                        "model_b": b,
                        "mean_a": statistics.fmean(r[metric] for r in by_model[a]),
                        "mean_b": statistics.fmean(r[metric] for r in by_model[b]),
                        "diff": statistics.fmean(diffs),
                        "diff_low": low,
                        "diff_high": high,
                        "excludes_zero": low > 0 or high < 0,
                        "share_a_better": sum(map(better, diffs)) / n,
                        "share_b_better": sum(better(-d) for d in diffs) / n,
                    }
                )
            # Micro CER: both models share the reference lengths, so resample pages and
            # take the difference of the two ratios on each resample.
            lengths = [r["ref_chars"] for r in by_model[a]]
            total = sum(lengths)
            ea = [r["char_errors"] for r in by_model[a]]
            eb = [r["char_errors"] for r in by_model[b]]
            diff_errors = [x - y for x, y in zip(ea, eb, strict=True)]
            low, high = micro_interval(diff_errors, lengths, indices)
            out.append(
                {
                    "protocol": protocol,
                    "metric": "cer_micro",
                    "model_a": a,
                    "model_b": b,
                    "mean_a": sum(ea) / total,
                    "mean_b": sum(eb) / total,
                    "diff": sum(diff_errors) / total,
                    "diff_low": low,
                    "diff_high": high,
                    "excludes_zero": low > 0 or high < 0,
                    "share_a_better": sum(d < 0 for d in diff_errors) / n,
                    "share_b_better": sum(d > 0 for d in diff_errors) / n,
                }
            )
    return out


def category_gap(table, metrics=("cer", "f1")) -> list[dict]:
    """nontext minus text, overall and within each layout.

    `nontext` pages are far more often two-column than `text` pages, so a raw gap mixes the
    region with the layout; the per-layout rows separate the two.
    """
    out = []
    for protocol, by_model in table.items():
        for model, rows in by_model.items():
            for layout in ("all", "1col", "2col"):
                subset = [r for r in rows if layout == "all" or r["layout"] == layout]
                groups = {c: [r for r in subset if r["category"] == c] for c in ("text", "nontext")}
                row = {"protocol": protocol, "model": model, "layout": layout,
                       "n_text": len(groups["text"]), "n_nontext": len(groups["nontext"])}
                if not groups["text"] or not groups["nontext"]:
                    out.append(row)  # nothing to compare; the counts still show why
                    continue
                for metric in metrics:
                    t = [r[metric] for r in groups["text"]]
                    nt = [r[metric] for r in groups["nontext"]]
                    low, high = difference_interval(nt, t)
                    row |= {
                        f"{metric}_text": statistics.fmean(t),
                        f"{metric}_nontext": statistics.fmean(nt),
                        f"{metric}_gap": statistics.fmean(nt) - statistics.fmean(t),
                        f"{metric}_gap_low": low,
                        f"{metric}_gap_high": high,
                    }
                for c in ("text", "nontext"):
                    row[f"cer_median_{c}"] = statistics.median(r["cer"] for r in groups[c])
                    row[f"cer_micro_{c}"] = micro_cer(groups[c])
                row["cer_micro_gap"] = row["cer_micro_nontext"] - row["cer_micro_text"]
                out.append(row)
    return out


def reference_lengths(table, bins=(0, 10, 50, 200, 1000)) -> list[dict]:
    """Pages per region by reference length in characters. Very short references make
    per-page CER explode (one extra word on a 2-character reference is a CER of several)."""
    out = []
    for protocol, by_model in table.items():
        rows = next(iter(by_model.values()))
        edges = [*bins, float("inf")]
        for category in ("text", "nontext"):
            lengths = [r["ref_chars"] for r in rows if r["category"] == category]
            for low, high in zip(edges, edges[1:]):
                out.append(
                    {
                        "protocol": protocol,
                        "category": category,
                        "ref_chars_from": low,
                        "ref_chars_below": high,
                        "pages": sum(low <= x < high for x in lengths),
                    }
                )
    return out


def by_group(table, field: str, metrics=("cer", "f1")) -> list[dict]:
    """Means by region and `field` (occlusion type or density), with each model's rank in a cell."""
    out = []
    for protocol, by_model in table.items():
        cells: dict[tuple, dict[str, list[dict]]] = defaultdict(dict)
        for model, rows in by_model.items():
            groups: dict[tuple, list[dict]] = defaultdict(list)
            for r in rows:
                groups[(r["category"], r[field])].append(r)
            for key, members in groups.items():
                cells[key][model] = members
        for (category, value), members_by_model in sorted(cells.items()):
            means = {m: {k: statistics.fmean(r[k] for r in rows) for k in metrics}
                     for m, rows in members_by_model.items()}
            rank = {k: ranks({m: means[m][k] for m in means}, LOWER_IS_BETTER[k]) for k in metrics}
            micro = {m: micro_cer(rows) for m, rows in members_by_model.items()}
            micro_rank = ranks(micro, True)
            for model, rows in members_by_model.items():
                out.append(
                    {
                        "protocol": protocol,
                        "category": category,
                        field: value,
                        "model": model,
                        "n": len(rows),
                        "n_truncated": sum(r["truncated"] for r in rows),
                        **{f"{k}_mean": means[model][k] for k in metrics},
                        "cer_median": statistics.median(r["cer"] for r in rows),
                        "cer_micro": micro[model],
                        **{f"rank_{k}": rank[k][model] for k in metrics},
                        "rank_cer_micro": micro_rank[model],
                    }
                )
    return out


def spread(group_rows: list[dict], field: str, metric: str = "f1") -> list[dict]:
    """For each model and region: best and worst `field` value and the range between them."""
    out = []
    keyed: dict[tuple, list[dict]] = defaultdict(list)
    for r in group_rows:
        keyed[(r["protocol"], r["category"], r["model"])].append(r)
    for (protocol, category, model), rows in sorted(keyed.items()):
        best = max(rows, key=lambda r: r[f"{metric}_mean"])
        worst = min(rows, key=lambda r: r[f"{metric}_mean"])
        out.append(
            {
                "protocol": protocol,
                "category": category,
                "model": model,
                "metric": metric,
                "best": best[field],
                "best_value": best[f"{metric}_mean"],
                "worst": worst[field],
                "worst_value": worst[f"{metric}_mean"],
                "range": best[f"{metric}_mean"] - worst[f"{metric}_mean"],
            }
        )
    return out


def length_ratio(row: dict) -> float:
    return row["hyp_words"] / row["ref_words"]


def error_direction(table, protocol: str = "normalized") -> list[dict]:
    """Does a model drop text (short output, low recall) or add text (long output, low precision)?"""
    out = []
    for model, rows in table[protocol].items():
        ratios = [length_ratio(r) for r in rows]
        precision = statistics.fmean(r["precision"] for r in rows)
        recall = statistics.fmean(r["recall"] for r in rows)
        out.append(
            {
                "protocol": protocol,
                "model": model,
                "precision_mean": precision,
                "recall_mean": recall,
                "precision_minus_recall": precision - recall,
                "words_ratio_median": statistics.median(ratios),
                "words_ratio_micro": sum(r["hyp_words"] for r in rows) / sum(r["ref_words"] for r in rows),
                "share_under_half": sum(x < 0.5 for x in ratios) / len(ratios),
                "share_over_1_5": sum(x > 1.5 for x in ratios) / len(ratios),
                "share_empty": sum(r["hyp_words"] == 0 for r in rows) / len(rows),
            }
        )
    return out


def quantile_row(rows: list[dict], metric: str, q: float) -> dict:
    """Row at quantile `q` of `metric`, ties broken by key, so the choice is reproducible."""
    ordered = sorted(rows, key=lambda r: (r[metric], r["key"]))
    return ordered[round(q * (len(ordered) - 1))]


def qualitative_picks(table, protocol: str = "normalized") -> list[dict]:
    """Pages for the qualitative section, chosen by fixed rules on scores only.

    - shared_median: the page whose F1, averaged over the five models, is the median;
      it shows all five models on one typical page.
    - per model: the median-F1 page, the 5th-percentile (bad) page, the page with the most
      extra words (largest output/reference length) and the most missing words (smallest).
    """
    by_model = table[protocol]
    models = list(by_model)
    n = len(by_model[models[0]])
    out = []
    shared = [
        {"key": by_model[models[0]][i]["key"],
         "f1": statistics.fmean(by_model[m][i]["f1"] for m in models)}
        for i in range(n)
    ]
    pick = quantile_row(shared, "f1", 0.5)
    out.append({"rule": "shared_median", "model": "all", "key": pick["key"], "value": pick["f1"]})
    for model in models:
        rows = [dict(r, ratio=length_ratio(r)) for r in by_model[model]]
        for rule, metric, q in (
            ("median_f1", "f1", 0.5),
            ("p05_f1", "f1", 0.05),
            ("most_extra_words", "ratio", 1.0),
            ("most_missing_words", "ratio", 0.0),
        ):
            row = quantile_row(rows, metric, q)
            out.append({"rule": rule, "model": model, "key": row["key"], "value": row[metric]})
    return out


def run(results: Path, tag: str, out: Path, models=MODELS) -> dict[str, list[dict]]:
    table = load_all(results, tag, models)
    types = by_group(table, "occlusion_type")
    densities = by_group(table, "density")
    tables = {
        "overall": overall(table),
        "paired": paired(table),
        "category_gap": category_gap(table),
        "reference_lengths": reference_lengths(table),
        "occlusion_type": types,
        "occlusion_type_spread": spread(types, "occlusion_type"),
        "density": densities,
        "density_spread": spread(densities, "density"),
        "error_direction": error_direction(table),
        "qualitative_picks": qualitative_picks(table),
    }
    for name, rows in tables.items():
        write_csv(out / f"{name}_{tag}.csv", rows)
    return tables
