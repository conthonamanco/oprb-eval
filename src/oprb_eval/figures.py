"""Figures for the report, drawn from the tables of `analysis`. Needs matplotlib (extra `analysis`)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .analysis import MODELS  # noqa: E402

NAMES = {
    "tesseract": "Tesseract 5",
    "ppocrv5_mobile_en": "PP-OCRv5",
    "doctr_fast_parseq": "docTR",
    "lightonocr2_1b": "LightOnOCR-2-1B",
    "qwen3_vl_2b": "Qwen3-VL-2B",
}
# Categorical slots 1-5 in fixed order (validated: adjacent CVD dE >= 9.1). Three of them are
# under 3:1 against the surface, so every series also gets a direct label and its own marker.
COLORS = dict(zip(MODELS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"), strict=True))
MARKERS = dict(zip(MODELS, ("o", "s", "^", "D", "v"), strict=True))
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("worse_better", ["#e34948", "#f0efec", "#2a78d6"])
TYPES = ("Black_Ink", "White_Whitener", "White_Burnt", "Through_Dust", "Through_Stamp", "Sim")
DENSITIES = (("0p5pct", 0.5), ("1p0pct", 1.0), ("2p0pct", 2.0))

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def occlusion_heatmap(analysis_dir: Path, tag: str, out: Path, protocol: str = "columns") -> None:
    """F1 under each occlusion type minus the model's mean F1 in that region (points).

    Subtracting each model's own mean removes the gap between models, so a cell shows how
    much that occlusion hurts or helps that model compared with its other pages.
    """
    rows = [r for r in _read(analysis_dir / f"occlusion_type_{tag}.csv") if r["protocol"] == protocol]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.4), sharey=True)
    grids = []
    for category in ("text", "nontext"):
        grid = []
        for model in MODELS:
            cells = {r["occlusion_type"]: r for r in rows if r["category"] == category and r["model"] == model}
            n_total = sum(int(c["n"]) for c in cells.values())
            mean = sum(float(c["f1_mean"]) * int(c["n"]) for c in cells.values()) / n_total
            grid.append([100 * (float(cells[t]["f1_mean"]) - mean) for t in TYPES])
        grids.append((category, grid, [int(cells[t]["n"]) for t in TYPES]))
    limit = max(abs(v) for _, g, _ in grids for row in g for v in row)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for ax, (category, grid, counts) in zip(axes, grids, strict=True):
        image = ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_xticks(range(len(TYPES)), [f"{t.replace('_', chr(10))}\nn={n}" for t, n in zip(TYPES, counts)],
                      rotation=0, fontsize=6)
        ax.set_yticks(range(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.set_title(f"Trang {category}", color=TEXT, fontsize=8.5, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                strong = abs(value) > 0.6 * limit
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if strong else TEXT)
    bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("F1 trừ trung bình của mô hình (điểm)", color=MUTED, fontsize=7)
    bar.outline.set_visible(False)
    _save(fig, out, "fig_occlusion_type")


def density_lines(analysis_dir: Path, tag: str, out: Path, protocol: str = "columns") -> None:
    """Mean F1 at each occlusion density. Only the four types that come in densities are
    included; Sim and Through_Stamp pages have no density level."""
    rows = [r for r in _read(analysis_dir / f"density_{tag}.csv") if r["protocol"] == protocol]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    xs = [x for _, x in DENSITIES]
    for ax, category in zip(axes, ("text", "nontext"), strict=True):
        for model in MODELS:
            cells = {r["density"]: r for r in rows if r["category"] == category and r["model"] == model}
            ys = [float(cells[d]["f1_mean"]) for d, _ in DENSITIES]
            ax.plot(xs, ys, color=COLORS[model], linewidth=2, marker=MARKERS[model], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=NAMES[model])
            if category == "nontext":  # its lines end far apart; on text pages labels would collide
                ax.annotate(NAMES[model], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=6.5, color=TEXT)
        counts = [int(next(r for r in rows if r["category"] == category and r["density"] == d)["n"])
                  for d, _ in DENSITIES]
        ax.set_xticks(xs, [f"{x:g}%\nn={n}" for x, n in zip(xs, counts, strict=True)])
        ax.set_xlim(0.35, 2.9)
        ax.set_title(f"Trang {category}", color=TEXT, fontsize=8.5, loc="left")
        ax.set_xlabel("diện tích bị che", color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1 trung bình")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center", ncol=len(MODELS),
               bbox_to_anchor=(0.5, -0.2))
    _save(fig, out, "fig_density")


def run(analysis_dir: Path, tag: str, out: Path) -> None:
    occlusion_heatmap(analysis_dir, tag, out)
    density_lines(analysis_dir, tag, out)
