"""
visualize_results.py
====================
Publication-quality figures for the quality-dataset degradation results.

Produces three figures:
  1. sensitivity_grid.png   -- 5-panel sensitivity curves with annotations
  2. convergence.png        -- combined degradation gap-closing chart
  3. quality_predictor.png  -- quality score vs recall scatter, coloured by
                               degradation type

Usage
-----
python Run/Quality/visualize_results.py --dataset quality
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       180,
})

PAR_COLOR  = {"naive": "#1565C0", "graph": "#B71C1C", "hybrid": "#2E7D32"}
PAR_LABEL  = {"naive": "Naive RAG", "graph": "Graph RAG", "hybrid": "Hybrid RAG"}
PAR_MARKER = {"naive": "o", "graph": "s", "hybrid": "^"}

DEG_ORDER  = ["text_truncation", "entity_variation", "graph_edge_removal",
              "document_dropout", "combined"]
DEG_LABEL  = {
    "text_truncation":    "Text Truncation",
    "entity_variation":   "Entity Variation",
    "graph_edge_removal": "Graph Edge Removal",
    "document_dropout":   "Document Dropout",
    "combined":           "Combined",
}
DEG_COLOR  = {
    "text_truncation":    "#7B1FA2",
    "entity_variation":   "#0277BD",
    "graph_edge_removal": "#E65100",
    "document_dropout":   "#2E7D32",
    "combined":           "#37474F",
}

METRIC = "recall@5"
RATES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


# ── Data loading ──────────────────────────────────────────────────────────────

def load(dataset: str) -> dict:
    path = Path(f"Dataset/EvaluationData/RetrievalEval/{dataset}/summary.json")
    with open(path) as f:
        return json.load(f)


def get(summary, deg, rate, paradigm, metric=METRIC):
    return (summary.get(deg, {})
                   .get(str(rate), {})
                   .get("paradigms", {})
                   .get(paradigm, {})
                   .get(metric, 0.0))


def quality(summary, deg, rate):
    return (summary.get(deg, {})
                   .get(str(rate), {})
                   .get("overall_quality", 0.0))


# ── Figure 1: Sensitivity grid ────────────────────────────────────────────────

def fig_sensitivity_grid(summary: dict, out_dir: Path, dataset_name: str = "") -> None:
    active = [d for d in DEG_ORDER if d in summary]
    ncols, nrows = 3, 2

    # Compute data-driven y-axis ceiling so curves fill the panel
    all_vals = [
        get(summary, deg, r, par)
        for deg in active for r in RATES
        for par in ["naive", "graph", "hybrid"]
    ]
    y_ceil = min(1.05, round(max(all_vals) * 1.15 + 0.05, 1)) if all_vals else 1.05

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.52, "wspace": 0.32})
    axes_flat = axes.flatten()

    for idx, deg in enumerate(active):
        ax = axes_flat[idx]

        for par in ["naive", "graph", "hybrid"]:
            vals = [get(summary, deg, r, par) for r in RATES]
            if not any(v > 0 for v in vals):
                continue
            ax.plot(RATES, vals,
                    color=PAR_COLOR[par], marker=PAR_MARKER[par],
                    linewidth=2, markersize=5, label=PAR_LABEL[par])

        # Annotate final values for naive and hybrid (most relevant on musique)
        for par in ["naive", "hybrid", "graph"]:
            final = get(summary, deg, 0.7, par)
            if final > 0:
                ax.annotate(f"{final:.2f}",
                            xy=(0.7, final),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color=PAR_COLOR[par], va="center")

        ax.set_title(DEG_LABEL.get(deg, deg), fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("Degradation rate", fontsize=9)
        ax.set_ylabel(METRIC, fontsize=9)
        ax.set_xlim(-0.03, 0.78)
        ax.set_ylim(0, y_ceil)
        ax.set_xticks(RATES)
        ax.tick_params(labelsize=8)

        if idx == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    for idx in range(len(active), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Sensitivity to corpus degradation  ({dataset_name}, recall@5)",
                 fontsize=13, fontweight="bold", y=1.01)

    out = out_dir / "sensitivity_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Combined degradation convergence ────────────────────────────────

def fig_convergence(summary: dict, out_dir: Path, dataset_name: str = "") -> None:
    """Shows how paradigm gaps close under combined degradation.
    Right panel adapts: shows naive-vs-hybrid gap when hybrid overtakes naive
    (multi-hop corpora), otherwise naive-vs-graph gap (narrative corpora)."""
    if "combined" not in summary:
        return

    rates = RATES
    naive_vals  = [get(summary, "combined", r, "naive")  for r in rates]
    graph_vals  = [get(summary, "combined", r, "graph")  for r in rates]
    hybrid_vals = [get(summary, "combined", r, "hybrid") for r in rates]

    # Determine whether hybrid overtakes naive (musique pattern)
    hybrid_wins = any(h >= n for h, n in zip(hybrid_vals, naive_vals))
    # Find first crossover rate
    crossover_rate = next(
        (r for r, h, n in zip(rates, hybrid_vals, naive_vals) if h >= n), None
    )

    y_ceil = min(1.05, round(max(naive_vals + hybrid_vals + graph_vals) * 1.15 + 0.05, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"wspace": 0.38})

    # Left: absolute recall curves
    for par, vals in [("naive", naive_vals), ("graph", graph_vals),
                      ("hybrid", hybrid_vals)]:
        ax1.plot(rates, vals, color=PAR_COLOR[par], marker=PAR_MARKER[par],
                 linewidth=2.5, markersize=6, label=PAR_LABEL[par])

    if hybrid_wins and crossover_rate is not None:
        # Shade the region where hybrid is ahead of naive
        ax1.fill_between(rates, naive_vals, hybrid_vals,
                         where=[h >= n for h, n in zip(hybrid_vals, naive_vals)],
                         color=PAR_COLOR["hybrid"], alpha=0.12,
                         label="hybrid leads naive")
        ax1.axvline(crossover_rate, color=PAR_COLOR["hybrid"],
                    linestyle=":", linewidth=1.5, alpha=0.7,
                    label=f"crossover r={crossover_rate:.1f}")
    else:
        ax1.fill_between(rates, graph_vals, naive_vals,
                         color=PAR_COLOR["naive"], alpha=0.08, label="naive–graph gap")
        ax1.axvspan(0.6, 0.7, color="gold", alpha=0.18, label="near-convergence zone")

    ax1.set_xlabel("Combined degradation rate", fontsize=10)
    ax1.set_ylabel(METRIC, fontsize=10)
    ax1.set_title("Paradigm recall under combined degradation", fontsize=11, fontweight="bold")
    ax1.set_xlim(-0.03, 0.78)
    ax1.set_ylim(0, y_ceil)
    ax1.set_xticks(rates)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Right: bar chart of the relevant gap
    if hybrid_wins:
        # Show naive lead over hybrid (negative = hybrid winning)
        gap = [n - h for n, h in zip(naive_vals, hybrid_vals)]
        bar_colors = [PAR_COLOR["naive"] if g > 0 else PAR_COLOR["hybrid"] for g in gap]
        ylabel = "Naive lead over hybrid (recall@5)"
        title  = "Hybrid overtakes naive under degradation"
    else:
        gap = [n - g for n, g in zip(naive_vals, graph_vals)]
        bar_colors = [PAR_COLOR["naive"] if g > 0.1 else "#FFA000" for g in gap]
        ylabel = "Naive lead over graph (recall@5)"
        title  = "Naive advantage narrows with degradation"

    ax2.bar(rates, gap, width=0.07, color=bar_colors,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Combined degradation rate", fontsize=10)
    ax2.set_ylabel(ylabel, fontsize=10)
    ax2.set_title(title, fontsize=11, fontweight="bold")
    ax2.set_xticks(rates)
    y_abs = max(abs(g) for g in gap) * 1.25 if gap else 1.0
    ax2.set_ylim(-y_abs, y_abs)

    for x, g in zip(rates, gap):
        offset = 0.015 if g >= 0 else -0.015
        va = "bottom" if g >= 0 else "top"
        ax2.text(x, g + offset, f"{g:+.2f}", ha="center", va=va,
                 fontsize=8, color="black")

    fig.suptitle(f"{dataset_name}  —  Combined degradation convergence",
                 fontsize=12, fontweight="bold", y=1.02)

    out = out_dir / "convergence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Quality score as predictor ─────────────────────────────────────

def fig_quality_predictor(summary: dict, out_dir: Path, dataset_name: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"wspace": 0.38})

    # Collect points: (quality, recall, paradigm, degradation_type)
    points = []
    for deg in DEG_ORDER:
        if deg not in summary:
            continue
        for rate in RATES:
            q = quality(summary, deg, rate)
            for par in ["naive", "graph", "hybrid"]:
                v = get(summary, deg, rate, par)
                points.append((q, v, par, deg))

    # Left panel: coloured by paradigm, trend lines
    ax = axes[0]
    for par in ["naive", "graph", "hybrid"]:
        xs = [p[0] for p in points if p[2] == par]
        ys = [p[1] for p in points if p[2] == par]
        ax.scatter(xs, ys, color=PAR_COLOR[par], alpha=0.55, s=45,
                   marker=PAR_MARKER[par], label=PAR_LABEL[par],
                   edgecolors="white", linewidths=0.4, zorder=3)
        z = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 100)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.plot(xline, np.poly1d(z)(xline), color=PAR_COLOR[par],
                linewidth=1.8, linestyle="--", alpha=0.9,
                label=f"  trend  r={r:+.2f}")

    ax.set_xlabel("Overall quality score", fontsize=10)
    ax.set_ylabel(METRIC, fontsize=10)
    ax.set_title("Recall vs quality score\n(coloured by paradigm)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
              ncol=2, columnspacing=0.5)

    # Right panel: coloured by degradation type
    ax2 = axes[1]
    for par in ["naive", "graph"]:
        for deg in DEG_ORDER:
            if deg not in summary:
                continue
            xs = [quality(summary, deg, r) for r in RATES]
            ys = [get(summary, deg, r, par) for r in RATES]
            ls = "-" if par == "naive" else "--"
            lw = 2.0 if par == "naive" else 1.5
            ax2.plot(xs, ys, color=DEG_COLOR[deg], linestyle=ls,
                     linewidth=lw, alpha=0.85, marker="o", markersize=4)

    # Legend: degradation types + paradigm line styles
    deg_handles = [mpatches.Patch(color=DEG_COLOR[d], label=DEG_LABEL[d])
                   for d in DEG_ORDER if d in summary]
    style_handles = [
        Line2D([0], [0], color="black", lw=2,   linestyle="-",  label="Naive (solid)"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Graph (dashed)"),
    ]
    ax2.legend(handles=deg_handles + style_handles, fontsize=7.5,
               loc="upper left", framealpha=0.9)
    ax2.set_xlabel("Overall quality score", fontsize=10)
    ax2.set_ylabel(METRIC, fontsize=10)
    ax2.set_title("Recall trajectory by degradation type\n(naive solid, graph dashed)",
                  fontsize=11, fontweight="bold")

    fig.suptitle(f"{dataset_name}  —  Corpus quality score as retrieval predictor",
                 fontsize=12, fontweight="bold", y=1.02)

    out = out_dir / "quality_predictor.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="quality")
    args = parser.parse_args()

    summary = load(args.dataset)
    out_dir = Path(f"Dataset/EvaluationData/RetrievalEval/{args.dataset}/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures -> {out_dir}\n")
    fig_sensitivity_grid(summary, out_dir, dataset_name=args.dataset)
    fig_convergence(summary, out_dir, dataset_name=args.dataset)
    fig_quality_predictor(summary, out_dir, dataset_name=args.dataset)
    print("\nDone.")


if __name__ == "__main__":
    main()
