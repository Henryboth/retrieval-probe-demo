"""
analyze_results.py
==================
Full analysis of degradation sensitivity results from run_retrieval_eval.py.

Steps
-----
1. Baseline consistency check  -- all r=0.0 naive values should match
2. Sensitivity curves          -- recall@5 vs rate per degradation type
3. Quality as predictor        -- scatter: overall_quality vs recall@5
4. Crossover analysis          -- where does the winning paradigm flip?
5. Per-query-type breakdown    -- factoid vs multi-hop vs abstractive

Usage
-----
python Run/Quality/analyze_results.py --dataset quality
python Run/Quality/analyze_results.py --dataset quality --metric ndcg@5
python Run/Quality/analyze_results.py --dataset quality --no-plots
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import argparse
from pathlib import Path
from collections import defaultdict

PARADIGMS   = ["naive", "graph", "hybrid"]
PARADIGM_COLORS = {"naive": "#2196F3", "graph": "#FF5722", "hybrid": "#4CAF50"}
DEG_ORDER   = ["text_truncation", "entity_variation", "graph_edge_removal",
               "document_dropout", "combined"]
DEG_LABELS  = {
    "text_truncation":    "Text Truncation",
    "entity_variation":   "Entity Variation",
    "graph_edge_removal": "Graph Edge Removal",
    "document_dropout":   "Document Dropout",
    "combined":           "Combined",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_summary(dataset_name: str) -> dict:
    path = Path(f"Dataset/EvaluationData/RetrievalEval/{dataset_name}/summary.json")
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}\n"
                                f"Run run_retrieval_eval.py first.")
    with open(path) as f:
        return json.load(f)


def load_per_query(dataset_name: str) -> dict:
    """Load all per-query result files into a nested dict."""
    base = Path(f"Dataset/EvaluationData/RetrievalEval/{dataset_name}")
    per_query = {}
    for fpath in sorted(base.glob("*.json")):
        if fpath.name == "summary.json":
            continue
        with open(fpath) as f:
            data = json.load(f)
        deg  = data.get("degradation_type", fpath.stem.rsplit("_", 1)[0])
        rate = str(data.get("rate", 0.0))
        per_query.setdefault(deg, {})[rate] = data
    return per_query


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Baseline consistency
# ─────────────────────────────────────────────────────────────────────────────

def check_baselines(summary: dict, metric: str) -> None:
    print("\n" + "=" * 60)
    print("  STEP 1 — Baseline consistency (r=0.0)")
    print("=" * 60)
    print(f"  {'Degradation':<24} {'naive':>7}  {'graph':>7}  {'hybrid':>7}  {'status'}")
    print("  " + "-" * 56)

    baselines = {}
    for deg in DEG_ORDER:
        if deg not in summary:
            continue
        p = summary[deg].get("0.0", {}).get("paradigms", {})
        vals = {par: p.get(par, {}).get(metric, 0.0) for par in PARADIGMS}
        baselines[deg] = vals
        status = "OK" if abs(vals.get("naive", 0) - list(baselines.values())[0].get("naive", 0)) < 0.05 else "MISMATCH"
        print(f"  {DEG_LABELS.get(deg, deg):<24} "
              f"{vals.get('naive', 0):>7.3f}  "
              f"{vals.get('graph', 0):>7.3f}  "
              f"{vals.get('hybrid', 0):>7.3f}  {status}")

    naive_vals = [v.get("naive", 0) for v in baselines.values()]
    spread = max(naive_vals) - min(naive_vals) if naive_vals else 0
    print(f"\n  Naive baseline spread: {spread:.3f}  "
          f"{'[OK — baselines consistent]' if spread < 0.05 else '[WARNING — baselines differ, re-check indexing]'}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Sensitivity curves
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_table(summary: dict, metric: str) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP 2 — Sensitivity ({metric})")
    print("=" * 60)

    for deg in DEG_ORDER:
        if deg not in summary:
            continue
        print(f"\n  {DEG_LABELS.get(deg, deg)}")
        print(f"  {'Rate':>5}  {'naive':>7}  {'graph':>7}  {'hybrid':>7}  {'quality':>8}  winner")
        print("  " + "-" * 52)
        for rate_str, info in sorted(summary[deg].items(), key=lambda x: float(x[0])):
            p = info.get("paradigms", {})
            n = p.get("naive",  {}).get(metric, 0.0)
            g = p.get("graph",  {}).get(metric, 0.0)
            h = p.get("hybrid", {}).get(metric, 0.0)
            q = info.get("overall_quality", 0.0)
            winner = max([(n,"naive"),(g,"graph"),(h,"hybrid")], key=lambda x: x[0])[1]
            print(f"  {float(rate_str):>5.1f}  {n:>7.3f}  {g:>7.3f}  {h:>7.3f}  {q:>8.3f}  {winner}")


def plot_sensitivity(summary: dict, metric: str, dataset_name: str,
                     out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    active = [d for d in DEG_ORDER if d in summary]
    ncols = 3
    nrows = -(-len(active) // ncols)   # ceiling division

    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)

    for idx, deg in enumerate(active):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        rates = sorted(summary[deg].keys(), key=float)
        for par in PARADIGMS:
            vals = [summary[deg][r]["paradigms"].get(par, {}).get(metric, 0.0)
                    for r in rates]
            if any(v > 0 for v in vals):
                ax.plot([float(r) for r in rates], vals,
                        marker="o", label=par, color=PARADIGM_COLORS[par],
                        linewidth=2, markersize=5)
        ax.set_title(DEG_LABELS.get(deg, deg), fontsize=11, fontweight="bold")
        ax.set_xlabel("Degradation rate", fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset_name}  —  Sensitivity curves ({metric})",
                 fontsize=13, fontweight="bold", y=1.01)
    out = out_dir / f"sensitivity_{metric.replace('@','_at_')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Quality score as predictor
# ─────────────────────────────────────────────────────────────────────────────

def quality_predictor_table(summary: dict, metric: str) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP 3 — Quality score as predictor ({metric})")
    print("=" * 60)

    # Collect all (quality, naive, graph, hybrid) points across all conditions
    points = []
    for deg, rates in summary.items():
        for rate_str, info in rates.items():
            q = info.get("overall_quality", 0.0)
            p = info.get("paradigms", {})
            n = p.get("naive",  {}).get(metric, 0.0)
            g = p.get("graph",  {}).get(metric, 0.0)
            h = p.get("hybrid", {}).get(metric, 0.0)
            points.append((float(rate_str), deg, q, n, g, h))

    # Bin by quality quartile
    points.sort(key=lambda x: x[2])
    n_pts  = len(points)
    q_size = max(1, n_pts // 4)

    print(f"\n  {'Quality range':<18}  {'naive':>7}  {'graph':>7}  {'hybrid':>7}  best")
    print("  " + "-" * 52)
    for i in range(0, n_pts, q_size):
        bucket = points[i:i + q_size]
        q_lo   = bucket[0][2]
        q_hi   = bucket[-1][2]
        avg_n  = sum(p[3] for p in bucket) / len(bucket)
        avg_g  = sum(p[4] for p in bucket) / len(bucket)
        avg_h  = sum(p[5] for p in bucket) / len(bucket)
        best   = max([("naive", avg_n), ("graph", avg_g), ("hybrid", avg_h)],
                     key=lambda x: x[1])[0]
        print(f"  [{q_lo:.3f} – {q_hi:.3f}]   {avg_n:>7.3f}  {avg_g:>7.3f}  {avg_h:>7.3f}  {best}")

    # Pearson correlation: quality vs naive/graph recall
    try:
        import numpy as np
        qs  = [p[2] for p in points]
        ns  = [p[3] for p in points]
        gs  = [p[4] for p in points]
        r_n = np.corrcoef(qs, ns)[0, 1]
        r_g = np.corrcoef(qs, gs)[0, 1]
        print(f"\n  Pearson r(quality, naive-{metric}):  {r_n:+.3f}")
        print(f"  Pearson r(quality, graph-{metric}):  {r_g:+.3f}")
        print(f"\n  Interpretation: {'quality score tracks naive recall well' if abs(r_n) > 0.5 else 'quality score is a weak predictor for naive'}")
    except Exception:
        pass


def plot_quality_predictor(summary: dict, metric: str, dataset_name: str,
                           out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    points = []
    for deg, rates in summary.items():
        for rate_str, info in rates.items():
            q = info.get("overall_quality", 0.0)
            p = info.get("paradigms", {})
            for par in PARADIGMS:
                v = p.get(par, {}).get(metric, 0.0)
                if v > 0:
                    points.append((q, v, par, deg))

    fig, ax = plt.subplots(figsize=(7, 5))
    for par in PARADIGMS:
        xs = [p[0] for p in points if p[2] == par]
        ys = [p[1] for p in points if p[2] == par]
        if xs:
            ax.scatter(xs, ys, label=par, color=PARADIGM_COLORS[par],
                       alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
            # Trend line
            z = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs), max(xs), 100)
            ax.plot(xline, np.poly1d(z)(xline),
                    color=PARADIGM_COLORS[par], linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("Overall quality score", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{dataset_name}  —  Quality as retrieval predictor ({metric})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = out_dir / f"quality_predictor_{metric.replace('@','_at_')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Crossover analysis
# ─────────────────────────────────────────────────────────────────────────────

def crossover_analysis(summary: dict, metric: str) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP 4 — Crossover analysis ({metric})")
    print("=" * 60)
    print("  (Conditions where the winning paradigm changes vs r=0.0 baseline)\n")

    crossovers_found = False
    for deg in DEG_ORDER:
        if deg not in summary:
            continue
        rates = sorted(summary[deg].keys(), key=float)
        if not rates:
            continue

        baseline_p = summary[deg][rates[0]]["paradigms"]
        baseline_winner = max(
            [(par, baseline_p.get(par, {}).get(metric, 0.0)) for par in PARADIGMS],
            key=lambda x: x[1]
        )[0]

        for rate_str in rates[1:]:
            p = summary[deg][rate_str]["paradigms"]
            winner = max(
                [(par, p.get(par, {}).get(metric, 0.0)) for par in PARADIGMS],
                key=lambda x: x[1]
            )[0]
            if winner != baseline_winner:
                n = p.get("naive",  {}).get(metric, 0.0)
                g = p.get("graph",  {}).get(metric, 0.0)
                h = p.get("hybrid", {}).get(metric, 0.0)
                q = summary[deg][rate_str].get("overall_quality", 0.0)
                print(f"  CROSSOVER  {DEG_LABELS.get(deg, deg):<24}  "
                      f"r={float(rate_str):.1f}  "
                      f"{baseline_winner} -> {winner}  "
                      f"(naive={n:.3f} graph={g:.3f} hybrid={h:.3f} quality={q:.3f})")
                crossovers_found = True

    if not crossovers_found:
        print("  No crossovers found — one paradigm dominates across all conditions.")
        print("  This is itself a finding: routing may not help for this dataset/metric.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Per-query-type breakdown
# ─────────────────────────────────────────────────────────────────────────────

def query_type_breakdown(per_query: dict, metric: str) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP 5 — Per-query-type breakdown ({metric}, r=0.0 baseline)")
    print("=" * 60)

    # Collect r=0.0 results across all degradation types (they should agree)
    type_scores = defaultdict(lambda: defaultdict(list))

    for deg, rates in per_query.items():
        rate_data = rates.get("0.0", rates.get("0", {}))
        for par, par_data in rate_data.get("paradigms", {}).items():
            for qr in par_data.get("per_query", []):
                qt = qr.get("query_type", "unknown")
                v  = qr.get("metrics", {}).get(metric, 0.0)
                type_scores[qt][par].append(v)

    if not type_scores:
        print("  No per-query data found (per-query results may not be in summary.json).")
        print("  Re-run with full results files present.")
        return

    print(f"\n  {'Query type':<20}  {'n':>5}  {'naive':>7}  {'graph':>7}  {'hybrid':>7}  best")
    print("  " + "-" * 58)
    for qt in sorted(type_scores.keys()):
        scores = type_scores[qt]
        n_q    = max(len(v) for v in scores.values()) if scores else 0
        avgs   = {par: sum(v)/len(v) for par, v in scores.items() if v}
        best   = max(avgs.items(), key=lambda x: x[1])[0] if avgs else "—"
        print(f"  {qt:<20}  {n_q:>5}  "
              f"{avgs.get('naive',  0):>7.3f}  "
              f"{avgs.get('graph',  0):>7.3f}  "
              f"{avgs.get('hybrid', 0):>7.3f}  {best}")


def plot_query_types(per_query: dict, metric: str, dataset_name: str,
                     out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    type_scores = defaultdict(lambda: defaultdict(list))
    for deg, rates in per_query.items():
        rate_data = rates.get("0.0", rates.get("0", {}))
        for par, par_data in rate_data.get("paradigms", {}).items():
            for qr in par_data.get("per_query", []):
                qt = qr.get("query_type", "unknown")
                v  = qr.get("metrics", {}).get(metric, 0.0)
                type_scores[qt][par].append(v)

    if not type_scores:
        return

    qtypes = sorted(type_scores.keys())
    x      = np.arange(len(qtypes))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(qtypes) * 1.8), 5))
    for i, par in enumerate(PARADIGMS):
        vals = [sum(type_scores[qt].get(par, [0])) /
                max(len(type_scores[qt].get(par, [0])), 1)
                for qt in qtypes]
        ax.bar(x + i * width, vals, width, label=par,
               color=PARADIGM_COLORS[par], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(qtypes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset_name}  —  Per-query-type breakdown ({metric}, r=0.0)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = out_dir / f"query_types_{metric.replace('@','_at_')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze retrieval evaluation results")
    parser.add_argument("--dataset",  default="quality",
                        help="Dataset name (must match a completed run)")
    parser.add_argument("--metric",   default="recall@5",
                        choices=["recall@1", "recall@5", "recall@10",
                                 "mrr", "ndcg@5", "ndcg@10", "hit@5"],
                        help="Primary metric to analyse (default: recall@5)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib figures (text output only)")
    args = parser.parse_args()

    summary   = load_summary(args.dataset)
    per_query = load_per_query(args.dataset)

    out_dir = Path(f"Dataset/EvaluationData/RetrievalEval/{args.dataset}/figures")

    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Figures will be saved to: {out_dir}")
        except ImportError:
            print("  matplotlib not found — running in --no-plots mode.")
            args.no_plots = True

    # ── Run all five analysis steps ──────────────────────────────────────────
    check_baselines(summary, args.metric)
    sensitivity_table(summary, args.metric)
    quality_predictor_table(summary, args.metric)
    crossover_analysis(summary, args.metric)
    query_type_breakdown(per_query, args.metric)

    if not args.no_plots:
        print("\n" + "=" * 60)
        print("  Saving figures...")
        print("=" * 60)
        plot_sensitivity(summary, args.metric, args.dataset, out_dir)
        plot_quality_predictor(summary, args.metric, args.dataset, out_dir)
        plot_query_types(per_query, args.metric, args.dataset, out_dir)

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
