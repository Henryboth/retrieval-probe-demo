"""
run_routing_uncertainty.py
==========================
Computes routing uncertainty metrics for each dataset.

Three metrics:

1. Per-query nDCG@5 variance (naive retrieval)
   -- How much does naive retrieval quality vary across queries?
   -- High variance = router has something to learn.

2. Paradigm diversity (oracle label entropy)
   -- How evenly distributed are oracle paradigm choices across queries?
   -- High entropy = no single paradigm dominates = routing matters.

3. Quality degradation sensitivity (from corpus quality assessor)
   -- How steeply does the corpus quality score fall as degradation increases?
   -- High sensitivity = corpus is vulnerable to degradation = routing matters more.
   -- This is the link between the quality measurement framework and cascade routing.

4. Routing opportunity score (composite)
   -- Geometric mean of all three normalised components.
   -- Single number summarising how much routing can help.
   -- Connects quality measurement → routing uncertainty → cascade routing gain.

These metrics are then correlated against:
   - In-domain per_query_feature_gain
   - Cross-dataset transfer gain (as receiver)

Output: Dataset/EvaluationData/RouterEval/routing_uncertainty.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score
from Run.Quality.run_router_training import load_retrieval_results, build_dataset

DATASETS = ["quality", "musique", "ultraDomain_legal"]


def probe_discriminability(rows: list) -> float:
    """
    AUC of the naive probe's own nDCG@5 score predicting whether escalation
    away from naive was needed (oracle label != naive). This is distinct
    from paradigm_entropy: entropy asks whether escalation opportunities
    exist at all, this asks whether the cheap, pre-router signal the cascade
    actually has access to (the probe's own score) can find them. AUC=0.5
    means no better than chance; capped below at 0 after rescaling.
    """
    naive_scores = np.array([
        r["paradigm_scores"].get("naive", {}).get("ndcg@5", 0.0) for r in rows
    ])
    y = np.array([0 if r["label"] == "naive" else 1 for r in rows])
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5  # degenerate: no variation in outcome to discriminate
    return float(roc_auc_score(y, -naive_scores))


def paradigm_entropy(rows: list) -> float:
    """Shannon entropy of oracle paradigm label distribution. Range [0, log(n_classes)]."""
    labels = [r["label"] for r in rows]
    counts = Counter(labels)
    total = len(labels)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    return float(entropy)


def naive_ndcg_variance(rows: list) -> float:
    """Variance of naive retrieval nDCG@5 across all queries."""
    naive_scores = [
        r["paradigm_scores"].get("naive", {}).get("ndcg@5", 0.0)
        for r in rows
    ]
    return float(np.var(naive_scores))


def naive_ndcg_mean(rows: list) -> float:
    naive_scores = [
        r["paradigm_scores"].get("naive", {}).get("ndcg@5", 0.0)
        for r in rows
    ]
    return float(np.mean(naive_scores))


def paradigm_win_rates(rows: list) -> dict:
    labels = [r["label"] for r in rows]
    counts = Counter(labels)
    total = len(labels)
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def quality_degradation_sensitivity(dataset_name: str) -> float:
    """
    Measures how steeply corpus quality drops as degradation rate increases.
    Computed as the mean absolute slope of quality score vs degradation rate
    across all 5 degradation types.

    This is the bridge between the quality measurement framework and cascade
    routing — corpora that degrade quickly produce more varied routing conditions.
    """
    base = Path(f"Dataset/EvaluationData/RetrievalEval/{dataset_name}")
    slopes = []
    degradation_types = ["text_truncation", "entity_variation", "graph_edge_removal",
                         "document_dropout", "combined"]
    for deg in degradation_types:
        rates, qualities = [], []
        for rate_int in range(0, 80, 10):
            fname = base / f"{deg}_{rate_int:03d}.json"
            if fname.exists():
                d = json.load(open(fname))
                q = d.get("quality_features", {}).get("overall", None)
                if q is not None:
                    rates.append(rate_int / 100)
                    qualities.append(q)
        if len(rates) >= 2:
            # Linear regression slope
            slope = np.polyfit(rates, qualities, 1)[0]
            slopes.append(abs(slope))
    return float(np.mean(slopes)) if slopes else 0.0


def compute_uncertainty(dataset_name: str) -> dict:
    print(f"\n  {dataset_name}")
    results_raw = load_retrieval_results(dataset_name)
    rows = build_dataset(results_raw)

    if not rows:
        print("    No data.")
        return {}

    var  = naive_ndcg_variance(rows)
    mean = naive_ndcg_mean(rows)
    ent  = paradigm_entropy(rows)
    wins = paradigm_win_rates(rows)
    qdeg = quality_degradation_sensitivity(dataset_name)
    auc  = probe_discriminability(rows)

    # Max entropy for 3 classes = log(3) = 1.0986
    max_entropy = np.log(3)
    norm_entropy   = ent / max_entropy
    # Normalise variance by mean^2 (coefficient of variation squared)
    norm_variance  = var / (mean ** 2) if mean > 0 else 0.0
    norm_variance  = min(norm_variance, 1.0)
    # Normalise quality sensitivity — typical slope range ~0.1 to 0.5
    norm_qdeg = min(qdeg / 0.5, 1.0)
    # Rescale AUC's meaningful range [0.5, 1.0] onto [0, 1]; at-or-below-chance -> 0
    norm_discriminability = max(0.0, 2 * (auc - 0.5))

    # Routing opportunity = geometric mean of all four components
    routing_opportunity = float(
        (norm_variance * norm_entropy * norm_qdeg * norm_discriminability) ** (1/4)
    )
    # Three-component routing opportunity (Omega_3) = geometric mean of variance,
    # entropy and quality sensitivity only, excluding probe discriminability.
    # Kept alongside routing_opportunity for comparison, not as a replacement.
    routing_opportunity_3c = float(
        (norm_variance * norm_entropy * norm_qdeg) ** (1/3)
    )

    print(f"    Instances           : {len(rows)}")
    print(f"    Naive nDCG mean     : {mean:.4f}")
    print(f"    Naive nDCG var      : {var:.4f}  (norm: {norm_variance:.4f})")
    print(f"    Paradigm entropy    : {ent:.4f}  (norm: {norm_entropy:.4f})")
    print(f"    Quality sensitivity : {qdeg:.4f}  (norm: {norm_qdeg:.4f})")
    print(f"    Probe discriminability (AUC): {auc:.4f}  (norm: {norm_discriminability:.4f})")
    print(f"    Routing opportunity : {routing_opportunity:.4f}")
    print(f"    Routing opportunity (3c): {routing_opportunity_3c:.4f}")
    print(f"    Paradigm win rates  : {wins}")

    return {
        "dataset":                    dataset_name,
        "n_instances":                len(rows),
        "naive_ndcg_mean":            round(mean, 4),
        "naive_ndcg_variance":        round(var, 4),
        "norm_variance":              round(norm_variance, 4),
        "paradigm_entropy":           round(ent, 4),
        "norm_entropy":               round(norm_entropy, 4),
        "quality_degradation_slope":  round(qdeg, 4),
        "norm_quality_sensitivity":   round(norm_qdeg, 4),
        "probe_discriminability_auc": round(auc, 4),
        "norm_discriminability":      round(norm_discriminability, 4),
        "routing_opportunity":        round(routing_opportunity, 4),
        "routing_opportunity_3c":     round(routing_opportunity_3c, 4),
        "paradigm_win_rates":         wins,
    }


def load_pq_gain(dataset_name: str) -> float:
    path = Path(f"Dataset/EvaluationData/RouterEval/{dataset_name}/oracle_gap_decomposition.json")
    if path.exists():
        d = json.load(open(path))
        return d.get("per_query_feature_gain", 0.0)
    return None


def load_gain_set(dataset_name: str) -> dict:
    """All realised-gain variants (grouped split) for the Ω correlation table:
    classifier PQ tier, classifier total (Full − Q-only), expected-value RA selector over
    Q-only and over the best static tier, plus the oracle gap."""
    path = Path(f"Dataset/EvaluationData/RouterEval/{dataset_name}/oracle_gap_decomposition.json")
    if not path.exists():
        return {}
    d = json.load(open(path)); ev = d.get("expected_value", {})
    return {
        "oracle_gap":                d.get("total_gap"),
        "pq_gain_classifier":        d.get("per_query_feature_gain"),
        "total_gain_classifier_pq":  d.get("total_router_gain_pq"),
        "static_tier_gain":          d.get("corpus_feature_gain"),
        "ra_gain_over_qonly_ev":     ev.get("total_router_gain_ra"),
        "ra_gain_over_static_ev":    (ev.get("Q+C+Quality+RA_correct_rate") - ev.get("Q+C+Quality_correct_rate"))
                                     if ev.get("Q+C+Quality+RA_correct_rate") is not None else None,
        "ra_gain_over_best_fixed_ev": _ra_gain_over_best_fixed(dataset_name, ev),
    }


def _ra_gain_over_best_fixed(dataset_name: str, ev: dict):
    """Selector (RA, expected-value) CR minus the best fixed paradigm's pooled CR over the 40 conditions."""
    import glob
    ra = ev.get("Q+C+Quality+RA_correct_rate")
    if ra is None:
        return None
    tot = {"naive": [], "graph": [], "hybrid": []}
    for f in glob.glob(f"Dataset/EvaluationData/RetrievalEval/{dataset_name}/*_[0-9][0-9][0-9].json"):
        # Phase 4 step 2 added graph_probe_{operator}_{rate:03d}.json files
        # (Run/Quality/compute_graph_probe.py), which match this same glob but
        # have no "paradigms" key -- skip them.
        if Path(f).name.startswith("graph_probe_"):
            continue
        d = json.load(open(f))
        for p in tot:
            tot[p] += [q["metrics"]["ndcg@5"] for q in d["paradigms"].get(p, {}).get("per_query", [])]
    best = max(float(np.mean(v)) for v in tot.values() if v)
    return round(ra - best, 4)


def load_cross_gains(dataset_name: str) -> dict:
    """Load PQ gains when this dataset is the TEST set (receiving transfer)."""
    gains = {}
    for train in DATASETS:
        if train == dataset_name:
            continue
        path = Path(f"Dataset/EvaluationData/RouterEval/cross_{train}_to_{dataset_name}/oracle_gap_decomposition.json")
        if path.exists():
            d = json.load(open(path))
            gains[train] = d.get("per_query_feature_gain", 0.0)
    return gains


def omega_from_rows(rows: list, dataset_name: str) -> float:
    """Compute Omega from an arbitrary row subset, for the sample-size
    stability check. Reuses the fitted quality-degradation sensitivity for
    the full corpus, since that is a property of the corpus's degradation
    behaviour, not of which queries happen to be sampled."""
    var, mean = naive_ndcg_variance(rows), naive_ndcg_mean(rows)
    norm_variance = min(var / mean ** 2, 1.0) if mean > 0 else 0.0
    norm_entropy = paradigm_entropy(rows) / np.log(3)
    norm_qdeg = min(quality_degradation_sensitivity(dataset_name) / 0.5, 1.0)
    norm_disc = max(0.0, 2 * (probe_discriminability(rows) - 0.5))
    return float((norm_variance * norm_entropy * norm_qdeg * norm_disc) ** (1/4))


def sample_size_stability(n_trials: int = 20, seed: int = 42) -> dict:
    """
    Checks how stable Omega is when computed from a random query subsample
    instead of the full instance set, at several candidate pilot sizes.
    This is the empirical basis for claiming Omega is usable as a cheap
    pre-deployment check: it quantifies how many queries (run across all
    three paradigms) are actually needed for a stable estimate, rather than
    asserting a sample-size claim without testing it.
    """
    rng = np.random.default_rng(seed)
    sizes = [200, 500, 1000, 2000]
    results = {}
    for ds in DATASETS:
        rows = build_dataset(load_retrieval_results(ds))
        full_omega = omega_from_rows(rows, ds)
        per_size = {}
        for n in sizes:
            n_eff = min(n, len(rows))
            vals = []
            for _ in range(n_trials):
                idx = rng.choice(len(rows), size=n_eff, replace=False)
                sub = [rows[i] for i in idx]
                vals.append(omega_from_rows(sub, ds))
            per_size[str(n)] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
            }
        results[ds] = {"full_data_omega": round(full_omega, 4), "by_sample_size": per_size}
        print(f"  {ds}: full-data Omega={full_omega:.4f}")
        for n in sizes:
            p = per_size[str(n)]
            print(f"    n={n:<5d} mean={p['mean']:.4f}  std={p['std']:.4f}")
    out_path = Path("Dataset/EvaluationData/RouterEval/omega_sample_size_stability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size-check", action="store_true",
                        help="Run the Omega sample-size stability check instead "
                             "of the main routing uncertainty analysis.")
    args = parser.parse_args()

    if args.sample_size_check:
        print(f"\n{'#'*65}")
        print(f"  Omega Sample-Size Stability Check")
        print(f"{'#'*65}")
        sample_size_stability()
        return

    print(f"\n{'#'*65}")
    print(f"  Routing Uncertainty Analysis")
    print(f"{'#'*65}")

    results = {}
    for ds in DATASETS:
        r = compute_uncertainty(ds)
        if r:
            results[ds] = r

    # Add in-domain PQ gain and cross-dataset transfer gains
    print(f"\n  {'Dataset':<25} {'Opportunity':>12} {'In-domain PQ gain':>18} {'Received transfer (mean)':>25}")
    print("  " + "-" * 85)

    summary = {}
    for ds, r in results.items():
        indomain = load_pq_gain(ds)
        cross = load_cross_gains(ds)
        mean_received = float(np.mean(list(cross.values()))) if cross else 0.0
        r["indomain_pq_gain"]       = indomain
        r["received_transfer"]      = cross
        r["mean_received_transfer"] = round(mean_received, 4)
        summary[ds] = r
        print(f"  {ds:<25} {r['routing_opportunity']:>12.4f} {indomain:>18.4f} {mean_received:>25.4f}")

    # Correlation analysis
    print(f"\n  Correlation: routing_opportunity vs in-domain PQ gain")
    opps   = [summary[ds]["routing_opportunity"] for ds in DATASETS if ds in summary]
    gains  = [summary[ds]["indomain_pq_gain"] for ds in DATASETS if ds in summary and summary[ds]["indomain_pq_gain"] is not None]
    if len(opps) >= 3:
        corr = np.corrcoef(opps, gains)[0, 1]
        print(f"    Pearson r = {corr:.4f}")
        summary["correlation_opportunity_vs_indomain_gain"] = round(float(corr), 4)
    # Correlations of Ω with every realised-gain variant (n = 3 corpora: characterisation only)
    gain_sets = {ds: load_gain_set(ds) for ds in DATASETS if ds in summary}
    corr_table = {}
    for key in ["oracle_gap", "pq_gain_classifier", "total_gain_classifier_pq", "static_tier_gain",
                "ra_gain_over_qonly_ev", "ra_gain_over_static_ev", "ra_gain_over_best_fixed_ev"]:
        vals = [gain_sets[ds].get(key) for ds in DATASETS if ds in gain_sets]
        if len(vals) == 3 and all(v is not None for v in vals) and np.std(vals) > 0:
            corr_table[key] = {"values": [round(float(v), 4) for v in vals],
                               "pearson_r_with_omega": round(float(np.corrcoef(opps, vals)[0, 1]), 4),
                               "omega_rank_matches": bool(list(np.argsort(opps)) == list(np.argsort(vals)))}
    summary["omega_correlation_table"] = {"datasets_order": [ds for ds in DATASETS if ds in gain_sets],
                                          "omega": [round(float(o), 4) for o in opps], **corr_table}
    print("\n  Ω vs realised gains (grouped split):")
    for k, v in corr_table.items():
        print(f"    {k:28s} r = {v['pearson_r_with_omega']:+.2f}  rank match = {v['omega_rank_matches']}  {v['values']}")

    print(f"\n  Interpretation:")
    for ds, r in summary.items():
        if not isinstance(r, dict) or "routing_opportunity" not in r:
            continue
        opp = r["routing_opportunity"]
        if opp > 0.3:
            level = "HIGH — cascade routing adds significant value"
        elif opp > 0.15:
            level = "MEDIUM — cascade routing adds moderate value"
        else:
            level = "LOW — corpus too uniform for routing to help"
        print(f"    {ds:<25}: {level}")

    # Save
    out_path = Path("Dataset/EvaluationData/RouterEval/routing_uncertainty.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
