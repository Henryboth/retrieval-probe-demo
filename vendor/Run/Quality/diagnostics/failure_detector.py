"""
failure_detector.py
====================
Pre-registered experiment (2026-09-20): the naive probe cannot select the winning
paradigm (settled, section 5.4.2). Can it instead detect, on an unseen query, that
retrieval has FAILED -- that no paradigm would return a relevant passage in its top
five -- so that a RAG system can abstain or fall back instead of generating from
irrelevant context? Spec: skripsie-workspace/reviews/2026-09-20_failure_detector_prereg.md
(binding; no definition, control, or criterion here deviates from it).

This script re-uses, without re-implementing, the folds / features of
`run_router_training.py`:
  load_retrieval_results, build_dataset, build_X, make_splitter, split_iter,
  PER_QUERY_FEATURE_NAMES, PARADIGMS.

Outer folds are reproduced exactly as `evaluate_router` (run_router_training.py,
approx. lines 441-470): y = LabelEncoder().fit_transform([r["label"] for r in rows]),
make_splitter(), split_iter(splitter, X, y, rows). The fold assignment depends only on
y (oracle label) and groups (query_id), not on which config's X is passed in, so it is
computed once per corpus (using the D-PQ feature matrix) and re-used for every
detector and both targets.

Targets:
  all_failed   (T2, primary)  : max over {naive, graph, hybrid} of nDCG@5 == 0
  naive_failed (T1, secondary): naive nDCG@5 == 0

Detectors (all XGBClassifier(200, depth 4, lr 0.1, seed 42), trained on training
folds, out-of-fold probabilities on eval folds):
  D-PQ  : the 12 probe features (pq_vec) -- the detector under test
  D-Q   : build_X(rows, "Q-only")          -- query type alone
  D-S   : build_X(rows, "Q+C+Quality")     -- condition-level base rate
  D-36  : build_X(rows, "Q+C+Quality+PQ")  -- the cascade's own input
  D-1   : single feature top1_similarity, no training; ranked directly, sign (feature
          or its negative) chosen by whichever gives AUC > 0.5 on the TRAINING fold,
          then applied unchanged to the eval fold.

Output: Dataset/EvaluationData/RouterEval/failure_detector.json
Run: PYTHONHASHSEED=42 ROUTER_GROUPED=1 /opt/anaconda3/envs/ragBench/bin/python \
     Run/Quality/diagnostics/failure_detector.py
(cwd: RAGRouter-Bench repo root)
"""

import os, sys, json, time
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUALITY_DIR = os.path.dirname(SCRIPT_DIR)          # Run/Quality
sys.path.insert(0, QUALITY_DIR)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

from run_router_training import (
    load_retrieval_results, build_dataset, build_X, make_splitter, split_iter,
    PER_QUERY_FEATURE_NAMES, PARADIGMS,
)

SPEC_FILE = "/Users/henrybothwell/Desktop/skripsie-workspace/reviews/2026-09-20_failure_detector_prereg.md"
DATASETS = ["quality", "musique", "ultraDomain_legal"]
XGB_KW = dict(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
TOP1_IDX = PER_QUERY_FEATURE_NAMES.index("top1_similarity")

DETECTOR_CONFIGS = {
    "D-PQ": None,            # pq_vec directly (handled specially, not via build_X)
    "D-Q":  "Q-only",
    "D-S":  "Q+C+Quality",
    "D-36": "Q+C+Quality+PQ",
}
DETECTOR_ORDER = ["D-PQ", "D-Q", "D-S", "D-36", "D-1"]
TARGET_NAMES = ["all_failed", "naive_failed"]

RESULTS_DIR = Path("Dataset/EvaluationData/RouterEval")
OUT_PATH = RESULTS_DIR / "failure_detector.json"


def rnd(x, digits=3):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        return round(float(x), digits)
    return x


def recall_at_precision(precision, recall, min_precision=0.80):
    """Max recall among PR-curve points with precision >= min_precision; 0 if none."""
    mask = precision >= min_precision
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def precision_at_recall(precision, recall, target_recall=0.50):
    """Precision at the PR point with the smallest recall >= target_recall."""
    mask = recall >= target_recall
    if not np.any(mask):
        return float("nan")
    idx = np.where(mask)[0]
    best = idx[np.argmin(recall[idx])]
    return float(precision[best])


def pr_metrics(y_true, scores):
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auc": auc,
        "ap": ap,
        "recall_at_precision0.80": recall_at_precision(precision, recall),
        "precision_at_recall0.50": precision_at_recall(precision, recall),
    }


def run_corpus(ds):
    print(f"\n{'='*70}\n{ds}\n{'='*70}")
    t0 = time.time()
    results_raw = load_retrieval_results(ds)
    rows = build_dataset(results_raw, ds)
    n = len(rows)
    for r in rows:
        assert all(p in r["paradigm_scores"] for p in PARADIGMS), \
            f"row {r['query_id']} missing a paradigm in paradigm_scores"

    all_failed = np.array(
        [1 if max(r["paradigm_scores"][p]["ndcg@5"] for p in PARADIGMS) == 0.0 else 0 for r in rows],
        dtype=np.int64,
    )
    naive_failed = np.array(
        [1 if r["paradigm_scores"]["naive"]["ndcg@5"] == 0.0 else 0 for r in rows],
        dtype=np.int64,
    )
    targets = {"all_failed": all_failed, "naive_failed": naive_failed}

    query_ids = np.array([r["query_id"] for r in rows])
    labels = [r["label"] for r in rows]
    le = LabelEncoder()
    y_label = le.fit_transform(labels)

    X_pq = np.vstack([r["pq_vec"] for r in rows]).astype(np.float64)
    X_by_detector = {"D-PQ": X_pq}
    for name, cfg in DETECTOR_CONFIGS.items():
        if cfg is not None:
            X_by_detector[name] = build_X(rows, cfg)

    splitter = make_splitter()
    fold_indices = list(split_iter(splitter, X_pq, y_label, rows))
    assert len(fold_indices) == 5

    # verification (1): no query_id appears in both train and eval of any fold
    for fi, (tr, ev) in enumerate(fold_indices):
        tr_q, ev_q = set(query_ids[tr]), set(query_ids[ev])
        overlap = tr_q & ev_q
        assert not overlap, f"{ds} fold {fi}: {len(overlap)} query_id(s) overlap train/eval"

    out = {
        "n_rows": n,
        "fold_sizes": [{"train": int(len(tr)), "eval": int(len(ev))} for tr, ev in fold_indices],
    }

    for tname in TARGET_NAMES:
        y_t = targets[tname]
        prevalence = float(np.mean(y_t))

        pooled_probs = {name: np.zeros(n) for name in X_by_detector}
        per_fold_auc = {name: [] for name in list(X_by_detector.keys()) + ["D-1"]}
        fi_gain_folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            y_train, y_val = y_t[train_idx], y_t[val_idx]

            for name, X in X_by_detector.items():
                clf = XGBClassifier(**XGB_KW)
                clf.fit(X[train_idx], y_train)
                p_val = clf.predict_proba(X[val_idx])[:, 1]
                pooled_probs[name][val_idx] = p_val
                fold_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else float("nan")
                per_fold_auc[name].append(float(fold_auc))
                if name == "D-PQ":
                    fi_gain_folds.append(clf.feature_importances_)

            # D-1: rank by top1_similarity; sign chosen on the training fold
            feat_train = X_pq[train_idx, TOP1_IDX]
            feat_val = X_pq[val_idx, TOP1_IDX]
            auc_pos = roc_auc_score(y_train, feat_train) if len(np.unique(y_train)) > 1 else 0.5
            sign = 1.0 if auc_pos > 0.5 else -1.0
            scored_val = sign * feat_val
            per_fold_auc["D-1"].append(
                float(roc_auc_score(y_val, scored_val)) if len(np.unique(y_val)) > 1 else float("nan")
            )
            if "D-1" not in pooled_probs:
                pooled_probs["D-1"] = np.zeros(n)
            pooled_probs["D-1"][val_idx] = scored_val

        pooled_metrics = {name: pr_metrics(y_t, pooled_probs[name]) for name in pooled_probs}

        diff_q = [per_fold_auc["D-PQ"][i] - per_fold_auc["D-Q"][i] for i in range(5)]
        diff_s = [per_fold_auc["D-PQ"][i] - per_fold_auc["D-S"][i] for i in range(5)]

        mean_fi = np.mean(fi_gain_folds, axis=0)

        bins = np.linspace(0.0, 1.0, 11)
        bin_idx = np.clip(np.digitize(pooled_probs["D-PQ"], bins[1:-1], right=False), 0, 9)
        reliability = []
        for b in range(10):
            mask = bin_idx == b
            cnt = int(np.sum(mask))
            reliability.append({
                "bin": b,
                "mean_p": float(np.mean(pooled_probs["D-PQ"][mask])) if cnt else None,
                "empirical_rate": float(np.mean(y_t[mask])) if cnt else None,
                "count": cnt,
            })

        out[tname] = {
            "prevalence": prevalence,
            "pooled": pooled_metrics,
            "per_fold_auc": per_fold_auc,
            "diff_pq_minus_q_per_fold": diff_q,
            "diff_pq_minus_q_positive_folds": int(sum(d > 0 for d in diff_q)),
            "diff_pq_minus_s_per_fold": diff_s,
            "diff_pq_minus_s_positive_folds": int(sum(d > 0 for d in diff_s)),
            "pq_fold_auc_ge_0.65_folds": int(sum(a >= 0.65 for a in per_fold_auc["D-PQ"])),
            "calibration": {
                "mean_predicted_p": float(np.mean(pooled_probs["D-PQ"])),
                "prevalence": prevalence,
                "reliability_10bin": reliability,
            },
            "pq_feature_importance_gain": {
                name: float(v) for name, v in zip(PER_QUERY_FEATURE_NAMES, mean_fi)
            },
        }

    elapsed = time.time() - t0
    out["_elapsed_s"] = elapsed
    print(f"  {ds}: {n} rows, {elapsed:.1f}s")
    return out


def evaluate_gate(entry):
    """Pre-registered success gate, T2 (all_failed), exactly as written in the spec."""
    pq = entry["pooled"]["D-PQ"]
    q = entry["pooled"]["D-Q"]
    s = entry["pooled"]["D-S"]

    c1 = pq["auc"] >= 0.70
    diff_q_pooled = pq["auc"] - q["auc"]
    diff_s_pooled = pq["auc"] - s["auc"]
    c2 = (
        diff_q_pooled >= 0.05
        and diff_s_pooled >= 0.05
        and entry["diff_pq_minus_q_positive_folds"] >= 4
        and entry["diff_pq_minus_s_positive_folds"] >= 4
    )
    c3 = pq["recall_at_precision0.80"] >= 0.30
    c4 = entry["pq_fold_auc_ge_0.65_folds"] >= 4

    criteria = {
        "1_pooled_auc_d_pq_ge_0.70": {"pass": bool(c1), "value": rnd(pq["auc"])},
        "2_discriminant_validity": {
            "pass": bool(c2),
            "auc_diff_vs_q_pooled": rnd(diff_q_pooled),
            "auc_diff_vs_s_pooled": rnd(diff_s_pooled),
            "diff_vs_q_positive_folds": entry["diff_pq_minus_q_positive_folds"],
            "diff_vs_s_positive_folds": entry["diff_pq_minus_s_positive_folds"],
        },
        "3_recall_at_precision0.80_ge_0.30": {
            "pass": bool(c3), "value": rnd(pq["recall_at_precision0.80"]),
        },
        "4_per_fold_auc_ge_0.65_on_ge4of5": {
            "pass": bool(c4), "folds_meeting": entry["pq_fold_auc_ge_0.65_folds"],
        },
    }
    return {"criteria": criteria, "pass": bool(c1 and c2 and c3 and c4)}


def round_pooled(pooled):
    return {name: {k: rnd(v, 3) for k, v in m.items()} for name, m in pooled.items()}


def round_corpus(entry):
    result = {"n_rows": entry["n_rows"], "fold_sizes": entry["fold_sizes"]}
    for tname in TARGET_NAMES:
        te = entry[tname]
        result[tname] = {
            "prevalence": rnd(te["prevalence"]),
            "pooled": round_pooled(te["pooled"]),
            "per_fold_auc": {name: [rnd(v) for v in vals] for name, vals in te["per_fold_auc"].items()},
            "diff_pq_minus_q_per_fold": [rnd(v) for v in te["diff_pq_minus_q_per_fold"]],
            "diff_pq_minus_q_positive_folds": te["diff_pq_minus_q_positive_folds"],
            "diff_pq_minus_s_per_fold": [rnd(v) for v in te["diff_pq_minus_s_per_fold"]],
            "diff_pq_minus_s_positive_folds": te["diff_pq_minus_s_positive_folds"],
            "pq_fold_auc_ge_0.65_folds": te["pq_fold_auc_ge_0.65_folds"],
            "calibration": {
                "mean_predicted_p": rnd(te["calibration"]["mean_predicted_p"]),
                "prevalence": rnd(te["calibration"]["prevalence"]),
                "reliability_10bin": [
                    {"bin": b["bin"], "mean_p": rnd(b["mean_p"]), "empirical_rate": rnd(b["empirical_rate"]),
                     "count": b["count"]}
                    for b in te["calibration"]["reliability_10bin"]
                ],
            },
            "pq_feature_importance_gain": {k: rnd(v, 4) for k, v in te["pq_feature_importance_gain"].items()},
        }
    return result


def print_table(ds, entry):
    print(f"\n  {ds}  (n={entry['n_rows']})")
    print(f"  prevalence: all_failed={entry['all_failed']['prevalence']:.3f}  "
          f"naive_failed={entry['naive_failed']['prevalence']:.3f}")
    print(f"  {'detector':<8}{'AUC(all_failed)':>16}{'AUC(naive_failed)':>19}"
          f"{'recall@P0.80':>14}{'prec@R0.50':>12}{'AP':>8}")
    for name in DETECTOR_ORDER:
        af = entry["all_failed"]["pooled"][name]
        nf = entry["naive_failed"]["pooled"][name]
        print(f"  {name:<8}{af['auc']:>16.3f}{nf['auc']:>19.3f}"
              f"{af['recall_at_precision0.80']:>14.3f}{af['precision_at_recall0.50']:>12.3f}{af['ap']:>8.3f}")


def main():
    assert os.environ.get("ROUTER_GROUPED") == "1", "ROUTER_GROUPED=1 must be set (grouped CV of record)."
    out = {}
    for ds in DATASETS:
        out[ds] = run_corpus(ds)
        print_table(ds, out[ds])

    gate = {ds: evaluate_gate(out[ds]["all_failed"]) for ds in DATASETS}
    overall_pass = all(g["pass"] for g in gate.values())

    final = {
        "spec": {
            "file": SPEC_FILE,
            "xgb_params": XGB_KW,
            "detectors": {
                "D-PQ": "12 naive-probe features (pq_vec) -- detector under test",
                "D-Q": "build_X(Q-only), 6-dim query type",
                "D-S": "build_X(Q+C+Quality), 24-dim static (corpus + quality)",
                "D-36": "build_X(Q+C+Quality+PQ), the cascade's own input",
                "D-1": "top1_similarity, ranked directly; sign chosen on training fold",
            },
            "targets": {
                "all_failed": "1 if max over {naive, graph, hybrid} of nDCG@5 == 0 (T2, primary)",
                "naive_failed": "1 if naive nDCG@5 == 0 (T1, secondary, not a criterion)",
            },
        },
    }
    for ds in DATASETS:
        final[ds] = round_corpus(out[ds])
    final["success_gate"] = {**gate, "overall_pass": bool(overall_pass)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {OUT_PATH}")

    print("\n" + "=" * 70)
    print("PRE-REGISTERED SUCCESS GATE (T2, all_failed)")
    print("=" * 70)
    for ds in DATASETS:
        print(f"\n{ds}: {'PASS' if gate[ds]['pass'] else 'FAIL'}")
        for cname, cval in gate[ds]["criteria"].items():
            print(f"  {cname}: {'PASS' if cval['pass'] else 'FAIL'}  {cval}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
