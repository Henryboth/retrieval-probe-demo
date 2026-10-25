"""
run_router_training.py
======================
Trains and evaluates the quality-aware router on the retrieval evaluation
results produced by run_retrieval_eval.py.

Three router configurations are compared, following the protocol of
Wang et al. (2026), plus a fourth (Q+C+Quality+PQ) and two H1 confound
controls (Q+QD, Q+C+Quality+PQ-QD) that isolate how much of the per-query
probe gain comes from the two query-only-derived features
(query_length_norm, query_entity_count) versus the ten genuine
probe-outcome features:

  Q-only            : query type features only  (3-dim one-hot)
  Q+C               : query + corpus features   (3 + 6 = 9-dim)  [RAGRouter-Bench baseline]
  Q+C+Quality       : query + corpus + quality  (3 + 6 + 12 = 21-dim) [thesis contribution]
  Q+C+Quality+PQ    : + naive per-query probe features (12-dim)  [thesis contribution]
  Q+QD              : query type + query-difficulty features only (2-dim)  [H1 control]
  Q+C+Quality+PQ-QD : Q+C+Quality + the ten probe-outcome features, excluding
                       the two query-only features  [H1 control]

Two architectures are trained for each configuration:
  XGBoost  : gradient-boosted trees (also used for feature importance analysis)
  MLP      : multi-layer perceptron

Evaluation: 5-fold stratified cross-validation.

Metrics reported:
  routing_accuracy  : rate of selecting the best-performing paradigm
  macro_f1          : balanced classification performance across paradigm classes
  oracle_gap        : oracle_accuracy - router_accuracy
  quality_gain      : accuracy improvement of Q+C+Quality over Q+C

Output
------
Dataset/EvaluationData/RouterEval/results.json
Dataset/EvaluationData/RouterEval/feature_importance.json
Dataset/EvaluationData/RouterEval/oracle_gap_decomposition.json

Usage
-----
python Run/Quality/run_router_training.py --dataset quality
python Run/Quality/run_router_training.py --dataset all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import os
# Leakage control (2026-09-17): ROUTER_GROUPED=1 groups folds by base query id so that no
# query appears in both training and validation under different degradation conditions.
# ROUTER_OUT_ROOT redirects outputs (default RouterEval) so a check run never overwrites live data.
ROUTER_GROUPED  = os.environ.get('ROUTER_GROUPED', '1') == '1'   # default: grouped by query id (leakage control)
ROUTER_OUT_ROOT = os.environ.get('ROUTER_OUT_ROOT', 'RouterEval')

def make_splitter():
    if ROUTER_GROUPED:
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def split_iter(splitter, X, y, rows):
    if ROUTER_GROUPED:
        groups = [r['query_id'] for r in rows]
        return splitter.split(X, y, groups)
    return splitter.split(X, y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

QUERY_TYPES   = ["factual", "reasoning", "summary", "multi_hop", "single_hop", "unknown"]
PARADIGMS     = ["naive", "graph", "hybrid"]
RA_DIFF_FEATURES = ["semantic_coverage", "top1_similarity", "score_mean", "entity_coverage"]
EV_MARGIN_GRID = np.round(np.arange(0.0, 0.31, 0.01), 2)
CORPUS_FEATURES_DIM = 6   # LCC, avg_degree, clustering, intrinsic_dim, dispersion, hubness
QUALITY_FEATURES_DIM = 12  # from QualityAssessor

# Corpus-level structural/semantic features from RAGRouter-Bench
# These are computed once per (dataset, degradation_condition) from the graph
# and embedding space. For the baseline Q+C router we use the CLEAN corpus
# features (as in Wang et al. 2026). For Q+C+Quality we add the quality vector.
CORPUS_FEATURE_NAMES = [
    "lcc_ratio", "avg_degree_norm", "clustering_coeff",
    "intrinsic_dim_norm", "dispersion_norm", "hubness_norm",
]

QUALITY_FEATURE_NAMES = [
    "text_completeness",          "avg_length_score",      "corpus_retention_rate",
    "entity_surface_consistency", "relation_diversity",    "no_duplicate_doc_rate",
    "graph_node_completeness",    "graph_lcc_ratio",       "graph_density_score",
    "vocab_richness",             "length_uniformity",     "content_density",
]

# Per-query features from naive RAG probe (12-dim)
# The router uses naive's retrieval quality as input to decide escalation.
PER_QUERY_FEATURE_NAMES = [
    "chunk_completeness",
    "n_retrieved_ratio",
    "score_mean",
    "semantic_coverage",
    "semantic_variance",
    "top1_similarity",
    "entity_coverage",
    "entity_density",
    "score_gap",
    "score_concentration",
    "query_length_norm",
    "query_entity_count",
]

# Graph-side probe (GP, phase 4 step 2): 8-dim, PPR-free seeding-only signal
# from Run/Quality/compute_graph_probe.py. Zeros if the graph_probe_*.json
# file for the record's condition is missing.
GP_FEATURE_NAMES = [
    "gp_seed_sim_top1",
    "gp_seed_sim_mean",
    "gp_seed_degree_mean",
    "gp_seed_degree_max",
    "gp_seed_edge_share",
    "gp_seed_components",
    "gp_seed_docs",
    "gp_seed_lcc_share",
]
GP_FEATURES_DIM = len(GP_FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and feature construction
# ─────────────────────────────────────────────────────────────────────────────

def load_retrieval_results(dataset_name: str) -> list:
    """Load all retrieval evaluation results for a dataset."""
    base = Path(f"Dataset/EvaluationData/RetrievalEval/{dataset_name}")
    if not base.exists():
        raise FileNotFoundError(f"No retrieval results found at {base}. "
                                f"Run run_retrieval_eval.py first.")
    records = []
    for result_file in sorted(base.glob("*.json")):
        if result_file.name == "summary.json":
            continue
        with open(result_file) as f:
            records.append(json.load(f))
    print(f"  Loaded {len(records)} result files for {dataset_name}")
    return records


def compute_corpus_features_from_quality(quality_dict: dict) -> np.ndarray:
    """
    Approximate the RAGRouter-Bench corpus features from the quality feature
    dict. In the full RAGRouter-Bench setup, these 6 features are computed
    on the CLEAN corpus only. We use the graph quality sub-features as a
    proxy for structural features, and text quality features for semantic ones.

    Mapping:
      lcc_ratio          <- graph_lcc_ratio
      avg_degree_norm    <- graph_density_score
      clustering_coeff   <- graph_node_completeness
      intrinsic_dim_norm <- 1 - vocab_richness  (higher dim = less uniform)
      dispersion_norm    <- length_uniformity
      hubness_norm       <- 1 - no_duplicate_doc_rate  (duplicates ~ hubness)
    """
    feat = quality_dict.get("features", {})
    return np.array([
        feat.get("graph_lcc_ratio",         0.0),
        feat.get("graph_density_score",      0.0),
        feat.get("graph_node_completeness",  0.0),
        1.0 - feat.get("vocab_richness",     0.0),
        feat.get("length_uniformity",        0.0),
        1.0 - feat.get("no_duplicate_doc_rate", 1.0),
    ], dtype=np.float32)


def load_graph_probe_features(dataset_name: str, degradation_type: str, rate: float) -> dict:
    """Load graph_probe_{degradation_type}_{rate:03d}.json's {query_id: {name: value}}
    map for one condition, or {} if the file is missing (step 1 not yet run for it)."""
    rate3 = int(round(rate * 100))
    path = Path(
        f"Dataset/EvaluationData/RetrievalEval/{dataset_name}/"
        f"graph_probe_{degradation_type}_{rate3:03d}.json"
    )
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("features", {})


def build_dataset(results: list, dataset_name: str = None) -> dict:
    """
    Build a dataset of (features, label) tuples from retrieval results.

    Each record in results covers one (degradation_type, rate, dataset) combo.
    For each query within that record, we build:
      - query features (one-hot query type)
      - corpus features (from quality dict, approximated)
      - quality features (12-dim vector from QualityAssessor)
      - label (best paradigm for this query)

    Label assignment: paradigm with highest nDCG@5 for this query.
    Ties broken by: naive > graph > hybrid (cheapest paradigm wins, cost ordering).
    If no paradigm retrieved anything, label = "cannot_answer".

    dataset_name is used only to locate the graph_probe_{operator}_{rate}.json
    files for the GP feature block (Run/Quality/compute_graph_probe.py); if
    omitted, gp_vec is zero for every row.
    """
    rows = []
    gp_missing = 0
    gp_total   = 0

    for record in results:
        quality_dict   = record.get("quality_features", {})
        quality_vector = np.array(
            quality_dict.get("vector", [0.0] * QUALITY_FEATURES_DIM),
            dtype=np.float32,
        )
        corpus_vector = compute_corpus_features_from_quality(quality_dict)
        paradigm_data = record.get("paradigms", {})

        if not paradigm_data:
            continue

        gp_features = {}
        if dataset_name is not None:
            gp_features = load_graph_probe_features(
                dataset_name, record.get("degradation_type"), float(record.get("rate", 0.0))
            )

        # Build per-query lookup: {query_id: {paradigm: metrics}}
        query_paradigm_scores = defaultdict(dict)
        query_types           = {}
        naive_pq_features     = {}  # query_id -> naive per-query feature dict
        other_pq_features     = defaultdict(dict)  # query_id -> {graph|hybrid: result-set feature dict}

        for paradigm, pdata in paradigm_data.items():
            for q_result in pdata.get("per_query", []):
                q_id = q_result["id"]
                query_paradigm_scores[q_id][paradigm] = q_result["metrics"]
                query_types[q_id] = q_result.get("query_type", "unknown")
                # Save naive probe features — used for all configs that include PQ
                if paradigm == "naive" and "pq_features" in q_result:
                    naive_pq_features[q_id] = q_result["pq_features"]
                elif paradigm in ("graph", "hybrid") and "pq_features" in q_result:
                    other_pq_features[q_id][paradigm] = q_result["pq_features"]

        for q_id, paradigm_scores in query_paradigm_scores.items():
            q_type = query_types.get(q_id, "unknown")

            # One-hot encode query type
            q_type_norm = q_type if q_type in QUERY_TYPES else "factual"
            query_vec = np.zeros(len(QUERY_TYPES), dtype=np.float32)
            if q_type_norm in QUERY_TYPES:
                query_vec[QUERY_TYPES.index(q_type_norm)] = 1.0

            # Per-query features from naive probe (zero if not available)
            pq_dict = naive_pq_features.get(q_id, {})
            pq_vec  = np.array(
                [pq_dict.get(n, 0.0) for n in PER_QUERY_FEATURE_NAMES],
                dtype=np.float32,
            )
            # Retrieve-all (RA) block: the same twelve result-set features computed on the
            # graph and hybrid result sets, plus pairwise differences of four semantic-fit
            # features across the three result sets (36-dim). Available only after all
            # three paradigms have been run for the query.
            g_dict = other_pq_features.get(q_id, {}).get("graph", {})
            h_dict = other_pq_features.get(q_id, {}).get("hybrid", {})
            ra = [g_dict.get(n, 0.0) for n in PER_QUERY_FEATURE_NAMES] + \
                 [h_dict.get(n, 0.0) for n in PER_QUERY_FEATURE_NAMES]
            for n in RA_DIFF_FEATURES:
                nv, gv, hv = pq_dict.get(n, 0.0), g_dict.get(n, 0.0), h_dict.get(n, 0.0)
                ra += [nv - gv, nv - hv, gv - hv]
            ra_vec = np.array(ra, dtype=np.float32)

            # Graph-side probe (GP) features for this query (zero if missing)
            gp_dict = gp_features.get(str(q_id), {})
            gp_total += 1
            if not gp_dict:
                gp_missing += 1
            gp_vec = np.array(
                [gp_dict.get(n, 0.0) for n in GP_FEATURE_NAMES],
                dtype=np.float32,
            )

            # Determine best paradigm (oracle label)
            best_paradigm = None
            best_score    = -1.0
            TIE_ORDER     = {"naive": 0, "graph": 1, "hybrid": 2}
            for paradigm, metrics in paradigm_scores.items():
                score = metrics.get("ndcg@5", 0.0)
                if score > best_score or (
                    score == best_score and
                    TIE_ORDER.get(paradigm, 9) <
                    TIE_ORDER.get(best_paradigm, 9)
                ):
                    best_score    = score
                    best_paradigm = paradigm

            label = best_paradigm if best_paradigm else "cannot_answer"

            rows.append({
                "query_id":        q_id,
                "query_type":      q_type,
                "query_vec":       query_vec,
                "corpus_vec":      corpus_vector,
                "quality_vec":     quality_vector,
                "pq_vec":          pq_vec,
                "ra_vec":          ra_vec,
                "gp_vec":          gp_vec,
                "label":           label,
                "oracle_ndcg5":    best_score,
                "paradigm_scores": paradigm_scores,
                "degradation_type": record.get("degradation_type"),
                "rate":            float(record.get("rate", 0.0)),
            })

    print(f"  Built {len(rows)} training instances")
    if dataset_name is not None:
        print(f"  Rows lacking graph-probe (GP) features: {gp_missing}/{gp_total}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Router evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_X(rows: list, config: str) -> np.ndarray:
    """Assemble feature matrix for the given router configuration.

    Configurations:
      Q-only          : query type one-hot (6-dim)
      Q+C             : + corpus structure features (6-dim)
      Q+C+Quality     : + corpus quality vector (12-dim)
      Q+C+Quality+PQ  : + naive per-query features (12-dim)  [thesis contribution]
      Q+QD            : query type one-hot + query-difficulty features only (2-dim)
                        [H1 confound control — isolates query_length_norm and
                        query_entity_count, the two probe features computable
                        from the query alone, without any retrieval]
      Q+C+Quality+PQ-QD : Q+C+Quality + the ten probe-outcome per-query features,
                        excluding query_length_norm and query_entity_count
                        [H1 confound control — PQ gain with the query-only
                        features held out]
      Q+C+Quality+GP    : + graph-side probe features (8-dim), no naive probe
                        [the cheap two-probe-free variant]
      Q+C+Quality+PQ+GP : + naive per-query (12-dim) + graph probe (8-dim)
                        [the two-probe cascade]
      Q+C+Quality+RA+GP : Q+C+Quality+RA + graph probe (8-dim) [everything]

    Column order (fixed, relied on by the ablation block offsets in
    run_ablation.py): query | corpus | quality | pq | ra | gp.
    """
    GP_CONFIGS = ("Q+C+Quality+GP", "Q+C+Quality+PQ+GP", "Q+C+Quality+RA+GP")
    parts = []
    for row in rows:
        feats = [row["query_vec"]]
        if config in ("Q+C", "Q+C+Quality", "Q+C+Quality+PQ", "Q+C+Quality+PQ-QD", "Q+C+Quality+RA") + GP_CONFIGS:
            feats.append(row["corpus_vec"])
        if config in ("Q+C+Quality", "Q+C+Quality+PQ", "Q+C+Quality+PQ-QD", "Q+C+Quality+RA") + GP_CONFIGS:
            feats.append(row["quality_vec"])
        if config in ("Q+C+Quality+PQ", "Q+C+Quality+RA", "Q+C+Quality+PQ+GP", "Q+C+Quality+RA+GP"):
            feats.append(row["pq_vec"])
        if config in ("Q+C+Quality+RA", "Q+C+Quality+RA+GP"):
            feats.append(row["ra_vec"])
        if config == "Q+C+Quality+PQ-QD":
            feats.append(row["pq_vec"][0:10])
        if config == "Q+QD":
            feats.append(row["pq_vec"][10:12])
        if config in GP_CONFIGS:
            feats.append(row["gp_vec"])
        parts.append(np.concatenate(feats))
    return np.vstack(parts)


def oracle_accuracy(rows: list, label_encoder: LabelEncoder) -> float:
    """Oracle always picks the best paradigm. Accuracy = 1.0 by definition."""
    return 1.0


def correct_rate_oracle(rows: list) -> float:
    """Oracle correct rate: mean nDCG@5 of the best paradigm per query."""
    return float(np.mean([r["oracle_ndcg5"] for r in rows]))


def correct_rate_router(rows: list, predictions: list) -> float:
    """Correct rate: mean nDCG@5 of the paradigm selected by the router."""
    scores = []
    for row, pred in zip(rows, predictions):
        paradigm_scores = row["paradigm_scores"]
        pred_ndcg = paradigm_scores.get(pred, {}).get("ndcg@5", 0.0)
        scores.append(pred_ndcg)
    return float(np.mean(scores))


def gain_by_band(rows: list, pred_labels) -> dict:
    """Out-of-fold CR minus always-naive CR, by degradation severity band."""
    bands = {"r=0": lambda r: r["rate"] == 0.0, "mild r<=0.3": lambda r: 0.0 < r["rate"] <= 0.3,
             "severe r>=0.4": lambda r: r["rate"] >= 0.4}
    out = {}
    for name, cond in bands.items():
        idx = [i for i, r in enumerate(rows) if cond(r)]
        if not idx: continue
        sel = [rows[i]["paradigm_scores"].get(pred_labels[i], {}).get("ndcg@5", 0.0) for i in idx]
        nai = [rows[i]["paradigm_scores"].get("naive", {}).get("ndcg@5", 0.0) for i in idx]
        out[name] = round(float(np.mean(sel) - np.mean(nai)), 4)
    return out


def fit_expected_value(X_train, train_rows, X_eval):
    """Expected-value rule: one XGBRegressor per paradigm predicts its nDCG@5. Route to the
    highest prediction, but stay with naive unless the predicted margin over naive exceeds a
    threshold chosen on the TRAINING rows only (grid EV_MARGIN_GRID). Returns predicted labels
    for X_eval, the chosen margin, and the mean regressor feature importances."""
    pred_tr = np.zeros((X_train.shape[0], len(PARADIGMS))); pred_ev = np.zeros((X_eval.shape[0], len(PARADIGMS)))
    fis = []
    for j, para in enumerate(PARADIGMS):
        target = np.array([r["paradigm_scores"].get(para, {}).get("ndcg@5", 0.0) for r in train_rows])
        reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
        reg.fit(X_train, target)
        pred_tr[:, j] = reg.predict(X_train); pred_ev[:, j] = reg.predict(X_eval); fis.append(reg.feature_importances_)
    n_idx = PARADIGMS.index("naive")
    def pick(pv, t):
        j = int(np.argmax(pv)); return PARADIGMS[j] if pv[j] - pv[n_idx] > t else "naive"
    best_t, best_cr = 0.0, -1.0
    for t in EV_MARGIN_GRID:
        c = correct_rate_router(train_rows, [pick(pv, t) for pv in pred_tr])
        if c > best_cr: best_cr, best_t = c, float(t)
    return [pick(pv, best_t) for pv in pred_ev], best_t, np.mean(fis, axis=0)


def evaluate_router(rows: list, config: str, architecture: str) -> dict:
    """Train and evaluate a router using 5-fold stratified cross-validation."""
    X = build_X(rows, config)
    labels = [r["label"] for r in rows]
    le = LabelEncoder()
    y = le.fit_transform(labels)

    skf = make_splitter()

    fold_accuracies = []
    fold_f1s        = []
    fold_correct_rates = []
    all_preds       = np.zeros(len(y), dtype=int)
    feature_importances = np.zeros(X.shape[1])

    ev_margins = []
    for fold, (train_idx, val_idx) in enumerate(split_iter(skf, X, y, rows)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if architecture == "xgb_ev":
            pred_labels, t, fi = fit_expected_value(X_train, [rows[i] for i in train_idx], X_val)
            ev_margins.append(t)
            preds = le.transform(pred_labels)
            all_preds[val_idx] = preds
            fold_accuracies.append(accuracy_score(y_val, preds))
            fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))
            fold_correct_rates.append(correct_rate_router([rows[i] for i in val_idx], pred_labels))
            feature_importances += fi
            continue

        if architecture == "xgboost":
            clf = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, verbosity=0,
            )
        else:  # mlp
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300,
                random_state=42, early_stopping=True,
            )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        all_preds[val_idx] = preds

        fold_accuracies.append(accuracy_score(y_val, preds))
        fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

        # Correct rate for this fold
        fold_rows = [rows[i] for i in val_idx]
        fold_pred_labels = le.inverse_transform(preds)
        fold_correct_rates.append(correct_rate_router(fold_rows, fold_pred_labels))

        # Accumulate feature importances (XGBoost only)
        if architecture == "xgboost" and hasattr(clf, "feature_importances_"):
            feature_importances += clf.feature_importances_

    # Average feature importances
    if architecture in ("xgboost", "xgb_ev"):
        feature_importances /= 5

    all_pred_labels = le.inverse_transform(all_preds)
    oracle_cr = correct_rate_oracle(rows)

    # Undegraded-condition (rate == 0.0) correct rate, using the out-of-fold CV
    # predictions above. All five operators' r = 0 files are identical (r = 0
    # is "no degradation applied" regardless of which operator would have been
    # applied), so only text_truncation's r = 0 rows are used here — pooling
    # all five would five-fold overweight the undegraded condition relative to
    # the degraded ones.
    undegraded_idx = [
        i for i, r in enumerate(rows)
        if r.get("rate", 0.0) == 0.0 and r.get("degradation_type") == "text_truncation"
    ]
    undegraded_rows = [rows[i] for i in undegraded_idx]
    if undegraded_rows:
        undegraded_pred_labels = [all_pred_labels[i] for i in undegraded_idx]
        undegraded_cr = correct_rate_router(undegraded_rows, undegraded_pred_labels)
        undegraded_oracle_cr = correct_rate_oracle(undegraded_rows)
        undegraded_fixed_cr = {
            p: float(np.mean([
                r["paradigm_scores"].get(p, {}).get("ndcg@5", 0.0)
                for r in undegraded_rows
            ]))
            for p in PARADIGMS
        }
    else:
        undegraded_cr = None
        undegraded_oracle_cr = None
        undegraded_fixed_cr = {p: None for p in PARADIGMS}

    return {
        "config":           config,
        "architecture":     architecture,
        "n_instances":      len(rows),
        "feature_dim":      X.shape[1],
        "routing_accuracy": float(np.mean(fold_accuracies)),
        "routing_acc_std":  float(np.std(fold_accuracies)),
        "macro_f1":         float(np.mean(fold_f1s)),
        "correct_rate":     float(np.mean(fold_correct_rates)),
        "correct_rate_std":   float(np.std(fold_correct_rates)),
        "fold_correct_rates": [float(x) for x in fold_correct_rates],
        "fold_accuracies":    [float(x) for x in fold_accuracies],
        "oracle_correct_rate": oracle_cr,
        "oracle_gap":       oracle_cr - float(np.mean(fold_correct_rates)),
        "feature_importances": feature_importances.tolist() if architecture in ("xgboost", "xgb_ev") else [],
        "ev_margins_per_fold": ev_margins,
        "pick_shares":        {p: float(np.mean(all_pred_labels == p)) for p in PARADIGMS},
        "gain_vs_naive_by_band": gain_by_band(rows, all_pred_labels),
        "split":              "grouped_by_query" if ROUTER_GROUPED else "instance_stratified",
        "undegraded_cr":        undegraded_cr,
        "undegraded_oracle_cr": undegraded_oracle_cr,
        "undegraded_fixed_cr":  undegraded_fixed_cr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Oracle gap decomposition
# ─────────────────────────────────────────────────────────────────────────────

def decompose_oracle_gap(results_by_config: dict, architecture: str = "xgboost") -> dict:
    """
    Oracle gap decomposition (novel contribution).

    Total gap  = oracle_correct_rate - Q_only_correct_rate
    Corpus gain = Q+C_correct_rate - Q_only_correct_rate
    Quality gain = Q+C+Quality_correct_rate - Q+C_correct_rate
    Remaining gap = oracle_correct_rate - Q+C+Quality_correct_rate
    """
    # Use XGBoost results for decomposition
    xgb = {r["config"]: r for r in results_by_config.values()
           if r["architecture"] == architecture}

    if len(xgb) < 3:
        return {"error": "Need at least three router configs for decomposition"}
    ra_cr = xgb.get("Q+C+Quality+RA", {}).get("correct_rate", None)
    gp_cr      = xgb.get("Q+C+Quality+GP",    {}).get("correct_rate", None)
    pqgp_cr    = xgb.get("Q+C+Quality+PQ+GP", {}).get("correct_rate", None)
    ragp_cr    = xgb.get("Q+C+Quality+RA+GP", {}).get("correct_rate", None)

    oracle_cr  = xgb.get("Q+C+Quality+PQ", xgb.get("Q+C+Quality", {})).get(
                     "oracle_correct_rate", 0.0)
    q_only_cr  = xgb.get("Q-only",         {}).get("correct_rate", 0.0)
    qc_cr      = xgb.get("Q+C",            {}).get("correct_rate", 0.0)
    qcq_cr     = xgb.get("Q+C+Quality",    {}).get("correct_rate", 0.0)
    qcqpq_cr   = xgb.get("Q+C+Quality+PQ", {}).get("correct_rate", qcq_cr)

    total_gap    = oracle_cr - q_only_cr
    corpus_gain  = qc_cr    - q_only_cr
    quality_gain = qcq_cr   - qc_cr
    pq_gain      = qcqpq_cr - qcq_cr
    remaining    = oracle_cr - qcqpq_cr

    return {
        "oracle_correct_rate":           round(oracle_cr,  4),
        "Q_only_correct_rate":           round(q_only_cr,  4),
        "Q+C_correct_rate":              round(qc_cr,      4),
        "Q+C+Quality_correct_rate":      round(qcq_cr,     4),
        "Q+C+Quality+PQ_correct_rate":   round(qcqpq_cr,   4),
        "total_gap":                     round(total_gap,   4),
        "corpus_feature_gain":           round(corpus_gain, 4),
        "quality_feature_gain":          round(quality_gain,4),
        "per_query_feature_gain":        round(pq_gain,     4),
        "remaining_unexplained_gap":     round(remaining,   4),
        "pq_pct_of_gap_closed":          round(pq_gain / total_gap, 4) if total_gap > 0 else 0.0,
        "quality_pct_of_gap_closed":     round(quality_gain / total_gap, 4) if total_gap > 0 else 0.0,
        "Q+C+Quality+RA_correct_rate":   round(ra_cr, 4) if ra_cr is not None else None,
        "ra_gain_over_pq":               round(ra_cr - qcqpq_cr, 4) if ra_cr is not None else None,
        "ra_gain_over_qcq":              round(ra_cr - qcq_cr, 4) if ra_cr is not None else None,
        "ra_pct_of_gap_closed":          round((ra_cr - q_only_cr) / total_gap, 4) if (ra_cr is not None and total_gap > 0) else None,
        "total_router_gain_pq":          round(qcqpq_cr - q_only_cr, 4),
        "total_router_gain_ra":          round(ra_cr - q_only_cr, 4) if ra_cr is not None else None,
        "Q+C+Quality+GP_correct_rate":      round(gp_cr, 4)   if gp_cr   is not None else None,
        "Q+C+Quality+PQ+GP_correct_rate":   round(pqgp_cr, 4) if pqgp_cr is not None else None,
        "Q+C+Quality+RA+GP_correct_rate":   round(ragp_cr, 4) if ragp_cr is not None else None,
        "gp_gain_over_qcq":              round(gp_cr - qcq_cr, 4)     if gp_cr   is not None else None,
        "gp_gain_over_pq":               round(pqgp_cr - qcqpq_cr, 4) if pqgp_cr is not None else None,
        "gp_gain_over_ra":               round(ragp_cr - ra_cr, 4)    if (ragp_cr is not None and ra_cr is not None) else None,
        "architecture":                  architecture,
        "split":                         "grouped_by_query" if ROUTER_GROUPED else "instance_stratified",
    }


def compute_control_decomposition(results_by_config: dict, architecture: str = "xgboost") -> dict:
    """
    H1 confound control decomposition (does not touch decompose_oracle_gap()).

    Isolates how much of the per-query-feature gain is attributable to the two
    query-only-derived features (query_length_norm, query_entity_count) versus
    the ten genuine probe-outcome features:

      qd_only_gain            = Q+QD_correct_rate - Q-only_correct_rate
      pq_without_qd_gain      = (Q+C+Quality+PQ-QD)_correct_rate - Q+C+Quality_correct_rate
      qd_gain_on_top_of_probe = Full_correct_rate - (Q+C+Quality+PQ-QD)_correct_rate
    """
    xgb = {r["config"]: r for r in results_by_config.values()
           if r["architecture"] == architecture}

    q_only_cr  = xgb.get("Q-only",              {}).get("correct_rate", 0.0)
    qd_cr      = xgb.get("Q+QD",                {}).get("correct_rate", 0.0)
    qcq_cr     = xgb.get("Q+C+Quality",         {}).get("correct_rate", 0.0)
    qcqpqmqd_cr = xgb.get("Q+C+Quality+PQ-QD",  {}).get("correct_rate", 0.0)
    full_cr    = xgb.get("Q+C+Quality+PQ",      {}).get("correct_rate", 0.0)

    qd_only_gain            = qd_cr        - q_only_cr
    pq_without_qd_gain      = qcqpqmqd_cr  - qcq_cr
    qd_gain_on_top_of_probe = full_cr      - qcqpqmqd_cr

    return {
        "Q+QD_correct_rate":              round(qd_cr,        4),
        "Q+C+Quality+PQ-QD_correct_rate": round(qcqpqmqd_cr,  4),
        "qd_only_gain":                   round(qd_only_gain, 4),
        "pq_without_qd_gain":             round(pq_without_qd_gain, 4),
        "qd_gain_on_top_of_probe":        round(qd_gain_on_top_of_probe, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance report
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_names(config: str) -> list:
    GP_CONFIGS = ("Q+C+Quality+GP", "Q+C+Quality+PQ+GP", "Q+C+Quality+RA+GP")
    names = [f"query_{qt}" for qt in QUERY_TYPES]
    if config in ("Q+C", "Q+C+Quality", "Q+C+Quality+PQ", "Q+C+Quality+PQ-QD", "Q+C+Quality+RA") + GP_CONFIGS:
        names += CORPUS_FEATURE_NAMES
    if config in ("Q+C+Quality", "Q+C+Quality+PQ", "Q+C+Quality+PQ-QD", "Q+C+Quality+RA") + GP_CONFIGS:
        names += QUALITY_FEATURE_NAMES
    if config in ("Q+C+Quality+PQ", "Q+C+Quality+RA", "Q+C+Quality+PQ+GP", "Q+C+Quality+RA+GP"):
        names += [f"pq_{n}" for n in PER_QUERY_FEATURE_NAMES]
    if config in ("Q+C+Quality+RA", "Q+C+Quality+RA+GP"):
        names += [f"graph_{n}" for n in PER_QUERY_FEATURE_NAMES] + [f"hybrid_{n}" for n in PER_QUERY_FEATURE_NAMES]
        for n in RA_DIFF_FEATURES:
            names += [f"diff_ng_{n}", f"diff_nh_{n}", f"diff_gh_{n}"]
    if config == "Q+C+Quality+PQ-QD":
        names += [f"pq_{n}" for n in PER_QUERY_FEATURE_NAMES[0:10]]
    if config == "Q+QD":
        names += [f"pq_{n}" for n in PER_QUERY_FEATURE_NAMES[10:12]]
    if config in GP_CONFIGS:
        # GP_FEATURE_NAMES already carry the "gp_" prefix (gp_seed_*).
        names += GP_FEATURE_NAMES
    return names


def feature_importance_report(result: dict) -> dict:
    if not result.get("feature_importances"):
        return {}
    names = build_feature_names(result["config"])
    imps  = result["feature_importances"]
    pairs = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
    return {name: round(imp, 6) for name, imp in pairs}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name: str) -> None:
    print(f"\n{'#'*65}")
    print(f"  Router Training & Evaluation: {dataset_name}")
    print(f"{'#'*65}")

    results_raw = load_retrieval_results(dataset_name)
    rows = build_dataset(results_raw, dataset_name=dataset_name)

    if len(rows) < 10:
        print(f"  Not enough data ({len(rows)} rows). Run run_retrieval_eval.py first.")
        return

    configs       = ["Q-only", "Q+C", "Q+C+Quality", "Q+C+Quality+PQ", "Q+C+Quality+RA", "Q+QD", "Q+C+Quality+PQ-QD",
                      "Q+C+Quality+GP", "Q+C+Quality+PQ+GP", "Q+C+Quality+RA+GP"]
    architectures = ["xgboost", "xgb_ev", "mlp"]

    all_results = {}
    print(f"\n  Training routers on {len(rows)} instances...")
    print(f"  {'Config':<16} {'Arch':<10} {'Acc':>6} {'F1':>6} {'CR':>6} {'Gap':>7}")
    print("  " + "-" * 55)

    for config in configs:
        for arch in architectures:
            result = evaluate_router(rows, config, arch)
            key = f"{config}_{arch}"
            all_results[key] = result
            print(f"  {config:<16} {arch:<10} "
                  f"{result['routing_accuracy']:>6.3f} "
                  f"{result['macro_f1']:>6.3f} "
                  f"{result['correct_rate']:>6.3f} "
                  f"{result['oracle_gap']:>7.4f}")

    # Oracle gap decomposition
    gap_decomp = decompose_oracle_gap(all_results)
    print(f"\n  Oracle Gap Decomposition:")
    print(f"    Oracle correct rate           : {gap_decomp.get('oracle_correct_rate', 0):.4f}")
    print(f"    Q-only correct rate           : {gap_decomp.get('Q_only_correct_rate', 0):.4f}")
    print(f"    Q+C correct rate              : {gap_decomp.get('Q+C_correct_rate', 0):.4f}")
    print(f"    Q+C+Quality correct rate      : {gap_decomp.get('Q+C+Quality_correct_rate', 0):.4f}")
    print(f"    Q+C+Quality+PQ correct rate   : {gap_decomp.get('Q+C+Quality+PQ_correct_rate', 0):.4f}")
    print(f"    ── Total gap                  : {gap_decomp.get('total_gap', 0):.4f}")
    print(f"       Corpus features gain       : {gap_decomp.get('corpus_feature_gain', 0):.4f}")
    print(f"       Quality features gain      : {gap_decomp.get('quality_feature_gain', 0):.4f}")
    print(f"       Per-query features gain    : {gap_decomp.get('per_query_feature_gain', 0):.4f}")
    print(f"       Remaining gap              : {gap_decomp.get('remaining_unexplained_gap', 0):.4f}")
    print(f"    Per-query features close {gap_decomp.get('pq_pct_of_gap_closed', 0)*100:.1f}% of total gap")
    print(f"    Quality features close   {gap_decomp.get('quality_pct_of_gap_closed', 0)*100:.1f}% of total gap")

    # H1 confound control decomposition (query-only features vs. genuine probe
    # outcome features) — does not touch decompose_oracle_gap() above.
    control_decomp = compute_control_decomposition(all_results)
    gap_decomp["control_decomposition"] = control_decomp
    # Expected-value rule: full decomposition and control, stored alongside the classifier's
    ev_decomp = decompose_oracle_gap(all_results, "xgb_ev")
    ev_decomp["control_decomposition"] = compute_control_decomposition(all_results, "xgb_ev")
    gap_decomp["expected_value"] = ev_decomp
    print(f"\n  Expected-value rule: PQ {ev_decomp.get('Q+C+Quality+PQ_correct_rate')}  RA {ev_decomp.get('Q+C+Quality+RA_correct_rate')}  (Q-only {ev_decomp.get('Q_only_correct_rate')})")
    print(f"  Classifier rule:     RA {gap_decomp.get('Q+C+Quality+RA_correct_rate')}")
    print(f"  GP (ev):   GP {ev_decomp.get('Q+C+Quality+GP_correct_rate')}  PQ+GP {ev_decomp.get('Q+C+Quality+PQ+GP_correct_rate')}  "
          f"RA+GP {ev_decomp.get('Q+C+Quality+RA+GP_correct_rate')}  "
          f"(gp_gain_over_qcq={ev_decomp.get('gp_gain_over_qcq')}  gp_gain_over_pq={ev_decomp.get('gp_gain_over_pq')}  gp_gain_over_ra={ev_decomp.get('gp_gain_over_ra')})")
    print(f"  GP (xgb):  GP {gap_decomp.get('Q+C+Quality+GP_correct_rate')}  PQ+GP {gap_decomp.get('Q+C+Quality+PQ+GP_correct_rate')}  "
          f"RA+GP {gap_decomp.get('Q+C+Quality+RA+GP_correct_rate')}")
    print(f"\n  Control Decomposition (H1 confound):")
    print(f"    Q+QD correct rate             : {control_decomp.get('Q+QD_correct_rate', 0):.4f}")
    print(f"    Q+C+Quality+PQ-QD correct rate: {control_decomp.get('Q+C+Quality+PQ-QD_correct_rate', 0):.4f}")
    print(f"    QD-only gain                  : {control_decomp.get('qd_only_gain', 0):.4f}")
    print(f"    PQ gain without QD            : {control_decomp.get('pq_without_qd_gain', 0):.4f}")
    print(f"    QD gain on top of full probe  : {control_decomp.get('qd_gain_on_top_of_probe', 0):.4f}")

    # Feature importance for Q+C+Quality XGBoost
    qcq_xgb_result = all_results.get("Q+C+Quality_xgboost", {})
    feat_imp = feature_importance_report(qcq_xgb_result)
    print(f"\n  Top quality features by importance:")
    quality_feats = {k: v for k, v in feat_imp.items()
                     if k in QUALITY_FEATURE_NAMES}
    for name, imp in sorted(quality_feats.items(), key=lambda x: x[1], reverse=True)[:5]:
        bar = "█" * int(imp * 200)
        print(f"    {name:<30} {imp:.4f}  {bar}")

    # Save all results
    out_dir = Path(f"Dataset/EvaluationData/{ROUTER_OUT_ROOT}/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "oracle_gap_decomposition.json", "w") as f:
        json.dump(gap_decomp, f, indent=2)

    with open(out_dir / "feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    print(f"\n  Results saved -> {out_dir}")


def run_crossdataset(train_name: str, test_name: str) -> None:
    """Train on one dataset, test on another — measures generalisation."""
    print(f"\n{'#'*65}")
    print(f"  Cross-dataset: train={train_name}  test={test_name}")
    print(f"{'#'*65}")

    train_rows = build_dataset(load_retrieval_results(train_name), dataset_name=train_name)
    test_rows  = build_dataset(load_retrieval_results(test_name), dataset_name=test_name)

    if len(train_rows) < 10 or len(test_rows) < 10:
        print("  Not enough data. Skipping.")
        return

    configs       = ["Q-only", "Q+C", "Q+C+Quality", "Q+C+Quality+PQ", "Q+C+Quality+RA", "Q+QD", "Q+C+Quality+PQ-QD",
                      "Q+C+Quality+GP", "Q+C+Quality+PQ+GP", "Q+C+Quality+RA+GP"]
    architectures = ["xgboost", "xgb_ev", "mlp"]

    all_results = {}
    print(f"  Train: {len(train_rows)} instances  |  Test: {len(test_rows)} instances")
    print(f"  {'Config':<16} {'Arch':<10} {'Acc':>6} {'F1':>6} {'CR':>6} {'Gap':>7}")
    print("  " + "-" * 55)

    for config in configs:
        for arch in architectures:
            X_train = build_X(train_rows, config)
            X_test  = build_X(test_rows,  config)
            labels_train = [r["label"] for r in train_rows]
            labels_test  = [r["label"] for r in test_rows]

            # Fit label encoder on union of labels
            le = LabelEncoder()
            le.fit(list(set(labels_train + labels_test)))
            y_train = le.transform(labels_train)
            y_test  = le.transform(labels_test)

            if arch == "xgb_ev":
                pred_labels, _, _ = fit_expected_value(X_train, train_rows, X_test)
                pred_labels = np.array(pred_labels); preds = le.transform(pred_labels)
            else:
                if arch == "xgboost":
                    clf = XGBClassifier(
                        n_estimators=200, max_depth=4, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="mlogloss",
                        random_state=42, verbosity=0,
                    )
                else:
                    clf = MLPClassifier(
                        hidden_layer_sizes=(64, 32), max_iter=300,
                        random_state=42, early_stopping=True,
                    )
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                pred_labels = le.inverse_transform(preds)

            acc    = float(accuracy_score(y_test, preds))
            f1     = float(f1_score(y_test, preds, average="macro", zero_division=0))
            cr     = correct_rate_router(test_rows, pred_labels)
            oracle = correct_rate_oracle(test_rows)
            gap    = oracle - cr

            key = f"{config}_{arch}"
            all_results[key] = {
                "config": config, "architecture": arch,
                "n_train": len(train_rows), "n_test": len(test_rows),
                "routing_accuracy": acc, "macro_f1": f1,
                "correct_rate": cr, "oracle_correct_rate": oracle, "oracle_gap": gap,
            }
            print(f"  {config:<16} {arch:<10} {acc:>6.3f} {f1:>6.3f} {cr:>6.3f} {gap:>7.4f}")

    gap_decomp = decompose_oracle_gap(all_results)
    gap_decomp["expected_value"] = decompose_oracle_gap(all_results, "xgb_ev")
    print(f"\n  Oracle Gap Decomposition (generalisation):")
    print(f"    Oracle correct rate         : {gap_decomp.get('oracle_correct_rate', 0):.4f}")
    print(f"    Q-only                      : {gap_decomp.get('Q_only_correct_rate', 0):.4f}")
    print(f"    Q+C                         : {gap_decomp.get('Q+C_correct_rate', 0):.4f}")
    print(f"    Q+C+Quality                 : {gap_decomp.get('Q+C+Quality_correct_rate', 0):.4f}")
    print(f"    Q+C+Quality+PQ              : {gap_decomp.get('Q+C+Quality+PQ_correct_rate', 0):.4f}")
    print(f"    Per-query feature gain      : {gap_decomp.get('per_query_feature_gain', 0):.4f}")
    print(f"    PQ closes {gap_decomp.get('pq_pct_of_gap_closed', 0)*100:.1f}% of oracle gap")

    out_dir = Path(f"Dataset/EvaluationData/{ROUTER_OUT_ROOT}/cross_{train_name}_to_{test_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / "oracle_gap_decomposition.json", "w") as f:
        json.dump(gap_decomp, f, indent=2)
    print(f"\n  Results saved -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate quality-aware RAG router"
    )
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--cross", action="store_true",
                        help="Run cross-dataset generalisation experiments")
    args = parser.parse_args()

    if args.cross:
        all_datasets = [
            name for name in ["quality", "musique", "ultraDomain_legal", "graphragBench_medical"]
            if Path(f"Dataset/EvaluationData/RetrievalEval/{name}").exists()
        ]
        for train in all_datasets:
            for test in all_datasets:
                if train != test:
                    run_crossdataset(train, test)
        return

    targets = ["musique", "quality"] if args.dataset == "all" else [args.dataset]
    for name in targets:
        path = Path(f"Dataset/EvaluationData/RetrievalEval/{name}")
        if path.exists():
            run_dataset(name)
        else:
            print(f"Skipping {name}: run run_retrieval_eval.py first")


if __name__ == "__main__":
    main()
