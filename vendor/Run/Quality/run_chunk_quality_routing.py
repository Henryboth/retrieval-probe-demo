"""
run_chunk_quality_routing.py
============================
Experiment: do chunk-level quality features improve routing?

Approach: no re-embedding. Instead, for each query we use the per-query
features already stored in the eval results (pq_features) to reconstruct
which chunks naive retrieval returned, then augment with per-chunk quality
proxies computed directly from the chunk text stored in the FAISS metadata.

Since we cannot reconstruct exact chunk order without re-running FAISS,
we instead compute chunk quality statistics across ALL chunks in the corpus
stratified by the query's retrieval quality signal — specifically:

  For each query, we sample the top-score-aligned chunks using the
  score_mean and score_gap features already stored, then score chunks
  in the score window above score_mean.

Actually, simpler and more defensible: compute chunk quality statistics
for the full corpus once, then use them as additional features alongside
the existing per-query PQ features. This tests whether knowing the
distribution of chunk quality in the corpus (beyond the 12-dim DQ vector)
helps the router.

Five new corpus-chunk-quality features (computed once per dataset):
  cq_mean     : mean quality across all corpus chunks
  cq_std      : std of chunk quality (spread)
  cq_p10      : 10th percentile (worst chunks)
  cq_p90      : 90th percentile (best chunks)
  cq_low_frac : fraction of chunks below 0.3 quality (poor chunk rate)

These are constant per corpus condition (like the 12-dim quality vector)
and are appended as 5 extra features to the Q+C+Quality+PQ config.
"""

import sys, os, re, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from Run.Quality.run_router_training import (
    load_retrieval_results,
    build_dataset,
    build_X,
    correct_rate_oracle,
    correct_rate_router,
)
from Config.PathConfig import PathConfig

_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]*)(?:\s+[A-Z][a-zA-Z]*){0,3}\b")


def chunk_quality_score(text: str, corpus_mean_chars: float) -> float:
    if not text.strip():
        return 0.0
    text_completeness = min(len(text) / max(corpus_mean_chars, 1.0), 1.0)
    tokens = text.lower().split()
    vocab_richness = len(set(tokens)) / max(len(tokens), 1)
    entities = _ENTITY_RE.findall(text)
    entity_density = min(len(entities) / max(len(tokens), 1) / 0.1, 1.0)
    return float(np.mean([text_completeness, vocab_richness, entity_density]))


def compute_corpus_chunk_quality(dataset_name: str) -> np.ndarray:
    """Score every chunk in the corpus and return 5 aggregate statistics."""
    index_dir = Path(PathConfig.get_index_path(dataset_name))
    metadata_path = index_dir / "metadata.json"
    with open(metadata_path) as f:
        chunks = json.load(f)

    print(f"  Scoring {len(chunks)} corpus chunks (no embedder needed)...")
    corpus_mean_chars = float(np.mean([len(c.get("text", "")) for c in chunks]))

    scores = np.array([
        chunk_quality_score(c.get("text", ""), corpus_mean_chars)
        for c in chunks
    ], dtype=np.float32)

    result = np.array([
        float(np.mean(scores)),
        float(np.std(scores)),
        float(np.percentile(scores, 10)),
        float(np.percentile(scores, 90)),
        float(np.mean(scores < 0.3)),
    ], dtype=np.float32)

    print(f"  Chunk quality: mean={result[0]:.3f} std={result[1]:.3f} "
          f"p10={result[2]:.3f} p90={result[3]:.3f} low_frac={result[4]:.3f}")
    return result


def run_cv(X: np.ndarray, y: np.ndarray, rows: list, le: LabelEncoder) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    all_preds = np.zeros(len(y), dtype=int)

    for train_idx, val_idx in skf.split(X, y):
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        all_preds[val_idx] = preds
        accs.append(float(np.mean(preds == y[val_idx])))
        f1s.append(float(f1_score(y[val_idx], preds, average="macro", zero_division=0)))

    pred_labels = le.inverse_transform(all_preds)
    cr = correct_rate_router(rows, list(pred_labels))
    return {"routing_accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)), "correct_rate": cr}


def run_chunk_quality_experiment(dataset_name: str):
    print(f"\n{'#'*65}")
    print(f"  Chunk Quality Routing — {dataset_name}")
    print(f"{'#'*65}\n")

    results_raw = load_retrieval_results(dataset_name)
    rows = build_dataset(results_raw)
    if not rows:
        print("  No training data.")
        return

    print(f"  Training instances : {len(rows)}")
    oracle_cr = correct_rate_oracle(rows)
    print(f"  Oracle correct rate: {oracle_cr:.4f}")

    # Corpus-level chunk quality vector (5-dim, constant per dataset)
    cq_vec = compute_corpus_chunk_quality(dataset_name)

    # Broadcast chunk quality vector across all rows
    cq_cols = np.tile(cq_vec, (len(rows), 1))

    X_qonly = build_X(rows, "Q-only")
    X_pq    = build_X(rows, "Q+C+Quality+PQ")
    X_pq_cq = np.hstack([X_pq, cq_cols])

    labels = [r["label"] for r in rows]
    le = LabelEncoder()
    y = le.fit_transform(labels)

    res_qonly = run_cv(X_qonly, y, rows, le)
    res_pq    = run_cv(X_pq,    y, rows, le)
    res_pq_cq = run_cv(X_pq_cq, y, rows, le)

    floor_cr  = res_qonly["correct_rate"]
    total_gap = oracle_cr - floor_cr
    pq_gap    = res_pq["correct_rate"]    - floor_cr
    cq_gap    = res_pq_cq["correct_rate"] - floor_cr
    delta     = res_pq_cq["correct_rate"] - res_pq["correct_rate"]

    print(f"\n  {'Config':<35} {'CR':>8} {'Acc':>8} {'F1':>8}")
    print("  " + "-" * 62)
    for label, res in [
        ("Q-only (floor)",        res_qonly),
        ("Q+C+Quality+PQ",        res_pq),
        ("Q+C+Quality+PQ+ChunkQ", res_pq_cq),
    ]:
        print(f"  {label:<35} {res['correct_rate']:>8.4f} "
              f"{res['routing_accuracy']:>8.4f} {res['macro_f1']:>8.4f}")

    print(f"\n  Oracle CR              : {oracle_cr:.4f}")
    print(f"  Total gap              : {total_gap:.4f}")
    print(f"  PQ gap closed          : {100*pq_gap/max(total_gap,1e-6):.1f}%")
    print(f"  PQ+ChunkQ gap closed   : {100*cq_gap/max(total_gap,1e-6):.1f}%")
    print(f"  Chunk quality delta    : {delta:+.4f}")

    if delta > 0.002:
        interp = "IMPROVEMENT — corpus chunk quality adds signal beyond DQ vector"
    elif delta > -0.002:
        interp = "NEUTRAL — corpus chunk quality is already captured by the 12-dim DQ vector"
    else:
        interp = "NEGATIVE — corpus chunk quality adds noise"
    print(f"  Interpretation         : {interp}")

    output = {
        "dataset":              dataset_name,
        "oracle_cr":            round(oracle_cr, 4),
        "floor_cr_qonly":       round(floor_cr, 4),
        "pq_cr":                round(res_pq["correct_rate"], 4),
        "pq_cq_cr":             round(res_pq_cq["correct_rate"], 4),
        "chunk_quality_delta":  round(delta, 4),
        "pq_gap_closed_pct":    round(100*pq_gap/max(total_gap,1e-6), 2),
        "pq_cq_gap_closed_pct": round(100*cq_gap/max(total_gap,1e-6), 2),
        "corpus_chunk_quality": {
            "mean": round(float(cq_vec[0]), 4),
            "std":  round(float(cq_vec[1]), 4),
            "p10":  round(float(cq_vec[2]), 4),
            "p90":  round(float(cq_vec[3]), 4),
            "low_frac": round(float(cq_vec[4]), 4),
        },
        "interpretation": interp,
    }

    out_dir = Path(f"Dataset/EvaluationData/RouterEval/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunk_quality_experiment.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="quality",
                        choices=["quality", "musique", "ultraDomain_legal"])
    args = parser.parse_args()
    run_chunk_quality_experiment(args.dataset)


if __name__ == "__main__":
    main()
