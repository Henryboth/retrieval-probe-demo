"""
run_latency_eval.py
====================
Wall-clock retrieval-cost benchmark for the cascade routing claim in Ch3:
that when the router selects naive, the naive probe's results are reused
at "no additional retrieval cost." That claim is argued from complexity
in the thesis and has never been measured. This script measures it.

Measures, per query, per corpus (undegraded r=0 condition only):
  t_naive          -- NaiveRetriever.retrieve() alone
  t_graph          -- GraphRetriever.retrieve() alone
  t_hybrid         -- HybridRetriever.retrieve() alone (re-runs naive internally)
  t_retrieve_all   -- t_naive + t_graph + t_hybrid (run-everything baseline)
  t_cascade        -- actually execute the cascade: probe (naive, reused from
                       t_naive's call, not re-run) -> per-query features ->
                       router.predict -> escalate (graph / hybrid-fuse / none)

Does NOT touch RAGCore/Retriever/ (the LLM answer-generation pipeline, used
by no experiment). Times exactly the NaiveRetriever / GraphRetriever /
HybridRetriever classes defined in run_retrieval_eval.py, called the same
way run_dataset() calls them.

Output
------
Dataset/EvaluationData/LatencyEval/{dataset}/latency.json

Usage
-----
python Run/Quality/run_latency_eval.py --dataset quality --limit 20
python Run/Quality/run_latency_eval.py --dataset musique --limit 500
python Run/Quality/run_latency_eval.py --dataset ultraDomain_legal
"""

import sys, os

# faiss and xgboost each bundle their own copy of libomp. Loading both in one
# process gives two duplicate OpenMP runtimes; depending on import order and
# workload size this has been observed to either segfault (XGBClassifier.fit
# on a small corpus, exit 139) or deadlock silently (faiss index.search()
# inside an OpenMP parallel region never returns -- 0% CPU, STAT S, no
# traceback -- reproduced on the musique corpus's larger FAISS/PPR workload).
# Import order alone (xgboost before faiss, below) only changes which failure
# mode shows up; it does not remove the duplicate runtime. These two env vars
# must be set before numpy/xgboost/faiss/torch are imported, so they are set
# first thing, right after `os` becomes available:
#   OMP_NUM_THREADS=1       -- forces single-threaded OpenMP, so a parallel
#                               region can never enter the barrier wait that
#                               deadlocks with the duplicate runtime present.
#                               Also makes latency measurements reproducible
#                               (no thread-count/scheduler variance).
#   KMP_DUPLICATE_LIB_OK=1  -- tells Intel's OpenMP runtime (which xgboost's
#                               libomp is derived from) to tolerate a second
#                               libomp already being loaded, instead of
#                               aborting/corrupting state.
# NOTE: this changes absolute wall-clock timings relative to any prior run
# that did not set these (e.g. the QuALITY run already on disk at
# Dataset/EvaluationData/LatencyEval/quality/latency.json). QuALITY must be
# rerun with this same setting for the three corpora's numbers to be
# comparable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import platform
import subprocess
import argparse
import datetime
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# xgboost MUST be imported before faiss. On this machine, importing faiss
# first and then calling XGBClassifier.fit() segfaults (exit 139) -- an
# OpenMP runtime conflict between faiss's and xgboost's bundled libomp.
# Reproduced in isolation: faiss-then-xgboost-fit crashes every time;
# xgboost-then-faiss does not. run_retrieval_eval (imported below) pulls
# in faiss at module level, so xgboost is imported first, here, to force
# the safe order. Kept alongside the OMP_NUM_THREADS/KMP_DUPLICATE_LIB_OK
# fix above as a belt-and-braces measure.
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from QualityCore import PerQueryAssessor
from Run.Quality.run_router_training import PARADIGMS, RA_DIFF_FEATURES

from Run.Quality.run_retrieval_eval import (
    NaiveRetriever, GraphRetriever, HybridRetriever,
    load_corpus, load_graph, load_questions, build_text_to_docid,
)
from Run.Quality.run_router_training import (
    load_retrieval_results, build_dataset, build_X,
    PER_QUERY_FEATURE_NAMES,
)


DATASETS = ["quality", "musique", "ultraDomain_legal"]
ROUTER_CONFIG = "Q+C+Quality+PQ"
N_REPS = 3
TOP_K = 10


# ─────────────────────────────────────────────────────────────────────────────
# RRF fusion, extracted from HybridRetriever.retrieve so the cascade's
# hybrid escalation path can fuse graph results with the naive results it
# already holds instead of re-running naive a second time.
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(naive_results: list, graph_results: list,
             top_k: int = TOP_K, rrf_k: int = 60) -> list:
    scores: dict = {}
    for rank, r in enumerate(naive_results, 1):
        doc_id = r["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
    for rank, r in enumerate(graph_results, 1):
        doc_id = r["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in fused[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Router: no persisted model exists anywhere in the codebase (verified: no
# joblib/pickle/save_model call for the XGBClassifier router in
# run_router_training.py or elsewhere). Train one in-script, in memory,
# on this corpus's full 40-condition instance set, using the exact
# feature-construction functions used to produce the thesis's router numbers.
# ─────────────────────────────────────────────────────────────────────────────

def train_selector(dataset_name: str, rows: list):
    """Retrieve-all selector of record: expected-value rule (one XGBRegressor per paradigm on the
    72-dim Q+C+Quality+RA design matrix), margin threshold chosen on the training rows, trained on
    the full 40-condition set (the deployed model; grouped CV is the evaluation protocol)."""
    from xgboost import XGBRegressor
    from Run.Quality.run_router_training import EV_MARGIN_GRID, correct_rate_router
    X = build_X(rows, "Q+C+Quality+RA")
    regs = {}; pred_tr = np.zeros((len(rows), len(PARADIGMS)))
    for j, para in enumerate(PARADIGMS):
        y = np.array([r["paradigm_scores"].get(para, {}).get("ndcg@5", 0.0) for r in rows])
        reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
        reg.fit(X, y); pred_tr[:, j] = reg.predict(X); regs[para] = reg
    n_idx = PARADIGMS.index("naive")
    def pick(pv, t):
        j = int(np.argmax(pv)); return PARADIGMS[j] if pv[j] - pv[n_idx] > t else "naive"
    best_t, best_cr = 0.0, -1.0
    for t in EV_MARGIN_GRID:
        c = correct_rate_router(rows, [pick(pv, t) for pv in pred_tr])
        if c > best_cr: best_cr, best_t = c, float(t)
    print(f"  Selector trained on {len(rows)} instances, {X.shape[1]}-dim, margin={best_t}")
    return regs, best_t


def train_router(dataset_name: str):
    print(f"  No persisted router found for {dataset_name} -- training in-script "
          f"on the 40-condition instance set (config={ROUTER_CONFIG}).")
    results_raw = load_retrieval_results(dataset_name)
    rows = build_dataset(results_raw)
    X = build_X(rows, ROUTER_CONFIG)
    labels = [r["label"] for r in rows]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    clf.fit(X, y)
    print(f"  Router trained on {len(rows)} instances, "
          f"{X.shape[1]}-dim features, classes={list(le.classes_)}")
    return clf, le, rows


def r0_feature_lookup(dataset_name: str, results_raw: list) -> dict:
    """Static (query_vec, corpus_vec, quality_vec, oracle label) per query_id,
    taken from the r=0 (undegraded) condition, as specified: the cascade's
    36-dim vector uses the r=0 condition's corpus/quality vectors."""
    r0_record = next((r for r in results_raw if r.get("rate") == 0.0), None)
    if r0_record is None:
        raise RuntimeError(f"No rate=0.0 record found for {dataset_name}")
    r0_rows = build_dataset([r0_record])
    return {row["query_id"]: row for row in r0_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def _stage(msg: str) -> None:
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def setup_corpus(dataset_name: str):
    _stage("loading embedder...")
    embedder = EmbeddingProcessor()
    _stage("embedder loaded; loading corpus...")
    corpus = load_corpus(dataset_name)
    _stage(f"corpus loaded ({len(corpus)} docs); loading graph (raw pickle)...")
    graph = load_graph(dataset_name)
    _stage("raw graph loaded; building text_to_docid map...")

    text_to_docid = build_text_to_docid(corpus)
    docid_to_text = {
        str(d.get("id", d.get("doc_id", ""))): d.get("context", d.get("text", ""))
        for d in corpus
    }
    corpus_mean_chars = float(np.mean([
        len(d.get("context", d.get("text", ""))) for d in corpus
    ])) if corpus else 1.0

    _stage("loading FAISS index (NaiveRetriever)...")
    naive = NaiveRetriever(dataset_name, embedder)
    _stage("FAISS index loaded; loading graph + entity index (GraphRetriever)...")
    graph_ret = GraphRetriever(dataset_name, embedder)  # r=0: undegraded disk graph
    _stage("graph + entity index loaded; constructing HybridRetriever...")
    hybrid = HybridRetriever(naive, graph_ret)

    pq_assessor = PerQueryAssessor(
        embedder=embedder, corpus_mean_chars=corpus_mean_chars, top_k=TOP_K,
    )
    _stage("setup_corpus complete.")

    return {
        "embedder": embedder, "naive": naive, "graph": graph_ret, "hybrid": hybrid,
        "pq_assessor": pq_assessor, "text_to_docid": text_to_docid,
        "docid_to_text": docid_to_text,
    }


def resolve_support(q: dict, text_to_docid: dict) -> set:
    support = set()
    for sf in q.get("supporting_facts", []):
        if isinstance(sf, dict):
            support.add(str(sf["doc_id"]))
        elif isinstance(sf, str):
            doc_id = text_to_docid.get(sf)
            if doc_id:
                support.add(doc_id)
    return support


def load_eval_queries(dataset_name: str, ctx: dict, r0_index: dict,
                      limit: int = None) -> list:
    """Same filter run_dataset() applies before timing/scoring a query:
    must have non-empty question text and resolvable supporting facts.
    Additionally require presence in the r=0 feature lookup, so every
    timed query has a static feature vector available."""
    questions = load_questions(dataset_name, limit=limit)
    out = []
    skipped_no_support = 0
    skipped_no_r0 = 0
    for q in questions:
        query = q.get("question", "")
        support = resolve_support(q, ctx["text_to_docid"])
        if not query or not support:
            skipped_no_support += 1
            continue
        qid = q.get("id", "")
        if qid not in r0_index:
            skipped_no_r0 += 1
            continue
        out.append(q)
    print(f"  Queries: {len(out)} eval-eligible "
          f"(skipped {skipped_no_support} no-text/no-support, "
          f"{skipped_no_r0} not in r=0 index)")
    return out


def warmup(ctx: dict, questions: list, n: int = 10):
    print(f"  Warming up retrievers with {n} throwaway queries...")
    _stage(f"warm-up start ({n} queries)...")
    for i, q in enumerate(questions[:n]):
        query = q.get("question", "")
        if not query:
            continue
        _stage(f"warm-up query {i}: naive...")
        ctx["naive"].retrieve(query, top_k=TOP_K)
        _stage(f"warm-up query {i}: graph...")
        ctx["graph"].retrieve(query, top_k=TOP_K)
        _stage(f"warm-up query {i}: hybrid...")
        ctx["hybrid"].retrieve(query, top_k=TOP_K)
    _stage("warm-up complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-query timing
# ─────────────────────────────────────────────────────────────────────────────

def build_pq_vector(pq_features: dict) -> np.ndarray:
    return np.array([pq_features.get(n, 0.0) for n in PER_QUERY_FEATURE_NAMES],
                     dtype=np.float32)


def time_query(query: str, ctx: dict, router, le, static_vec: np.ndarray,
              q_emb_cache: dict, selector=None):
    """One repetition: time naive / graph / hybrid alone, then actually
    execute the cascade reusing the naive result from this same repetition
    (no second naive call)."""
    embedder = ctx["embedder"]

    t0 = time.perf_counter()
    naive_result = ctx["naive"].retrieve(query, top_k=TOP_K)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph_result = ctx["graph"].retrieve(query, top_k=TOP_K)
    t_graph = time.perf_counter() - t0

    t0 = time.perf_counter()
    hybrid_result = ctx["hybrid"].retrieve(query, top_k=TOP_K)
    t_hybrid = time.perf_counter() - t0

    t_retrieve_all = t_naive + t_graph + t_hybrid   # three independent calls (historical baseline)

    # Deployed cost of obtaining all three result sets: naive once, graph once, hybrid by fusing the
    # two (hybrid retrieval runs naive and graph internally, so t_retrieve_all double-counts them).
    t0 = time.perf_counter()
    fused_result = rrf_fuse(naive_result, graph_result, top_k=TOP_K)
    t_rrf = time.perf_counter() - t0
    t_all3 = t_naive + t_graph + t_rrf
    fused_matches_hybrid = [r["doc_id"] for r in fused_result] == [r["doc_id"] for r in hybrid_result]

    # ── Cascade: probe reused from t_naive above (not re-run) ──────────────
    t_probe = t_naive

    t0 = time.perf_counter()
    q_emb = q_emb_cache.get(query)
    if q_emb is None:
        q_emb = embedder.embed_texts([query]).astype(np.float32)[0]
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)
        q_emb_cache[query] = q_emb
    pq_features = ctx["pq_assessor"].compute(
        query=query, query_emb=q_emb, retrieved=naive_result,
        docid_to_text=ctx["docid_to_text"],
    )
    pq_vec = build_pq_vector(pq_features)
    t_features = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_full = np.concatenate([static_vec, pq_vec]).reshape(1, -1)
    pred_idx = router.predict(x_full)[0]
    decision = le.inverse_transform([pred_idx])[0]
    t_predict = time.perf_counter() - t0

    t0 = time.perf_counter()
    escalated_to = "none"
    if decision == "graph":
        graph_esc = ctx["graph"].retrieve(query, top_k=TOP_K)
        escalated_to = "graph"
    elif decision == "hybrid":
        graph_esc = ctx["graph"].retrieve(query, top_k=TOP_K)
        _ = rrf_fuse(naive_result, graph_esc, top_k=TOP_K)
        escalated_to = "hybrid"
    # decision == "naive" (or any unexpected label, e.g. "cannot_answer"):
    # no further retrieval.
    t_escalation = time.perf_counter() - t0

    t_cascade = t_probe + t_features + t_predict + t_escalation

    # ── Retrieve-all selector: all three result sets already timed above; add the cost of
    #    computing result-set features for graph and hybrid and the three regressor predictions.
    t_sel_features = t_sel_predict = 0.0; sel_decision = "n/a"
    if selector is not None:
        regs, margin = selector
        t0 = time.perf_counter()
        g_feats = ctx["pq_assessor"].compute(query=query, query_emb=q_emb, retrieved=graph_result,
                                             docid_to_text=ctx["docid_to_text"])
        h_feats = ctx["pq_assessor"].compute(query=query, query_emb=q_emb, retrieved=fused_result,
                                             docid_to_text=ctx["docid_to_text"])
        ra = list(build_pq_vector(g_feats)) + list(build_pq_vector(h_feats))
        for n in RA_DIFF_FEATURES:
            nv, gv, hv = pq_features.get(n, 0.0), g_feats.get(n, 0.0), h_feats.get(n, 0.0)
            ra += [nv - gv, nv - hv, gv - hv]
        x_ra = np.concatenate([static_vec, pq_vec, np.array(ra, dtype=np.float32)]).reshape(1, -1)
        t_sel_features = time.perf_counter() - t0
        t0 = time.perf_counter()
        pv = np.array([regs[p].predict(x_ra)[0] for p in PARADIGMS])
        j = int(np.argmax(pv)); n_idx = PARADIGMS.index("naive")
        sel_decision = PARADIGMS[j] if pv[j] - pv[n_idx] > margin else "naive"
        t_sel_predict = time.perf_counter() - t0
    t_selector = t_all3 + t_sel_features + t_sel_predict            # deployed selector cost
    t_selector_3calls = t_retrieve_all + t_sel_features + t_sel_predict   # if the three were run independently

    return {
        "t_all3": t_all3, "t_rrf": t_rrf, "fused_matches_hybrid": fused_matches_hybrid,
        "t_selector": t_selector, "t_selector_3calls": t_selector_3calls,
        "t_selector_features": t_sel_features,
        "t_selector_predict": t_sel_predict, "selector_decision": sel_decision,
        "t_naive": t_naive, "t_graph": t_graph, "t_hybrid": t_hybrid,
        "t_retrieve_all": t_retrieve_all, "t_cascade": t_cascade,
        "t_cascade_probe": t_probe, "t_cascade_features": t_features,
        "t_cascade_predict": t_predict, "t_cascade_escalation": t_escalation,
        "decision": decision, "escalated_to": escalated_to,
    }


METRIC_KEYS = ["t_naive", "t_graph", "t_hybrid", "t_retrieve_all", "t_cascade",
               "t_cascade_probe", "t_cascade_features", "t_cascade_predict",
               "t_cascade_escalation", "t_all3", "t_rrf", "t_selector", "t_selector_3calls",
               "t_selector_features", "t_selector_predict"]


def run_query(qid, query, q_type, ctx, router, le, static_vec, q_emb_cache, selector=None):
    reps = [time_query(query, ctx, router, le, static_vec, q_emb_cache, selector)
            for _ in range(N_REPS)]
    medians = {k: float(np.median([r[k] for r in reps])) for k in METRIC_KEYS}
    decisions = [r["decision"] for r in reps]
    decision = Counter(decisions).most_common(1)[0][0]
    consistent = len(set(decisions)) == 1
    sel_decision = Counter([r["selector_decision"] for r in reps]).most_common(1)[0][0]
    return medians, decision, consistent, sel_decision


# ─────────────────────────────────────────────────────────────────────────────
# Machine info
# ─────────────────────────────────────────────────────────────────────────────

def get_machine_info() -> dict:
    chip = "unknown"
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        pass

    torch_device = "unknown"
    try:
        import torch
        if torch.backends.mps.is_available():
            torch_device = "mps"
        elif torch.cuda.is_available():
            torch_device = "cuda"
        else:
            torch_device = "cpu"
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "chip": chip,
        "torch_device": torch_device,
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name: str, limit: int = None) -> dict:
    print(f"\n{'#'*65}")
    print(f"  Latency Evaluation: {dataset_name}")
    print(f"{'#'*65}")

    ctx = setup_corpus(dataset_name)

    _stage("loading retrieval results (40-condition instance set)...")
    results_raw = load_retrieval_results(dataset_name)
    _stage("retrieval results loaded; building r=0 feature lookup (build_dataset)...")
    r0_index = r0_feature_lookup(dataset_name, results_raw)
    _stage("r=0 feature lookup built; training router (build_dataset + XGBClassifier.fit)...")
    router, le, _train_rows = train_router(dataset_name)
    _stage("router trained; training retrieve-all selector (3 x XGBRegressor)...")
    selector = train_selector(dataset_name, _train_rows)
    _stage("selector trained; loading eval queries...")

    questions = load_eval_queries(dataset_name, ctx, r0_index, limit=limit)
    _stage("eval queries loaded; starting warm-up...")
    warmup(ctx, questions, n=10)
    _stage("warm-up done; entering per-query timing loop.")

    q_emb_cache: dict = {}
    per_query = []
    decision_counts = Counter(); selector_counts = Counter()

    for q in tqdm(questions, desc=f"  latency {dataset_name}"):
        qid = q.get("id", "")
        query = q.get("question", "")
        q_type = q.get("query_type", "unknown")
        row = r0_index[qid]
        static_vec = np.concatenate(
            [row["query_vec"], row["corpus_vec"], row["quality_vec"]]
        )

        medians, decision, consistent, sel_decision = run_query(
            qid, query, q_type, ctx, router, le, static_vec, q_emb_cache, selector
        )
        decision_counts[decision] += 1; selector_counts[sel_decision] += 1

        per_query.append({
            "id": qid,
            "query_type": q_type,
            **medians,
            "routing_decision": decision,
            "selector_decision": sel_decision,
            "decision_consistent_across_reps": consistent,
            "oracle_label": row.get("label"),
        })

    # ── Aggregate ────────────────────────────────────────────────────────
    def agg_stats(key):
        vals = [pq[key] for pq in per_query]
        return {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    aggregate = {k: agg_stats(k) for k in METRIC_KEYS}

    n = len(per_query)
    routing_distribution = {
        k: {"count": v, "fraction": v / n if n else 0.0}
        for k, v in decision_counts.items()
    }
    selector_distribution = {
        k: {"count": v, "fraction": v / n if n else 0.0}
        for k, v in selector_counts.items()
    }

    cascade_median = aggregate["t_cascade"]["median"]
    naive_median = aggregate["t_naive"]["median"]
    hybrid_median = aggregate["t_hybrid"]["median"]
    retrieve_all_median = aggregate["t_retrieve_all"]["median"]

    headline_ratios = {
        "t_cascade_over_t_naive": cascade_median / naive_median if naive_median else None,
        "t_cascade_over_t_retrieve_all": cascade_median / retrieve_all_median if retrieve_all_median else None,
        "t_cascade_over_t_hybrid": cascade_median / hybrid_median if hybrid_median else None,
    }

    output = {
        "dataset": dataset_name,
        "n_queries": n,
        "n_reps": N_REPS,
        "top_k": TOP_K,
        "router_config": ROUTER_CONFIG,
        "router_persisted_model_found": False,
        "hybrid_note": ("HybridRetriever.retrieve() re-runs naive internally "
                        "(measured as-implemented for t_hybrid). The cascade's "
                        "hybrid escalation path reuses the already-held naive "
                        "result and calls the RRF fusion logic directly "
                        "(rrf_fuse, extracted from HybridRetriever.retrieve), "
                        "avoiding a second naive call."),
        "machine": get_machine_info(),
        "aggregate": aggregate,
        "routing_decision_distribution": routing_distribution,
        "headline_ratios": headline_ratios,
        "selector_distribution": selector_distribution,
        "selector_median_over_t_hybrid": aggregate["t_selector"]["median"] / hybrid_median if hybrid_median else None,
        "selector_median_over_t_naive": aggregate["t_selector"]["median"] / naive_median if naive_median else None,
        "selector_median_over_t_all3": aggregate["t_selector"]["median"] / aggregate["t_all3"]["median"] if aggregate["t_all3"]["median"] else None,
        "cost_accounting": "t_all3 = naive + graph + RRF fusion (deployed cost of all three result sets); "
                           "t_retrieve_all = three independent retriever calls (historical baseline, double-counts naive and graph); "
                           "result-set features embed at most the first 2000 characters of each text (identical embeddings under MiniLM's 256-word-piece limit)",
        "router_training": "final router and selector trained on the full 40-condition set of this corpus (deployed models); "
                           "routing quality is evaluated separately under grouped-by-query cross-validation",
        "per_query": per_query,
    }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Wall-clock retrieval-cost benchmark for the cascade's "
                    "'no additional retrieval cost' claim."
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions (fixed-seed stratified sample, "
                             "same as run_retrieval_eval.py). Omit for all "
                             "queries in the r=0 condition.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path override. Defaults to "
                             "Dataset/EvaluationData/LatencyEval/{dataset}/latency.json. "
                             "Use this for test/smoke runs (e.g. --limit 20) so the "
                             "full-run output is not overwritten.")
    args = parser.parse_args()

    output = run_dataset(args.dataset, limit=args.limit)

    out_path = Path(args.out) if args.out else Path(
        f"Dataset/EvaluationData/LatencyEval/{args.dataset}/latency.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    agg = output["aggregate"]
    print(f"\n  Aggregate (n={output['n_queries']} queries, {N_REPS} reps/query):")
    for k in METRIC_KEYS:
        a = agg[k]
        print(f"    {k:<22} median={a['median']*1000:8.2f}ms  "
              f"mean={a['mean']*1000:8.2f}ms  std={a['std']*1000:8.2f}ms")
    print(f"\n  Routing decisions: {output['routing_decision_distribution']}")
    print(f"  Headline ratios: {output['headline_ratios']}")
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
