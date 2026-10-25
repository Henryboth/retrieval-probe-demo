"""
embedding_truncation_pilot.py
==============================
Pre-registered pilot (2026-09-20): does the 256-word-piece encoder truncation
change the retrieval baseline? Chunks are 512 cl100k tokens (Config/ChunkConfig.py);
all-MiniLM-L6-v2 embeds only the first 256 word-pieces (max_seq_length=256), so
naive is potentially handicapped relative to graph on long-document corpora.
Spec: skripsie-workspace/reviews/2026-09-20_embedding_truncation_pilot_prereg.md
(binding; no definition, variant, or decision rule here deviates from it).

Re-uses the harness (Run/Quality/run_retrieval_eval.py) directly rather than
re-implementing retrieval or metrics:
  NaiveRetriever, GraphRetriever, HybridRetriever, EmbeddingProcessor,
  load_corpus, load_graph, load_questions, build_text_to_docid,
  compute_metrics, DEGRADERS["combined"], mean.
Gold-set resolution (raw supporting_facts -> str(doc_id) support set) is copied
verbatim from run_dataset (run_retrieval_eval.py, lines ~529-542) since the
harness does not expose it as a standalone function.

Variants (all at r=0, `combined` degrader, rate 0.0 -- deg_result.docs is then
identical to the undegraded corpus; run_dataset's own comment confirms all five
r=0 operator files are identical):
  A. Baseline of record -- default ChunkProcessor (512/100 cl100k tokens),
     default encoder limit (max_seq_length=256).
  B. Encoder limit raised -- st_model.max_seq_length=512, same chunks as A.
  C. Chunks fit the encoder -- ChunkProcessor with chunk_size=200,
     chunk_overlap=40 (cl100k), default encoder limit; if <95% of the
     resulting chunks are <=256 MiniLM word-pieces, retry with chunk_size=180.

Methodological note on the graph half of hybrid
------------------------------------------------
GraphRetriever.retrieve() embeds the query for entity-seed search using the
SAME EmbeddingProcessor object that variant B mutates (st_model.max_seq_length).
To keep this pilot a controlled comparison of the NAIVE/embedding change only
(graph is pre-registered as "unchanged"), graph retrieval is run exactly ONCE
per corpus, before any max_seq_length mutation, and its per-query output is
cached and replayed (via a frozen shim exposing the same .retrieve() signature)
into HybridRetriever's actual RRF fusion for all three variants. The reported
"graph nDCG@5" column is always the stored value from
Dataset/EvaluationData/RetrievalEval/{ds}/combined_000.json (the file of
record), per the pre-registration; variant A additionally re-runs
GraphRetriever live (uncached, via the harness class) once, purely to verify
the harness reproduces those stored numbers (reported as a max-abs-diff, not
used downstream in any aggregate).

Output
------
Dataset/EvaluationData/Pilots/embedding_truncation/{ds}.json  (per corpus)
Dataset/EvaluationData/Pilots/embedding_truncation/summary.json

Hard rules
----------
Never writes under RetrievalEval/, RouterEval/, or LatencyEval/; never calls
run_dataset(); does not modify any existing repo file (chunk-size and
max_seq_length overrides are applied to in-memory objects only, and
max_seq_length is restored after variant B).

Run
---
PYTHONHASHSEED=42 /opt/anaconda3/envs/ragBench/bin/python \\
    Run/Quality/diagnostics/embedding_truncation_pilot.py
(cwd: RAGRouter-Bench repo root)
"""

import os
import sys
import json
import time

import numpy as np
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUALITY_DIR = os.path.dirname(SCRIPT_DIR)          # Run/Quality
sys.path.insert(0, QUALITY_DIR)

import faiss  # noqa: E402  (after sys.path insert, matches sibling diagnostics scripts)

from run_retrieval_eval import (  # noqa: E402
    NaiveRetriever, GraphRetriever, HybridRetriever, EmbeddingProcessor,
    load_corpus, load_graph, load_questions, build_text_to_docid,
    compute_metrics, DEGRADERS, mean,
)
from RAGCore.Chunk.ChunkDo import ChunkProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SPEC_FILE = (
    "/Users/henrybothwell/Desktop/skripsie-workspace/reviews/"
    "2026-09-20_embedding_truncation_pilot_prereg.md"
)

DATASETS = ["quality", "ultraDomain_legal", "musique"]
TOP_K = 10
RRF_K = 60  # HybridRetriever's default rrf_k, matches the harness call site
RECORD_DIR = Path("Dataset/EvaluationData/RetrievalEval")
OUT_DIR = Path("Dataset/EvaluationData/Pilots/embedding_truncation")

_MINI_TOKENIZER = None


def mini_tokenizer():
    global _MINI_TOKENIZER
    if _MINI_TOKENIZER is None:
        _MINI_TOKENIZER = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _MINI_TOKENIZER


def wordpiece_coverage(texts: list, threshold: int = 256) -> float:
    """Share of texts whose MiniLM tokenisation is <= threshold word-pieces."""
    if not texts:
        return 0.0
    tok = mini_tokenizer()
    lengths = [len(tok.encode(t)) for t in texts]
    return sum(1 for l in lengths if l <= threshold) / len(lengths)


def resolve_support(raw_sf: list, text_to_docid: dict) -> set:
    """Verbatim copy of the gold-resolution logic in run_dataset
    (run_retrieval_eval.py, lines ~529-542)."""
    support = set()
    for sf in raw_sf:
        if isinstance(sf, dict):
            support.add(str(sf["doc_id"]))
        elif isinstance(sf, str):
            doc_id = text_to_docid.get(sf)
            if doc_id:
                support.add(doc_id)
    return support


def load_record(ds: str):
    path = RECORD_DIR / ds / "combined_000.json"
    with open(path) as f:
        d = json.load(f)
    naive_mean_record = d["paradigms"]["naive"]["aggregate"]["ndcg@5"]
    graph_per_query_record = {
        q["id"]: q["metrics"]["ndcg@5"] for q in d["paradigms"]["graph"]["per_query"]
    }
    graph_mean_record = d["paradigms"]["graph"]["aggregate"]["ndcg@5"]
    return d, naive_mean_record, graph_per_query_record, graph_mean_record


def chunk_corpus(deg_docs: list, chunker: ChunkProcessor):
    """Reproduces run_dataset's chunking loop (run_retrieval_eval.py, lines
    ~434-448) verbatim, parameterised by an already-configured chunker."""
    all_chunks, all_texts = [], []
    for doc in deg_docs:
        doc_id = str(doc.get("id", doc.get("doc_id", "")))
        title = doc.get("title", "")
        body = doc.get("text", doc.get("context", ""))
        full_text = f"{title}\n\n{body}" if title else body
        if not full_text.strip():
            continue
        for ct in chunker.chunk_text(full_text):
            all_chunks.append({"doc_id": doc_id, "text": ct})
            all_texts.append(ct)
    return all_chunks, all_texts


def embed_chunks(all_texts: list, embedder: EmbeddingProcessor, batch_size: int = 128):
    """Reproduces run_dataset's embedding + FAISS-build loop (lines ~450-460)."""
    if not all_texts:
        raise RuntimeError("no chunks produced (unexpected at rate 0.0)")
    emb_parts = []
    for i in range(0, len(all_texts), batch_size):
        emb_parts.append(
            embedder.embed_texts(all_texts[i:i + batch_size]).astype(np.float32)
        )
    emb_matrix = np.vstack(emb_parts)
    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)
    return index


def make_naive(all_chunks: list, index, embedder: EmbeddingProcessor) -> NaiveRetriever:
    naive = NaiveRetriever.__new__(NaiveRetriever)
    naive.embedder = embedder
    naive.index = index
    naive.chunks = all_chunks
    return naive


def build_variant_index(deg_docs: list, chunker: ChunkProcessor, embedder: EmbeddingProcessor):
    t0 = time.perf_counter()
    all_chunks, all_texts = chunk_corpus(deg_docs, chunker)
    index = embed_chunks(all_texts, embedder)
    elapsed = time.perf_counter() - t0
    naive = make_naive(all_chunks, index, embedder)
    return naive, len(all_chunks), elapsed


class FrozenGraphRetriever:
    """Shim exposing GraphRetriever's .retrieve(query, top_k) signature but
    returning cached, pre-computed results -- see module docstring for why
    the graph half of hybrid must be frozen across variants."""

    def __init__(self, cache: dict):
        self.cache = cache

    def retrieve(self, query: str, top_k: int = TOP_K, **kw) -> list:
        return self.cache.get(query, [])


def run_variant(label: str, resolved_questions: list, naive: NaiveRetriever,
                 graph_cache: dict, graph_per_query_record: dict) -> dict:
    hybrid = HybridRetriever(naive, FrozenGraphRetriever(graph_cache))
    per_query = []
    for q_id, query, support in tqdm(resolved_questions, desc=f"    variant {label}", leave=False):
        naive_results = naive.retrieve(query, top_k=TOP_K)
        naive_ids = [r["doc_id"] for r in naive_results]
        naive_ndcg = compute_metrics(naive_ids, support)["ndcg@5"]

        hybrid_results = hybrid.retrieve(query, top_k=TOP_K)
        hybrid_ids = [r["doc_id"] for r in hybrid_results]
        hybrid_ndcg = compute_metrics(hybrid_ids, support)["ndcg@5"]

        graph_ndcg = graph_per_query_record.get(q_id)
        candidates = [naive_ndcg, hybrid_ndcg] + ([graph_ndcg] if graph_ndcg is not None else [])
        oracle = max(candidates)
        routing_matters = (oracle - naive_ndcg) > 1e-12

        per_query.append({
            "id": q_id,
            "naive": naive_ndcg,
            "hybrid": hybrid_ndcg,
            "graph": graph_ndcg,
            "oracle": oracle,
            "routing_matters": bool(routing_matters),
        })

    return {
        "n_queries": len(per_query),
        "naive_ndcg5_mean": mean([r["naive"] for r in per_query]),
        "hybrid_ndcg5_mean": mean([r["hybrid"] for r in per_query]),
        "graph_ndcg5_mean": mean([r["graph"] for r in per_query if r["graph"] is not None]),
        "oracle_cr": mean([r["oracle"] for r in per_query]),
        "routing_matters_share": mean([1.0 if r["routing_matters"] else 0.0 for r in per_query]),
        "per_query": per_query,
    }


def run_pilot(ds: str) -> dict:
    print(f"\n{'=' * 70}\n  {ds}\n{'=' * 70}")

    record, naive_mean_record, graph_per_query_record, graph_mean_record = load_record(ds)

    embedder = EmbeddingProcessor()
    default_max_seq_length = embedder.st_model.max_seq_length
    print(f"  default encoder max_seq_length = {default_max_seq_length}")

    questions = load_questions(ds)
    corpus = load_corpus(ds)
    graph = load_graph(ds)
    text_to_docid = build_text_to_docid(corpus)

    degrader = DEGRADERS["combined"](rate=0.0, seed=42)
    deg_result = degrader.apply(corpus, graph)
    allowed_doc_ids = {str(d.get("id", d.get("doc_id", ""))) for d in deg_result.docs}

    resolved_questions = []
    for q in questions:
        q_id = q.get("id", "")
        query = q.get("question", "")
        support = resolve_support(q.get("supporting_facts", []), text_to_docid)
        if not query or not support:
            continue
        resolved_questions.append((q_id, query, support))
    print(f"  questions: {len(questions)} total, {len(resolved_questions)} resolved "
          f"(record naive n_queries = {record['paradigms']['naive']['aggregate']['n_queries']})")

    # ---- graph retriever built once, BEFORE any embedder mutation ----------
    graph_ret = GraphRetriever(ds, embedder, degraded_graph=deg_result.graph,
                                allowed_doc_ids=allowed_doc_ids)
    graph_cache, graph_recomputed_ndcg = {}, {}
    for q_id, query, support in tqdm(resolved_questions, desc=f"  {ds} graph (once, variant A check)"):
        g_results = graph_ret.retrieve(query, top_k=TOP_K)
        graph_cache[query] = g_results
        ret_ids = [r["doc_id"] for r in g_results]
        graph_recomputed_ndcg[q_id] = compute_metrics(ret_ids, support)["ndcg@5"]

    graph_diffs = [
        abs(graph_recomputed_ndcg[q_id] - graph_per_query_record[q_id])
        for q_id, _, _ in resolved_questions if q_id in graph_per_query_record
    ]
    n_graph_matched = len(graph_diffs)
    graph_max_abs_diff = max(graph_diffs) if graph_diffs else float("nan")
    print(f"  variant A graph reproduction: max|diff| = {graph_max_abs_diff:.6f} "
          f"over {n_graph_matched}/{len(resolved_questions)} matched queries")

    # ---- variant A: baseline of record --------------------------------------
    chunker_a = ChunkProcessor()  # defaults: 512 cl100k tokens, overlap 100
    naive_a, n_chunks_a, t_a = build_variant_index(deg_result.docs, chunker_a, embedder)
    res_a = run_variant("A", resolved_questions, naive_a, graph_cache, graph_per_query_record)
    res_a["n_chunks"], res_a["index_seconds"] = n_chunks_a, t_a
    print(f"  A: n_chunks={n_chunks_a}  index={t_a:.1f}s  naive={res_a['naive_ndcg5_mean']:.4f}  "
          f"hybrid={res_a['hybrid_ndcg5_mean']:.4f}  oracle_cr={res_a['oracle_cr']:.4f}")

    naive_a_matches_record = round(res_a["naive_ndcg5_mean"], 4) == round(naive_mean_record, 4)
    print(f"  variant A naive nDCG@5: computed={res_a['naive_ndcg5_mean']:.4f}  "
          f"record={naive_mean_record:.4f}  matches_4dp={naive_a_matches_record}")

    verification = {
        "variant_a_naive_mean_computed": res_a["naive_ndcg5_mean"],
        "variant_a_naive_mean_record": naive_mean_record,
        "variant_a_naive_matches_record_4dp": naive_a_matches_record,
        "variant_a_graph_max_abs_diff": graph_max_abs_diff,
        "variant_a_graph_n_matched": n_graph_matched,
        "default_max_seq_length": default_max_seq_length,
    }

    if not naive_a_matches_record:
        print(f"  !!! STOPPING {ds}: variant A does not reproduce the naive nDCG@5 of record "
              f"to 4dp -- not proceeding to B/C for this corpus, per pre-registration.")
        return {
            "dataset": ds,
            "n_questions_total": len(questions),
            "n_questions_resolved": len(resolved_questions),
            "top_k": TOP_K, "rrf_k": RRF_K,
            "record_file": str(RECORD_DIR / ds / "combined_000.json"),
            "verification": verification,
            "variants": {"A": res_a},
            "stopped_before_bc": True,
        }

    # ---- variant B: encoder limit raised -------------------------------------
    embedder.st_model.max_seq_length = 512
    try:
        chunker_b = ChunkProcessor()  # same defaults as A -- same chunks
        naive_b, n_chunks_b, t_b = build_variant_index(deg_result.docs, chunker_b, embedder)
        res_b = run_variant("B", resolved_questions, naive_b, graph_cache, graph_per_query_record)
        res_b["n_chunks"], res_b["index_seconds"] = n_chunks_b, t_b
    finally:
        embedder.st_model.max_seq_length = default_max_seq_length
    print(f"  B: n_chunks={n_chunks_b}  index={t_b:.1f}s  naive={res_b['naive_ndcg5_mean']:.4f}  "
          f"hybrid={res_b['hybrid_ndcg5_mean']:.4f}  oracle_cr={res_b['oracle_cr']:.4f}")

    # ---- variant C: chunks fit the encoder -----------------------------------
    chunker_c = ChunkProcessor()
    chunker_c.chunk_size = 200
    chunker_c.chunk_overlap = 40
    t0 = time.perf_counter()
    all_chunks_c, all_texts_c = chunk_corpus(deg_result.docs, chunker_c)
    coverage_200 = wordpiece_coverage(all_texts_c)
    coverage_180 = None
    chunk_size_used = 200
    print(f"  C (chunk_size=200): n_chunks={len(all_chunks_c)}  "
          f"coverage<=256wp={coverage_200:.4f}")
    if coverage_200 < 0.95:
        chunker_c.chunk_size = 180
        all_chunks_c, all_texts_c = chunk_corpus(deg_result.docs, chunker_c)
        coverage_180 = wordpiece_coverage(all_texts_c)
        chunk_size_used = 180
        print(f"  C (chunk_size=180, retry): n_chunks={len(all_chunks_c)}  "
              f"coverage<=256wp={coverage_180:.4f}")
    index_c = embed_chunks(all_texts_c, embedder)
    naive_c = make_naive(all_chunks_c, index_c, embedder)
    t_c = time.perf_counter() - t0
    res_c = run_variant("C", resolved_questions, naive_c, graph_cache, graph_per_query_record)
    res_c["n_chunks"], res_c["index_seconds"] = len(all_chunks_c), t_c
    res_c["chunk_size_used"] = chunk_size_used
    res_c["wordpiece_coverage_200"] = coverage_200
    res_c["wordpiece_coverage_180"] = coverage_180
    print(f"  C: n_chunks={len(all_chunks_c)}  index={t_c:.1f}s  naive={res_c['naive_ndcg5_mean']:.4f}  "
          f"hybrid={res_c['hybrid_ndcg5_mean']:.4f}  oracle_cr={res_c['oracle_cr']:.4f}")

    return {
        "dataset": ds,
        "n_questions_total": len(questions),
        "n_questions_resolved": len(resolved_questions),
        "top_k": TOP_K, "rrf_k": RRF_K,
        "record_file": str(RECORD_DIR / ds / "combined_000.json"),
        "verification": verification,
        "variants": {"A": res_a, "B": res_b, "C": res_c},
        "stopped_before_bc": False,
    }


def r4(x):
    return round(x, 4) if x is not None else None


def build_summary(results: dict) -> dict:
    summary = {}
    for ds, out in results.items():
        if out.get("stopped_before_bc"):
            summary[ds] = {"stopped_before_bc": True,
                            "naive_A": r4(out["variants"]["A"]["naive_ndcg5_mean"])}
            continue
        a, b, c = out["variants"]["A"], out["variants"]["B"], out["variants"]["C"]
        summary[ds] = {
            "naive_A": r4(a["naive_ndcg5_mean"]), "naive_B": r4(b["naive_ndcg5_mean"]), "naive_C": r4(c["naive_ndcg5_mean"]),
            "delta_naive_B_minus_A": r4(b["naive_ndcg5_mean"] - a["naive_ndcg5_mean"]),
            "delta_naive_C_minus_A": r4(c["naive_ndcg5_mean"] - a["naive_ndcg5_mean"]),
            "hybrid_A": r4(a["hybrid_ndcg5_mean"]), "hybrid_B": r4(b["hybrid_ndcg5_mean"]), "hybrid_C": r4(c["hybrid_ndcg5_mean"]),
            "delta_hybrid_B_minus_A": r4(b["hybrid_ndcg5_mean"] - a["hybrid_ndcg5_mean"]),
            "delta_hybrid_C_minus_A": r4(c["hybrid_ndcg5_mean"] - a["hybrid_ndcg5_mean"]),
            "oracle_cr_A": r4(a["oracle_cr"]), "oracle_cr_B": r4(b["oracle_cr"]), "oracle_cr_C": r4(c["oracle_cr"]),
            "delta_oracle_cr_B_minus_A": r4(b["oracle_cr"] - a["oracle_cr"]),
            "delta_oracle_cr_C_minus_A": r4(c["oracle_cr"] - a["oracle_cr"]),
            "routing_matters_A": r4(a["routing_matters_share"]), "routing_matters_B": r4(b["routing_matters_share"]),
            "routing_matters_C": r4(c["routing_matters_share"]),
            "delta_routing_matters_B_minus_A": r4(b["routing_matters_share"] - a["routing_matters_share"]),
            "delta_routing_matters_C_minus_A": r4(c["routing_matters_share"] - a["routing_matters_share"]),
            "chunk_size_used_C": c["chunk_size_used"],
            "wordpiece_coverage_200": r4(c["wordpiece_coverage_200"]),
            "wordpiece_coverage_180": r4(c["wordpiece_coverage_180"]),
            "n_chunks_A": a["n_chunks"], "n_chunks_B": b["n_chunks"], "n_chunks_C": c["n_chunks"],
            "index_seconds_A": round(a["index_seconds"], 1),
            "index_seconds_B": round(b["index_seconds"], 1),
            "index_seconds_C": round(c["index_seconds"], 1),
        }

    gate_datasets = ("quality", "ultraDomain_legal")
    can_decide = all(
        ds in summary and not summary[ds].get("stopped_before_bc") for ds in gate_datasets
    )
    if can_decide:
        decision_ok = all(
            abs(summary[ds]["delta_naive_B_minus_A"]) < 0.005
            and abs(summary[ds]["delta_naive_C_minus_A"]) < 0.005
            for ds in gate_datasets
        )
        decision = "STANDS" if decision_ok else "ESCALATE"
    else:
        decision = "ESCALATE"

    summary["decision"] = decision
    summary["decision_rule"] = (
        "STANDS if |naive_B-naive_A| < 0.005 AND |naive_C-naive_A| < 0.005 on BOTH "
        "quality and ultraDomain_legal; else ESCALATE."
    )
    summary["spec_file"] = SPEC_FILE
    return summary


def print_table(results: dict, summary: dict):
    print(f"\n{'=' * 100}")
    print("  SUMMARY")
    print(f"{'=' * 100}")
    header = f"{'dataset':<18}{'variant':<9}{'n_chunks':>9}{'idx_s':>8}{'naive':>8}{'hybrid':>8}{'graph':>8}{'oracle_CR':>11}{'route%':>8}"
    print(header)
    print("-" * len(header))
    for ds in DATASETS:
        out = results.get(ds)
        if out is None:
            continue
        if out.get("stopped_before_bc"):
            a = out["variants"]["A"]
            print(f"{ds:<18}{'A':<9}{a['n_chunks']:>9}{a['index_seconds']:>8.1f}"
                  f"{a['naive_ndcg5_mean']:>8.4f}{a['hybrid_ndcg5_mean']:>8.4f}"
                  f"{a['graph_ndcg5_mean']:>8.4f}{a['oracle_cr']:>11.4f}{a['routing_matters_share']*100:>7.1f}%")
            print(f"{ds:<18}  STOPPED before B/C -- variant A did not reproduce record naive nDCG@5")
            continue
        for label in ("A", "B", "C"):
            v = out["variants"][label]
            print(f"{ds:<18}{label:<9}{v['n_chunks']:>9}{v['index_seconds']:>8.1f}"
                  f"{v['naive_ndcg5_mean']:>8.4f}{v['hybrid_ndcg5_mean']:>8.4f}"
                  f"{v['graph_ndcg5_mean']:>8.4f}{v['oracle_cr']:>11.4f}{v['routing_matters_share']*100:>7.1f}%")
    print(f"\n  decision = {summary['decision']}")
    print(f"  rule     = {summary['decision_rule']}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in DATASETS:
        t0 = time.perf_counter()
        try:
            out = run_pilot(ds)
        except Exception as e:
            print(f"  !!! {ds} FAILED: {e!r}")
            import traceback
            traceback.print_exc()
            continue
        results[ds] = out
        with open(OUT_DIR / f"{ds}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"  {ds} done in {time.perf_counter() - t0:.1f}s -> {OUT_DIR / f'{ds}.json'}")

    summary = build_summary(results)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_table(results, summary)
    print(f"\n  summary -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
