"""
run_retrieval_eval.py
=====================
Retrieval-only evaluation across all paradigms, datasets, and quality
degradation conditions. No LLM calls required -- uses ground-truth
supporting document IDs from Question.json to compute retrieval metrics.

This is the primary experiment script for Phase 3 (degradation sensitivity).

Metrics computed per (query, paradigm, quality_condition) tuple:
  - Recall@1, Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - nDCG@5, nDCG@10
  - Hit rate (whether any supporting doc was retrieved in top-K)

Output
------
Dataset/EvaluationData/RetrievalEval/{dataset}/{paradigm}/{degradation_type}_{rate}.json
Dataset/EvaluationData/RetrievalEval/{dataset}/summary.json

Usage
-----
python Run/Quality/run_retrieval_eval.py --dataset quality
python Run/Quality/run_retrieval_eval.py --dataset all --limit 200
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import math
import pickle
import argparse
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from Config.PathConfig import PathConfig
from Config.RetrieverConfig import RetrieverConfig
from Config.EmbConfig import EmbConfig
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from QualityCore import (
    QualityAssessor,
    TextTruncation, EntityVariation,
    GraphEdgeRemoval, DocumentDropout, CombinedDegradation,
    PerQueryAssessor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metric functions
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    hits = len(set(retrieved_doc_ids[:k]) & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr(retrieved_doc_ids: list, relevant_doc_ids: set) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int) -> float:
    def dcg(ids, rel, k):
        score = 0.0
        seen = set()
        for i, d in enumerate(ids[:k], 1):
            if d in rel and d not in seen:
                score += 1.0 / math.log2(i + 1)
                seen.add(d)
        return score
    actual = dcg(retrieved_doc_ids, relevant_doc_ids, k)
    ideal_ids = list(relevant_doc_ids) + [d for d in retrieved_doc_ids if d not in relevant_doc_ids]
    ideal = dcg(ideal_ids, relevant_doc_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def compute_metrics(retrieved_doc_ids: list, relevant_doc_ids: set) -> dict:
    return {
        "recall@1":  recall_at_k(retrieved_doc_ids, relevant_doc_ids, 1),
        "recall@5":  recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
        "recall@10": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10),
        "mrr":       mrr(retrieved_doc_ids, relevant_doc_ids),
        "ndcg@5":    ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
        "ndcg@10":   ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, 10),
        "hit@5":     float(any(d in relevant_doc_ids for d in retrieved_doc_ids[:5])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Naive retriever (dense vector, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

class NaiveRetriever:
    """Dense vector retrieval using the pre-built FAISS index."""

    def __init__(self, dataset_name: str, embedder: EmbeddingProcessor):
        self.embedder = embedder
        index_dir = PathConfig.get_index_path(dataset_name)
        self.index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        with open(Path(index_dir) / "metadata.json") as f:
            self.chunks = json.load(f)
        print(f"  NaiveRetriever: {len(self.chunks)} chunks loaded")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        emb = self.embedder.embed_texts([query]).astype(np.float32)
        faiss.normalize_L2(emb)
        distances, indices = self.index.search(emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "doc_id": str(chunk.get("doc_id", "")),
                "text":   chunk.get("text", ""),
                "score":  float(dist),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Graph retriever (PPR-based, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

class GraphRetriever:
    """Entity-propagation graph retrieval using Personalized PageRank."""

    def __init__(self, dataset_name: str, embedder: EmbeddingProcessor,
                 degraded_graph=None, allowed_doc_ids: set = None):
        import networkx as nx
        import pandas as pd

        self.allowed_doc_ids = allowed_doc_ids

        graph_dir = Path(PathConfig.get_graph_path(dataset_name))

        if degraded_graph is not None:
            self.graph = degraded_graph
        else:
            with open(graph_dir / "graph.gpickle", "rb") as f:
                self.graph = pickle.load(f)

        mapping_df = pd.read_parquet(graph_dir / "entity_mapping.parquet")
        col = next((c for c in mapping_df.columns if "entity" in c.lower()),
                   mapping_df.columns[0])
        self.id_to_entity = dict(enumerate(mapping_df[col].tolist()))

        self.entity_index = faiss.read_index(str(graph_dir / "entity_index.faiss"))
        self.embedder = embedder

        # Build node -> doc_id mapping from graph edge attributes
        node_to_docs_sets: dict = {}
        for u, v, data in self.graph.edges(data=True):
            doc_id = str(data.get("doc_id", ""))
            if doc_id:
                node_to_docs_sets.setdefault(u, set()).add(doc_id)
                node_to_docs_sets.setdefault(v, set()).add(doc_id)
        # Freeze to sorted lists: set iteration order for strings depends on
        # PYTHONHASHSEED, which is unset by default, making graph retrieval
        # order nondeterministic across processes. Sorting fixes the order.
        self.node_to_docs: dict = {n: sorted(docs) for n, docs in node_to_docs_sets.items()}

        print(f"  GraphRetriever: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

        # Sanity check: what fraction of doc_ids carried by the graph edges
        # actually fall inside allowed_doc_ids. Expect 1.00 at r=0.0 for every
        # operator (id formats match); under document_dropout at rate r,
        # expect approximately 1 - r.
        if self.allowed_doc_ids is not None:
            edge_doc_ids = {d for docs in self.node_to_docs.values() for d in docs}
            in_allowed = sum(1 for d in edge_doc_ids if d in self.allowed_doc_ids)
            frac = in_allowed / len(edge_doc_ids) if edge_doc_ids else 1.0
            print(f"  GraphRetriever: allowed_doc_ids sanity fraction = {frac:.2f} "
                  f"({in_allowed}/{len(edge_doc_ids)})")

    def retrieve(self, query: str, top_k: int = 10,
                 n_seeds: int = 20, ppr_alpha: float = 0.85) -> list:
        import networkx as nx

        if self.graph.number_of_edges() == 0:
            return []

        # Step 1: Find seed entities
        q_emb = self.embedder.embed_texts([query]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        k_seed = min(n_seeds, self.entity_index.ntotal)
        if k_seed == 0:
            return []
        _, indices = self.entity_index.search(q_emb, k_seed)
        seed_nodes = [int(i) for i in indices[0] if i >= 0 and i in self.graph.nodes]

        if not seed_nodes:
            return []

        # Step 2: Personalized PageRank from seeds
        personalisation = {n: 1.0 / len(seed_nodes) for n in seed_nodes}
        try:
            ppr_scores = nx.pagerank(
                self.graph, alpha=ppr_alpha,
                personalization=personalisation, max_iter=100
            )
        except Exception:
            return []

        # Step 3: Map top nodes to doc_ids
        top_nodes = sorted(ppr_scores, key=ppr_scores.get, reverse=True)[:top_k * 2]
        seen_docs = []
        seen_set = set()
        for node in top_nodes:
            for doc_id in self.node_to_docs.get(node, []):
                if self.allowed_doc_ids is not None and doc_id not in self.allowed_doc_ids:
                    continue
                if doc_id not in seen_set:
                    seen_docs.append({"doc_id": doc_id, "score": ppr_scores.get(node, 0)})
                    seen_set.add(doc_id)
            if len(seen_docs) >= top_k:
                break

        return seen_docs[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever (RRF fusion)
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """Reciprocal Rank Fusion of NaiveRetriever and GraphRetriever."""

    def __init__(self, naive: NaiveRetriever, graph: GraphRetriever):
        self.naive = naive
        self.graph = graph

    def retrieve(self, query: str, top_k: int = 10, rrf_k: int = 60) -> list:
        naive_results = self.naive.retrieve(query, top_k=top_k)
        graph_results = self.graph.retrieve(query, top_k=top_k)

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
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

DATASETS  = ["musique", "quality"]
RATES     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DEGRADERS = {
    "text_truncation":    TextTruncation,
    "entity_variation":   EntityVariation,
    "graph_edge_removal": GraphEdgeRemoval,
    "document_dropout":   DocumentDropout,
    "combined":           CombinedDegradation,
}
PARADIGMS = ["naive", "graph", "hybrid"]


def load_corpus(dataset_name: str) -> list:
    with open(PathConfig.get_corpus_path(dataset_name)) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    # Try line-by-line first (standard JSONL)
    docs = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # Skip whitespace
        while pos < len(content) and content[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, pos)
            docs.append(obj)
            pos = end
        except json.JSONDecodeError:
            # Skip to next newline and retry
            next_nl = content.find('\n', pos + 1)
            if next_nl == -1:
                break
            pos = next_nl + 1
    return docs


def load_graph(dataset_name: str):
    path = Path(PathConfig.get_graph_path(dataset_name)) / "graph.gpickle"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_questions(dataset_name: str, limit: int = None) -> list:
    with open(PathConfig.get_question_path(dataset_name)) as f:
        content = f.read().strip()
        if content.startswith("["):
            questions = json.loads(content)
        else:
            questions = [json.loads(l) for l in content.splitlines() if l.strip()]

    # Normalise query type field — musique uses "type", others use "query_type"
    for q in questions:
        if "query_type" not in q or not q["query_type"]:
            q["query_type"] = q.get("type", "unknown")

    # Stratified sample: equal numbers per query type
    if limit and len(questions) > limit:
        from collections import defaultdict
        import random
        by_type = defaultdict(list)
        for q in questions:
            by_type[q.get("query_type", "unknown")].append(q)
        per_type = limit // max(len(by_type), 1)
        sampled = []
        rng = random.Random(42)
        for qtype, qs in by_type.items():
            sampled.extend(rng.sample(qs, min(per_type, len(qs))))
        questions = sampled[:limit]
    return questions


def build_text_to_docid(corpus: list) -> dict:
    """Map context/text string -> str(doc_id) for datasets where supporting
    facts are stored as raw passage text rather than doc_id references."""
    mapping = {}
    for doc in corpus:
        doc_id = str(doc.get("id", doc.get("doc_id", "")))
        for field in ("context", "text"):
            val = doc.get(field, "")
            if val:
                mapping[val] = doc_id
    return mapping


def mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def run_dataset(dataset_name: str, limit: int = None,
                degradation_filter: str = None, rates_filter: list = None,
                paradigms_filter: set = None) -> None:
    print(f"\n{'#'*65}")
    print(f"  Retrieval Evaluation: {dataset_name}")
    print(f"{'#'*65}")

    embedder = EmbeddingProcessor()
    questions = load_questions(dataset_name, limit=limit)
    corpus    = load_corpus(dataset_name)
    graph     = load_graph(dataset_name)

    # Build text->doc_id lookup for datasets where supporting facts are
    # stored as raw passage text rather than doc_id references (e.g. musique)
    text_to_docid = build_text_to_docid(corpus)

    # doc_id -> undegraded document text. Retained only for the r = 0.0 case and
    # as a fallback; the per-condition map built inside the degradation loop is
    # what the result-set measurements actually read. Degradation writes to
    # "text" and never to "context", so a map built here can only ever hold the
    # original text -- see the per-condition map below.
    base_docid_to_text = {
        str(d.get("id", d.get("doc_id", ""))): d.get("text", d.get("context", ""))
        for d in corpus
    }

    # Corpus mean character length per document — used to normalise
    # chunk_completeness in per-query features
    corpus_mean_chars = float(np.mean([
        len(d.get("text", d.get("context", ""))) for d in corpus
    ])) if corpus else 1.0

    # Pre-compute query embeddings once (reused across all paradigms)
    print("  Pre-computing query embeddings...")
    unique_queries = list({q.get("question", "") for q in questions
                           if q.get("question", "")})
    q_emb_matrix = embedder.embed_texts(unique_queries).astype(np.float32)
    norms = np.linalg.norm(q_emb_matrix, axis=1, keepdims=True)
    q_emb_matrix = q_emb_matrix / np.where(norms == 0, 1.0, norms)
    query_emb_cache = {q: q_emb_matrix[i] for i, q in enumerate(unique_queries)}

    print(f"  Questions: {len(questions)}  |  Corpus: {len(corpus)} docs")

    summary_all = {}

    active_degraders = {
        k: v for k, v in DEGRADERS.items()
        if degradation_filter is None or k == degradation_filter
    }
    active_rates = rates_filter if rates_filter is not None else RATES

    for deg_name, DegClass in active_degraders.items():
        print(f"\n  ── Degradation: {deg_name} ──")
        deg_summary = {}

        for rate in active_rates:
            # Apply degradation
            degrader = DegClass(rate=rate, seed=42)
            deg_result = degrader.apply(corpus, graph)

            # Compute quality features on degraded corpus, passing the original
            # corpus/graph as reference so relative features equal 1.0 at baseline.
            assessor = QualityAssessor(
                dataset_name, project_root=".",
                degraded_docs=deg_result.docs,
                degraded_graph=deg_result.graph,
                reference_docs=corpus,
                reference_graph=graph,
            )
            quality_result = assessor.compute()

            # Re-index from scratch for every degradation, including
            # graph_edge_removal. The pre-built disk index (RAGCore/Chunk/
            # ChunkDo.py) always chunks "{title}\n\n{context}", even when
            # title is empty, which differs from the rebuild path below for
            # corpora with empty titles (e.g. Legal). Rebuilding in-process
            # for every operator keeps the naive/hybrid chunking identical
            # across all five degradation types.
            #
            # Key invariant: degradation classes write modified text to d["text"].
            # The original corpus field is "context". We must read "text" first so
            # we embed the degraded content, not the original.
            from RAGCore.Chunk.ChunkDo import ChunkProcessor
            chunker = ChunkProcessor()
            all_chunks, all_texts = [], []
            for doc in deg_result.docs:
                doc_id = str(doc.get("id", doc.get("doc_id", "")))
                title  = doc.get("title", "")
                # Degradation classes write to "text"; fall back to "context"
                # for docs that only carry the original corpus field.
                body   = doc.get("text", doc.get("context", ""))
                full_text = f"{title}\n\n{body}" if title else body
                if not full_text.strip():
                    continue
                for ct in chunker.chunk_text(full_text):
                    all_chunks.append({"doc_id": doc_id, "text": ct})
                    all_texts.append(ct)

            if all_texts:
                batch_size = 128
                emb_parts = []
                for i in range(0, len(all_texts), batch_size):
                    emb_parts.append(
                        embedder.embed_texts(all_texts[i:i + batch_size]).astype(np.float32)
                    )
                emb_matrix = np.vstack(emb_parts)
                faiss.normalize_L2(emb_matrix)
                temp_index = faiss.IndexFlatIP(emb_matrix.shape[1])
                temp_index.add(emb_matrix)
                naive = NaiveRetriever.__new__(NaiveRetriever)
                naive.embedder = embedder
                naive.index    = temp_index
                naive.chunks   = all_chunks
            else:
                naive = NaiveRetriever(dataset_name, embedder)

            needs_graph = paradigms_filter is None or bool(
                paradigms_filter & {"graph", "hybrid"}
            )

            graph_ret = None
            if needs_graph:
                # Graph retriever with degraded graph. allowed_doc_ids restricts
                # returned doc_ids to documents that survived this degradation
                # step -- the graph itself is not rebuilt (see DegradationDo),
                # so without this filter a dropped document's doc_id can still
                # be reachable via a surviving edge and get credited. This is a
                # no-op for the three operators that do not drop documents.
                allowed_doc_ids = {
                    str(d.get("id", d.get("doc_id", ""))) for d in deg_result.docs
                }
                try:
                    graph_ret = GraphRetriever(dataset_name, embedder,
                                               degraded_graph=deg_result.graph,
                                               allowed_doc_ids=allowed_doc_ids)
                except Exception as e:
                    print(f"    Warning: GraphRetriever failed ({e}), skipping graph/hybrid")
                    graph_ret = None

            hybrid = HybridRetriever(naive, graph_ret) if graph_ret else None

            retrievers = {}
            if paradigms_filter is None or "naive" in paradigms_filter:
                retrievers["naive"] = naive
            if graph_ret and (paradigms_filter is None or "graph" in paradigms_filter):
                retrievers["graph"] = graph_ret
            if hybrid and (paradigms_filter is None or "hybrid" in paradigms_filter):
                retrievers["hybrid"] = hybrid

            # Per-query feature assessor — uses degraded corpus mean length
            deg_mean_chars = float(np.mean([
                len(d.get("text", d.get("context", "")))
                for d in deg_result.docs
            ])) if deg_result.docs else corpus_mean_chars

            # doc_id -> text as it stands UNDER THIS CONDITION. Graph and hybrid
            # return doc-level results with no chunk text, so their result-set
            # measurements resolve text through this map. It must be rebuilt per
            # condition and must prefer "text" (which degradation writes) over
            # "context" (which it never touches), or those measurements describe
            # the undegraded corpus while naive's describe the degraded one.
            deg_docid_to_text = {
                str(d.get("id", d.get("doc_id", ""))): d.get("text", d.get("context", ""))
                for d in deg_result.docs
            } if deg_result.docs else base_docid_to_text
            pq_assessor = PerQueryAssessor(
                embedder=embedder,
                corpus_mean_chars=deg_mean_chars,
                top_k=10,
            )

            rate_results = {
                "rate": rate,
                "degradation_type": deg_name,
                "quality_features": quality_result.to_dict(),
                "paradigms": {},
            }

            for paradigm_name, retriever in retrievers.items():
                query_results = []
                for q in tqdm(questions,
                               desc=f"  {deg_name} r={rate:.1f} {paradigm_name}",
                               leave=False):
                    q_id   = q.get("id", "")
                    query  = q.get("question", "")
                    q_type = q.get("query_type", "unknown")
                    raw_sf = q.get("supporting_facts", [])

                    # Resolve supporting facts to doc_ids.
                    # Dicts: use sf["doc_id"] directly.
                    # Strings: look up in text->doc_id map (musique format).
                    support = set()
                    for sf in raw_sf:
                        if isinstance(sf, dict):
                            support.add(str(sf["doc_id"]))
                        elif isinstance(sf, str):
                            doc_id = text_to_docid.get(sf)
                            if doc_id:
                                support.add(doc_id)

                    if not query or not support:
                        continue

                    retrieved = retriever.retrieve(query, top_k=10)
                    ret_ids   = [r["doc_id"] for r in retrieved]
                    metrics   = compute_metrics(ret_ids, support)

                    # Per-query retrieval quality features
                    q_emb = query_emb_cache.get(query)
                    pq_features = pq_assessor.compute(
                        query=query,
                        query_emb=q_emb,
                        retrieved=retrieved,
                        docid_to_text=deg_docid_to_text,
                    )

                    query_results.append({
                        "id":              q_id,
                        "query_type":      q_type,
                        "metrics":         metrics,
                        "pq_features":     pq_features,
                    })

                # Aggregate
                if query_results:
                    agg = {
                        metric: mean([r["metrics"][metric] for r in query_results])
                        for metric in query_results[0]["metrics"]
                    }
                    agg["n_queries"] = len(query_results)
                else:
                    agg = {"n_queries": 0}

                rate_results["paradigms"][paradigm_name] = {
                    "aggregate": agg,
                    "per_query": query_results,
                }

                print(f"    [{paradigm_name:7s}] r={rate:.1f}  "
                      f"R@5={agg.get('recall@5', 0):.3f}  "
                      f"nDCG@5={agg.get('ndcg@5', 0):.3f}  "
                      f"MRR={agg.get('mrr', 0):.3f}  "
                      f"quality={quality_result.overall:.3f}")

            # Save this rate's results. If an earlier run already produced this
            # file with a different (or fuller) set of paradigms, merge rather
            # than overwrite, so re-running a paradigm subset (e.g. --paradigms
            # naive) cannot silently delete already-correct results for the
            # paradigms that were not re-run this time.
            out_path = Path(
                f"Dataset/EvaluationData/RetrievalEval/"
                f"{dataset_name}/{deg_name}_{int(rate*100):03d}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                with open(out_path) as f:
                    existing = json.load(f)
                existing.setdefault("paradigms", {}).update(rate_results["paradigms"])
                existing["quality_features"] = rate_results["quality_features"]
                rate_results = existing
            with open(out_path, "w") as f:
                json.dump(rate_results, f, indent=2)

            deg_summary[str(rate)] = {
                "rate": rate,
                "overall_quality": quality_result.overall,
                "completeness": quality_result.completeness,
                "consistency": quality_result.consistency,
                "paradigms": {
                    p: rate_results["paradigms"][p]["aggregate"]
                    for p in rate_results["paradigms"]
                },
            }

        summary_all[deg_name] = deg_summary

    # Save dataset summary
    summary_path = Path(
        f"Dataset/EvaluationData/RetrievalEval/{dataset_name}/summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary_all, f, indent=2)

    print(f"\n  Summary saved -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval-only evaluation across paradigms and quality conditions"
    )
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions per dataset (stratified sample). "
                             "Recommended: 150 for full degradation study.")
    parser.add_argument("--degradation", type=str, default=None,
                        choices=list(DEGRADERS.keys()),
                        help="Run only this degradation type (omit for all).")
    parser.add_argument("--rates", type=str, default=None,
                        help="Comma-separated rates to test, e.g. '0.0,0.5' (omit for all).")
    parser.add_argument("--paradigms", type=str, default=None,
                        help="Comma-separated paradigms to (re-)run, e.g. 'naive' "
                             "(omit for all three). Results for paradigms not listed "
                             "are left untouched in the saved JSON, not deleted.")
    args = parser.parse_args()

    rates_filter = (
        [float(r) for r in args.rates.split(",")]
        if args.rates else None
    )
    paradigms_filter = (
        set(p.strip() for p in args.paradigms.split(","))
        if args.paradigms else None
    )

    targets = DATASETS if args.dataset == "all" else [args.dataset]

    for name in targets:
        corpus_path = Path(PathConfig.get_corpus_path(name))
        if corpus_path.exists():
            run_dataset(name, limit=args.limit,
                        degradation_filter=args.degradation,
                        rates_filter=rates_filter,
                        paradigms_filter=paradigms_filter)
        else:
            print(f"Skipping {name}: corpus not found at {corpus_path}")

    print("\nAll done. Results in Dataset/EvaluationData/RetrievalEval/")


if __name__ == "__main__":
    main()
