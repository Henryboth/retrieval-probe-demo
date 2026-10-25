"""
compute_graph_probe.py
=======================
Graph-side probe (GP): a cheap, PPR-free signal of whether the knowledge
graph has anything to offer a query. In milliseconds, WITHOUT running PPR,
this repeats only the seeding step of GraphRetriever.retrieve (query
embedding -> FAISS search over the entity index -> up to 20 seed nodes,
see Run/Quality/run_retrieval_eval.py GraphRetriever.retrieve step 1) and
computes cheap statistics of the seed set on the condition's graph.

Feature vector layout (8 dimensions, all in [0, 1])
----------------------------------------------------
  gp_seed_sim_top1      max cosine similarity of query to a seed entity
  gp_seed_sim_mean      mean cosine similarity over the seeds
  gp_seed_degree_mean   mean degree of seeds / max degree in the graph
  gp_seed_degree_max    max seed degree / max degree
  gp_seed_edge_share    edges among seed pairs (in G) / C(n_seeds, 2)
  gp_seed_components    1 - (components of seed-induced subgraph - 1) / (n_seeds - 1)
  gp_seed_docs          |union node_to_docs[seed]| / 10, capped at 1
  gp_seed_lcc_share     share of seeds inside the graph's largest connected component

If no seeds are found, all features are 0.0. Both guards that precede seeding in
GraphRetriever.retrieve are replicated exactly: a zero-edge graph short-circuits
to all-zero GP features before any FAISS search, and k_seed = min(20, ntotal).

Condition invariance (see QualityCore/Degradation/DegradationDo.py): text_truncation,
entity_variation and document_dropout pass the graph through unchanged, so gp_vec is
identical to its r=0 value for those three operators; only graph_edge_removal and
combined actually change the graph (recorded per-condition as "graph_varies").

t_gp_ms is search + statistics only. The query embedding is precomputed once per
unique query text and reused across every condition/paradigm -- as a DEPLOYED
single-embed router would -- so it is NOT counted as part of this probe's cost
(today's offline eval scripts embed the query separately per paradigm instead).

Output
------
Dataset/EvaluationData/RetrievalEval/{dataset}/graph_probe_{degradation_type}_{rate:03d}.json
  = {"dataset", "degradation_type", "rate", "n_seeds": 20, "graph_varies": bool,
     "features": {query_id: {name: value}}, "t_gp_ms_median"}

Usage
-----
python Run/Quality/compute_graph_probe.py --dataset quality
python Run/Quality/compute_graph_probe.py --dataset quality --rates 0.0 --limit 5
python Run/Quality/compute_graph_probe.py --dataset musique
python Run/Quality/compute_graph_probe.py --dataset ultraDomain_legal
# (invoke each dataset by name -- --dataset all silently drops datasets in the
#  router/ablation scripts and is avoided here too for consistency)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import json
import argparse
import itertools
from pathlib import Path
from time import perf_counter

import numpy as np
import faiss
import pandas as pd
import networkx as nx

from Config.PathConfig import PathConfig
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from Run.Quality.run_retrieval_eval import DEGRADERS, load_corpus, load_graph, load_questions


DATASETS = ["quality", "musique", "ultraDomain_legal"]
N_SEEDS  = 20

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

# Operators whose DEGRADERS[op](rate=rate, seed=42).apply(corpus, graph).graph is
# passed straight through unchanged (see QualityCore/Degradation/DegradationDo.py:
# TextTruncation, EntityVariation and DocumentDropout only touch docs, never graph).
# It is therefore identical to the r=0 graph at every rate; computed once and reused.
GRAPH_STATIC_OPERATORS = {"text_truncation", "entity_variation", "document_dropout"}
GRAPH_VARYING_OPERATORS = {"graph_edge_removal", "combined"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph / entity-index loading (mirrors GraphRetriever.__init__ / .retrieve step 1)
# ─────────────────────────────────────────────────────────────────────────────

def load_entity_index(dataset_name: str):
    graph_dir = Path(PathConfig.get_graph_path(dataset_name))
    entity_index = faiss.read_index(str(graph_dir / "entity_index.faiss"))
    mapping_df = pd.read_parquet(graph_dir / "entity_mapping.parquet")
    col = next((c for c in mapping_df.columns if "entity" in c.lower()),
               mapping_df.columns[0])
    id_to_entity = dict(enumerate(mapping_df[col].tolist()))
    return entity_index, id_to_entity


def build_node_to_docs(graph: nx.Graph) -> dict:
    node_to_docs_sets: dict = {}
    for u, v, data in graph.edges(data=True):
        doc_id = str(data.get("doc_id", ""))
        if doc_id:
            node_to_docs_sets.setdefault(u, set()).add(doc_id)
            node_to_docs_sets.setdefault(v, set()).add(doc_id)
    return {n: sorted(docs) for n, docs in node_to_docs_sets.items()}


def compute_graph_stats(graph: nx.Graph) -> dict:
    """Precompute per-graph statistics reused across every query in a condition."""
    deg = dict(graph.degree())
    max_degree = max(deg.values()) if deg else 0
    G_u = graph.to_undirected() if graph.is_directed() else graph
    components = list(nx.connected_components(G_u))
    lcc = max(components, key=len) if components else set()
    return {
        "deg":        deg,
        "max_degree": max_degree,
        "G_u":        G_u,
        "lcc":        lcc,
        "node_to_docs": build_node_to_docs(graph),
    }


def get_seed_nodes(q_emb: np.ndarray, entity_index, graph: nx.Graph, n_seeds: int = N_SEEDS):
    """Identical to GraphRetriever.retrieve step 1, incl. the two guards that
    precede seeding in GraphRetriever.retrieve (zero-edge graph -> no seeding
    at all; k_seed = min(n_seeds, entity_index.ntotal)) and `i in self.graph.nodes`,
    but also returns the FAISS inner-product similarity for each kept seed."""
    if graph.number_of_edges() == 0:
        return [], []
    k_seed = min(n_seeds, entity_index.ntotal)
    if k_seed == 0:
        return [], []
    sims, indices = entity_index.search(q_emb, k_seed)
    seed_nodes, seed_sims = [], []
    for sim, idx in zip(sims[0], indices[0]):
        if idx >= 0 and int(idx) in graph.nodes:
            seed_nodes.append(int(idx))
            seed_sims.append(float(sim))
    return seed_nodes, seed_sims


def compute_seed_features(seed_nodes: list, seed_sims: list, stats: dict, graph: nx.Graph) -> dict:
    n = len(seed_nodes)
    if n == 0:
        return {name: 0.0 for name in GP_FEATURE_NAMES}

    deg         = stats["deg"]
    max_degree  = max(stats["max_degree"], 1)
    G_u         = stats["G_u"]
    lcc         = stats["lcc"]
    node_to_docs = stats["node_to_docs"]

    sims = [float(np.clip(s, 0.0, 1.0)) for s in seed_sims]
    sim_top1 = max(sims)
    sim_mean = sum(sims) / n

    degrees = [deg.get(node, 0) for node in seed_nodes]
    degree_mean = (sum(degrees) / n) / max_degree
    degree_max  = max(degrees) / max_degree

    n_pairs = n * (n - 1) / 2
    if n_pairs > 0:
        n_edges = sum(
            1 for a, b in itertools.combinations(seed_nodes, 2)
            if graph.has_edge(a, b)
        )
        edge_share = n_edges / n_pairs
    else:
        edge_share = 0.0

    if n > 1:
        seed_subgraph = G_u.subgraph(seed_nodes)
        n_components = nx.number_connected_components(seed_subgraph)
        components_score = 1.0 - (n_components - 1) / (n - 1)
    else:
        # A single seed is trivially "all connected to itself".
        components_score = 1.0

    docs_touched = set()
    for node in seed_nodes:
        docs_touched.update(node_to_docs.get(node, []))
    docs_score = min(len(docs_touched) / 10.0, 1.0)

    lcc_share = sum(1 for node in seed_nodes if node in lcc) / n

    return {
        "gp_seed_sim_top1":    float(np.clip(sim_top1, 0.0, 1.0)),
        "gp_seed_sim_mean":    float(np.clip(sim_mean, 0.0, 1.0)),
        "gp_seed_degree_mean": float(np.clip(degree_mean, 0.0, 1.0)),
        "gp_seed_degree_max":  float(np.clip(degree_max, 0.0, 1.0)),
        "gp_seed_edge_share":  float(np.clip(edge_share, 0.0, 1.0)),
        "gp_seed_components":  float(np.clip(components_score, 0.0, 1.0)),
        "gp_seed_docs":        float(docs_score),
        "gp_seed_lcc_share":   float(np.clip(lcc_share, 0.0, 1.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition processing
# ─────────────────────────────────────────────────────────────────────────────

def process_condition(dataset_name: str, deg_name: str, rate: float,
                       corpus: list, base_graph: nx.Graph, graph_stats_cache: dict,
                       entity_index, query_emb_cache: dict, questions_by_id: dict,
                       base_edges: int, limit: int = None) -> dict:
    rate3 = int(round(rate * 100))
    retrieval_path = Path(
        f"Dataset/EvaluationData/RetrievalEval/{dataset_name}/{deg_name}_{rate3:03d}.json"
    )
    with open(retrieval_path) as f:
        retrieval_data = json.load(f)
    naive_per_query = retrieval_data.get("paradigms", {}).get("naive", {}).get("per_query", [])
    query_ids = [q["id"] for q in naive_per_query]
    if limit is not None:
        query_ids = query_ids[:limit]

    # Condition graph, exactly as run_retrieval_eval.py obtains it.
    if deg_name in GRAPH_VARYING_OPERATORS:
        cond_graph = DEGRADERS[deg_name](rate=rate, seed=42).apply(corpus, base_graph).graph
    else:
        cond_graph = base_graph

    cache_key = id(cond_graph)
    if cache_key not in graph_stats_cache:
        graph_stats_cache[cache_key] = compute_graph_stats(cond_graph)
    stats = graph_stats_cache[cache_key]

    # Edge-count verification (spec: graph_edge_removal at r=0.3 -> round(E*0.7)).
    # Computed as E - round(E*rate), exactly matching GraphEdgeRemoval.apply()'s
    # own n_remove = round(len(removable)*rate); round(E*(1-rate)) is NOT always
    # equal to this under Python's round-half-to-even at exact .5 boundaries
    # (e.g. E=12109, rate=0.5: E-round(E*0.5)=6055 but round(E*0.5)=6054).
    e = cond_graph.number_of_edges()
    if deg_name in GRAPH_VARYING_OPERATORS:
        expected = base_edges - round(base_edges * rate)
        print(f"    [check] {deg_name:<20} r={rate:.1f}: edges={e}  "
              f"expected=E-round(E*{rate:.1f})={expected}  match={e == expected}")
    else:
        print(f"    [check] {deg_name:<20} r={rate:.1f}: edges={e}  "
              f"expected(unchanged E)={base_edges}  match={e == base_edges}")

    features = {}
    times_ms = []
    for q_id in query_ids:
        query = questions_by_id.get(str(q_id))
        if not query:
            features[str(q_id)] = {name: 0.0 for name in GP_FEATURE_NAMES}
            continue
        q_emb = query_emb_cache.get(query)
        if q_emb is None:
            features[str(q_id)] = {name: 0.0 for name in GP_FEATURE_NAMES}
            continue

        t0 = perf_counter()
        seed_nodes, seed_sims = get_seed_nodes(q_emb.reshape(1, -1), entity_index, cond_graph)
        feats = compute_seed_features(seed_nodes, seed_sims, stats, cond_graph)
        t1 = perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

        features[str(q_id)] = feats

    t_gp_ms_median = float(np.median(times_ms)) if times_ms else 0.0

    # A2: text_truncation / entity_variation / document_dropout pass the graph
    # through unchanged (DegradationDo.py), so gp_vec is identical to its r=0
    # value for those three operators; only graph_edge_removal and combined
    # actually change the graph. Downstream reporting must never describe GP
    # as "responding to severity" for the constant-graph operators.
    out = {
        "dataset":          dataset_name,
        "degradation_type": deg_name,
        "rate":             rate,
        "n_seeds":          N_SEEDS,
        "graph_varies":     deg_name in GRAPH_VARYING_OPERATORS,
        "features":         features,
        # search + statistics only (embedding reused, precomputed once per
        # unique query text -- see run_dataset()); not the embedding cost.
        "t_gp_ms_median":   t_gp_ms_median,
    }
    out_path = Path(
        f"Dataset/EvaluationData/RetrievalEval/{dataset_name}/graph_probe_{deg_name}_{rate3:03d}.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name: str, rates_filter: list = None,
                 degradation_filter: str = None, limit: int = None) -> None:
    print(f"\n{'#'*65}")
    print(f"  Graph Probe: {dataset_name}")
    print(f"{'#'*65}")

    corpus = load_corpus(dataset_name)
    graph  = load_graph(dataset_name)
    if graph is None:
        print(f"  No graph for {dataset_name}, skipping.")
        return

    questions = load_questions(dataset_name)
    questions_by_id = {str(q.get("id", "")): q.get("question", "") for q in questions}

    entity_index, _ = load_entity_index(dataset_name)
    embedder = EmbeddingProcessor()

    # A6 pre-flight: query ids must be unique and non-empty before computing anything.
    ids = [q.get("id", "") for q in questions]
    n_ids, n_unique_ids = len(ids), len(set(ids))
    assert all(i != "" and i is not None for i in ids), \
        f"{dataset_name}: found empty/None query id(s) in Question.json"
    assert n_ids == n_unique_ids, \
        f"{dataset_name}: query ids are not unique ({n_ids} rows, {n_unique_ids} unique ids)"
    print(f"  Pre-flight: {n_ids} questions, {n_unique_ids} unique non-empty ids -- OK")

    # Query embeddings: computed once per unique query text, reused across every
    # condition. This mirrors a DEPLOYED single-embed router, where the query is
    # embedded once and shared by the naive and graph probes; it is NOT part of
    # the graph probe's own cost. (Today's offline eval scripts, by contrast,
    # embed the query separately per paradigm -- see run_retrieval_eval.py.)
    unique_queries = sorted({q for q in questions_by_id.values() if q})
    print(f"  Pre-computing {len(unique_queries)} unique query embeddings...")
    q_emb_matrix = embedder.embed_texts(unique_queries).astype(np.float32)
    faiss.normalize_L2(q_emb_matrix)
    query_emb_cache = {q: q_emb_matrix[i] for i, q in enumerate(unique_queries)}

    base_edges = graph.number_of_edges()
    graph_stats_cache: dict = {}

    retrieval_dir = Path(f"Dataset/EvaluationData/RetrievalEval/{dataset_name}")
    condition_files = sorted(
        p for p in retrieval_dir.glob("*.json")
        if p.stem != "summary" and not p.name.startswith("graph_probe_")
    )

    medians = []
    r0_features = None  # feature dict of the text_truncation r=0 condition, for the summary print

    for p in condition_files:
        m = re.match(r"^(.+)_(\d{3})$", p.stem)
        if not m:
            continue
        deg_name, rate3str = m.group(1), m.group(2)
        if deg_name not in DEGRADERS:
            continue
        rate = int(rate3str) / 100.0
        if degradation_filter is not None and deg_name != degradation_filter:
            continue
        if rates_filter is not None and rate not in rates_filter:
            continue

        out = process_condition(
            dataset_name, deg_name, rate, corpus, graph, graph_stats_cache,
            entity_index, query_emb_cache, questions_by_id, base_edges, limit=limit,
        )
        medians.append(out["t_gp_ms_median"])
        n_q = len(out["features"])
        print(f"    [{deg_name:<20}] r={rate:.1f}  n_queries={n_q}  "
              f"t_gp_ms_median={out['t_gp_ms_median']:.3f}")

        if deg_name == "text_truncation" and rate == 0.0:
            r0_features = out["features"]

    if medians:
        print(f"\n  {dataset_name}: median t_gp_ms across conditions = {float(np.median(medians)):.3f} ms")

    if r0_features:
        vals = {name: float(np.mean([f[name] for f in r0_features.values()]))
                for name in GP_FEATURE_NAMES}
        print(f"  Feature means at r=0 (text_truncation, n={len(r0_features)}):")
        print(f"    sim_top1={vals['gp_seed_sim_top1']:.3f}  sim_mean={vals['gp_seed_sim_mean']:.3f}  "
              f"degree_mean={vals['gp_seed_degree_mean']:.3f}  degree_max={vals['gp_seed_degree_max']:.3f}")
        print(f"    edge_share={vals['gp_seed_edge_share']:.3f}  components={vals['gp_seed_components']:.3f}")
        print(f"    docs={vals['gp_seed_docs']:.3f}  lcc_share={vals['gp_seed_lcc_share']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Graph-side (GP) probe feature computation")
    parser.add_argument("--dataset", default="all",
                        help="Dataset name (quality, musique, ultraDomain_legal) or 'all'")
    parser.add_argument("--rates", type=str, default=None,
                        help="Comma-separated rates to test, e.g. '0.0' (omit for all).")
    parser.add_argument("--degradation", type=str, default=None,
                        choices=list(DEGRADERS.keys()),
                        help="Run only this degradation type (omit for all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max queries per condition (smoke-testing only).")
    args = parser.parse_args()

    rates_filter = (
        [float(r) for r in args.rates.split(",")]
        if args.rates else None
    )

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for name in targets:
        corpus_path = Path(PathConfig.get_corpus_path(name))
        if corpus_path.exists():
            run_dataset(name, rates_filter=rates_filter,
                        degradation_filter=args.degradation, limit=args.limit)
        else:
            print(f"Skipping {name}: corpus not found at {corpus_path}")

    print("\nAll done. Graph probe results in Dataset/EvaluationData/RetrievalEval/{dataset}/graph_probe_*.json")


if __name__ == "__main__":
    main()
