"""
Full-Legal confirmation (2026-09-30) of the PPR-diffusion diagnosis.

Uses the PRODUCTION GraphRetriever (same node_to_docs construction, same allowed_doc_ids logic).
Query embeddings and seed sets are cached once -- edge removal deletes edges, never nodes, so the
seed set is identical across every configuration -- and the PPR + document-collection step is then
replicated exactly as GraphRetriever.retrieve performs it.

Step 0 validates the cached fast path against production .retrieve() on 50 queries at alpha=0.85.
Writes only to Dataset/EvaluationData/Pilots/graph_alpha/.
"""
import os, sys, json, math, random, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np, faiss, networkx as nx
from pathlib import Path
from Run.Quality.run_retrieval_eval import GraphRetriever, compute_metrics
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from Config.PathConfig import PathConfig
from QualityCore.Degradation.DegradationDo import GraphEdgeRemoval

DS = "ultraDomain_legal"
ALPHAS = [0.05, 0.25, 0.50, 0.85]
RATES = [0.0, 0.7]
OUT = os.path.join("Dataset", "EvaluationData", "Pilots", "graph_alpha")
os.makedirs(OUT, exist_ok=True)


def collect(gr, ppr, top_k=10):
    """Exactly GraphRetriever.retrieve's step 3."""
    top_nodes = sorted(ppr, key=ppr.get, reverse=True)[:top_k * 2]
    seen_docs, seen = [], set()
    for node in top_nodes:
        for doc_id in gr.node_to_docs.get(node, []):
            if gr.allowed_doc_ids is not None and doc_id not in gr.allowed_doc_ids:
                continue
            if doc_id not in seen:
                seen_docs.append(doc_id); seen.add(doc_id)
        if len(seen_docs) >= top_k:
            break
    return seen_docs[:top_k], seen


def main():
    emb = EmbeddingProcessor()
    docs = [json.loads(l) for l in open(PathConfig.get_corpus_path(DS))]
    qs = [json.loads(l) for l in open(PathConfig.get_question_path(DS))]
    allowed = {str(d.get("id", d.get("doc_id", ""))) for d in docs}
    base_graph = pickle.load(open(Path(PathConfig.get_graph_path(DS)) / "graph.gpickle", "rb"))

    work = []
    for q in qs:
        gold = {str(sf["doc_id"]) for sf in q.get("supporting_facts", []) if isinstance(sf, dict)}
        if q.get("question") and gold:
            work.append((q["question"], gold))
    print(f"queries with gold: {len(work)} of {len(qs)}", flush=True)

    gr0 = GraphRetriever(DS, emb, degraded_graph=base_graph, allowed_doc_ids=allowed)

    # cache seeds once (edge removal never deletes nodes)
    seeds_all = []
    B = 128
    for i in range(0, len(work), B):
        chunk = [w[0] for w in work[i:i+B]]
        qe = emb.embed_texts(chunk).astype(np.float32); faiss.normalize_L2(qe)
        _, I = gr0.entity_index.search(qe, 20)
        for row in I:
            seeds_all.append([int(x) for x in row if x >= 0 and x in base_graph.nodes])
    print("seeds cached", flush=True)

    # ---- step 0: validate the fast path against production .retrieve() ----
    mism = 0
    for j in range(50):
        prod = [r["doc_id"] for r in gr0.retrieve(work[j][0], top_k=10, ppr_alpha=0.85)]
        s = seeds_all[j]
        ppr = nx.pagerank(base_graph, alpha=0.85,
                          personalization={n: 1.0/len(s) for n in s}, max_iter=100)
        fast, _ = collect(gr0, ppr)
        if prod != fast:
            mism += 1
    print(f"VALIDATION: fast path differs from production on {mism}/50 queries", flush=True)

    results = {}
    for rate in RATES:
        g = base_graph if rate == 0.0 else GraphEdgeRemoval(rate=rate, seed=42).apply(docs, base_graph).graph
        gr = GraphRetriever(DS, emb, degraded_graph=g, allowed_doc_ids=allowed)
        for alpha in ALPHAS:
            sc, pool, ceil = [], [], []
            for (q, gold), s in zip(work, seeds_all):
                s2 = [n for n in s if n in g.nodes]
                if not s2:
                    continue
                ppr = nx.pagerank(g, alpha=alpha,
                                  personalization={n: 1.0/len(s2) for n in s2}, max_iter=100)
                top10, allseen = collect(gr, ppr)
                sc.append(compute_metrics(top10, gold)["ndcg@5"])
                pool.append(len(allseen))
                ceil.append(compute_metrics(sorted(allseen, key=lambda d: d not in gold)[:10], gold)["ndcg@5"])
            results[f"r{rate}_a{alpha}"] = dict(
                rate=rate, alpha=alpha, n=len(sc), ndcg5=float(np.mean(sc)),
                pool=float(np.mean(pool)), ceiling=float(np.mean(ceil)))
            print(f"  r={rate}  alpha={alpha:.2f}  nDCG@5={np.mean(sc):.4f}  "
                  f"pool={np.mean(pool):.1f}  ceiling={np.mean(ceil):.4f}  n={len(sc)}", flush=True)
    json.dump(results, open(os.path.join(OUT, "legal_full_alpha.json"), "w"), indent=1)
    print("\n== EDGE-REMOVAL CHANGE BY ALPHA ==")
    for a in ALPHAS:
        d = results[f"r0.7_a{a}"]["ndcg5"] - results[f"r0.0_a{a}"]["ndcg5"]
        print(f"  alpha={a:.2f}: {results[f'r0.0_a{a}']['ndcg5']:.4f} -> "
              f"{results[f'r0.7_a{a}']['ndcg5']:.4f}   change {d:+.4f}")


if __name__ == "__main__":
    main()
