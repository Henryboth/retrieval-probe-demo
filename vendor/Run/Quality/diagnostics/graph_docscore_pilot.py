"""
Pilot (2026-09-22): does ranking DOCUMENTS by PPR mass, instead of appending each node's
doc_id-sorted list until ten are held, change graph retrieval on QuALITY?

Pre-registration: reviews/2026-09-22_graph_docscore_pilot_prereg.md (skripsie-workspace).
Reads the data of record read-only; writes ONLY to Dataset/EvaluationData/Pilots/graph_docscore/.
Scope: QuALITY, operator graph_edge_removal, all eight rates. Retrieval only.
"""
import sys, os, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np, faiss
from statistics import mean
from Run.Quality.run_retrieval_eval import (
    NaiveRetriever, GraphRetriever, HybridRetriever, compute_metrics,
)
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from RAGCore.Chunk.ChunkDo import ChunkProcessor
from QualityCore.Degradation.DegradationDo import GraphEdgeRemoval
from Config.PathConfig import PathConfig

DS = "quality"
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
OUT = os.path.join("Dataset", "EvaluationData", "Pilots", "graph_docscore")
os.makedirs(OUT, exist_ok=True)


class FixedGraphRetriever(GraphRetriever):
    """Identical to GraphRetriever except that documents are ranked by the maximum PPR
    mass over the nodes that surface them, rather than appended in node order."""

    def retrieve(self, query, top_k=10, n_seeds=20, ppr_alpha=0.85):
        import networkx as nx
        if self.graph.number_of_edges() == 0:
            return []
        q_emb = self.embedder.embed_texts([query]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        k_seed = min(n_seeds, self.entity_index.ntotal)
        if k_seed == 0:
            return []
        _, indices = self.entity_index.search(q_emb, k_seed)
        seed_nodes = [int(i) for i in indices[0] if i >= 0 and i in self.graph.nodes]
        if not seed_nodes:
            return []
        pers = {n: 1.0 / len(seed_nodes) for n in seed_nodes}
        try:
            ppr = nx.pagerank(self.graph, alpha=ppr_alpha, personalization=pers, max_iter=100)
        except Exception:
            return []
        top_nodes = sorted(ppr, key=ppr.get, reverse=True)[:top_k * 2]
        doc_score = {}
        for node in top_nodes:                       # score every candidate, do not stop at ten
            s = ppr.get(node, 0.0)
            for doc_id in self.node_to_docs.get(node, []):
                if self.allowed_doc_ids is not None and doc_id not in self.allowed_doc_ids:
                    continue
                if doc_id not in doc_score or s > doc_score[doc_id]:
                    doc_score[doc_id] = s
        ranked = sorted(doc_score.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{"doc_id": d, "score": s} for d, s in ranked]


def load_corpus():
    docs = [json.loads(l) for l in open(PathConfig.get_corpus_path(DS))]
    qs = [json.loads(l) for l in open(PathConfig.get_question_path(DS))]
    return docs, qs


def main():
    import pickle
    from pathlib import Path
    embedder = EmbeddingProcessor()
    docs, questions = load_corpus()
    gd = Path(PathConfig.get_graph_path(DS))
    base_graph = pickle.load(open(gd / "graph.gpickle", "rb"))

    # naive index is identical across edge-removal rates (documents untouched); build once
    chunker = ChunkProcessor()
    all_chunks, all_texts = [], []
    for doc in docs:
        doc_id = str(doc.get("id", doc.get("doc_id", "")))
        title = doc.get("title", "")
        body = doc.get("text", doc.get("context", ""))
        full = f"{title}\n\n{body}" if title else body
        if not full.strip():
            continue
        for ct in chunker.chunk_text(full):
            all_chunks.append({"doc_id": doc_id, "text": ct}); all_texts.append(ct)
    parts = [embedder.embed_texts(all_texts[i:i+128]).astype(np.float32) for i in range(0, len(all_texts), 128)]
    M = np.vstack(parts); faiss.normalize_L2(M)
    index = faiss.IndexFlatIP(M.shape[1]); index.add(M)
    naive = NaiveRetriever.__new__(NaiveRetriever)
    naive.embedder, naive.index, naive.chunks = embedder, index, all_chunks
    print(f"naive index: {len(all_chunks)} chunks")

    allowed = {str(d.get("id", d.get("doc_id", ""))) for d in docs}
    results = {}
    for rate in RATES:
        deg = GraphEdgeRemoval(rate=rate, seed=42).apply(docs, base_graph)
        row = {}
        for label, cls in (("current", GraphRetriever), ("fixed", FixedGraphRetriever)):
            g = cls(DS, embedder, degraded_graph=deg.graph, allowed_doc_ids=allowed)
            h = HybridRetriever(naive, g)
            for pname, ret in (("graph", g), ("hybrid", h)):
                sc = []
                for q in questions:
                    support = set()
                    for sf in q.get("supporting_facts", []):
                        if isinstance(sf, dict):
                            support.add(str(sf["doc_id"]))
                    if not q.get("question") or not support:
                        continue
                    ids = [r["doc_id"] for r in ret.retrieve(q["question"], top_k=10)]
                    sc.append(compute_metrics(ids, support)["ndcg@5"])
                row[f"{pname}_{label}"] = float(mean(sc)); row["n"] = len(sc)
        results[f"{rate:.1f}"] = row
        print(f"r={rate:.1f}  graph cur={row['graph_current']:.4f} fix={row['graph_fixed']:.4f} "
              f"| hybrid cur={row['hybrid_current']:.4f} fix={row['hybrid_fixed']:.4f}", flush=True)
    json.dump(results, open(os.path.join(OUT, "quality_edge_removal.json"), "w"), indent=1)
    print("written", os.path.join(OUT, "quality_edge_removal.json"))


if __name__ == "__main__":
    main()
