"""
engine.py — live retrieval for the demo, over benchmark corpora or one a user
has dropped in.

Everything here calls the project's own components rather than reimplementing
them: the three retrievers and the corpus-mean-characters normaliser come from
Run/Quality/run_retrieval_eval.py, and the twelve probe measurements from
QualityCore/PerQueryFeatures/PerQueryDo.py.

Corpus storage is kept apart. Benchmark corpora are read from code/Dataset,
read-only. A corpus a user adds is written under demo/corpora/<name>/, never
into the research tree. PathConfig resolves its directories at call time, so
pointing it at a corpus root before building the retrievers is enough to
redirect the whole pipeline.
"""

from __future__ import annotations

import os
import sys

# These must be set before numpy, xgboost, faiss or torch is imported, and
# xgboost must be imported before faiss. Both requirements are the project's,
# documented in Run/Quality/run_latency_eval.py: faiss and xgboost each bundle
# their own OpenMP runtime, and with both loaded a parallel region can enter a
# barrier wait that never returns. Without this the first retrieval hangs at
# 0 per cent CPU, and importing faiss before xgboost segfaults on fit().
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import xgboost  # noqa: F401  (import order matters — before anything pulling faiss)

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
# In the published bundle the project's modules and one corpus travel under
# vendor/. On this machine there is no vendor/, so the research tree is used
# exactly as before.
CODE = DEMO / "vendor" if (DEMO / "vendor").is_dir() else DEMO.parent / "code"
USER_CORPORA = DEMO / "corpora"
BENCHMARKS = ["quality", "musique", "ultraDomain_legal"]

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if str(CODE / "Run" / "Quality") not in sys.path:
    sys.path.insert(0, str(CODE / "Run" / "Quality"))


def _require_cwd() -> None:
    """The experiment's modules resolve Dataset/ relative to code/. Done at
    import as well, so a background thread never races the main one over it."""
    if Path.cwd() != CODE:
        os.chdir(CODE)


_require_cwd()

# ── corpus cache ────────────────────────────────────────────────────────────
# Held here rather than in Streamlit's cache so that a background warming
# thread fills the same cache the interface reads from. Loads are serialised
# under one lock: they are memory-heavy, and the pipeline modules resolve paths
# through PathConfig's class attributes, which are process-global.
_CACHE: dict = {}
_STATUS: dict = {}
_LOCK = threading.Lock()
_WARMING = threading.Event()


def status() -> dict:
    with _LOCK:
        return dict(_STATUS)


def is_ready(name: str) -> bool:
    return status().get(name) == "ready"


def get_corpus(name: str, with_graph: bool = True):
    """Return a loaded corpus, loading it if necessary. Safe to call from the
    interface and the warming thread at the same time."""
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        _STATUS[name] = "loading"
    try:
        corpus = LiveCorpus(name, with_graph=with_graph)
    except Exception as exc:
        with _LOCK:
            _STATUS[name] = f"failed: {exc}"
        raise
    with _LOCK:
        _CACHE[name] = corpus
        _STATUS[name] = "ready"
    return corpus


def prewarm(names: list[str] | None = None, include_llm: bool = True):
    """Load corpora in the background so the first question does not pay for
    it. Starts once per process; later calls are ignored."""
    if _WARMING.is_set():
        return None
    _WARMING.set()
    targets = names if names is not None else (BENCHMARKS + list_user_corpora())

    def run():
        if include_llm:
            try:
                import llm
                llm.preload()
            except Exception:
                pass
        for name in targets:
            try:
                get_corpus(name)
            except Exception:
                continue

    thread = threading.Thread(target=run, name="prewarm", daemon=True)
    thread.start()
    return thread


def corpus_root(name: str) -> Path:
    """Where a corpus lives: the research tree for benchmarks, demo/corpora
    for anything a user added."""
    return CODE / "Dataset" if name in BENCHMARKS else USER_CORPORA / name


def point_paths_at(name: str) -> None:
    from Config.PathConfig import PathConfig
    root = corpus_root(name)
    processed = root / "ProcessedData"
    PathConfig.RAW_DATA_DIR = str(root / "RawData")
    PathConfig.PROCESSED_DATA_DIR = str(processed)
    PathConfig.TRIPLET_DIR = str(processed / "Triplet")
    PathConfig.EMBEDDING_DIR = str(processed / "Embedding")
    PathConfig.INDEX_DIR = str(processed / "Index")
    PathConfig.GRAPH_DIR = str(processed / "Graph")


def list_user_corpora() -> list[str]:
    if not USER_CORPORA.exists():
        return []
    return sorted(p.name for p in USER_CORPORA.iterdir()
                  if (p / "ProcessedData" / "Index" / p.name / "index.faiss").exists())


def available_benchmarks() -> list[str]:
    """Benchmarks whose index is present. All three on the research machine;
    only the bundled one in the published copy."""
    return [c for c in BENCHMARKS
            if (corpus_root(c) / "ProcessedData" / "Index" / c / "index.faiss").exists()]


def corpus_is_ready(name: str) -> bool:
    return name in available_benchmarks() or name in list_user_corpora()


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    rank: int


class LiveCorpus:
    """Naive, graph and hybrid retrieval plus the twelve probe measurements,
    for one corpus held in memory."""

    def __init__(self, name: str, with_graph: bool = True):
        _require_cwd()
        point_paths_at(name)
        self.name = name

        from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
        from QualityCore.PerQueryFeatures.PerQueryDo import PerQueryAssessor
        import run_retrieval_eval as rre

        self.embedder = EmbeddingProcessor()
        self.naive = rre.NaiveRetriever(name, self.embedder)
        self.graph = rre.GraphRetriever(name, self.embedder) if with_graph else None
        self.hybrid = rre.HybridRetriever(self.naive, self.graph) if with_graph else None

        corpus = rre.load_corpus(name)
        self.docid_to_text = {
            str(d.get("id", d.get("doc_id", ""))): d.get("text", d.get("context", ""))
            for d in corpus
        }
        # identical to run_retrieval_eval.py's normaliser for chunk_completeness
        self.corpus_mean_chars = float(np.mean([
            len(d.get("text", d.get("context", ""))) for d in corpus
        ])) if corpus else 1.0
        self.n_docs = len(corpus)
        self.n_chunks = len(getattr(self.naive, "chunks", []))
        self.assessor = PerQueryAssessor(self.embedder, self.corpus_mean_chars, top_k=10)

    def _hits(self, results: list) -> list[Hit]:
        return [
            Hit(doc_id=str(r.get("doc_id", "")),
                text=r.get("text", "") or self.docid_to_text.get(str(r.get("doc_id", "")), ""),
                score=float(r.get("score", 0.0)),
                rank=i + 1)
            for i, r in enumerate(results)
        ]

    def ask(self, question: str, top_k: int = 10, run_all: bool = True) -> dict:
        """Run the probe, read the twelve measurements, and optionally run the
        other two paradigms so their result sets can be compared."""
        import faiss

        emb = self.embedder.embed_texts([question]).astype(np.float32)
        faiss.normalize_L2(emb)
        query_emb = emb[0]

        naive_raw = self.naive.retrieve(question, top_k=top_k)
        features = self.assessor.compute(question, query_emb, naive_raw, self.docid_to_text)

        out = {
            "question": question,
            "corpus": self.name,
            "features": features,
            "naive": self._hits(naive_raw),
            "graph": [],
            "hybrid": [],
        }
        if run_all and self.graph is not None:
            out["graph"] = self._hits(self.graph.retrieve(question, top_k=top_k))
            out["hybrid"] = self._hits(self.hybrid.retrieve(question, top_k=top_k))
        return out


class Predictor:
    """The two models the demo ships, both reading the twelve measurements only."""

    def __init__(self, models_dir: Path | None = None):
        import joblib
        d = models_dir or (DEMO / "data" / "models")
        self.detector = joblib.load(d / "detector.joblib")
        self.gate = joblib.load(d / "gate_pq.joblib")
        self.feature_order = [c[2:] for c in self.detector["features"]]
        self.transfer = json.loads((DEMO / "data" / "transfer.json").read_text())

    def _vector(self, features: dict) -> np.ndarray:
        return np.array([[float(features.get(n, 0.0)) for n in self.feature_order]],
                        dtype=np.float32)

    def predict(self, features: dict) -> dict:
        v = self._vector(features)
        return {
            "p_all_failed": float(self.detector["model"].predict_proba(v)[0, 1]),
            "p_escalate": float(self.gate["model"].predict_proba(v)[0, 1]),
        }

    def unseen_corpus_auc(self, model: str = "detector") -> list[float]:
        return self.transfer["models"][model]["unseen_corpus_auc_range"]
