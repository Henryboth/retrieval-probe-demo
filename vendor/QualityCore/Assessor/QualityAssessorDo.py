"""
QualityAssessorDo.py
====================
Computes a 12-dimensional data-quality feature vector from a RAGRouter-Bench
corpus, grounded in the low-level functionality taxonomy of:

    Papastergios, V., Ehrlinger, L., & Gounaris, A. (2026).
    Unfolding data quality dimensions in practice: A survey.
    ACM Journal of Data and Information Quality, 18(1), 1-37.
    https://doi.org/10.1145/3786328

Each feature maps to a named functionality (f-XX) and a high-level
ISO/IEC 25012 dimension. All features are normalised to [0, 1] where
1 always represents higher quality.

To compute meaningful relative features under degradation, pass the
original (undegraded) corpus and graph as reference_docs / reference_graph.
Without a reference, absolute features are used as fallbacks.

Feature vector layout
---------------------
Completeness  (f-15, f-19, f-20)
  [0]  text_completeness         -- proportion of non-empty documents
  [1]  avg_length_score          -- mean doc length relative to reference
  [2]  corpus_retention_rate     -- proportion of original documents retained

Consistency   (f-09, f-16, f-17)
  [3]  entity_surface_consistency -- canonical surface-form consistency in text
  [4]  relation_diversity         -- distinct relation types / total graph edges
  [5]  no_duplicate_doc_rate      -- 1 - fraction of exact-duplicate documents

Graph quality (f-15, f-19)
  [6]  graph_node_completeness   -- 1 - isolated-node rate
  [7]  graph_lcc_ratio           -- largest connected component / total nodes
  [8]  graph_density_score       -- edge density relative to reference density

Text quality  (f-07, f-16)
  [9]  vocab_richness            -- type-token ratio across the corpus
  [10] length_uniformity         -- 1 - normalised coefficient of variation
  [11] content_density           -- mean word count relative to reference
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-scale * (x - midpoint)))


def _doc_text(doc: dict) -> str:
    """Return the best available text field, preferring degraded 'text' over
    original 'context' so degradation effects are captured."""
    return doc.get("text", doc.get("context", ""))


def _load_corpus(corpus_path: Path) -> list:
    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def _load_graph(graph_path: Path) -> Optional[nx.Graph]:
    if not graph_path.exists():
        return None
    import pickle
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def _tokenise(text: str) -> list:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# -------------------------------------------------------------------------
# Reference stats (computed once from the original corpus/graph)
# -------------------------------------------------------------------------

class _ReferenceStats:
    """Baseline statistics computed from the undegraded corpus and graph.
    Used to normalise relative features so they equal 1.0 at baseline."""

    def __init__(self, docs: list, graph: Optional[nx.Graph]):
        texts = [_doc_text(d) for d in docs]

        self.n_docs       = len(docs)
        self.mean_chars   = float(np.mean([len(t) for t in texts])) if texts else 1.0
        self.mean_words   = float(np.mean([len(t.split()) for t in texts])) if texts else 1.0
        self.graph_density = nx.density(graph) if (graph and graph.number_of_nodes() > 1) else 0.0

    @classmethod
    def from_docs_graph(cls, docs, graph):
        return cls(docs, graph)


# -------------------------------------------------------------------------
# Feature groups
# -------------------------------------------------------------------------

class _CompletenessFeatures:
    """f-15, f-19, f-20 -> Completeness"""

    @staticmethod
    def text_completeness(docs: list) -> float:
        """f-19: proportion of documents with non-empty text."""
        if not docs:
            return 0.0
        return sum(1 for d in docs if _doc_text(d).strip()) / len(docs)

    @staticmethod
    def avg_length_score(docs: list, ref: Optional[_ReferenceStats]) -> float:
        """f-15: mean document character length relative to reference.
        Returns 1.0 at baseline; drops as text is truncated.
        Falls back to log-sigmoid when no reference is available."""
        if not docs:
            return 0.0
        mean_len = float(np.mean([len(_doc_text(d)) for d in docs]))
        if ref is not None and ref.mean_chars > 0:
            return min(mean_len / ref.mean_chars, 1.0)
        # Fallback: log-sigmoid calibrated for medium-length documents
        return _sigmoid(math.log1p(mean_len), midpoint=math.log1p(1000), scale=1.5)

    @staticmethod
    def corpus_retention_rate(docs: list, ref: Optional[_ReferenceStats]) -> float:
        """f-15: proportion of original documents still present.
        Directly tracks document dropout; equals 1.0 on the full corpus."""
        if ref is None or ref.n_docs == 0:
            return 1.0
        return min(len(docs) / ref.n_docs, 1.0)


class _ConsistencyFeatures:
    """f-09, f-16, f-17 -> Consistency"""

    # Capitalised 1-4 word sequences (same pattern as EntityVariation degrader)
    _ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]*)(?:\s+[A-Z][a-zA-Z]*){0,3}\b")

    @classmethod
    def entity_surface_consistency(cls, docs: list) -> float:
        """f-17: for each frequently-occurring entity, fraction of occurrences
        that use the single most common surface form.  High = consistent naming;
        low = entity variation has introduced surface-form noise.
        Computed directly from text so it responds to EntityVariation degradation."""
        texts = [_doc_text(d) for d in docs[:300]]  # cap for speed
        if not texts:
            return 1.0

        # Build surface-form frequency table per canonical (lowercased) entity
        canon_surfaces: dict = {}
        for text in texts:
            for m in cls._ENTITY_RE.finditer(text):
                surface = m.group(0)
                canon   = surface.lower()
                if len(canon) > 3:
                    canon_surfaces.setdefault(canon, Counter())[surface] += 1

        # Only consider entities that appear >= 3 times (noise filter)
        scores = []
        for canon, surf_counts in canon_surfaces.items():
            total = sum(surf_counts.values())
            if total < 3:
                continue
            top = surf_counts.most_common(1)[0][1]
            scores.append(top / total)

        return float(np.mean(scores)) if scores else 1.0

    @staticmethod
    def relation_diversity(graph: Optional[nx.Graph]) -> float:
        """f-16: distinct relation types / total edges.
        Rich relation vocabularies support precise graph traversal."""
        if graph is None or graph.number_of_edges() == 0:
            return 0.0
        relations = [d.get("relation", "unknown") for _, _, d in graph.edges(data=True)]
        return min(len(set(relations)) / len(relations), 1.0)

    @staticmethod
    def no_duplicate_doc_rate(docs: list) -> float:
        """f-17: 1 - duplicate fraction. Duplicates inflate retrieval scores."""
        if not docs:
            return 1.0
        texts = [_doc_text(d).strip() for d in docs]
        n_dup = sum(c - 1 for c in Counter(texts).values() if c > 1)
        return 1.0 - (n_dup / len(texts))


class _GraphQualityFeatures:
    """f-15, f-19 -> Graph structural quality"""

    @staticmethod
    def graph_node_completeness(graph: Optional[nx.Graph]) -> float:
        """f-19: 1 - isolated-node rate. Isolated nodes are unreachable by traversal."""
        if graph is None or graph.number_of_nodes() == 0:
            return 0.0
        G_u = graph.to_undirected() if graph.is_directed() else graph
        isolated = sum(1 for n in G_u.nodes() if G_u.degree(n) == 0)
        return 1.0 - (isolated / graph.number_of_nodes())

    @staticmethod
    def graph_lcc_ratio(graph: Optional[nx.Graph]) -> float:
        """f-15: largest weakly connected component / total nodes."""
        if graph is None or graph.number_of_nodes() == 0:
            return 0.0
        G_u = graph.to_undirected() if graph.is_directed() else graph
        lcc = max(len(c) for c in nx.connected_components(G_u))
        return lcc / G_u.number_of_nodes()

    @staticmethod
    def graph_density_score(graph: Optional[nx.Graph],
                            ref: Optional[_ReferenceStats]) -> float:
        """f-15: graph density relative to reference density.
        Returns 1.0 at baseline; drops as edges are removed.
        Falls back to a sigmoid when no reference is available."""
        if graph is None or graph.number_of_nodes() < 2:
            return 0.0
        density = nx.density(graph)
        if ref is not None and ref.graph_density > 0:
            return min(density / ref.graph_density, 1.0)
        # Fallback sigmoid — midpoint at a typical sparse-graph density
        return _sigmoid(density, midpoint=0.0005, scale=5000)


class _TextQualityFeatures:
    """f-07, f-16 -> Distributional text quality"""

    @staticmethod
    def vocab_richness(docs: list, sample_size: int = 500) -> float:
        """f-16: type-token ratio over a sample. Higher = more lexically diverse."""
        if not docs:
            return 0.0
        tokens = []
        for d in docs[:sample_size]:
            tokens.extend(_tokenise(_doc_text(d)))
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def length_uniformity(docs: list) -> float:
        """f-07: 1 - normalised CV of document lengths. High CV = uneven corpus."""
        if not docs:
            return 1.0
        lengths = np.array([len(_doc_text(d)) for d in docs], dtype=float)
        mean = lengths.mean()
        if mean == 0:
            return 1.0
        cv = min(lengths.std() / mean, 2.0)
        return 1.0 - (cv / 2.0)

    @staticmethod
    def content_density(docs: list, ref: Optional[_ReferenceStats]) -> float:
        """f-19: mean word count per document relative to reference.
        Complements avg_length_score at the token level; drops with truncation.
        Falls back to absolute log-sigmoid when no reference is available."""
        if not docs:
            return 0.0
        mean_words = float(np.mean([len(_doc_text(d).split()) for d in docs]))
        if ref is not None and ref.mean_words > 0:
            return min(mean_words / ref.mean_words, 1.0)
        return _sigmoid(math.log1p(mean_words), midpoint=math.log1p(200), scale=1.5)


# -------------------------------------------------------------------------
# Main assessor
# -------------------------------------------------------------------------

class QualityAssessor:
    """
    Computes a standardised 12-dimensional quality feature vector for a
    RAGRouter-Bench dataset, grounded in the Papastergios et al. (2026)
    low-level functionality taxonomy.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    project_root : str or Path
        Path to the root of the RAGRouter-Bench project directory.
    degraded_docs : list or None
        Pre-degraded document list. If None, loaded from disk.
    degraded_graph : nx.Graph or None
        Pre-degraded graph. If None, loaded from disk.
    reference_docs : list or None
        Original (undegraded) documents used to compute relative features.
        Pass this in degradation experiments for meaningful sensitivity scores.
    reference_graph : nx.Graph or None
        Original (undegraded) graph for relative graph features.
    """

    FEATURE_NAMES = [
        "text_completeness",        "avg_length_score",      "corpus_retention_rate",
        "entity_surface_consistency","relation_diversity",    "no_duplicate_doc_rate",
        "graph_node_completeness",  "graph_lcc_ratio",       "graph_density_score",
        "vocab_richness",           "length_uniformity",     "content_density",
    ]
    DIMENSIONS = [
        "Completeness", "Completeness", "Completeness",
        "Consistency",  "Consistency",  "Consistency",
        "Completeness", "Completeness", "Completeness",
        "Consistency",  "Accuracy",     "Completeness",
    ]
    PAPASTERGIOS = [
        "f-19", "f-15", "f-15",
        "f-17", "f-16", "f-17",
        "f-19", "f-15", "f-15",
        "f-16", "f-07", "f-19",
    ]

    def __init__(self, dataset_name: str, project_root=".",
                 degraded_docs=None, degraded_graph=None,
                 reference_docs=None, reference_graph=None):
        self.dataset_name    = dataset_name
        self.root            = Path(project_root)
        self._degraded_docs  = degraded_docs
        self._degraded_graph = degraded_graph
        self._ref_docs       = reference_docs
        self._ref_graph      = reference_graph

    def compute(self) -> "QualityFeatureResult":
        """Load data and compute the full quality feature vector."""
        root = self.root
        name = self.dataset_name

        docs = self._degraded_docs if self._degraded_docs is not None else \
               _load_corpus(root / "Dataset/RawData" / name / "Corpus.json")

        graph = self._degraded_graph if self._degraded_graph is not None else \
                _load_graph(root / "Dataset/ProcessedData/Graph" / name / "graph.gpickle")

        # Build reference stats if reference data was provided
        if self._ref_docs is not None:
            ref_graph = self._ref_graph if self._ref_graph is not None else graph
            ref = _ReferenceStats.from_docs_graph(self._ref_docs, ref_graph)
        else:
            ref = None

        print(f"[QualityAssessor] {name}: {len(docs)} docs, "
              f"{graph.number_of_nodes() if graph else 0} graph nodes"
              + (f"  [ref: {ref.n_docs} docs]" if ref else ""))

        comp = _CompletenessFeatures()
        cons = _ConsistencyFeatures()
        gq   = _GraphQualityFeatures()
        tq   = _TextQualityFeatures()

        scores = {
            "text_completeness":          comp.text_completeness(docs),
            "avg_length_score":           comp.avg_length_score(docs, ref),
            "corpus_retention_rate":      comp.corpus_retention_rate(docs, ref),
            "entity_surface_consistency": cons.entity_surface_consistency(docs),
            "relation_diversity":         cons.relation_diversity(graph),
            "no_duplicate_doc_rate":      cons.no_duplicate_doc_rate(docs),
            "graph_node_completeness":    gq.graph_node_completeness(graph),
            "graph_lcc_ratio":            gq.graph_lcc_ratio(graph),
            "graph_density_score":        gq.graph_density_score(graph, ref),
            "vocab_richness":             tq.vocab_richness(docs),
            "length_uniformity":          tq.length_uniformity(docs),
            "content_density":            tq.content_density(docs, ref),
        }

        vector = np.array([scores[n] for n in self.FEATURE_NAMES], dtype=np.float32)
        return QualityFeatureResult(name, scores, vector,
                                    self.FEATURE_NAMES, self.DIMENSIONS, self.PAPASTERGIOS)


# -------------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------------

class QualityFeatureResult:
    """
    Holds the quality feature vector and provides reporting utilities.

    Attributes
    ----------
    dataset_name : str
    scores : dict[str, float]    Per-feature scores in [0, 1].
    vector : np.ndarray           Shape (12,) float32 array.
    """

    def __init__(self, dataset_name, scores, vector, feature_names,
                 dimensions, papastergios_refs):
        self.dataset_name      = dataset_name
        self.scores            = scores
        self.vector            = vector
        self.feature_names     = feature_names
        self.dimensions        = dimensions
        self.papastergios_refs = papastergios_refs

    @property
    def completeness(self) -> float:
        idxs = [i for i, d in enumerate(self.dimensions) if d == "Completeness"]
        return float(np.mean([self.vector[i] for i in idxs]))

    @property
    def consistency(self) -> float:
        idxs = [i for i, d in enumerate(self.dimensions) if d == "Consistency"]
        return float(np.mean([self.vector[i] for i in idxs]))

    @property
    def accuracy(self) -> float:
        idxs = [i for i, d in enumerate(self.dimensions) if d == "Accuracy"]
        return float(np.mean([self.vector[i] for i in idxs]))

    @property
    def overall(self) -> float:
        return float(np.mean(self.vector))

    def report(self) -> str:
        w = 65
        lines = [
            "\n" + "=" * w,
            f"  Quality Assessment  |  {self.dataset_name}",
            "=" * w,
            f"  {'Feature':<32} {'Dim':<14} {'Ref':<12} {'Score':>5}",
            "  " + "-" * (w - 2),
        ]
        prev_dim = None
        for name, dim, ref in zip(self.feature_names, self.dimensions,
                                  self.papastergios_refs):
            if dim != prev_dim:
                lines.append(f"  -- {dim} --")
                prev_dim = dim
            score = self.scores[name]
            bar = ("█" * int(score * 12)).ljust(12, "░")
            lines.append(f"  {name:<32} {dim:<14} {ref:<12} {score:>5.3f}  {bar}")
        lines += [
            "  " + "-" * (w - 2),
            f"  Completeness  : {self.completeness:.3f}",
            f"  Consistency   : {self.consistency:.3f}",
            f"  Accuracy      : {self.accuracy:.3f}",
            f"  Overall       : {self.overall:.3f}",
            "=" * w + "\n",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset":      self.dataset_name,
            "overall":      round(self.overall, 4),
            "completeness": round(self.completeness, 4),
            "consistency":  round(self.consistency, 4),
            "accuracy":     round(self.accuracy, 4),
            "features":     {k: round(v, 4) for k, v in self.scores.items()},
            "vector":       [round(float(v), 4) for v in self.vector],
        }

    def save(self, path) -> None:
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[QualityAssessor] Saved -> {p}")
