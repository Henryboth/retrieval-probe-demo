"""
DegradationDo.py
================
Controlled quality degradation of RAGRouter-Bench corpora for benchmarking.

Each degradation type is grounded in a Papastergios et al. (2026) dimension
and simulates a realistic data-quality problem found in organisational data.

Degradation types
-----------------
TextTruncation      (Completeness / f-19)
    Truncates document text to a fraction of its original length.
    Simulates poor OCR, incomplete extraction, or storage truncation.

EntityVariation     (Consistency / f-17)
    Replaces entity surface forms with plausible alternatives
    (abbreviations, acronyms, alternate spellings).
    Simulates the common organisational problem of inconsistent naming
    conventions across systems, which directly degrades BM25 retrieval.

GraphEdgeRemoval    (Completeness / f-19)
    Randomly removes edges from the knowledge graph.
    Simulates incomplete relation extraction or knowledge graph sparsity,
    which directly degrades GraphRAG and HybridRAG traversal.

DocumentDropout     (Completeness / f-15)
    Randomly removes entire documents from the corpus.
    Simulates data loss, access restrictions, or ingestion failures.

CombinedDegradation
    Applies all four degradation types simultaneously at the same rate.
    Used in Phase 3 experiments to study compound quality effects.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np


# -------------------------------------------------------------------------
# Random name variation helpers (for EntityVariation)
# -------------------------------------------------------------------------

def _abbreviate(name: str, rng: random.Random) -> str:
    """Generate a plausible surface-form variant of an entity name."""
    words = name.split()
    if len(words) == 1:
        # Single word: lowercase, or add/remove trailing punctuation
        variants = [
            name.lower(),
            name.upper(),
            name + "s",   # naive pluralisation
            name[:max(3, len(name)-2)],  # truncation
        ]
        return rng.choice(variants)
    # Multi-word: initialism, drop last word, or first-word only
    initialism = "".join(w[0].upper() for w in words if w)
    variants = [
        initialism,
        initialism + ".",
        words[0],
        " ".join(words[:-1]),
        name.lower(),
        name.replace(" ", "-"),
    ]
    return rng.choice(variants)


# -------------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------------

@dataclass
class DegradationResult:
    """
    Holds the output of a degradation operation.

    Attributes
    ----------
    docs : list[dict]              Degraded document list (or original if unchanged).
    graph : nx.Graph | None        Degraded graph (or original if unchanged).
    degradation_type : str         Which degradation was applied.
    rate : float                   Degradation rate in [0, 1].
    stats : dict                   Diagnostic statistics about the degradation.
    """
    docs: list
    graph: Optional[nx.Graph]
    degradation_type: str
    rate: float
    stats: dict = field(default_factory=dict)


# -------------------------------------------------------------------------
# Degradation classes
# -------------------------------------------------------------------------

class TextTruncation:
    """
    Papastergios f-19 | Completeness
    Truncates each document's text to (1 - rate) of its original length.
    At rate=0.0 the corpus is unchanged; at rate=0.9 only the first 10% of
    each document's text is retained.

    Parameters
    ----------
    rate : float    Fraction of text to remove (0.0 = no change, 1.0 = all blank).
    seed : int      Random seed for reproducibility.
    """

    def __init__(self, rate: float = 0.3, seed: int = 42):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.seed = seed

    def apply(self, docs: list, graph: Optional[nx.Graph] = None) -> DegradationResult:
        rng = random.Random(self.seed)
        keep_fraction = 1.0 - self.rate
        degraded = []
        chars_removed = 0

        for doc in docs:
            d = copy.copy(doc)
            text = d.get("text", d.get("context", ""))
            if text and keep_fraction < 1.0:
                # Split into sentences and keep the leading fraction
                sentences = re.split(r"(?<=[.!?])\s+", text)
                n_keep = max(1, round(len(sentences) * keep_fraction))
                new_text = " ".join(sentences[:n_keep])
                chars_removed += len(text) - len(new_text)
                d["text"] = new_text
            degraded.append(d)

        return DegradationResult(
            docs=degraded,
            graph=graph,
            degradation_type="text_truncation",
            rate=self.rate,
            stats={
                "docs_affected": len(degraded),
                "chars_removed": chars_removed,
                "keep_fraction": keep_fraction,
            },
        )


class EntityVariation:
    """
    Papastergios f-17 | Consistency
    Replaces entity surface forms in document text with alternate surface
    variants at the specified rate.

    In practice: the degrader identifies the most frequent named-entity-like
    tokens (capitalised multi-word phrases) and replaces a fraction of their
    occurrences with a generated variant. This simulates the common
    organisational problem of inconsistent naming (e.g., "European Central
    Bank" vs "ECB" vs "the Central Bank") which directly degrades BM25
    retrieval by breaking lexical overlap.

    Parameters
    ----------
    rate : float    Probability of replacing each entity occurrence.
    seed : int      Random seed.
    """

    # Simple pattern: capitalised sequences of 1-4 words
    _ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]*)(?:\s+[A-Z][a-zA-Z]*){0,3}\b")

    def __init__(self, rate: float = 0.3, seed: int = 42):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.seed = seed

    def _build_entity_map(self, docs: list, rng: random.Random) -> dict:
        """Find frequent entities and assign a surface variant to each."""
        freq: dict = {}
        for doc in docs:
            for match in self._ENTITY_PATTERN.finditer(doc.get("text", doc.get("context", ""))):
                entity = match.group(0)
                if len(entity) > 3:
                    freq[entity] = freq.get(entity, 0) + 1
        # Only vary entities that appear frequently enough to matter
        frequent = {e for e, c in freq.items() if c >= 3}
        return {e: _abbreviate(e, rng) for e in frequent}

    def apply(self, docs: list, graph: Optional[nx.Graph] = None) -> DegradationResult:
        rng = random.Random(self.seed)
        entity_map = self._build_entity_map(docs, rng)
        degraded = []
        replacements_made = 0

        def _replace(text: str) -> str:
            nonlocal replacements_made
            result = text
            for canonical, variant in entity_map.items():
                if canonical in result and rng.random() < self.rate:
                    new_result = result.replace(canonical, variant, 1)
                    if new_result != result:
                        replacements_made += 1
                    result = new_result
            return result

        for doc in docs:
            d = copy.copy(doc)
            d["text"] = _replace(d.get("text", d.get("context", "")))
            degraded.append(d)

        return DegradationResult(
            docs=degraded,
            graph=graph,
            degradation_type="entity_variation",
            rate=self.rate,
            stats={
                "entities_varied": len(entity_map),
                "replacements_made": replacements_made,
                "sample_variants": dict(list(entity_map.items())[:5]),
            },
        )


class GraphEdgeRemoval:
    """
    Papastergios f-19 | Completeness
    Randomly removes edges from the knowledge graph at the specified rate.

    This simulates incomplete relation extraction -- a common problem when
    knowledge graphs are built automatically from text and the extraction
    model fails to capture all relations. Missing edges break graph traversal
    paths, directly degrading GraphRAG and HybridRAG for queries that depend
    on the removed relationships.

    Parameters
    ----------
    rate : float    Fraction of edges to remove (0.0 = unchanged, 1.0 = no edges).
    seed : int      Random seed.
    protect_relation : str | None
        If set, edges with this relation type are never removed.
        Useful for preserving critical structural edges during experiments.
    """

    def __init__(self, rate: float = 0.3, seed: int = 42,
                 protect_relation: Optional[str] = None):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.seed = seed
        self.protect_relation = protect_relation

    def apply(self, docs: list, graph: Optional[nx.Graph] = None) -> DegradationResult:
        if graph is None:
            return DegradationResult(
                docs=docs, graph=None,
                degradation_type="graph_edge_removal",
                rate=self.rate,
                stats={"warning": "No graph provided; degradation skipped."},
            )

        rng = random.Random(self.seed)
        G_degraded = copy.deepcopy(graph)

        all_edges = list(G_degraded.edges(data=True))
        removable = [
            (u, v) for u, v, d in all_edges
            if self.protect_relation is None or
               d.get("relation") != self.protect_relation
        ]
        n_remove = round(len(removable) * self.rate)
        edges_to_remove = rng.sample(removable, min(n_remove, len(removable)))

        G_degraded.remove_edges_from(edges_to_remove)

        # Recompute isolated nodes after removal
        G_u = G_degraded.to_undirected() if G_degraded.is_directed() else G_degraded
        n_isolated = sum(1 for n in G_u.nodes() if G_u.degree(n) == 0)

        return DegradationResult(
            docs=docs,
            graph=G_degraded,
            degradation_type="graph_edge_removal",
            rate=self.rate,
            stats={
                "original_edges": len(all_edges),
                "edges_removed": len(edges_to_remove),
                "remaining_edges": G_degraded.number_of_edges(),
                "isolated_nodes_after": n_isolated,
            },
        )


class DocumentDropout:
    """
    Papastergios f-15 | Completeness
    Randomly removes entire documents from the corpus at the specified rate.

    Simulates data loss, access restrictions, failed ingestion, or
    records that were purged. This degrades all retrieval methods by
    reducing the evidence available, but affects NaiveRAG and HybridRAG
    most when the dropped documents contain the answer.

    Parameters
    ----------
    rate : float    Fraction of documents to remove.
    seed : int      Random seed.
    """

    def __init__(self, rate: float = 0.3, seed: int = 42):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.seed = seed

    def apply(self, docs: list, graph: Optional[nx.Graph] = None) -> DegradationResult:
        rng = random.Random(self.seed)
        n_remove = round(len(docs) * self.rate)
        remove_indices = set(rng.sample(range(len(docs)), min(n_remove, len(docs))))
        degraded = [d for i, d in enumerate(docs) if i not in remove_indices]

        return DegradationResult(
            docs=degraded,
            graph=graph,
            degradation_type="document_dropout",
            rate=self.rate,
            stats={
                "original_docs": len(docs),
                "docs_removed": len(remove_indices),
                "remaining_docs": len(degraded),
            },
        )


class CombinedDegradation:
    """
    Applies all four degradation types simultaneously at the same rate.
    Used in Phase 3 experiments to study compound quality effects and to
    test whether the quality-aware router can detect and respond to
    multi-dimensional degradation.

    The application order is:
      1. DocumentDropout (reduces corpus size)
      2. TextTruncation  (truncates remaining documents)
      3. EntityVariation (introduces naming inconsistency)
      4. GraphEdgeRemoval (removes graph relations)

    Parameters
    ----------
    rate : float    Degradation rate applied uniformly to all four types.
    seed : int      Base random seed (each sub-degrader uses seed + offset).
    """

    def __init__(self, rate: float = 0.3, seed: int = 42):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.seed = seed

    def apply(self, docs: list, graph: Optional[nx.Graph] = None) -> DegradationResult:
        r = self.rate
        s = self.seed

        r1 = DocumentDropout(rate=r, seed=s).apply(docs, graph)
        r2 = TextTruncation(rate=r, seed=s+1).apply(r1.docs, r1.graph)
        r3 = EntityVariation(rate=r, seed=s+2).apply(r2.docs, r2.graph)
        r4 = GraphEdgeRemoval(rate=r, seed=s+3).apply(r3.docs, r3.graph)

        combined_stats = {
            "document_dropout":  r1.stats,
            "text_truncation":   r2.stats,
            "entity_variation":  r3.stats,
            "graph_edge_removal": r4.stats,
        }

        return DegradationResult(
            docs=r4.docs,
            graph=r4.graph,
            degradation_type="combined",
            rate=self.rate,
            stats=combined_stats,
        )
