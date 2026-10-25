import json
import asyncio
import networkx as nx
import numpy as np
import spacy
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from Config.LLMConfig import LLMConfig
from Config.EmbConfig import EmbConfig
from Config.GraphConfig import GraphConfig


class GraphProcessor:
    """Extract entities and relations from chunks using spaCy, then build knowledge graph."""

    def __init__(self):
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000

        # Sentence-transformers for entity embeddings
        from sentence_transformers import SentenceTransformer
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.st_model = SentenceTransformer(EmbConfig.HUGGINGFACE_MODEL, device=device)

    def _extract_triplets_from_chunk(self, chunk: str, doc_id: int, chunk_idx: int) -> List[Dict]:
        """Extract (subject, relation, object) triplets using spaCy dependency parsing."""
        triplets = []
        try:
            doc = self.nlp(chunk[:1000])  # limit chunk length for speed
            for sent in doc.sents:
                # Find subject-verb-object patterns
                subj = None
                obj = None
                root = None
                for token in sent:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.ent_type_:
                        subj = token.text
                        root = token.head.lemma_
                    if token.dep_ in ("dobj", "pobj", "attr") and token.ent_type_:
                        obj = token.text
                if subj and obj and root and subj != obj:
                    triplets.append({
                        "triplet": [subj, root, obj],
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "source_text": sent.text
                    })
                # Also add co-occurrence edges between named entities in same sentence
                ents = [e.text for e in sent.ents if len(e.text) > 2]
                for i in range(len(ents)):
                    for j in range(i+1, min(i+3, len(ents))):
                        triplets.append({
                            "triplet": [ents[i], "co-occurs-with", ents[j]],
                            "doc_id": doc_id,
                            "chunk_idx": chunk_idx,
                            "source_text": sent.text
                        })
        except Exception:
            pass
        return triplets

    def extract_triplets_incremental(self, chunks_by_doc, dataset_name, existing_triplets):
        docs_to_process = [(doc_id, chunks) for doc_id, chunks in chunks_by_doc.items()
                          if doc_id not in existing_triplets]
        if not docs_to_process:
            return existing_triplets

        total_chunks = sum(len(c) for _, c in docs_to_process)
        print(f"Extracting triplets from {len(docs_to_process)} docs ({total_chunks} chunks) using spaCy...")

        with tqdm(total=total_chunks, desc="Extracting triplets", unit="chunk") as pbar:
            for doc_id, chunks in docs_to_process:
                doc_triplets = []
                for chunk_idx, chunk in enumerate(chunks):
                    doc_triplets.extend(self._extract_triplets_from_chunk(chunk, doc_id, chunk_idx))
                    pbar.update(1)
                existing_triplets[doc_id] = doc_triplets
                if dataset_name:
                    from RAGCore.Graph.GraphSave import GraphSaver
                    GraphSaver.save_batch_triplets({doc_id: doc_triplets}, dataset_name)

        return existing_triplets

    def _embed_entities(self, entities: List[str]) -> Dict[str, np.ndarray]:
        if not entities:
            return {}
        valid = [e for e in entities if e and e.strip()]
        if not valid:
            return {}
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(valid), batch_size):
            batch = valid[i:i+batch_size]
            embs = self.st_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(embs)
        all_embs = np.vstack(all_embeddings)
        return {entity: all_embs[i] for i, entity in enumerate(valid)}

    def build_graph(self, triplets_by_doc):
        G = nx.DiGraph() if GraphConfig.DIRECTED else nx.Graph()
        all_entities = set()
        triplet_sources = {}

        for doc_id, triplet_records in triplets_by_doc.items():
            for record in triplet_records:
                triplet = record["triplet"] if isinstance(record, dict) else record
                if len(triplet) != 3:
                    continue
                s, p, o = triplet
                if not (s and p and o and isinstance(s,str) and isinstance(o,str)):
                    continue
                all_entities.add(s); all_entities.add(o)
                key = (s, p, o)
                triplet_sources.setdefault(key, []).append(
                    {"doc_id": doc_id, "chunk_idx": record.get("chunk_idx",0) if isinstance(record,dict) else 0,
                     "source_text": record.get("source_text","") if isinstance(record,dict) else ""}
                )

        entity_to_id = {e: i for i, e in enumerate(sorted(all_entities))}
        id_to_entity = {i: e for e, i in entity_to_id.items()}

        for doc_id, triplet_records in triplets_by_doc.items():
            for record in triplet_records:
                triplet = record["triplet"] if isinstance(record, dict) else record
                if len(triplet) != 3: continue
                s, p, o = triplet
                if not (s and p and o and isinstance(s,str) and isinstance(o,str)): continue
                si, oi = entity_to_id[s], entity_to_id[o]
                if si not in G: G.add_node(si, entity_id=si)
                if oi not in G: G.add_node(oi, entity_id=oi)
                G.add_edge(si, oi, relation=p, doc_id=doc_id)

        if not all_entities:
            return {"graph": G, "entity_mapping": {}, "entity_embeddings": {}, "triplet_sources": {}}

        print(f"Generating embeddings for {len(all_entities)} entities...")
        entity_list = [id_to_entity[i] for i in range(len(all_entities))]
        text_to_emb = self._embed_entities(entity_list)
        entity_embeddings = {entity_to_id[e]: emb for e, emb in text_to_emb.items()}

        return {"graph": G, "entity_mapping": id_to_entity,
                "entity_embeddings": entity_embeddings, "triplet_sources": triplet_sources}

    def _derive_entities_from_triplets(self, triplets_by_doc):
        entities_by_doc = {}
        for doc_id, records in triplets_by_doc.items():
            doc_entities = set()
            for record in records:
                triplet = record["triplet"] if isinstance(record, dict) else record
                if len(triplet) == 3:
                    doc_entities.add(triplet[0]); doc_entities.add(triplet[2])
            entities_by_doc[doc_id] = list(doc_entities)
        return entities_by_doc

    def process(self, chunks_by_doc, dataset_name=None, resume=True):
        if resume and dataset_name:
            from RAGCore.Graph.GraphSave import GraphSaver
            triplets_by_doc = GraphSaver.load_existing_triplets(dataset_name)
            if triplets_by_doc:
                print(f"Resuming: {len(triplets_by_doc)} docs already processed")
        else:
            triplets_by_doc = {}

        triplets_by_doc = self.extract_triplets_incremental(chunks_by_doc, dataset_name, triplets_by_doc)

        print("Building knowledge graph...")
        graph_data = self.build_graph(triplets_by_doc)
        entities_by_doc = self._derive_entities_from_triplets(triplets_by_doc)

        return {
            "entities": entities_by_doc,
            "triplets": triplets_by_doc,
            "graph": graph_data["graph"],
            "entity_mapping": graph_data["entity_mapping"],
            "entity_embeddings": graph_data["entity_embeddings"],
            "triplet_sources": graph_data["triplet_sources"]
        }
