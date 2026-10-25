import os
import json
import pickle
import networkx as nx
from typing import Dict, List, Tuple, Any
from Config.PathConfig import PathConfig


class GraphSaver:
    """Save graph data including triplets and graph object (entities are derived from triplets)"""

    @staticmethod
    def save_incremental_triplets(doc_id: int, triplets: List[Tuple[str, str, str]], dataset_name: str):
        """Save triplets for a single document (incremental save)

        NOTE: This method is kept for backward compatibility but is slow for large datasets.
        Use save_batch_triplets() for better performance with batch processing.

        Args:
            doc_id: Document ID
            triplets: List of triplets for this document
            dataset_name: Name of the dataset
        """
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        os.makedirs(triplet_dir, exist_ok=True)
        resource_path = os.path.join(triplet_dir, "triplet_resource.json")

        # Load existing data
        if os.path.exists(resource_path):
            with open(resource_path, 'r', encoding='utf-8') as f:
                triplet_resource = json.load(f)
        else:
            triplet_resource = {}

        # Update with new triplets (format: "s|||p|||o" -> [sources])
        for triplet in triplets:
            if len(triplet) >= 3:
                key = f"{triplet[0]}|||{triplet[1]}|||{triplet[2]}"
                if key not in triplet_resource:
                    triplet_resource[key] = []
                triplet_resource[key].append({"doc_id": doc_id})

        # Save back
        with open(resource_path, 'w', encoding='utf-8') as f:
            json.dump(triplet_resource, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_batch_triplets(triplets_by_doc_batch: Dict[int, List[Any]], dataset_name: str):
        """Save a batch of documents' triplets (batch incremental save)

        More efficient than save_incremental_triplets for batch processing because it:
        - Reads the file once
        - Updates multiple documents
        - Writes the file once

        Args:
            triplets_by_doc_batch: Dict of {doc_id: [triplet_records]} for a batch of documents
                triplet_records format: {"triplet": [s, p, o], "doc_id": int, "chunk_idx": int, "source_text": str}
            dataset_name: Name of the dataset
        """
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        os.makedirs(triplet_dir, exist_ok=True)
        resource_path = os.path.join(triplet_dir, "triplet_resource.json")

        # Load existing data
        if os.path.exists(resource_path):
            with open(resource_path, 'r', encoding='utf-8') as f:
                triplet_resource = json.load(f)
        else:
            triplet_resource = {}

        # Update with batch
        for doc_id, triplets in triplets_by_doc_batch.items():
            for record in triplets:
                if isinstance(record, dict) and "triplet" in record:
                    triplet = record["triplet"]
                    source_info = {
                        "doc_id": record.get("doc_id", doc_id),
                        "chunk_idx": record.get("chunk_idx", 0),
                        "source_text": record.get("source_text", "")
                    }
                else:
                    triplet = record
                    source_info = {"doc_id": doc_id}

                if len(triplet) >= 3:
                    key = f"{triplet[0]}|||{triplet[1]}|||{triplet[2]}"
                    if key not in triplet_resource:
                        triplet_resource[key] = []
                    triplet_resource[key].append(source_info)

        # Save back
        with open(resource_path, 'w', encoding='utf-8') as f:
            json.dump(triplet_resource, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_existing_triplets(dataset_name: str) -> Dict[int, List[Any]]:
        """Load existing triplets for resume capability

        Reads triplet_resource.json and converts it back to {doc_id: [triplet_records]} format
        for compatibility with GraphDo.process() and build_graph().

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict of {doc_id: [{"triplet": [s,p,o], "doc_id": int, "chunk_idx": int, "source_text": str}, ...]}
            or empty dict if not exists
        """
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        resource_path = os.path.join(triplet_dir, "triplet_resource.json")

        if not os.path.exists(resource_path):
            return {}

        with open(resource_path, 'r', encoding='utf-8') as f:
            triplet_resource = json.load(f)

        # Convert from {"A|||rel|||B": [sources]} to {doc_id: [triplet_records]}
        triplets_by_doc = {}
        for key, sources in triplet_resource.items():
            parts = key.split("|||")
            if len(parts) != 3:
                continue
            triplet = parts
            for source_info in sources:
                doc_id = source_info.get("doc_id", 0)
                if doc_id not in triplets_by_doc:
                    triplets_by_doc[doc_id] = []
                triplets_by_doc[doc_id].append({
                    "triplet": triplet,
                    "doc_id": doc_id,
                    "chunk_idx": source_info.get("chunk_idx", 0),
                    "source_text": source_info.get("source_text", "")
                })

        return triplets_by_doc

    @staticmethod
    def save_failed_chunk(doc_id: int, chunk_idx: int, chunk_text: str, error: str, dataset_name: str):
        """Save a single failed chunk (incremental save)

        Args:
            doc_id: Document ID
            chunk_idx: Chunk index within the document
            chunk_text: The chunk text that failed
            error: Error message
            dataset_name: Name of the dataset
        """
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        os.makedirs(triplet_dir, exist_ok=True)
        failed_chunks_path = os.path.join(triplet_dir, "failed_chunks.json")

        # Load existing failed chunks
        if os.path.exists(failed_chunks_path):
            with open(failed_chunks_path, 'r', encoding='utf-8') as f:
                failed_chunks = json.load(f)
        else:
            failed_chunks = []

        # Add new failed chunk
        failed_chunks.append({
            "doc_id": doc_id,
            "chunk_idx": chunk_idx,
            "chunk_text": chunk_text,
            "error": str(error)
        })

        # Save back
        with open(failed_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(failed_chunks, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_failed_chunks(dataset_name: str) -> List[Dict[str, Any]]:
        """Load failed chunks for retry

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of failed chunk records: [{"doc_id": int, "chunk_idx": int, "chunk_text": str, "error": str}, ...]
        """
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        failed_chunks_path = os.path.join(triplet_dir, "failed_chunks.json")

        if not os.path.exists(failed_chunks_path):
            return []

        with open(failed_chunks_path, 'r', encoding='utf-8') as f:
            failed_chunks = json.load(f)

        return failed_chunks

    @staticmethod
    def clear_failed_chunks(dataset_name: str):
        """Clear failed chunks file after successful retry

        Args:
            dataset_name: Name of the dataset
        """
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        failed_chunks_path = os.path.join(triplet_dir, "failed_chunks.json")

        if os.path.exists(failed_chunks_path):
            os.remove(failed_chunks_path)
            print(f"Cleared failed chunks file: {failed_chunks_path}")

    @staticmethod
    def save(result: Dict[str, Any], dataset_name: str):
        """Save graph processing results to disk with optimized storage

        Args:
            result: Output from GraphProcessor.process()
                   Format: {
                       "entities": {doc_id: [entity1, ...]},  # Derived from triplets, not saved separately
                       "triplets": {doc_id: [{"triplet": [s, p, o], "doc_id": int, "chunk_idx": int, "source_text": str}, ...]},
                       "graph": NetworkX graph object (nodes contain entity_id only),
                       "entity_mapping": {entity_id: entity_text},
                       "entity_embeddings": {entity_id: embedding_vector},
                       "triplet_sources": {(s, p, o): [{"doc_id": int, "chunk_idx": int, "source_text": str}, ...]}
                   }
            dataset_name: Name of the dataset (e.g., "hotpotqa")
        """
        graph = result["graph"]
        entity_mapping = result["entity_mapping"]
        entity_embeddings = result["entity_embeddings"]

        # Note: triplet_resource.json is already saved by save_batch_triplets() during extraction
        # No need to save it again here

        # Save graph structure
        graph_dir = PathConfig.get_graph_path(dataset_name)
        os.makedirs(graph_dir, exist_ok=True)
        graph_path = os.path.join(graph_dir, "graph.gpickle")

        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved graph to: {graph_path}")

        # Save entity mapping as Parquet
        import pandas as pd

        mapping_path = os.path.join(graph_dir, "entity_mapping.parquet")

        # Create DataFrame with entity_id and entity columns
        df = pd.DataFrame([
            {"entity_id": eid, "entity": text}
            for eid, text in entity_mapping.items()
        ])

        df.to_parquet(mapping_path, index=False, engine='pyarrow')
        print(f"Saved entity mapping to: {mapping_path}")

        # Save entity embeddings as FAISS index
        if entity_embeddings:
            import faiss
            import numpy as np

            # Create embeddings matrix (sorted by entity_id)
            entity_ids = sorted(entity_embeddings.keys())
            embeddings_matrix = np.array([entity_embeddings[eid] for eid in entity_ids]).astype('float32')

            # Build FAISS index
            dimension = embeddings_matrix.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_matrix)

            # Save FAISS index
            index_path = os.path.join(graph_dir, "entity_index.faiss")
            faiss.write_index(index, index_path)

            print(f"Saved entity index to: {index_path}")
            print(f"Entity index contains {index.ntotal} vectors, dimension {dimension}")

        # Print file sizes
        graph_size_kb = os.path.getsize(graph_path) / 1024
        mapping_size_kb = os.path.getsize(mapping_path) / 1024
        print(f"\nFile sizes:")
        print(f"  Graph: {graph_size_kb:.2f} KB")
        print(f"  Entity mapping (Parquet): {mapping_size_kb:.2f} KB")
        if entity_embeddings:
            index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
            print(f"  Entity index: {index_size_mb:.2f} MB")

    @staticmethod
    def load(dataset_name: str) -> Dict[str, Any]:
        """Load graph processing results from disk

        Args:
            dataset_name: Name of the dataset (e.g., "hotpotqa")

        Returns:
            Dict containing graph, entity_mapping, entity_index, and triplet_sources
            Format: {
                "graph": NetworkX graph object,
                "entity_mapping": {entity_id: entity_text},
                "entity_index": FAISS index,
                "triplet_sources": {(s, p, o): [{"doc_id": int, "chunk_idx": int, "source_text": str}, ...]}
            }
        """
        # Load triplet_resource.json
        triplet_dir = PathConfig.get_triplet_path(dataset_name)
        resource_path = os.path.join(triplet_dir, "triplet_resource.json")

        triplet_sources = {}
        if os.path.exists(resource_path):
            with open(resource_path, 'r', encoding='utf-8') as f:
                resource_data = json.load(f)
            # Convert string keys back to tuples
            for key, sources in resource_data.items():
                parts = key.split("|||")
                if len(parts) == 3:
                    triplet_sources[tuple(parts)] = sources
            print(f"Loaded triplet resource: {len(triplet_sources)} triplets")

        # Load graph
        graph_dir = PathConfig.get_graph_path(dataset_name)
        graph_path = os.path.join(graph_dir, "graph.gpickle")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        # Load entity mapping from Parquet
        import pandas as pd

        mapping_path = os.path.join(graph_dir, "entity_mapping.parquet")

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Entity mapping file not found: {mapping_path}")

        df = pd.read_parquet(mapping_path, engine='pyarrow')

        # Convert DataFrame to dict
        entity_mapping = dict(zip(df['entity_id'], df['entity']))

        # Load entity index
        index_path = os.path.join(graph_dir, "entity_index.faiss")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Entity index file not found: {index_path}")

        import faiss
        entity_index = faiss.read_index(index_path)

        print(f"Loaded graph data for dataset: {dataset_name}")
        print(f"  Triplet sources: {len(triplet_sources)}")
        print(f"  Graph nodes: {graph.number_of_nodes()}")
        print(f"  Graph edges: {graph.number_of_edges()}")
        print(f"  Entity mapping: {len(entity_mapping)} entities")
        print(f"  Entity index: {entity_index.ntotal} vectors")

        return {
            "graph": graph,
            "entity_mapping": entity_mapping,
            "entity_index": entity_index,
            "triplet_sources": triplet_sources
        }


# Usage example
if __name__ == "__main__":
    from RAGCore.Chunk.ChunkDo import ChunkProcessor
    from RAGCore.Graph.GraphDo import GraphProcessor

    dataset_name = "hotpotqa"

    # Step 1: Load and chunk corpus
    chunk_processor = ChunkProcessor()
    corpus_path = PathConfig.get_corpus_path(dataset_name)
    corpus = chunk_processor.load_corpus(corpus_path)
    chunks_by_doc = chunk_processor.process_corpus(corpus)

    # Step 2: Process graph
    graph_processor = GraphProcessor()
    result = graph_processor.process(chunks_by_doc)

    print(f"\nProcessing Results:")
    print(f"Documents: {len(result['entities'])}")
    print(f"Total entities: {sum(len(e) for e in result['entities'].values())}")
    print(f"Total triplets: {sum(len(t) for t in result['triplets'].values())}")
    print(f"Graph nodes: {result['graph'].number_of_nodes()}")
    print(f"Graph edges: {result['graph'].number_of_edges()}")

    # Step 3: Save graph data
    GraphSaver.save(result, dataset_name)

    # Step 4: Load graph data (to verify)
    loaded_result = GraphSaver.load(dataset_name)

    print(f"\nVerification:")
    print(f"Original docs: {len(result['entities'])}")
    print(f"Loaded docs: {len(loaded_result['entities'])}")
    print(f"Original graph nodes: {result['graph'].number_of_nodes()}")
    print(f"Loaded graph nodes: {loaded_result['graph'].number_of_nodes()}")
