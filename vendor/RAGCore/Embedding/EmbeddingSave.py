import os
import numpy as np
from typing import Dict, Any
from Config.PathConfig import PathConfig


class EmbeddingSaver:
    """Save embeddings to compressed .npz format"""

    @staticmethod
    def save_incremental(doc_id: int, chunks: list, embeddings: np.ndarray, dataset_name: str):
        """Save embeddings for a single document (incremental save)

        Args:
            doc_id: Document ID
            chunks: List of chunk texts for this document
            embeddings: Embeddings array for this document
            dataset_name: Name of the dataset
        """
        save_dir = PathConfig.get_embedding_path(dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save individual document file
        doc_path = os.path.join(save_dir, f"doc_{doc_id}.npz")
        np.savez_compressed(
            doc_path,
            chunks=np.array(chunks, dtype=object),
            embeddings=embeddings
        )

    @staticmethod
    def load_existing(dataset_name: str) -> Dict[int, Dict[str, Any]]:
        """Load existing embeddings for resume capability

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict of {doc_id: {"chunks": [...], "embeddings": array}} or empty dict if not exists
        """
        save_dir = PathConfig.get_embedding_path(dataset_name)

        if not os.path.exists(save_dir):
            return {}

        embeddings_by_doc = {}

        # Load all doc_*.npz files
        for filename in os.listdir(save_dir):
            if filename.startswith("doc_") and filename.endswith(".npz"):
                doc_id = int(filename.replace("doc_", "").replace(".npz", ""))
                doc_path = os.path.join(save_dir, filename)

                data = np.load(doc_path, allow_pickle=True)
                embeddings_by_doc[doc_id] = {
                    "chunks": data["chunks"].tolist(),
                    "embeddings": data["embeddings"]
                }

        return embeddings_by_doc

    @staticmethod
    def save(embeddings_by_doc: Dict[int, Dict[str, Any]], dataset_name: str):
        """Save embeddings to disk in compressed format and clean up incremental files

        Args:
            embeddings_by_doc: Output from EmbeddingProcessor.process_chunks()
                              Format: {doc_id: {"chunks": [...], "embeddings": array}}
            dataset_name: Name of the dataset (e.g., "hotpotqa")
        """
        # Get save directory
        save_dir = PathConfig.get_embedding_path(dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # Prepare data for saving
        save_dict = {}
        for doc_id, data in embeddings_by_doc.items():
            save_dict[f'doc_{doc_id}_chunks'] = np.array(data['chunks'], dtype=object)
            save_dict[f'doc_{doc_id}_embeddings'] = data['embeddings']

        # Save as compressed .npz file
        save_path = os.path.join(save_dir, "embeddings.npz")
        np.savez_compressed(save_path, **save_dict)

        print(f"Saved embeddings to: {save_path}")

        # Print file size
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        # Clean up incremental files (doc_*.npz)
        print("Cleaning up incremental checkpoint files...")
        cleanup_count = 0
        for filename in os.listdir(save_dir):
            if filename.startswith("doc_") and filename.endswith(".npz"):
                file_path = os.path.join(save_dir, filename)
                os.remove(file_path)
                cleanup_count += 1

        if cleanup_count > 0:
            print(f"Removed {cleanup_count} incremental checkpoint files")

    @staticmethod
    def load(dataset_name: str) -> Dict[int, Dict[str, Any]]:
        """Load embeddings from disk

        Args:
            dataset_name: Name of the dataset (e.g., "hotpotqa")

        Returns:
            Dict mapping doc_id to chunk data with embeddings
            Format: {doc_id: {"chunks": [...], "embeddings": array}}
        """
        # Get load path
        load_dir = PathConfig.get_embedding_path(dataset_name)
        load_path = os.path.join(load_dir, "embeddings.npz")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")

        # Load compressed file
        data = np.load(load_path, allow_pickle=True)

        # Reconstruct embeddings_by_doc
        embeddings_by_doc = {}
        doc_ids = set()

        # Extract doc_ids from keys
        for key in data.keys():
            if key.startswith('doc_') and key.endswith('_chunks'):
                doc_id = int(key.replace('doc_', '').replace('_chunks', ''))
                doc_ids.add(doc_id)

        # Reconstruct data for each doc
        for doc_id in sorted(doc_ids):
            embeddings_by_doc[doc_id] = {
                'chunks': data[f'doc_{doc_id}_chunks'].tolist(),
                'embeddings': data[f'doc_{doc_id}_embeddings']
            }

        print(f"Loaded embeddings for {len(embeddings_by_doc)} documents from: {load_path}")

        return embeddings_by_doc


# Usage example
if __name__ == "__main__":
    from RAGCore.Chunk.ChunkDo import ChunkProcessor
    from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor

    dataset_name = "hotpotqa"

    # Step 1: Generate embeddings
    chunk_processor = ChunkProcessor()
    corpus_path = PathConfig.get_corpus_path(dataset_name)
    corpus = chunk_processor.load_corpus(corpus_path)
    chunks_by_doc = chunk_processor.process_corpus(corpus)

    embedding_processor = EmbeddingProcessor()
    embeddings_by_doc = embedding_processor.process_chunks(chunks_by_doc)

    # Step 2: Save embeddings
    EmbeddingSaver.save(embeddings_by_doc, dataset_name)

    # Step 3: Load embeddings (to verify)
    loaded_embeddings = EmbeddingSaver.load(dataset_name)

    print(f"\nVerification:")
    print(f"Original docs: {len(embeddings_by_doc)}")
    print(f"Loaded docs: {len(loaded_embeddings)}")
