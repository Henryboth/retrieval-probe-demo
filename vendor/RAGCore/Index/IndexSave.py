import os
import json
import faiss
from typing import Dict, List, Any
from Config.PathConfig import PathConfig


class IndexSaver:
    """Save and load FAISS index and metadata"""

    @staticmethod
    def save(index_data: Dict[str, Any], dataset_name: str):
        """Save FAISS index and metadata to disk

        Args:
            index_data: Output from IndexProcessor.build_index()
                       Format: {"index": faiss.Index, "metadata": [...]}
            dataset_name: Name of the dataset (e.g., "hotpotqa")
        """
        index = index_data["index"]
        metadata = index_data["metadata"]

        # Get save directory
        save_dir = PathConfig.get_index_path(dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(save_dir, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"Saved FAISS index to: {index_path}")

        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to: {metadata_path}")

        # Print file sizes
        index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
        metadata_size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
        print(f"Index file size: {index_size_mb:.2f} MB")
        print(f"Metadata file size: {metadata_size_mb:.2f} MB")

    @staticmethod
    def load(dataset_name: str) -> Dict[str, Any]:
        """Load FAISS index and metadata from disk

        Args:
            dataset_name: Name of the dataset (e.g., "hotpotqa")

        Returns:
            Dict containing index and metadata
            Format: {"index": faiss.Index, "metadata": [...]}
        """
        # Get load directory
        load_dir = PathConfig.get_index_path(dataset_name)

        # Load FAISS index
        index_path = os.path.join(load_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        index = faiss.read_index(index_path)
        print(f"Loaded FAISS index from: {index_path}")

        # Load metadata
        metadata_path = os.path.join(load_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from: {metadata_path}")

        print(f"Index contains {index.ntotal} vectors")
        print(f"Metadata contains {len(metadata)} entries")

        return {
            "index": index,
            "metadata": metadata
        }


# Usage example
if __name__ == "__main__":
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
    from RAGCore.Index.IndexDo import IndexProcessor

    dataset_name = "hotpotqa"

    # Step 1: Load embeddings
    print("Loading embeddings...")
    embeddings_by_doc = EmbeddingSaver.load(dataset_name)

    # Step 2: Build index
    print("\nBuilding FAISS index...")
    index_processor = IndexProcessor()
    index_data = index_processor.build_index(embeddings_by_doc)

    print(f"\nIndex built:")
    print(f"  Total vectors: {index_data['index'].ntotal}")
    print(f"  Metadata entries: {len(index_data['metadata'])}")

    # Step 3: Save index
    print("\nSaving index...")
    IndexSaver.save(index_data, dataset_name)

    # Step 4: Load index (to verify)
    print("\nLoading index to verify...")
    loaded_index_data = IndexSaver.load(dataset_name)

    print(f"\nVerification:")
    print(f"  Original vectors: {index_data['index'].ntotal}")
    print(f"  Loaded vectors: {loaded_index_data['index'].ntotal}")
    print(f"  Original metadata: {len(index_data['metadata'])}")
    print(f"  Loaded metadata: {len(loaded_index_data['metadata'])}")
