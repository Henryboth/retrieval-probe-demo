import faiss
import numpy as np
from typing import Dict, List, Any
from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver


class IndexProcessor:
    """Build FAISS index from embeddings for fast similarity search"""

    def __init__(self):
        pass

    def build_index(self, embeddings_by_doc: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Build FAISS index and metadata from embeddings

        Args:
            embeddings_by_doc: Output from EmbeddingSaver.load()
                              Format: {doc_id: {"chunks": [...], "embeddings": array}}

        Returns:
            {
                "index": faiss.Index,
                "metadata": [{"doc_id": 1, "text": "..."}, ...]
            }
        """
        all_embeddings = []
        metadata = []

        # Collect all embeddings and metadata
        for doc_id in sorted(embeddings_by_doc.keys()):
            doc_data = embeddings_by_doc[doc_id]
            chunks = doc_data["chunks"]
            embeddings = doc_data["embeddings"]

            # Add each chunk to metadata
            for i, chunk_text in enumerate(chunks):
                all_embeddings.append(embeddings[i])
                metadata.append({
                    "doc_id": doc_id,
                    "text": chunk_text
                })

        # Convert to numpy array
        embeddings_matrix = np.array(all_embeddings).astype('float32')

        # Build FAISS index
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors = cosine similarity)

        # Add vectors to index
        index.add(embeddings_matrix)

        print(f"Built FAISS index with {index.ntotal} vectors, dimension {dimension}")

        return {
            "index": index,
            "metadata": metadata
        }


# Usage example
if __name__ == "__main__":
    from Config.PathConfig import PathConfig

    dataset_name = "hotpotqa"

    # Step 1: Load embeddings
    print("Loading embeddings...")
    embeddings_by_doc = EmbeddingSaver.load(dataset_name)

    # Step 2: Build index
    print("\nBuilding FAISS index...")
    index_processor = IndexProcessor()
    index_data = index_processor.build_index(embeddings_by_doc)

    print(f"\nIndex built successfully:")
    print(f"  Total vectors: {index_data['index'].ntotal}")
    print(f"  Metadata entries: {len(index_data['metadata'])}")
    print(f"  First metadata entry: {index_data['metadata'][0]}")
