class ChunkConfig:
    """Configuration for text chunking"""

    # 2026-09-20: CHUNK_SIZE 512 -> 180, CHUNK_OVERLAP 100 -> 36 (same ~20% overlap
    # ratio). Reason: pilot (Dataset/EvaluationData/Pilots/embedding_truncation/
    # summary.json, decision ESCALATE) found 512 cl100k-token chunks exceed
    # MiniLM's 256 word-piece input limit; re-chunking to fit lifts naive nDCG@5
    # at r=0 by +0.0162 (QuALITY) / +0.0058 (Legal). Old values: CHUNK_SIZE = 512,
    # CHUNK_OVERLAP = 100. Backed up to Config/ChunkConfig.py.pre_rechunk.
    CHUNK_SIZE = 180               # Number of tokens/characters per chunk
    CHUNK_OVERLAP = 36             # Overlap between consecutive chunks
    TOKENIZER_ENCODING = "cl100k_base"  # OpenAI's tokenizer encoding


# Usage example
if __name__ == "__main__":
    print(f"Chunk Size: {ChunkConfig.CHUNK_SIZE}")
    print(f"Chunk Overlap: {ChunkConfig.CHUNK_OVERLAP}")
