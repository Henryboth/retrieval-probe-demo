"""
Configuration for Retrieval Methods (Naive RAG, Graph RAG, etc.)
"""


class RetrieverConfig:
    """Configuration for retrieval-based QA methods"""

    # Retrieval Parameters
    TOP_K = 100                                  # Maximum chunks to retrieve (upper bound)
    SIMILARITY_METRIC = "cosine"                # "cosine", "dot_product", "euclidean"
    MIN_SIMILARITY_THRESHOLD = 0.4              # Minimum similarity score to include a chunk (same as GraphRAG)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count using simple approximation (len/4)
        Args:
            text: Input text
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) // 4


    # Context Construction (Token Budget Matching)
    CHUNK_SEPARATOR = "\n\n"                    # Separator between retrieved chunks
    MAX_CONTEXT_LENGTH = 4000                   # Maximum context length in characters (deprecated, use TOKEN_BUDGET)
    INCLUDE_CHUNK_METADATA = False              # Whether to include chunk metadata (doc_id, chunk_id)

    # Token Budget for fair comparison between methods
    # Both NaiveRAG and GraphRAG will use the same token budget
    # Content is sorted by similarity (highest first), so reducing budget keeps the most relevant content
    CONTEXT_TOKEN_BUDGET = 8000                 # Maximum tokens for context (reduced to save API costs)

    # LLM for RAG (Answer Generation)
    # Note: Uses LLMConfig.PROVIDER and model settings
    # This section is for RAG-specific LLM parameters
    RAG_TEMPERATURE = 0.3                       # Lower temperature for more focused answers
    RAG_MAX_TOKENS = 1000                        # Maximum tokens for RAG answers
    RAG_TIMEOUT = 120                            # Request timeout in seconds

    # Graph RAG Specific Parameters
    GRAPH_SEED_ENTITIES = 20                     # Top-k seed entities to retrieve
    GRAPH_ENTITY_SIMILARITY_THRESHOLD = 0.4     # Minimum similarity to include entity (> threshold)

    # PPR-based retrieval (no BFS hop limit, adaptive exploration)
    GRAPH_PPR_MAX_NODES = 100                   # Maximum nodes to keep after PPR
    GRAPH_PPR_ALPHA = 0.85                      # PPR damping factor (higher = more local)
    GRAPH_PPR_MIN_SCORE = 1e-5                  # Minimum PPR score to include node
    GRAPH_MAX_TRIPLETS = 500                    # Maximum triplets to serialize

    # Reranking (Optional - for future use)
    ENABLE_RERANKING = False                    # Whether to rerank retrieved chunks
    RERANK_TOP_K = 10                            # Number of chunks after reranking

    # Caching
    CACHE_EMBEDDINGS = True                     # Cache query embeddings
    CACHE_RETRIEVAL_RESULTS = False             # Cache retrieval results (for debugging)

    # Iterative RAG Parameters
    ITERATIVE_RETRIEVER = "graph"               # "naive" or "graph"
    ITERATIVE_MAX_ITERATIONS = 3                # Maximum retrieval iterations
    ITERATIVE_EVAL_TEMPERATURE = 0.1            # Temperature for evaluator LLM


# Usage example
if __name__ == "__main__":
    print(f"Top-K: {RetrieverConfig.TOP_K}")
    print(f"Similarity Metric: {RetrieverConfig.SIMILARITY_METRIC}")
    print(f"RAG Temperature: {RetrieverConfig.RAG_TEMPERATURE}")
    print(f"RAG Max Tokens: {RetrieverConfig.RAG_MAX_TOKENS}")
    print(f"Context Separator: {repr(RetrieverConfig.CHUNK_SEPARATOR)}")
    print(f"Max Context Length: {RetrieverConfig.MAX_CONTEXT_LENGTH}")
