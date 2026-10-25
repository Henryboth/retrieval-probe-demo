class GraphConfig:
    """Configuration for knowledge graph construction"""

    # Graph Type
    DIRECTED = False  # Use undirected graph for bidirectional traversal

    # Async Triplet Extraction 
    MAX_CONCURRENT = 15            # Maximum concurrent requests (10-20 recommended for DeepSeek API)
    REQUEST_TIMEOUT = 60           # Request timeout in seconds (increased from 30 to handle complex chunks)
    MAX_RETRIES = 3                # Maximum retry attempts for failed requests
    RETRY_DELAY = 2                # Base delay for exponential backoff retry (seconds)

    # Batch Processing
    BATCH_SIZE = 500               # Number of documents to process per batch (for memory control and incremental save)
