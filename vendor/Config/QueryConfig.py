"""
Configuration for Query Classification
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("")


class QueryConfig:
    """Configuration for query classification methods"""
    # LLM Configuration for Classification
    # Uses DeepSeek by default for cost efficiency
    PROVIDER = "deepseek"

    # API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = "https://api.openai.com/v1"
    OPENAI_MODEL = "gpt-4o-mini"

    # Classification Parameters
    TEMPERATURE = 0.0                # Deterministic classification
    MAX_TOKENS = 50                  # Short response expected
    TIMEOUT = 30                     # Request timeout in seconds

    # Batch Processing
    BATCH_SIZE = 10                  # Questions to process before progress update
    REQUEST_DELAY = 0.1              # Delay between requests (rate limiting)
    MAX_RETRIES = 3                  # Retries on failure
    RETRY_DELAY = 2                  # Delay between retries in seconds


    # MemoRAG Classification Scheme
    # Based on MemoRAG paper's task categorization
    MEMORAG_TYPES = {
        "distributed": "Distributed Information Gathering",
        "query_focused": "Query-Focused Summarization",
        "full_context": "Full Context Summarization"
    }

    # Helper Methods
    @staticmethod
    def get_llm_config():
        """Get LLM configuration based on PROVIDER
        Returns:
            Dictionary with keys: model, api_key, base_url
        """
        provider = QueryConfig.PROVIDER.lower()

        if provider == "deepseek":
            return {
                "model": QueryConfig.DEEPSEEK_MODEL,
                "api_key": QueryConfig.DEEPSEEK_API_KEY,
                "base_url": QueryConfig.DEEPSEEK_BASE_URL
            }
        elif provider == "openai":
            return {
                "model": QueryConfig.OPENAI_MODEL,
                "api_key": QueryConfig.OPENAI_API_KEY,
                "base_url": QueryConfig.OPENAI_BASE_URL
            }
        else:
            raise ValueError(f"Unsupported provider: {QueryConfig.PROVIDER}")


# Usage example
if __name__ == "__main__":
    print(f"Provider: {QueryConfig.PROVIDER}")
    config = QueryConfig.get_llm_config()
    print(f"Model: {config['model']}")
    print(f"Temperature: {QueryConfig.TEMPERATURE}")
    print(f"MemoRAG Types: {QueryConfig.MEMORAG_TYPES}")
