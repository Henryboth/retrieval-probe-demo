import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("")

class LLMConfig:
    """Configuration for Large Language Models"""

    # LLM Provider
    # Options: "openai", "anthropic", "deepseek", "llama" (local vLLM)
    PROVIDER = "llama"


    # Model Selection
    OPENAI_MODEL = "gpt-4o"                    # or "gpt-4", "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    DEEPSEEK_MODEL = "deepseek-chat"
    HUGGINGFACE_MODEL = "meta-llama/Llama-2-70b-chat-hf"
    LLAMA_MODEL = "llama3.1:8b"
    LLAMA_MODEL_NAME = "llama3.1:8b"  # Simple name for file paths (no slashes)


    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


    # API Base URLs
    OPENAI_BASE_URL = "https://api.openai.com/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
    LLAMA_BASE_URL = "http://localhost:11434/v1"


    # Generation Parameters
    TEMPERATURE = 0.7                          # Randomness (0.0-2.0)
    MAX_TOKENS = 1000                          # Maximum tokens in response
    TOP_P = 0.9                                # Nucleus sampling


    # API Request Settings
    TIMEOUT = 60                               # Request timeout in seconds
    MAX_RETRIES = 3                            # Number of retries on failure
    RETRY_DELAY = 2                            # Delay between retries in seconds


    # Response Format
    STREAM = False                             # Stream responses
    RESPONSE_FORMAT = "text"                   # "text" or "json"


    # Helper Methods
    @staticmethod
    def get_model_config():
        """Get current model configuration based on PROVIDER

        Returns:
            Dictionary with keys: model, api_key, base_url
        """
        provider = LLMConfig.PROVIDER.lower()

        if provider == "openai":
            return {
                "model": LLMConfig.OPENAI_MODEL,
                "api_key": LLMConfig.OPENAI_API_KEY,
                "base_url": LLMConfig.OPENAI_BASE_URL
            }
        elif provider == "deepseek":
            return {
                "model": LLMConfig.DEEPSEEK_MODEL,
                "api_key": LLMConfig.DEEPSEEK_API_KEY,
                "base_url": LLMConfig.DEEPSEEK_BASE_URL
            }
        elif provider == "anthropic":
            return {
                "model": LLMConfig.ANTHROPIC_MODEL,
                "api_key": LLMConfig.ANTHROPIC_API_KEY,
                "base_url": LLMConfig.ANTHROPIC_BASE_URL
            }
        elif provider == "llama":
            return {
                "model": LLMConfig.LLAMA_MODEL,
                "model_name": LLMConfig.LLAMA_MODEL_NAME,  # For file paths
                "api_key": "not-needed",  # vLLM doesn't require API key
                "base_url": LLMConfig.LLAMA_BASE_URL
            }
        else:
            raise ValueError(f"Unsupported provider: {LLMConfig.PROVIDER}")


# Usage example
if __name__ == "__main__":
    print(f"Provider: {LLMConfig.PROVIDER}")
    print(f"Model: {LLMConfig.OPENAI_MODEL}")
    print(f"Temperature: {LLMConfig.TEMPERATURE}")
    print(f"Max Tokens: {LLMConfig.MAX_TOKENS}")
