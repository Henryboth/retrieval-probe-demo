import os

class PathConfig:
    """Configuration for all file paths"""

    # Root Directories
    PROJECT_ROOT = ""
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "Dataset")


    # Raw Data Paths
    RAW_DATA_DIR = os.path.join(DATASET_ROOT, "RawData")

    @staticmethod
    def get_corpus_path(dataset_name: str) -> str:
        """Get corpus file path for a dataset"""
        return os.path.join(PathConfig.RAW_DATA_DIR, dataset_name, "Corpus.json")

    @staticmethod
    def get_question_path(dataset_name: str) -> str:
        """Get question file path for a dataset"""
        return os.path.join(PathConfig.RAW_DATA_DIR, dataset_name, "Question.json")


    # Processed Data Paths
    PROCESSED_DATA_DIR = os.path.join(DATASET_ROOT, "ProcessedData")

    # Subdirectories for different processed data types
    TRIPLET_DIR = os.path.join(PROCESSED_DATA_DIR, "Triplet")
    EMBEDDING_DIR = os.path.join(PROCESSED_DATA_DIR, "Embedding")
    INDEX_DIR = os.path.join(PROCESSED_DATA_DIR, "Index")
    GRAPH_DIR = os.path.join(PROCESSED_DATA_DIR, "Graph")

    @staticmethod
    def get_triplet_path(dataset_name: str) -> str:
        """Get triplet extraction result path for a dataset"""
        return os.path.join(PathConfig.TRIPLET_DIR, dataset_name)

    @staticmethod
    def get_embedding_path(dataset_name: str) -> str:
        """Get embedding result path for a dataset"""
        return os.path.join(PathConfig.EMBEDDING_DIR, dataset_name)

    @staticmethod
    def get_index_path(dataset_name: str) -> str:
        """Get index result path for a dataset"""
        return os.path.join(PathConfig.INDEX_DIR, dataset_name)

    @staticmethod
    def get_graph_path(dataset_name: str) -> str:
        """Get graph result path for a dataset"""
        return os.path.join(PathConfig.GRAPH_DIR, dataset_name)


    # Retrieval Result Data Paths
    RETRIEVAL_RESULT_DIR = os.path.join(DATASET_ROOT, "RetrievalResultData")

    @staticmethod
    def get_llm_direct_path(model_name: str, dataset_name: str) -> str:
        """Get LLM Direct result path
        Returns: /Dataset/RetrievalResultData/LLMDirect/{model}/{dataset}/answer.jsonl
        """
        result_dir = os.path.join(PathConfig.RETRIEVAL_RESULT_DIR, "LLMDirect", model_name, dataset_name)
        return os.path.join(result_dir, "answer.jsonl")

    @staticmethod
    def get_naive_rag_path(model_name: str, dataset_name: str) -> str:
        """Get Naive RAG result path
        Returns: /Dataset/RetrievalResultData/NaiveRAG/{model}/{dataset}/answer.jsonl
        """
        result_dir = os.path.join(PathConfig.RETRIEVAL_RESULT_DIR, "NaiveRAG", model_name, dataset_name)
        return os.path.join(result_dir, "answer.jsonl")

    @staticmethod
    def get_graph_rag_path(model_name: str, dataset_name: str) -> str:
        """Get Graph RAG result path
        Returns: /Dataset/RetrievalResultData/GraphRAG/{model}/{dataset}/answer.jsonl
        """
        result_dir = os.path.join(PathConfig.RETRIEVAL_RESULT_DIR, "GraphRAG", model_name, dataset_name)
        return os.path.join(result_dir, "answer.jsonl")

    @staticmethod
    def get_hybrid_rag_path(model_name: str, dataset_name: str) -> str:
        """Get Hybrid RAG result path
        Returns: /Dataset/RetrievalResultData/HybridRAG/{model}/{dataset}/answer.jsonl
        """
        result_dir = os.path.join(PathConfig.RETRIEVAL_RESULT_DIR, "HybridRAG", model_name, dataset_name)
        return os.path.join(result_dir, "answer.jsonl")

    @staticmethod
    def get_iterative_rag_path(model_name: str, dataset_name: str, retriever_type: str = "naive") -> str:
        """Get Iterative RAG result path

        Args:
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")

        Returns: /Dataset/RetrievalResultData/IterativeRAG/{model}/{retriever_type}/{dataset}/answer.jsonl
        """
        result_dir = os.path.join(PathConfig.RETRIEVAL_RESULT_DIR, "IterativeRAG", model_name, retriever_type, dataset_name)
        return os.path.join(result_dir, "answer.jsonl")


    # Query Generation Data Paths
    QUERY_GENERATION_DIR = os.path.join(DATASET_ROOT, "QueryGenerationData")

    @staticmethod
    def get_query_generation_dir(dataset_name: str) -> str:
        """Get query generation directory for a dataset
        Returns: /Dataset/QueryGenerationData/{dataset}/
        """
        return os.path.join(PathConfig.QUERY_GENERATION_DIR, dataset_name)

    @staticmethod
    def get_query_raw_dir(dataset_name: str) -> str:
        """Get raw query output directory
        Returns: /Dataset/QueryGenerationData/{dataset}/raw/
        """
        return os.path.join(PathConfig.QUERY_GENERATION_DIR, dataset_name, "raw")

    @staticmethod
    def get_query_validation_dir(dataset_name: str) -> str:
        """Get validation result directory
        Returns: /Dataset/QueryGenerationData/{dataset}/validation/
        """
        return os.path.join(PathConfig.QUERY_GENERATION_DIR, dataset_name, "validation")

    @staticmethod
    def get_query_final_path(dataset_name: str) -> str:
        """Get final questions.json path
        Returns: /Dataset/QueryGenerationData/{dataset}/questions.json
        """
        return os.path.join(PathConfig.QUERY_GENERATION_DIR, dataset_name, "questions.json")


    # Evaluation Data Paths

    EVALUATION_DATA_DIR = os.path.join(DATASET_ROOT, "EvaluationData")
    RESULT_EVAL_DIR = os.path.join(EVALUATION_DATA_DIR, "ResultEvaluation")
    CORPUS_EVAL_DIR = os.path.join(EVALUATION_DATA_DIR, "CorpusEvaluation")

    @staticmethod
    def get_result_eval_path(model_name: str, dataset_name: str, method: str, retriever_type: str = None) -> str:
        """Get result evaluation path

        Args:
            model_name: Model name (e.g., "deepseek-chat")
            dataset_name: Dataset name
            method: "naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")

        Returns: /Dataset/EvaluationData/ResultEvaluation/{model}/{dataset}/{method}.json
                 For iterative_rag: {method}_{retriever_type}.json
        """
        eval_dir = os.path.join(PathConfig.RESULT_EVAL_DIR, model_name, dataset_name)
        # For iterative_rag, include retriever_type in filename
        if method == "iterative_rag" and retriever_type:
            return os.path.join(eval_dir, f"{method}_{retriever_type}.json")
        return os.path.join(eval_dir, f"{method}.json")


    # Corpus Evaluation Paths
    @staticmethod
    def get_corpus_eval_path(dataset_name: str, eval_type: str = None) -> str:
        """Get corpus evaluation result path"""
        if eval_type:
            return os.path.join(PathConfig.CORPUS_EVAL_DIR, eval_type, dataset_name)
        return os.path.join(PathConfig.CORPUS_EVAL_DIR, dataset_name)


    # Cache Paths
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
    LLM_CACHE_DIR = os.path.join(CACHE_DIR, "llm_responses")


    # Log Paths
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


    # Helper Methods
    @staticmethod
    def ensure_dir(path: str) -> None:
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def ensure_all_dirs() -> None:
        """Create all necessary directories"""
        dirs = [
            # Raw and Processed Data
            PathConfig.RAW_DATA_DIR,
            PathConfig.PROCESSED_DATA_DIR,
            PathConfig.TRIPLET_DIR,
            PathConfig.EMBEDDING_DIR,
            PathConfig.INDEX_DIR,
            PathConfig.GRAPH_DIR,
            # Retrieval Results
            PathConfig.RETRIEVAL_RESULT_DIR,
            # Query Generation
            PathConfig.QUERY_GENERATION_DIR,
            # Evaluation Data
            PathConfig.EVALUATION_DATA_DIR,
            PathConfig.RESULT_EVAL_DIR,
            PathConfig.CORPUS_EVAL_DIR,
            # Cache and Logs
            PathConfig.CACHE_DIR,
            PathConfig.EMBEDDING_CACHE_DIR,
            PathConfig.LLM_CACHE_DIR,
            PathConfig.LOG_DIR,
        ]
        for d in dirs:
            PathConfig.ensure_dir(d)


# Usage example
if __name__ == "__main__":
    dataset = "hotpotqa"
    model = "deepseek-chat"
    print(f"Corpus Path: {PathConfig.get_corpus_path(dataset)}")
    print(f"Question Path: {PathConfig.get_question_path(dataset)}")
    print(f"Triplet Path: {PathConfig.get_triplet_path(dataset)}")
    print(f"Embedding Path: {PathConfig.get_embedding_path(dataset)}")
    print(f"Result Eval Path: {PathConfig.get_result_eval_path(model, dataset, 'naive_rag')}")
