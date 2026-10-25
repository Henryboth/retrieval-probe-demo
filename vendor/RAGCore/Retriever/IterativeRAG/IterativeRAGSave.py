"""
Iterative RAG Result Saver

Handles saving and loading of Iterative RAG QA results in JSONL format.
"""
import json
import os
from typing import Dict, List, Any, Optional

from Config.PathConfig import PathConfig


class IterativeRAGSaver:
    """Save and load Iterative RAG results"""

    @staticmethod
    def save_answer(answer_data: Dict[str, Any], model_name: str, dataset_name: str, retriever_type: str = "naive") -> None:
        """Save a single answer to JSONL file (incremental save)

        Args:
            answer_data: Dictionary with keys: id, rag_answer, rounds
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")
        """
        answer_file = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        # Append to JSONL file
        with open(answer_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(answer_data, ensure_ascii=False) + '\n')

    @staticmethod
    def save_retrieval(retrieval_data: Dict[str, Any], model_name: str, dataset_name: str, retriever_type: str = "naive") -> None:
        """Save a single retrieval result to JSONL file (incremental save)

        Args:
            retrieval_data: Dictionary with keys: id, retrieved_chunks, history
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")
        """
        answer_file = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        retrieval_file = os.path.join(output_dir, "retrieval.jsonl")

        # Append to JSONL file
        with open(retrieval_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(retrieval_data, ensure_ascii=False) + '\n')

    @staticmethod
    def save_all(results: List[Dict[str, Any]], model_name: str, dataset_name: str, retriever_type: str = "naive") -> None:
        """Save all answer results to JSONL file

        Args:
            results: List of answer dictionaries
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")
        """
        answer_file = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        # Write all answers to JSONL
        with open(answer_file, 'w', encoding='utf-8') as f:
            for answer in results:
                f.write(json.dumps(answer, ensure_ascii=False) + '\n')

        print(f"Results saved to: {output_dir}")
        print(f"  - answer.jsonl: {len(results)} answers")

    @staticmethod
    def save_all_retrievals(retrievals: List[Dict[str, Any]], model_name: str, dataset_name: str, retriever_type: str = "naive") -> None:
        """Save all retrieval results to JSONL file

        Args:
            retrievals: List of retrieval dictionaries
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")
        """
        answer_file = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        retrieval_file = os.path.join(output_dir, "retrieval.jsonl")

        # Write all retrievals to JSONL
        with open(retrieval_file, 'w', encoding='utf-8') as f:
            for retrieval in retrievals:
                f.write(json.dumps(retrieval, ensure_ascii=False) + '\n')

        print(f"  - retrieval.jsonl: {len(retrievals)} retrieval results")

    @staticmethod
    def load_answers(model_name: str, dataset_name: str, retriever_type: str = "naive") -> Optional[List[Dict[str, Any]]]:
        """Load existing answers from JSONL file

        Args:
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")

        Returns:
            List of answer dictionaries, or None if file doesn't exist
        """
        answer_file = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)

        if not os.path.exists(answer_file):
            return None

        answers = []
        with open(answer_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    answers.append(json.loads(line))

        return answers

    @staticmethod
    def load_retrievals(model_name: str, dataset_name: str, retriever_type: str = "naive") -> Optional[List[Dict[str, Any]]]:
        """Load existing retrieval results from JSONL file

        Args:
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            retriever_type: Type of base retriever ("naive" or "graph")

        Returns:
            List of retrieval dictionaries, or None if file doesn't exist
        """
        answer_file = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        output_dir = os.path.dirname(answer_file)
        retrieval_file = os.path.join(output_dir, "retrieval.jsonl")

        if not os.path.exists(retrieval_file):
            return None

        retrievals = []
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    retrievals.append(json.loads(line))

        return retrievals


# Usage example
if __name__ == "__main__":
    # Test save and load
    test_results = [
        {"id": 0, "rag_answer": "Basal cell carcinoma", "rounds": 1},
        {"id": 1, "rag_answer": "Surgery is the most common treatment", "rounds": 2},
        {"id": 2, "rag_answer": "I cannot answer", "rounds": 3}
    ]

    # Save
    IterativeRAGSaver.save_all(test_results, "deepseek-chat", "test")

    # Load
    loaded = IterativeRAGSaver.load_answers("deepseek-chat", "test")
    print(f"Loaded {len(loaded)} answers")
