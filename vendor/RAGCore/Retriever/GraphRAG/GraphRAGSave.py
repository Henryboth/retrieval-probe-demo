"""
Graph RAG Result Saver

Handles saving and loading of Graph RAG QA results in JSONL format.
"""
import json
import os
from typing import Dict, List, Any, Optional

from Config.PathConfig import PathConfig


class GraphRAGSaver:
    """Save and load Graph RAG results"""

    @staticmethod
    def save_answer(answer_data: Dict[str, Any], model_name: str, dataset_name: str) -> None:
        """Save a single answer to JSONL file (incremental save)

        Args:
            answer_data: Dictionary with keys: id, rag_answer
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
        """
        # get_graph_rag_path returns the full file path (answer.jsonl), so get its directory
        answer_file = PathConfig.get_graph_rag_path(model_name, dataset_name)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        # Append to JSONL file
        with open(answer_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(answer_data, ensure_ascii=False) + '\n')

    @staticmethod
    def save_retrieval(retrieval_data: Dict[str, Any], model_name: str, dataset_name: str) -> None:
        """Save a single retrieval result to JSONL file (incremental save)

        Args:
            retrieval_data: Dictionary with keys: id, nodes, triplets
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
        """
        # get_graph_rag_path returns the full file path (answer.jsonl), so get its directory
        answer_file = PathConfig.get_graph_rag_path(model_name, dataset_name)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        retrieval_file = os.path.join(output_dir, "retrieval.jsonl")

        # Append to JSONL file
        with open(retrieval_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(retrieval_data, ensure_ascii=False) + '\n')

    @staticmethod
    def save_all(results: List[Dict[str, Any]], model_name: str, dataset_name: str) -> None:
        """Save all answer results to JSONL file

        Args:
            results: List of answer dictionaries
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
        """
        # get_graph_rag_path returns the full file path (answer.jsonl), so get its directory
        answer_file = PathConfig.get_graph_rag_path(model_name, dataset_name)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        # Write all answers to JSONL
        with open(answer_file, 'w', encoding='utf-8') as f:
            for answer in results:
                f.write(json.dumps(answer, ensure_ascii=False) + '\n')

        print(f"Results saved to: {output_dir}")
        print(f"  - answer.jsonl: {len(results)} answers")

    @staticmethod
    def save_all_retrievals(retrievals: List[Dict[str, Any]], model_name: str, dataset_name: str) -> None:
        """Save all retrieval results to JSONL file

        Args:
            retrievals: List of retrieval dictionaries
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
        """
        # get_graph_rag_path returns the full file path (answer.jsonl), so get its directory
        answer_file = PathConfig.get_graph_rag_path(model_name, dataset_name)
        output_dir = os.path.dirname(answer_file)
        PathConfig.ensure_dir(output_dir)

        retrieval_file = os.path.join(output_dir, "retrieval.jsonl")

        # Write all retrievals to JSONL
        with open(retrieval_file, 'w', encoding='utf-8') as f:
            for retrieval in retrievals:
                f.write(json.dumps(retrieval, ensure_ascii=False) + '\n')

        print(f"  - retrieval.jsonl: {len(retrievals)} retrieval results")

    @staticmethod
    def load_answers(model_name: str, dataset_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load existing answers from JSONL file

        Args:
            model_name: Name of the LLM model
            dataset_name: Name of the dataset

        Returns:
            List of answer dictionaries, or None if file doesn't exist
        """
        # get_graph_rag_path returns the full file path (answer.jsonl)
        answer_file = PathConfig.get_graph_rag_path(model_name, dataset_name)

        if not os.path.exists(answer_file):
            return None

        answers = []
        with open(answer_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    answers.append(json.loads(line))

        return answers

if __name__ == "__main__":
    # Test save and load
    test_results = [
        {"id": 0, "rag_answer": "yes"},
        {"id": 1, "rag_answer": "Chief of Protocol"},
        {"id": 2, "rag_answer": None}
    ]

    # Save
    GraphRAGSaver.save_all(test_results, "deepseek-chat", "test")

    # Load
    loaded = GraphRAGSaver.load_answers("deepseek-chat", "test")
    print(f"Loaded {len(loaded)} answers")
