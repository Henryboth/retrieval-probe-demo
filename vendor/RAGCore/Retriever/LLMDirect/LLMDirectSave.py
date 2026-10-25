"""
LLM Direct Result Saver

Handles saving and loading of LLM Direct QA results in JSONL format.
"""
import json
import os
from typing import Dict, List, Any, Optional

from Config.PathConfig import PathConfig


class LLMDirectSaver:
    """Save and load LLM Direct results"""

    @staticmethod
    def save_answer(answer_data: Dict[str, Any], model_name: str, dataset_name: str) -> None:
        """Save a single answer to JSONL file (incremental save)

        Args:
            answer_data: Dictionary with keys: id, llm_answer
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
        """
        answer_file = PathConfig.get_llm_direct_path(model_name, dataset_name)
        PathConfig.ensure_dir(os.path.dirname(answer_file))

        # Append to JSONL file
        with open(answer_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(answer_data, ensure_ascii=False) + '\n')

    @staticmethod
    def save_all(results: List[Dict[str, Any]], model_name: str, dataset_name: str) -> None:
        """Save all results to JSONL file

        Args:
            results: List of answer dictionaries
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
        """
        answer_file = PathConfig.get_llm_direct_path(model_name, dataset_name)
        PathConfig.ensure_dir(os.path.dirname(answer_file))

        # Write all answers to JSONL
        with open(answer_file, 'w', encoding='utf-8') as f:
            for answer in results:
                f.write(json.dumps(answer, ensure_ascii=False) + '\n')

        print(f"Results saved to: {answer_file}")
        print(f"  - answer.jsonl: {len(results)} answers")

    @staticmethod
    def load_answers(model_name: str, dataset_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load existing answers from JSONL file

        Args:
            model_name: Name of the LLM model
            dataset_name: Name of the dataset

        Returns:
            List of answer dictionaries, or None if file doesn't exist
        """
        answer_file = PathConfig.get_llm_direct_path(model_name, dataset_name)

        if not os.path.exists(answer_file):
            return None

        answers = []
        with open(answer_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    answers.append(json.loads(line))

        return answers


# Usage example
if __name__ == "__main__":
    # Test save and load
    test_results = [
        {"id": 0, "llm_answer": "yes"},
        {"id": 1, "llm_answer": "Chief of Protocol"},
        {"id": 2, "llm_answer": None}
    ]

    # Save
    LLMDirectSaver.save_all(test_results, "deepseek-chat", "test")

    # Load
    loaded = LLMDirectSaver.load_answers("deepseek-chat", "test")
    print(f"Loaded {len(loaded)} answers")
