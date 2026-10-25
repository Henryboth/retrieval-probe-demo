"""
Query Classification Save Module

Handles loading and saving of classified questions to QuestionType.json
"""
import os
import json
from typing import List, Dict, Any
from Config.PathConfig import PathConfig


class QueryClassifySaver:
    """Save and load question type classifications"""

    @staticmethod
    def get_question_type_path(dataset_name: str) -> str:
        """Get path to QuestionType.json

        Args:
            dataset_name: Name of dataset

        Returns:
            Path to QuestionType.json
        """
        raw_data_dir = os.path.join(PathConfig.RAW_DATA_DIR, dataset_name)
        return os.path.join(raw_data_dir, "QuestionType.json")

    @staticmethod
    def load_questions(dataset_name: str) -> List[Dict]:
        """Load questions from Question.json

        Args:
            dataset_name: Name of dataset (e.g., "UltraDomain_mix")

        Returns:
            List of question dictionaries
        """
        question_path = PathConfig.get_question_path(dataset_name)
        questions = []

        if not os.path.exists(question_path):
            raise FileNotFoundError(f"Question file not found: {question_path}")

        with open(question_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

        print(f"Loaded {len(questions)} questions from {question_path}")
        return questions

    @staticmethod
    def load_question_types(dataset_name: str, scheme: str = None) -> Dict[int, Dict]:
        """Load existing question types from QuestionType.json

        Args:
            dataset_name: Name of dataset
            scheme: If provided, only return types for this scheme

        Returns:
            Dict of {question_id: {scheme: type, ...}} or {question_id: type} if scheme specified
        """
        type_path = QueryClassifySaver.get_question_type_path(dataset_name)

        if not os.path.exists(type_path):
            return {}

        with open(type_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert string keys to int
        result = {int(k): v for k, v in data.items()}

        if scheme:
            # Return only the specified scheme's types
            return {qid: types.get(scheme) for qid, types in result.items() if scheme in types}

        return result

    @staticmethod
    def save_question_types(classifications: Dict[int, str], dataset_name: str, scheme: str):
        """Save question type classifications to QuestionType.json

        Args:
            classifications: Dict of {question_id: type}
            dataset_name: Name of dataset
            scheme: Classification scheme name (e.g., "memorag")
        """
        type_path = QueryClassifySaver.get_question_type_path(dataset_name)

        # Load existing data
        existing = {}
        if os.path.exists(type_path):
            with open(type_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                # Convert to int keys for merging
                existing = {int(k): v for k, v in existing.items()}

        # Merge new classifications
        for qid, qtype in classifications.items():
            if qid not in existing:
                existing[qid] = {}
            existing[qid][scheme] = qtype

        # Sort by question id and convert back to string keys for JSON
        sorted_data = {str(k): existing[k] for k in sorted(existing.keys())}

        # Save
        with open(type_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(classifications)} classifications to {type_path}")

    @staticmethod
    def extract_from_questions(dataset_name: str, scheme: str) -> Dict[int, str]:
        """Extract question_type from Question.json if it exists

        Args:
            dataset_name: Name of dataset
            scheme: Classification scheme name

        Returns:
            Dict of {question_id: type} extracted from Question.json
        """
        questions = QueryClassifySaver.load_questions(dataset_name)
        extracted = {}

        for q in questions:
            if "question_type" in q:
                extracted[q["id"]] = q["question_type"]

        if extracted:
            print(f"Extracted {len(extracted)} existing classifications from Question.json")

        return extracted

    @staticmethod
    def get_statistics(dataset_name: str, scheme: str) -> Dict[str, Any]:
        """Get classification statistics

        Args:
            dataset_name: Name of dataset
            scheme: Classification scheme

        Returns:
            Statistics dict
        """
        types = QueryClassifySaver.load_question_types(dataset_name, scheme)

        if not types:
            return {"total": 0, "statistics": {}, "distribution": {}}

        # Count types
        stats = {}
        for qtype in types.values():
            if qtype:
                stats[qtype] = stats.get(qtype, 0) + 1

        total = len(types)
        distribution = {k: round(v / total * 100, 2) for k, v in stats.items()}

        return {
            "dataset": dataset_name,
            "scheme": scheme,
            "total": total,
            "statistics": stats,
            "distribution": distribution
        }


# Usage example
if __name__ == "__main__":
    dataset_name = "UltraDomain_mix"
    scheme = "memorag"

    # Test loading
    try:
        # Check if QuestionType.json exists
        types = QueryClassifySaver.load_question_types(dataset_name, scheme)
        print(f"\nLoaded {len(types)} classifications for scheme '{scheme}'")

        # Get statistics
        stats = QueryClassifySaver.get_statistics(dataset_name, scheme)
        print(f"\nStatistics: {stats}")

    except Exception as e:
        print(f"Error: {e}")
