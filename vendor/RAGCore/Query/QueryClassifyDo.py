"""
Query Classification Module

Supports multiple classification schemes from different papers.
Currently implemented:
- MemoRAG: Distributed Information Gathering, Query-Focused Summarization, Full Context Summarization
"""
import time
from openai import OpenAI
from typing import List, Dict, Optional
from Config.QueryConfig import QueryConfig


class QueryClassifier:
    """Classify questions using LLM based on different classification schemes"""

    def __init__(self, scheme: str = "memorag"):
        """Initialize classifier with specified scheme

        Args:
            scheme: Classification scheme to use ("memorag", etc.)
        """
        self.scheme = scheme.lower()

        # Initialize LLM client
        config = QueryConfig.get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model = config["model"]

        # Get scheme-specific settings
        if self.scheme == "memorag":
            self.prompt_template = QueryConfig.MEMORAG_PROMPT
            self.valid_types = list(QueryConfig.MEMORAG_TYPES.keys())
            self.type_names = QueryConfig.MEMORAG_TYPES
            self.default_type = "query_focused"
        else:
            raise ValueError(f"Unknown classification scheme: {scheme}")

    def classify_single(self, question: str, answer: str) -> str:
        """Classify a single question

        Args:
            question: The question text
            answer: The answer text (used for context)

        Returns:
            Classification type string
        """
        prompt = self.prompt_template.format(question=question, answer=answer)

        for attempt in range(QueryConfig.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=QueryConfig.TEMPERATURE,
                    max_tokens=QueryConfig.MAX_TOKENS,
                    timeout=QueryConfig.TIMEOUT
                )

                result = response.choices[0].message.content.strip().lower()

                # Validate and extract result
                if result in self.valid_types:
                    return result

                # Try to extract valid type from response
                for qtype in self.valid_types:
                    if qtype in result:
                        return qtype

                print(f"Warning: Invalid classification '{result}', using default")
                return self.default_type

            except Exception as e:
                if attempt < QueryConfig.MAX_RETRIES - 1:
                    print(f"Retry {attempt + 1}/{QueryConfig.MAX_RETRIES} after error: {e}")
                    time.sleep(QueryConfig.RETRY_DELAY)
                else:
                    print(f"Error classifying question after {QueryConfig.MAX_RETRIES} attempts: {e}")
                    return self.default_type

    def classify_batch(self, questions: List[Dict],
                       question_key: str = "question",
                       answer_key: str = "answer") -> List[Dict]:
        """Classify a batch of questions

        Args:
            questions: List of question dictionaries
            question_key: Key for question text in dict
            answer_key: Key for answer text in dict

        Returns:
            List of questions with added 'question_type' field
        """
        total = len(questions)
        classified = []

        for i, q in enumerate(questions):
            qtype = self.classify_single(q[question_key], q[answer_key])
            q["question_type"] = qtype
            classified.append(q)

            if (i + 1) % QueryConfig.BATCH_SIZE == 0:
                print(f"Progress: {i + 1}/{total} questions classified")

            # Rate limiting
            time.sleep(QueryConfig.REQUEST_DELAY)

        return classified

    def get_statistics(self, questions: List[Dict]) -> Dict[str, int]:
        """Get classification statistics

        Args:
            questions: List of classified questions

        Returns:
            Dict of {type: count}
        """
        stats = {t: 0 for t in self.valid_types}

        for q in questions:
            qtype = q.get("question_type", "unknown")
            if qtype in stats:
                stats[qtype] += 1

        return stats

    def print_statistics(self, questions: List[Dict]):
        """Print classification statistics

        Args:
            questions: List of classified questions
        """
        stats = self.get_statistics(questions)
        total = len(questions)

        print("\n" + "=" * 50)
        print(f"Classification Statistics ({self.scheme.upper()})")
        print("=" * 50)

        for qtype, count in stats.items():
            pct = count / total * 100 if total > 0 else 0
            type_name = self.type_names.get(qtype, qtype)
            print(f"  {type_name}: {count} ({pct:.1f}%)")

        print(f"  Total: {total}")
        print("=" * 50)


# Usage example
if __name__ == "__main__":
    # Test classification
    classifier = QueryClassifier(scheme="memorag")

    test_questions = [
        {
            "question": "What are all the awards won by the horse Charisma?",
            "answer": "Charisma won individual gold at 1984 and 1988 Olympics, and the Luhmuhlen Three-Day Event in 1986."
        },
        {
            "question": "How did the Black Death impact Europe?",
            "answer": "The Black Death caused significant religious, social, and economic upheavals, leading to labor shortages and restructured power dynamics."
        },
        {
            "question": "What is the main theme of the film?",
            "answer": "The main theme explores the struggle for independence and the search for leadership in the fight against colonialism."
        }
    ]

    print("Testing MemoRAG classification:")
    for q in test_questions:
        qtype = classifier.classify_single(q["question"], q["answer"])
        print(f"  Q: {q['question'][:50]}...")
        print(f"  Type: {qtype}")
        print()
