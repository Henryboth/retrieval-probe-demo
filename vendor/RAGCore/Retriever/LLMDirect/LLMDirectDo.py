"""
LLM Direct Question Answering Processor

This module handles direct question answering using LLM without any retrieval.
Supports both synchronous and asynchronous (parallel) processing.
"""
import json
import asyncio
from typing import Dict, List, Any, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from Config.LLMConfig import LLMConfig
from Config.PathConfig import PathConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


class LLMDirectProcessor:
    """Process questions using direct LLM inference"""

    def __init__(self):
        """Initialize LLM Direct Processor

        The model and API configuration are read from LLMConfig.
        To change the model, modify PROVIDER in Config/LLMConfig.py
        """
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup LLM client from LLMConfig (both sync and async)"""
        model_config = LLMConfig.get_model_config()

        self.provider = LLMConfig.PROVIDER
        self.model = model_config["model"]  # Full path for API calls
        # Use model_name for file paths if available, otherwise use model
        self.model_name = model_config.get("model_name", model_config["model"])

        # Sync client
        self.llm_client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        # Async client for parallel processing
        self.async_llm_client = AsyncOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        print(f"Initialized LLM: Provider={self.provider}, Model={self.model_name}")

    def load_questions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load questions from dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of question dictionaries with keys: id, question, answer
        """
        question_path = PathConfig.get_question_path(dataset_name)

        with open(question_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                questions = json.load(f)
            else:
                questions = [json.loads(line) for line in f if line.strip()]

        return questions

    def answer_question(self, question: str) -> str:
        """Answer a single question using LLM

        Args:
            question: The question text

        Returns:
            Answer string, or None if failed
        """
        try:
            messages = PromptTemplate.get_qa_messages(question)

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error answering question: {e}")
            return None

    async def answer_question_async(self, question: str) -> Optional[str]:
        """Async version: Answer a single question using LLM

        Args:
            question: The question text

        Returns:
            Answer string, or None if failed
        """
        try:
            messages = PromptTemplate.get_qa_messages(question)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error answering question: {e}")
            return None

    def process(self, dataset_name: str, resume: bool = True) -> List[Dict[str, Any]]:
        """Process all questions in the dataset

        Args:
            dataset_name: Name of the dataset to process
            resume: If True, skip already processed questions

        Returns:
            List of results with format [{id, llm_answer}, ...]
        """
        print(f"Processing questions for dataset: {dataset_name}")
        print(f"Using model: {self.model_name}")

        # Load questions
        questions = self.load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Initialize results tracking
        results = []
        processed_ids = set()

        # Load existing results if resuming
        if resume:
            from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
            existing_results = LLMDirectSaver.load_answers(self.model_name, dataset_name)

            if existing_results:
                results = existing_results
                processed_ids = {r['id'] for r in existing_results}
                print(f"Resuming: Found {len(processed_ids)} already processed questions")

        # Calculate number of questions to process
        num_to_process = total_questions - len(processed_ids)

        if num_to_process == 0:
            print("All questions already processed!")
            return results
        else:
            print(f"Processing {num_to_process} questions...")

        # Process each question with progress bar
        pbar = tqdm(questions, desc="Processing", unit="q")

        for q_data in pbar:
            q_id = q_data.get('id')
            question = q_data.get('question')

            if q_id is None or not question:
                continue

            # Skip if already processed
            if q_id in processed_ids:
                continue

            # Answer question
            answer = self.answer_question(question)

            if answer is not None:
                result = {"id": q_id, "llm_answer": answer}
                results.append(result)
                processed_ids.add(q_id)

                # Save incrementally
                if resume:
                    from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
                    LLMDirectSaver.save_answer(result, self.model_name, dataset_name)
            else:
                # Record failure
                error_result = {"id": q_id, "llm_answer": None}
                results.append(error_result)

                if resume:
                    from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
                    LLMDirectSaver.save_answer(error_result, self.model_name, dataset_name)

        pbar.close()
        print(f"✓ Processing Complete! Processed {len(processed_ids)}/{total_questions} questions")

        return results

    def process_async(
        self,
        dataset_name: str,
        resume: bool = True,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Process all questions using async parallel processing

        Args:
            dataset_name: Name of the dataset to process
            resume: If True, skip already processed questions
            max_concurrent: Maximum concurrent LLM calls

        Returns:
            List of results with format [{id, llm_answer}, ...]
        """
        return asyncio.run(self._process_async_impl(dataset_name, resume, max_concurrent))

    async def _process_async_impl(
        self,
        dataset_name: str,
        resume: bool = True,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Internal async implementation of parallel processing"""
        print(f"Processing questions for dataset: {dataset_name}")
        print(f"Using model: {self.model_name}")
        print(f"Max concurrent requests: {max_concurrent}")

        # Load questions
        questions = self.load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Load existing results if resuming
        from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
        results = []
        processed_ids = set()

        if resume:
            existing_results = LLMDirectSaver.load_answers(self.model_name, dataset_name)
            if existing_results:
                results = existing_results
                processed_ids = {r['id'] for r in existing_results}
                print(f"Resuming: Found {len(processed_ids)} already processed questions")

        # Filter questions to process
        questions_to_process = [
            q for q in questions
            if q.get('id') and q.get('question') and q.get('id') not in processed_ids
        ]

        if not questions_to_process:
            print("All questions already processed!")
            return results

        print(f"Processing {len(questions_to_process)} questions...")

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(q_data: Dict) -> Optional[Dict[str, Any]]:
            """Process a single question with semaphore"""
            async with semaphore:
                q_id = q_data['id']
                question = q_data['question']

                # Answer question
                answer = await self.answer_question_async(question)

                return {
                    "id": q_id,
                    "llm_answer": answer
                }

        # Create all tasks
        tasks = [
            asyncio.create_task(process_single(q))
            for q in questions_to_process
        ]

        # Process with progress bar
        pbar = tqdm(total=len(tasks), desc="Processing (parallel)", unit="q")

        for coro in asyncio.as_completed(tasks):
            result = await coro
            pbar.update(1)

            if result:
                q_id = result["id"]
                results.append(result)
                processed_ids.add(q_id)

                # Save incrementally
                if resume:
                    LLMDirectSaver.save_answer(result, self.model_name, dataset_name)

        pbar.close()
        print(f"✓ Processing Complete! Processed {len(processed_ids)}/{total_questions} questions")

        return results


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM Direct QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    processor = LLMDirectProcessor()
    results = processor.process(dataset_name=args.dataset, resume=not args.no_resume)

    # Save final results
    from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
    LLMDirectSaver.save_all(results, processor.model_name, args.dataset)

    print(f"\nResults saved to: {PathConfig.get_llm_direct_path(processor.model_name, args.dataset)}")
