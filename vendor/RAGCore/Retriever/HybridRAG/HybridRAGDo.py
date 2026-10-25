"""
Hybrid RAG Retrieval and Answer Processor

This module implements Hybrid RAG: combining NaiveRAG and GraphRAG retrieval results,
then generating answers with LLM.
Supports both synchronous and asynchronous (parallel) processing.
"""
import json
import os
import asyncio
from typing import Dict, List, Any, Tuple, Set, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

from Config.LLMConfig import LLMConfig
from Config.RetrieverConfig import RetrieverConfig
from Config.PathConfig import PathConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


class HybridRAGProcessor:
    """Process questions using Hybrid RAG (NaiveRAG + GraphRAG)"""

    def __init__(self, dataset_name: str):
        """Initialize Hybrid RAG Processor

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
        self._setup_llm_client()
        self._load_retrieval_data()

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

    def _load_retrieval_data(self):
        """Load NaiveRAG and GraphRAG retrieval results"""
        # Load NaiveRAG retrieval
        naive_path = PathConfig.get_naive_rag_path(self.model_name, self.dataset_name)
        naive_dir = os.path.dirname(naive_path)
        naive_retrieval_file = os.path.join(naive_dir, "retrieval.jsonl")

        self.naive_data = {}
        if os.path.exists(naive_retrieval_file):
            with open(naive_retrieval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.naive_data[r['id']] = r
            print(f"Loaded NaiveRAG retrieval: {len(self.naive_data)} questions")
        else:
            raise FileNotFoundError(f"NaiveRAG retrieval not found: {naive_retrieval_file}")

        # Load GraphRAG retrieval
        graph_path = PathConfig.get_graph_rag_path(self.model_name, self.dataset_name)
        graph_dir = os.path.dirname(graph_path)
        graph_retrieval_file = os.path.join(graph_dir, "retrieval.jsonl")

        self.graph_data = {}
        if os.path.exists(graph_retrieval_file):
            with open(graph_retrieval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.graph_data[r['id']] = r
            print(f"Loaded GraphRAG retrieval: {len(self.graph_data)} questions")
        else:
            raise FileNotFoundError(f"GraphRAG retrieval not found: {graph_retrieval_file}")

        # Load questions
        question_path = PathConfig.get_question_path(self.dataset_name)
        self.questions = {}
        if os.path.exists(question_path):
            with open(question_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    for q in json.loads(content):
                        self.questions[q['id']] = q
                else:
                    for line in content.split('\n'):
                        if line.strip():
                            q = json.loads(line)
                            self.questions[q['id']] = q
            print(f"Loaded questions: {len(self.questions)}")
        else:
            raise FileNotFoundError(f"Questions not found: {question_path}")

    def get_naive_chunks(self, qid: int) -> List[Dict[str, Any]]:
        """Get NaiveRAG chunks for a question

        Args:
            qid: Question ID

        Returns:
            List of chunks with keys: doc_id, chunk_idx, text
        """
        if qid not in self.naive_data:
            return []

        chunks = self.naive_data[qid].get('chunks', [])
        # Normalize format: ensure doc_id, chunk_idx, text
        normalized = []
        for chunk in chunks:
            normalized.append({
                "doc_id": chunk.get("doc_id"),
                "chunk_idx": chunk.get("chunk_idx"),
                "text": chunk.get("text", "")
            })
        return normalized

    def get_graph_sources(self, qid: int) -> List[Dict[str, Any]]:
        """Get GraphRAG source sentences for a question

        Args:
            qid: Question ID

        Returns:
            List of sources with keys: doc_id, chunk_idx, text
        """
        if qid not in self.graph_data:
            return []

        sources = self.graph_data[qid].get('source_sentences', [])
        # Normalize format: ensure doc_id, chunk_idx, text
        normalized = []
        for src in sources:
            if isinstance(src, dict):
                normalized.append({
                    "doc_id": src.get("doc_id"),
                    "chunk_idx": src.get("chunk_idx"),
                    "text": src.get("text", "")
                })
            elif isinstance(src, str):
                # Old format: just text string
                normalized.append({
                    "doc_id": None,
                    "chunk_idx": None,
                    "text": src
                })
        return normalized

    def merge_interleave(
        self,
        naive_chunks: List[Dict[str, Any]],
        graph_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge chunks using interleave strategy with deduplication

        Deduplication is based on (doc_id, chunk_idx).
        Interleave: naive[0], graph[0], naive[1], graph[1], ...
        If one side finishes, append remaining from the other side.

        Args:
            naive_chunks: List of NaiveRAG chunks
            graph_sources: List of GraphRAG source sentences

        Returns:
            Merged and deduplicated list of chunks
        """
        seen_keys: Set[Tuple] = set()
        merged: List[Dict[str, Any]] = []

        def add_chunk(chunk: Dict[str, Any]) -> bool:
            """Add chunk if not seen, return True if added"""
            doc_id = chunk.get("doc_id")
            chunk_idx = chunk.get("chunk_idx")
            text = chunk.get("text", "")

            if not text:
                return False

            # Use (doc_id, chunk_idx) as key if available, otherwise use text hash
            if doc_id is not None and chunk_idx is not None:
                key = (doc_id, chunk_idx)
            else:
                key = hash(text)

            if key in seen_keys:
                return False

            seen_keys.add(key)
            merged.append({
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "text": text
            })
            return True

        # Interleave: naive[i], graph[i] alternately
        max_len = max(len(naive_chunks), len(graph_sources))

        for i in range(max_len):
            # Add naive chunk at position i
            if i < len(naive_chunks):
                add_chunk(naive_chunks[i])

            # Add graph chunk at position i
            if i < len(graph_sources):
                add_chunk(graph_sources[i])

        return merged

    def apply_token_budget(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply token budget to limit context size

        Args:
            chunks: List of chunks

        Returns:
            Truncated list within token budget
        """
        result = []
        total_tokens = 0
        token_budget = RetrieverConfig.CONTEXT_TOKEN_BUDGET

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_tokens = RetrieverConfig.estimate_tokens(text)

            if total_tokens + chunk_tokens > token_budget:
                break

            result.append(chunk)
            total_tokens += chunk_tokens

        return result

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from chunks

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Context string with chunks joined by separator
        """
        texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
        return RetrieverConfig.CHUNK_SEPARATOR.join(texts)

    def answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context

        Args:
            question: The question text
            context: Retrieved context

        Returns:
            Answer string, or None if failed
        """
        if not context.strip():
            return "I cannot answer this question based on the provided context."

        try:
            messages = PromptTemplate.get_rag_messages(context, question)

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=RetrieverConfig.RAG_TEMPERATURE,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=RetrieverConfig.RAG_TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

    async def answer_with_context_async(self, question: str, context: str) -> Optional[str]:
        """Async version: Generate answer using LLM with retrieved context

        Args:
            question: The question text
            context: Retrieved context

        Returns:
            Answer string, or None if failed
        """
        if not context.strip():
            return "I cannot answer this question based on the provided context."

        try:
            messages = PromptTemplate.get_rag_messages(context, question)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=RetrieverConfig.RAG_TEMPERATURE,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=RetrieverConfig.RAG_TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

    def process(self, dataset_name: str, resume: bool = True) -> List[Dict[str, Any]]:
        """Process all questions: merge retrieval + generate answers

        Args:
            dataset_name: Name of the dataset
            resume: If True, skip already processed questions

        Returns:
            List of answer results
        """
        print(f"Processing questions for dataset: {dataset_name}")
        print(f"Using Hybrid RAG with model: {self.model_name}")

        # Get common question IDs
        qids = sorted(set(self.naive_data.keys()) & set(self.graph_data.keys()))
        total_questions = len(qids)
        print(f"Common questions: {total_questions}")

        # Initialize results tracking
        results = []
        processed_ids = set()

        # Load existing results if resuming
        if resume:
            from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
            existing_results = HybridRAGSaver.load_answers(self.model_name, dataset_name)

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
        pbar = tqdm(qids, desc="Processing", unit="q")

        for qid in pbar:
            # Skip if already processed
            if qid in processed_ids:
                continue

            # Get question text
            question = self.questions.get(qid, {}).get('question', '')
            if not question:
                continue

            # Step 1: Get chunks from both sources
            naive_chunks = self.get_naive_chunks(qid)
            graph_sources = self.get_graph_sources(qid)

            # Step 2: Merge with deduplication
            merged = self.merge_interleave(naive_chunks, graph_sources)

            # Step 3: Apply token budget
            truncated = self.apply_token_budget(merged)

            # Step 4: Build context
            context = self.build_context(truncated)

            # Step 5: Generate answer with LLM
            answer = self.answer_with_context(question, context)

            # Prepare retrieval result
            retrieval_result = {
                "id": qid,
                "retrieved_chunks": truncated
            }

            if answer is not None:
                # Prepare answer result
                result = {"id": qid, "rag_answer": answer}
                results.append(result)
                processed_ids.add(qid)

                # Save incrementally
                if resume:
                    from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
                    HybridRAGSaver.save_answer(result, self.model_name, dataset_name)
                    HybridRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)
            else:
                # Record failure
                error_result = {"id": qid, "rag_answer": None}
                results.append(error_result)

                if resume:
                    from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
                    HybridRAGSaver.save_answer(error_result, self.model_name, dataset_name)
                    HybridRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)

        pbar.close()
        print(f"Processing Complete! Processed {len(processed_ids)}/{total_questions} questions")

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
            List of results with format [{id, rag_answer}, ...]
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
        print(f"Using Hybrid RAG with model: {self.model_name}")
        print(f"Max concurrent requests: {max_concurrent}")

        # Get common question IDs
        qids = sorted(set(self.naive_data.keys()) & set(self.graph_data.keys()))
        total_questions = len(qids)
        print(f"Common questions: {total_questions}")

        # Load existing results if resuming
        from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
        results = []
        processed_ids = set()

        if resume:
            existing_results = HybridRAGSaver.load_answers(self.model_name, dataset_name)
            if existing_results:
                results = existing_results
                processed_ids = {r['id'] for r in existing_results}
                print(f"Resuming: Found {len(processed_ids)} already processed questions")

        # Filter questions to process
        qids_to_process = [qid for qid in qids if qid not in processed_ids]

        if not qids_to_process:
            print("All questions already processed!")
            return results

        print(f"Processing {len(qids_to_process)} questions...")

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(qid: int) -> Optional[Dict[str, Any]]:
            """Process a single question with semaphore"""
            async with semaphore:
                # Get question text
                question = self.questions.get(qid, {}).get('question', '')
                if not question:
                    return None

                # Step 1: Get chunks from both sources
                naive_chunks = self.get_naive_chunks(qid)
                graph_sources = self.get_graph_sources(qid)

                # Step 2: Merge with deduplication
                merged = self.merge_interleave(naive_chunks, graph_sources)

                # Step 3: Apply token budget
                truncated = self.apply_token_budget(merged)

                # Step 4: Build context
                context = self.build_context(truncated)

                # Step 5: Generate answer with LLM (async)
                answer = await self.answer_with_context_async(question, context)

                return {
                    "id": qid,
                    "rag_answer": answer,
                    "retrieved_chunks": truncated
                }

        # Create all tasks
        tasks = [
            asyncio.create_task(process_single(qid))
            for qid in qids_to_process
        ]

        # Process with progress bar
        pbar = tqdm(total=len(tasks), desc="Processing (parallel)", unit="q")

        for coro in asyncio.as_completed(tasks):
            result = await coro
            pbar.update(1)

            if result:
                qid = result["id"]
                answer_result = {"id": qid, "rag_answer": result["rag_answer"]}
                results.append(answer_result)
                processed_ids.add(qid)

                # Save incrementally
                if resume:
                    HybridRAGSaver.save_answer(answer_result, self.model_name, dataset_name)
                    retrieval_result = {"id": qid, "retrieved_chunks": result["retrieved_chunks"]}
                    HybridRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)

        pbar.close()
        print(f"Processing Complete! Processed {len(processed_ids)}/{total_questions} questions")

        return results


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Hybrid RAG QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    processor = HybridRAGProcessor(dataset_name=args.dataset)
    results = processor.process(dataset_name=args.dataset, resume=not args.no_resume)

    # Save final results
    from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
    HybridRAGSaver.save_all(results, processor.model_name, args.dataset)

    print(f"\nResults saved to: {PathConfig.get_hybrid_rag_path(processor.model_name, args.dataset)}")
