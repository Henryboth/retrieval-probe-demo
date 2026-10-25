"""
Naive RAG Question Answering Processor

This module implements standard Naive RAG: vector retrieval + LLM generation.
Supports both synchronous and asynchronous (parallel) processing.
"""
import json
import os
import asyncio
import numpy as np
import faiss
from typing import Dict, List, Any, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from Config.LLMConfig import LLMConfig
from Config.EmbConfig import EmbConfig
from Config.RetrieverConfig import RetrieverConfig
from Config.PathConfig import PathConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor


class NaiveRAGProcessor:
    """Process questions using Naive RAG (Vector Retrieval + LLM)"""

    def __init__(self, dataset_name: str):
        """Initialize Naive RAG Processor

        Args:
            dataset_name: Name of the dataset (needed to load index)
        """
        self.dataset_name = dataset_name
        self._setup_llm_client()
        self._setup_embedding_client()
        self._load_index()

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

    def _setup_embedding_client(self):
        """Setup embedding client"""
        self.embedding_processor = EmbeddingProcessor()
        print(f"Initialized Embedding: Provider={EmbConfig.PROVIDER}, Model={EmbConfig.OPENAI_MODEL}")

    def _load_index(self):
        """Load FAISS index and chunk metadata"""
        index_dir = PathConfig.get_index_path(self.dataset_name)

        # Load FAISS index
        index_file = os.path.join(index_dir, "index.faiss")
        self.index = faiss.read_index(index_file)

        # Load chunk metadata
        metadata_file = os.path.join(index_dir, "metadata.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        print(f"Loaded index: {len(self.chunks)} chunks")

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

    def retrieve_chunks(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks for a question

        Args:
            question: The question text
            top_k: Number of chunks to retrieve (default: from RetrieverConfig)

        Returns:
            List of retrieved chunks with similarity scores
        """
        if top_k is None:
            top_k = RetrieverConfig.TOP_K

        # Step 1: Embed the question
        query_embedding = self.embedding_processor.embed_texts([question])  # shape: (1, dim)

        # Ensure correct dtype and memory layout for FAISS
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        # Step 2: Search in FAISS index
        # The index is built as IndexFlatIP (see RAGCore/Index/IndexDo.py), so on
        # unit-normalised vectors it returns cosine similarity directly, not L2 distance.
        if RetrieverConfig.SIMILARITY_METRIC == "cosine":
            faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        # Step 3: Get chunk texts and compute similarity scores
        retrieved_chunks = []
        for idx, (dist, chunk_idx) in enumerate(zip(distances[0], indices[0])):
            if chunk_idx == -1:  # FAISS returns -1 for empty results
                continue

            chunk = self.chunks[chunk_idx]

            # IndexFlatIP already returns cosine similarity for normalised vectors
            if RetrieverConfig.SIMILARITY_METRIC == "cosine":
                similarity = dist
            else:
                similarity = -dist  # negative distance as similarity

            # Apply similarity threshold
            if similarity < RetrieverConfig.MIN_SIMILARITY_THRESHOLD:
                continue

            retrieved_chunks.append({
                "rank": idx + 1,
                "chunk_idx": int(chunk_idx),
                "doc_id": chunk.get("doc_id"),
                "text": chunk["text"],
                "similarity": float(similarity)
            })

        return retrieved_chunks

    def build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks using token budget

        Uses CONTEXT_TOKEN_BUDGET for fair comparison with other retrieval methods.
        Chunks are already sorted by similarity (highest first).

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries

        Returns:
            Context string with chunks separated by CHUNK_SEPARATOR
        """
        context_parts = []
        current_tokens = 0
        token_budget = RetrieverConfig.CONTEXT_TOKEN_BUDGET

        for chunk in retrieved_chunks:
            chunk_text = chunk["text"]
            chunk_tokens = RetrieverConfig.estimate_tokens(chunk_text)

            # Check if adding this chunk would exceed token budget
            if current_tokens + chunk_tokens > token_budget:
                break

            if RetrieverConfig.INCLUDE_CHUNK_METADATA:
                # Include metadata like doc_id
                chunk_text = f"[Doc {chunk['doc_id']}] {chunk_text}"

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        return RetrieverConfig.CHUNK_SEPARATOR.join(context_parts)

    def answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context

        Args:
            question: The question text
            context: Retrieved context

        Returns:
            Answer string, or None if failed
        """
        try:
            # Build RAG prompt using PromptTemplate
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
        """Process all questions in the dataset using Naive RAG

        Args:
            dataset_name: Name of the dataset to process
            resume: If True, skip already processed questions

        Returns:
            List of results with format [{id, rag_answer}, ...]
        """
        print(f"Processing questions for dataset: {dataset_name}")
        print(f"Using Naive RAG with model: {self.model_name}")

        # Load questions
        questions = self.load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Initialize results tracking
        results = []
        processed_ids = set()

        # Load existing results if resuming
        if resume:
            from RAGCore.Retriever.NaiveRAG.NaiveRAGSave import NaiveRAGSaver
            existing_results = NaiveRAGSaver.load_answers(self.model_name, dataset_name)

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

            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retrieve_chunks(question)

            # Step 2: Build context
            context = self.build_context(retrieved_chunks)

            # Step 3: Generate answer with LLM
            answer = self.answer_with_context(question, context)

            if answer is not None:
                # Prepare answer result
                result = {"id": q_id, "rag_answer": answer}
                results.append(result)
                processed_ids.add(q_id)

                # Save incrementally
                if resume:
                    from RAGCore.Retriever.NaiveRAG.NaiveRAGSave import NaiveRAGSaver
                    NaiveRAGSaver.save_answer(result, self.model_name, dataset_name)
                    # Save retrieval result
                    retrieval_result = {"id": q_id, "chunks": retrieved_chunks}
                    NaiveRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)
            else:
                # Record failure
                error_result = {"id": q_id, "rag_answer": None}
                results.append(error_result)

                if resume:
                    from RAGCore.Retriever.NaiveRAG.NaiveRAGSave import NaiveRAGSaver
                    NaiveRAGSaver.save_answer(error_result, self.model_name, dataset_name)
                    # Save empty retrieval result for consistency
                    retrieval_result = {"id": q_id, "chunks": retrieved_chunks}
                    NaiveRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)

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
        print(f"Using Naive RAG with model: {self.model_name}")
        print(f"Max concurrent requests: {max_concurrent}")

        # Load questions
        questions = self.load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Load existing results if resuming
        from RAGCore.Retriever.NaiveRAG.NaiveRAGSave import NaiveRAGSaver
        results = []
        processed_ids = set()

        if resume:
            existing_results = NaiveRAGSaver.load_answers(self.model_name, dataset_name)
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

                # Step 1: Retrieve relevant chunks (sync, fast)
                retrieved_chunks = self.retrieve_chunks(question)

                # Step 2: Build context
                context = self.build_context(retrieved_chunks)

                # Step 3: Generate answer with LLM (async)
                answer = await self.answer_with_context_async(question, context)

                if answer is not None:
                    return {
                        "id": q_id,
                        "rag_answer": answer,
                        "chunks": retrieved_chunks
                    }
                else:
                    return {
                        "id": q_id,
                        "rag_answer": None,
                        "chunks": retrieved_chunks
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
                answer_result = {"id": q_id, "rag_answer": result["rag_answer"]}
                results.append(answer_result)
                processed_ids.add(q_id)

                # Save incrementally
                if resume:
                    NaiveRAGSaver.save_answer(answer_result, self.model_name, dataset_name)
                    retrieval_result = {"id": q_id, "chunks": result["chunks"]}
                    NaiveRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)

        pbar.close()
        print(f"Processing Complete! Processed {len(processed_ids)}/{total_questions} questions")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Naive RAG QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    processor = NaiveRAGProcessor(dataset_name=args.dataset)
    results = processor.process(dataset_name=args.dataset, resume=not args.no_resume)

    # Save final results
    from RAGCore.Retriever.NaiveRAG.NaiveRAGSave import NaiveRAGSaver
    NaiveRAGSaver.save_all(results, processor.model_name, args.dataset)

    print(f"\nResults saved to: {PathConfig.get_naive_rag_path(processor.model_name, args.dataset)}")
