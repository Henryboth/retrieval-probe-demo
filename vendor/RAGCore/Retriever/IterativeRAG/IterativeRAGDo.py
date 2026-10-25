"""
Iterative RAG Processor

This module implements Iterative RAG: multi-round retrieval with LLM evaluation.
Flow:
- Round 0: LLM direct answer → Evaluate → if sufficient, done
- Round 1+: Retrieve → Generate with context → Evaluate → Continue or stop
- Stop conditions: sufficient=true, max_iterations, no sub_question, duplicate sub_question
"""
import json
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

from Config.LLMConfig import LLMConfig
from Config.RetrieverConfig import RetrieverConfig
from Config.PathConfig import PathConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


class IterativeRAGProcessor:
    """Process questions using Iterative RAG (Multi-round Retrieval + Evaluation)"""

    def __init__(self, dataset_name: str):
        """Initialize Iterative RAG Processor

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
        self._setup_llm_client()
        self._setup_retriever()

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

    def _setup_retriever(self):
        """Setup retriever based on config"""
        self.retriever_type = RetrieverConfig.ITERATIVE_RETRIEVER.lower()

        if self.retriever_type == "naive":
            from RAGCore.Retriever.NaiveRAG.NaiveRAGDo import NaiveRAGProcessor
            self.retriever = NaiveRAGProcessor(dataset_name=self.dataset_name)
            print(f"Initialized Retriever: NaiveRAG")
        elif self.retriever_type == "graph":
            from RAGCore.Retriever.GraphRAG.GraphRAGDo import GraphRAGProcessor
            self.retriever = GraphRAGProcessor(dataset_name=self.dataset_name)
            print(f"Initialized Retriever: GraphRAG")
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")

    def retrieve_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve chunks using configured retriever"""
        return self.retriever.retrieve_chunks(query)

    def generate_direct(self, question: str) -> str:
        """Generate answer without context (Round 0)"""
        try:
            messages = PromptTemplate.get_qa_messages(question)
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=RetrieverConfig.RAG_TEMPERATURE,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=RetrieverConfig.RAG_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating direct answer: {e}")
            return "I cannot answer this question."

    def generate_with_context(self, question: str, context: str) -> str:
        """Generate answer with retrieved context (Round 1+)"""
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
            print(f"Error generating answer with context: {e}")
            return "I cannot answer this question."

    def evaluate_answer(self, question: str, answer: str, context: str = "None") -> Dict[str, Any]:
        """Evaluate if answer is sufficient and generate sub_question if needed"""
        try:
            messages = PromptTemplate.get_iterative_eval_messages(question, answer, context)
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=RetrieverConfig.ITERATIVE_EVAL_TEMPERATURE,
                max_tokens=300,
                timeout=RetrieverConfig.RAG_TIMEOUT
            )
            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            if "```" in result_text:
                result_text = result_text.split("```")[1].replace("json", "").strip()
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"sufficient": False, "reason": "Parse error", "sub_question": None}
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {"sufficient": False, "reason": str(e), "sub_question": None}

    # Async LLM Methods for Parallel Processing
    async def generate_direct_async(self, question: str) -> str:
        """Async: Generate answer without context (Round 0)"""
        try:
            messages = PromptTemplate.get_qa_messages(question)
            response = await self.async_llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=RetrieverConfig.RAG_TEMPERATURE,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=RetrieverConfig.RAG_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating direct answer: {e}")
            return "I cannot answer this question."

    async def generate_with_context_async(self, question: str, context: str) -> str:
        """Async: Generate answer with retrieved context (Round 1+)"""
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
            print(f"Error generating answer with context: {e}")
            return "I cannot answer this question."

    async def evaluate_answer_async(self, question: str, answer: str, context: str = "None") -> Dict[str, Any]:
        """Async: Evaluate if answer is sufficient and generate sub_question if needed"""
        try:
            messages = PromptTemplate.get_iterative_eval_messages(question, answer, context)
            response = await self.async_llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=RetrieverConfig.ITERATIVE_EVAL_TEMPERATURE,
                max_tokens=300,
                timeout=RetrieverConfig.RAG_TIMEOUT
            )
            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            if "```" in result_text:
                result_text = result_text.split("```")[1].replace("json", "").strip()
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"sufficient": False, "reason": "Parse error", "sub_question": None}
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {"sufficient": False, "reason": str(e), "sub_question": None}

    async def process_single_async(self, question: str) -> Dict[str, Any]:
        """Async: Process a single question using Iterative RAG

        Note: Iterative rounds within a question are still sequential,
        but multiple questions can be processed in parallel.
        """
        max_iterations = RetrieverConfig.ITERATIVE_MAX_ITERATIONS
        all_queries, all_chunks, history = [question], [], []

        # Round 0: Direct LLM answer
        answer = await self.generate_direct_async(question)
        eval_result = await self.evaluate_answer_async(question, answer)
        history.append({"round": 0, "query": question, "answer": answer, "evaluation": eval_result, "new_chunks_count": 0})

        if eval_result.get("sufficient", False):
            return {"final_answer": answer, "rounds": 0, "chunks": all_chunks, "history": history}

        current_query = eval_result.get("sub_question") or question
        if current_query and current_query not in all_queries:
            all_queries.append(current_query)
        else:
            current_query = question

        # Round 1+: Retrieval loop (retrieval is sync, but LLM calls are async)
        for iteration in range(1, max_iterations + 1):
            # Retrieval is synchronous (uses vector DB)
            new_chunks = self.retrieve_chunks(current_query)
            all_chunks, added_chunks = self.merge_chunks(all_chunks, new_chunks)
            truncated_chunks = self.apply_token_budget(all_chunks)
            context = self.build_context(truncated_chunks)

            # LLM calls are async
            answer = await self.generate_with_context_async(question, context)
            eval_result = await self.evaluate_answer_async(question, answer, context[:2000])
            history.append({"round": iteration, "query": current_query, "answer": answer, "evaluation": eval_result, "new_chunks_count": len(added_chunks)})

            if eval_result.get("sufficient", False):
                return {"final_answer": answer, "rounds": iteration, "chunks": truncated_chunks, "history": history}

            sub_question = eval_result.get("sub_question")
            if not sub_question or sub_question in all_queries:
                return {"final_answer": answer, "rounds": iteration, "chunks": truncated_chunks, "history": history}

            all_queries.append(sub_question)
            current_query = sub_question

        return {"final_answer": history[-1]["answer"], "rounds": max_iterations, "chunks": self.apply_token_budget(all_chunks), "history": history}

    def merge_chunks(self, existing_chunks: List[Dict], new_chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Merge new chunks with existing, deduplicating by (doc_id, chunk_idx)"""
        seen = {(c.get("doc_id"), c.get("chunk_idx")) for c in existing_chunks if c.get("doc_id") is not None}
        added = [c for c in new_chunks if (c.get("doc_id"), c.get("chunk_idx")) not in seen]
        all_chunks = existing_chunks + added
        return all_chunks, added

    def apply_token_budget(self, chunks: List[Dict]) -> List[Dict]:
        """Apply token budget to limit context size"""
        result, total = [], 0
        for chunk in chunks:
            tokens = RetrieverConfig.estimate_tokens(chunk.get("text", ""))
            if total + tokens > RetrieverConfig.CONTEXT_TOKEN_BUDGET:
                break
            result.append(chunk)
            total += tokens
        return result

    def build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks"""
        texts = [c.get("text", "") for c in chunks if c.get("text")]
        return RetrieverConfig.CHUNK_SEPARATOR.join(texts)

    def process_single(self, question: str) -> Dict[str, Any]:
        """Process a single question using Iterative RAG"""
        max_iterations = RetrieverConfig.ITERATIVE_MAX_ITERATIONS
        all_queries, all_chunks, history = [question], [], []

        # Round 0: Direct LLM answer
        answer = self.generate_direct(question)
        eval_result = self.evaluate_answer(question, answer)
        history.append({"round": 0, "query": question, "answer": answer, "evaluation": eval_result, "new_chunks_count": 0})

        if eval_result.get("sufficient", False):
            return {"final_answer": answer, "rounds": 0, "chunks": all_chunks, "history": history}

        current_query = eval_result.get("sub_question") or question
        if current_query and current_query not in all_queries:
            all_queries.append(current_query)
        else:
            current_query = question

        # Round 1+: Retrieval loop
        for iteration in range(1, max_iterations + 1):
            new_chunks = self.retrieve_chunks(current_query)
            all_chunks, added_chunks = self.merge_chunks(all_chunks, new_chunks)
            truncated_chunks = self.apply_token_budget(all_chunks)
            context = self.build_context(truncated_chunks)

            answer = self.generate_with_context(question, context)
            eval_result = self.evaluate_answer(question, answer, context[:2000])
            history.append({"round": iteration, "query": current_query, "answer": answer, "evaluation": eval_result, "new_chunks_count": len(added_chunks)})

            if eval_result.get("sufficient", False):
                return {"final_answer": answer, "rounds": iteration, "chunks": truncated_chunks, "history": history}

            sub_question = eval_result.get("sub_question")
            if not sub_question or sub_question in all_queries:
                return {"final_answer": answer, "rounds": iteration, "chunks": truncated_chunks, "history": history}

            all_queries.append(sub_question)
            current_query = sub_question

        return {"final_answer": history[-1]["answer"], "rounds": max_iterations, "chunks": self.apply_token_budget(all_chunks), "history": history}

    def process(self, dataset_name: str, resume: bool = True) -> List[Dict[str, Any]]:
        """Process all questions in the dataset"""
        print(f"Processing: {dataset_name}, Retriever: {self.retriever_type}, Max iterations: {RetrieverConfig.ITERATIVE_MAX_ITERATIONS}")

        questions = self._load_questions(dataset_name)
        print(f"Loaded {len(questions)} questions")

        results, processed_ids = [], set()

        if resume:
            from RAGCore.Retriever.IterativeRAG.IterativeRAGSave import IterativeRAGSaver
            existing = IterativeRAGSaver.load_answers(self.model_name, dataset_name, self.retriever_type)
            if existing:
                results = existing
                processed_ids = {r['id'] for r in existing}
                print(f"Resuming: {len(processed_ids)} already processed")

        if len(questions) == len(processed_ids):
            print("All questions already processed!")
            return results

        pbar = tqdm(questions, desc="Processing", unit="q")
        for q_data in pbar:
            q_id, question = q_data.get('id'), q_data.get('question')
            if q_id is None or not question or q_id in processed_ids:
                continue

            result = self.process_single(question)
            answer_result = {"id": q_id, "rag_answer": result["final_answer"], "rounds": result["rounds"]}
            results.append(answer_result)
            processed_ids.add(q_id)

            if resume:
                from RAGCore.Retriever.IterativeRAG.IterativeRAGSave import IterativeRAGSaver
                IterativeRAGSaver.save_answer(answer_result, self.model_name, dataset_name, self.retriever_type)
                IterativeRAGSaver.save_retrieval({"id": q_id, "retrieved_chunks": result["chunks"], "history": result["history"]}, self.model_name, dataset_name, self.retriever_type)

        pbar.close()
        print(f"Complete! Processed {len(processed_ids)}/{len(questions)} questions")
        return results

    def process_async(
        self,
        dataset_name: str,
        resume: bool = True,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Process all questions using async parallel processing

        Args:
            dataset_name: Name of the dataset to process
            resume: If True, skip already processed questions
            max_concurrent: Maximum concurrent question processing (default 3 for iterative)

        Returns:
            List of results with format [{id, rag_answer, rounds}, ...]
        """
        return asyncio.run(self._process_async_impl(dataset_name, resume, max_concurrent))

    async def _process_async_impl(
        self,
        dataset_name: str,
        resume: bool = True,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Internal async implementation of parallel processing"""
        print(f"Processing (parallel): {dataset_name}, Retriever: {self.retriever_type}")
        print(f"Max iterations: {RetrieverConfig.ITERATIVE_MAX_ITERATIONS}, Max concurrent: {max_concurrent}")

        # Load questions
        questions = self._load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Load existing results if resuming
        from RAGCore.Retriever.IterativeRAG.IterativeRAGSave import IterativeRAGSaver
        results = []
        processed_ids = set()

        if resume:
            existing = IterativeRAGSaver.load_answers(self.model_name, dataset_name, self.retriever_type)
            if existing:
                results = existing
                processed_ids = {r['id'] for r in existing}
                print(f"Resuming: {len(processed_ids)} already processed")

        # Filter questions to process
        questions_to_process = [
            q for q in questions
            if q.get('id') is not None and q.get('question') and q.get('id') not in processed_ids
        ]

        if not questions_to_process:
            print("All questions already processed!")
            return results

        print(f"Processing {len(questions_to_process)} questions...")

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(q_data: Dict) -> Optional[Dict[str, Any]]:
            """Process a single question with semaphore"""
            async with semaphore:
                q_id = q_data['id']
                question = q_data['question']

                # Process question (iterative rounds are sequential within)
                result = await self.process_single_async(question)

                return {
                    "id": q_id,
                    "result": result
                }

        # Create all tasks
        tasks = [
            asyncio.create_task(process_with_semaphore(q))
            for q in questions_to_process
        ]

        # Process with progress bar
        pbar = tqdm(total=len(tasks), desc="Processing (parallel)", unit="q")

        for coro in asyncio.as_completed(tasks):
            task_result = await coro
            pbar.update(1)

            if task_result:
                q_id = task_result["id"]
                result = task_result["result"]

                answer_result = {
                    "id": q_id,
                    "rag_answer": result["final_answer"],
                    "rounds": result["rounds"]
                }
                results.append(answer_result)
                processed_ids.add(q_id)

                # Save incrementally
                if resume:
                    IterativeRAGSaver.save_answer(answer_result, self.model_name, dataset_name, self.retriever_type)
                    IterativeRAGSaver.save_retrieval(
                        {"id": q_id, "retrieved_chunks": result["chunks"], "history": result["history"]},
                        self.model_name, dataset_name, self.retriever_type
                    )

        pbar.close()
        print(f"Complete! Processed {len(processed_ids)}/{total_questions} questions")

        return results

    def _load_questions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load questions from dataset"""
        question_path = PathConfig.get_question_path(dataset_name)
        with open(question_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return json.loads(content) if content.startswith('[') else [json.loads(line) for line in content.split('\n') if line.strip()]
