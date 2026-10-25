"""
Graph RAG Question Answering Processor

This module implements Graph RAG: entity retrieval + graph traversal + LLM generation.
Supports both synchronous and asynchronous (parallel) processing.
"""
import json
import os
import pickle
import asyncio
import numpy as np
import faiss
import pandas as pd
from typing import Dict, List, Any, Tuple, Set, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

from Config.LLMConfig import LLMConfig
from Config.EmbConfig import EmbConfig
from Config.RetrieverConfig import RetrieverConfig
from Config.PathConfig import PathConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor


class GraphRAGProcessor:
    """Process questions using Graph RAG (Graph Retrieval + LLM)"""

    def __init__(self, dataset_name: str):
        """Initialize Graph RAG Processor

        Args:
            dataset_name: Name of the dataset (needed to load graph)
        """
        self.dataset_name = dataset_name
        self._setup_llm_client()
        self._setup_embedding_client()
        self._load_graph_data()

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

    def _load_graph_data(self):
        """Load graph data: graph structure, entity mapping, entity index, triplet_sources"""
        graph_dir = PathConfig.get_graph_path(self.dataset_name)
        triplet_dir = PathConfig.get_triplet_path(self.dataset_name)

        # Load NetworkX graph
        graph_path = os.path.join(graph_dir, "graph.gpickle")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        print(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Load entity mapping (Parquet)
        mapping_path = os.path.join(graph_dir, "entity_mapping.parquet")
        df = pd.read_parquet(mapping_path)

        # Create bidirectional mapping
        self.entity_id_to_text = {row['entity_id']: row['entity'] for _, row in df.iterrows()}
        self.entity_text_to_id = {row['entity']: row['entity_id'] for _, row in df.iterrows()}

        # Create entity_id list (sorted by entity_id for FAISS index correspondence)
        self.entity_id_list = sorted(self.entity_id_to_text.keys())

        print(f"Loaded entity mapping: {len(self.entity_id_to_text)} entities")

        # Load FAISS entity index
        index_path = os.path.join(graph_dir, "entity_index.faiss")
        self.entity_index = faiss.read_index(index_path)

        print(f"Loaded entity index: {self.entity_index.ntotal} vectors")

        # Load triplet_resource.json for sentence mapping (REQUIRED for fair comparison)
        resource_path = os.path.join(triplet_dir, "triplet_resource.json")
        self.triplet_sources = {}
        if os.path.exists(resource_path):
            with open(resource_path, 'r', encoding='utf-8') as f:
                resource_data = json.load(f)
            # Convert string keys back to tuples
            for key, sources in resource_data.items():
                parts = key.split("|||")
                if len(parts) == 3:
                    self.triplet_sources[tuple(parts)] = sources
            print(f"Loaded triplet resource: {len(self.triplet_sources)} triplets with source info")
        else:
            # CRITICAL: triplet_resource.json is required for fair comparison with NaiveRAG
            raise FileNotFoundError(
                f"triplet_resource.json not found at {resource_path}\n"
                f"This file is REQUIRED for fair comparison with NaiveRAG.\n"
                f"Please re-run graph generation with the updated GraphDo.py:\n"
                f"  python Run/Process/run_graph.py --dataset {self.dataset_name}"
            )

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

    def extract_query_entities(self, question: str) -> List[str]:
        """Extract entities from question using LLM

        Args:
            question: The question text

        Returns:
            List of extracted entity strings
        """
        try:
            messages = PromptTemplate.get_entity_messages(question)
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            entities = json.loads(response.choices[0].message.content)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Warning: Failed to extract entities from question: {e}")
            return []

    def retrieve_seed_entities(self, question: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Retrieve top-k seed entities from entity index

        Now uses entity extraction first, then matches each extracted entity
        against the graph entities for better precision.

        Args:
            question: The question text
            top_k: Number of seed entities to retrieve

        Returns:
            List of (entity_id, similarity) tuples
        """
        if top_k is None:
            top_k = RetrieverConfig.GRAPH_SEED_ENTITIES

        # Step 1: Extract entities from question using LLM
        query_entities = self.extract_query_entities(question)

        if not query_entities:
            # Fallback: use entire question if no entities extracted
            print("Warning: No entities extracted, falling back to question embedding")
            query_entities = [question]
        else:
            print(f"Extracted {len(query_entities)} entities: {query_entities}")

        # Step 2: For each extracted entity, find matching graph entities
        all_matches = {}  # entity_id -> max_similarity

        for query_ent in query_entities:
            # Embed the extracted entity
            ent_embedding = self.embedding_processor.embed_texts([query_ent])
            ent_embedding = np.ascontiguousarray(ent_embedding, dtype=np.float32)
            faiss.normalize_L2(ent_embedding)

            # Search in entity index
            distances, indices = self.entity_index.search(ent_embedding, top_k)

            # Collect matches (filter by similarity threshold)
            for score, entity_idx in zip(distances[0], indices[0]):
                if entity_idx == -1:
                    continue

                similarity = float(score)

                # Skip low similarity matches
                if similarity <= RetrieverConfig.GRAPH_ENTITY_SIMILARITY_THRESHOLD:
                    continue

                entity_id = self.entity_id_list[entity_idx]

                # Keep max similarity for each entity
                if entity_id not in all_matches or similarity > all_matches[entity_id]:
                    all_matches[entity_id] = similarity

        # Step 3: Sort by similarity and return top-k
        seed_entities = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return seed_entities

    def retrieve_subgraph_by_ppr(self, seed_entities: List[Tuple[int, float]]) -> Set[int]:
        """Use PPR directly on full graph to find relevant nodes

        No BFS hop limit - PPR naturally explores based on graph structure.
        This allows adaptive multi-hop retrieval without hardcoded depth.

        Args:
            seed_entities: List of (entity_id, similarity) tuples

        Returns:
            Set of entity IDs selected by PPR
        """
        import networkx as nx

        if not seed_entities:
            return set()

        # Setup personalization vector from seed entities
        personalization = {}
        total_sim = sum(sim for _, sim in seed_entities)

        for entity_id, sim in seed_entities:
            if entity_id in self.graph:
                personalization[entity_id] = sim / total_sim

        if not personalization:
            print("Warning: No seed entities found in graph")
            return set()

        # Run PPR on FULL graph (not subgraph!)
        try:
            ppr_scores = nx.pagerank(
                self.graph,
                personalization=personalization,
                alpha=RetrieverConfig.GRAPH_PPR_ALPHA,
                max_iter=100
            )
        except Exception as e:
            print(f"Warning: PPR failed ({e}), returning seed entities only")
            return {eid for eid, _ in seed_entities}

        # Filter by minimum PPR score threshold
        min_score = RetrieverConfig.GRAPH_PPR_MIN_SCORE
        relevant_nodes = {
            node for node, score in ppr_scores.items()
            if score >= min_score
        }

        # Sort by PPR score and keep top-k
        sorted_nodes = sorted(
            [(n, s) for n, s in ppr_scores.items() if n in relevant_nodes],
            key=lambda x: x[1],
            reverse=True
        )[:RetrieverConfig.GRAPH_PPR_MAX_NODES]

        final_nodes = {node for node, _ in sorted_nodes}

        # Always include seed entities
        seed_ids = {eid for eid, _ in seed_entities}
        final_nodes.update(seed_ids)

        print(f"PPR on full graph: {len(final_nodes)} nodes selected (from {self.graph.number_of_nodes()} total)")

        return final_nodes

    def expand_subgraph(self, seed_entities: List[Tuple[int, float]]) -> Set[int]:
        """[DEPRECATED] Expand subgraph from seed entities using BFS

        NOTE: This method is deprecated. Use retrieve_subgraph_by_ppr() instead,
        which does adaptive exploration without hardcoded hop limits.

        Args:
            seed_entities: List of (entity_id, similarity) tuples

        Returns:
            Set of entity IDs in the subgraph
        """
        depth = 2  # Fixed depth (deprecated - use retrieve_subgraph_by_ppr instead)
        subgraph_nodes = set()

        # Extract entity IDs
        seed_entity_ids = [entity_id for entity_id, _ in seed_entities]

        # Add seed entities to subgraph
        subgraph_nodes.update(seed_entity_ids)

        # BFS expansion
        current_layer = set(seed_entity_ids)

        for hop in range(depth):
            next_layer = set()

            for node in current_layer:
                if node in self.graph:
                    # For undirected graph, neighbors() returns all connected nodes
                    neighbors = list(self.graph.neighbors(node))
                    next_layer.update(neighbors)
                    subgraph_nodes.update(neighbors)

            current_layer = next_layer

        return subgraph_nodes

    def filter_subgraph_by_ppr(self, subgraph_nodes: Set[int], seed_entities: List[Tuple[int, float]]) -> Set[int]:
        """[DEPRECATED] Filter subgraph using Personalized PageRank

        NOTE: This method is deprecated. Use retrieve_subgraph_by_ppr() instead,
        which runs PPR on the full graph without BFS expansion.

        Args:
            subgraph_nodes: Set of entity IDs in the expanded subgraph
            seed_entities: List of (entity_id, similarity) tuples

        Returns:
            Filtered set of entity IDs (top nodes by PPR score)
        """
        import networkx as nx

        # Step 1: Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes).copy()

        if subgraph.number_of_nodes() <= RetrieverConfig.GRAPH_PPR_MAX_NODES:
            # Already small enough, no need to filter
            return subgraph_nodes

        # Step 2: Setup personalization vector
        # Seed entities get higher initial probability based on their similarity scores
        personalization = {}
        total_similarity = sum(sim for _, sim in seed_entities)

        for entity_id, similarity in seed_entities:
            if entity_id in subgraph:
                # Normalize by total similarity
                personalization[entity_id] = similarity / total_similarity

        # All other nodes get 0 initial probability
        for node in subgraph.nodes():
            if node not in personalization:
                personalization[node] = 0.0

        # Step 3: Run Personalized PageRank
        try:
            ppr_scores = nx.pagerank(
                subgraph,
                personalization=personalization,
                alpha=RetrieverConfig.GRAPH_PPR_ALPHA,
                max_iter=100
            )
        except Exception as e:
            print(f"Warning: PPR failed ({e}), using original subgraph")
            return subgraph_nodes

        # Step 4: Select top-k nodes by PPR score
        # Always keep all seed entities
        seed_ids = {eid for eid, _ in seed_entities}

        # Get top nodes by PPR score
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # Keep top-k nodes, ensuring all seeds are included
        filtered_nodes = set(seed_ids)
        for node, score in sorted_nodes:
            if len(filtered_nodes) >= RetrieverConfig.GRAPH_PPR_MAX_NODES:
                break
            filtered_nodes.add(node)

        print(f"PPR filtering: {len(subgraph_nodes)} → {len(filtered_nodes)} nodes")

        return filtered_nodes

    def extract_subgraph_data(self, subgraph_nodes: Set[int], seed_entities: List[Tuple[int, float]]) -> Dict[str, Any]:
        """Extract subgraph data for retrieval result

        Args:
            subgraph_nodes: Set of entity IDs in subgraph
            seed_entities: List of (entity_id, similarity) tuples

        Returns:
            Dictionary with nodes and triplets
        """
        # Build nodes list
        nodes = []
        seed_entity_ids = {entity_id for entity_id, _ in seed_entities}

        # Add seed entities with rank and similarity
        for rank, (entity_id, similarity) in enumerate(seed_entities, 1):
            entity_text = self.entity_id_to_text.get(entity_id, f"Entity_{entity_id}")
            nodes.append({
                "rank": rank,
                "entity": entity_text,
                "similarity": similarity,
                "is_seed": True
            })

        # Add expanded entities (no rank, no similarity)
        for entity_id in subgraph_nodes:
            if entity_id not in seed_entity_ids:
                entity_text = self.entity_id_to_text.get(entity_id, f"Entity_{entity_id}")
                nodes.append({
                    "entity": entity_text,
                    "is_seed": False
                })

        # Extract triplets
        subgraph = self.graph.subgraph(subgraph_nodes)
        triplets = []

        for u, v, data in subgraph.edges(data=True):
            entity1 = self.entity_id_to_text.get(u, f"Entity_{u}")
            entity2 = self.entity_id_to_text.get(v, f"Entity_{v}")
            relation = data.get('relation', 'related_to')

            # Use list format: [subject, relation, object]
            triplets.append([entity1, relation, entity2])

        return {
            "nodes": nodes,
            "triplets": triplets
        }

    def collect_source_sentences(self, triplets: List[List[str]], entity_similarities: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Collect source sentences from triplets using triplet_sources mapping

        Prioritizes sentences by max entity similarity for better relevance.

        Args:
            triplets: List of [subject, relation, object] triplets
            entity_similarities: Dict mapping entity_name -> similarity score

        Returns:
            List of dicts with keys: doc_id, chunk_idx, text
            Sorted by max entity similarity (descending)
        """
        if not self.triplet_sources:
            return []

        entity_sim_map = entity_similarities if entity_similarities else {}

        # Collect all source texts with relevance info
        # (max_similarity, doc_id, chunk_idx, source_text)
        sources_with_score = []
        seen_chunks = set()  # Use (doc_id, chunk_idx) as unique key

        for triplet in triplets:
            if len(triplet) != 3:
                continue
            subject, relation, obj = triplet
            triplet_key = (subject, relation, obj)

            # Get max similarity from entities in this triplet
            triplet_sim = max(
                entity_sim_map.get(subject, 0.0),
                entity_sim_map.get(obj, 0.0)
            )

            if triplet_key in self.triplet_sources:
                for source_info in self.triplet_sources[triplet_key]:
                    source_text = source_info.get("source_text", "")
                    doc_id = source_info.get("doc_id", 0)
                    chunk_idx = source_info.get("chunk_idx", 0)
                    chunk_key = (doc_id, chunk_idx)

                    if source_text and chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        sources_with_score.append((triplet_sim, doc_id, chunk_idx, source_text))

        # Sort by: 1) max_similarity (descending), 2) doc_id, 3) chunk_idx
        sources_with_score.sort(key=lambda x: (-x[0], x[1], x[2]))

        # Return list of dicts with metadata
        return [
            {"doc_id": doc_id, "chunk_idx": chunk_idx, "text": text}
            for _, doc_id, chunk_idx, text in sources_with_score
        ]

    def retrieve_chunks(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve source sentences for a question (IterativeRAG compatibility)

        This method provides the same interface as NaiveRAGProcessor.retrieve_chunks()
        to enable GraphRAG to be used as the retriever in IterativeRAG.

        Args:
            question: The question text
            top_k: Number of seed entities to retrieve (default: from config)

        Returns:
            List of retrieved chunks with format: {doc_id, chunk_idx, text, similarity}
        """
        # Step 1: Retrieve seed entities
        seed_entities = self.retrieve_seed_entities(question, top_k)

        if not seed_entities:
            return []

        # Step 2: Expand subgraph using PPR
        subgraph_nodes = self.retrieve_subgraph_by_ppr(seed_entities)

        if not subgraph_nodes:
            return []

        # Step 3: Extract subgraph data (triplets)
        subgraph_data = self.extract_subgraph_data(subgraph_nodes, seed_entities)
        triplets = subgraph_data.get("triplets", [])

        if not triplets:
            return []

        # Step 4: Build entity similarity map for ranking
        entity_similarities = {
            self.entity_id_to_text.get(eid, ""): sim
            for eid, sim in seed_entities
        }

        # Step 5: Collect source sentences (same format as NaiveRAG chunks)
        source_sentences = self.collect_source_sentences(triplets, entity_similarities)

        # Add similarity scores to match NaiveRAG chunk format
        for i, chunk in enumerate(source_sentences):
            chunk["rank"] = i + 1
            # Use a default similarity based on rank (actual similarity is embedded in sorting)
            chunk["similarity"] = 1.0 - (i * 0.01)  # Decreasing similarity by rank

        return source_sentences

    def serialize_subgraph(self, triplets: List[List[str]], entity_similarities: Dict[str, float] = None) -> str:
        """Serialize subgraph to text format for LLM using token budget

        Uses CONTEXT_TOKEN_BUDGET for fair comparison with NaiveRAG.
        If triplet_sources is available, returns source sentences.
        Otherwise, falls back to triplet format.

        Args:
            triplets: List of [subject, relation, object] triplets
            entity_similarities: Dict mapping entity_name -> similarity score

        Returns:
            Formatted text string (limited by CONTEXT_TOKEN_BUDGET)
        """
        token_budget = RetrieverConfig.CONTEXT_TOKEN_BUDGET

        # Use source sentences (triplet_sources is required, checked in __init__)
        source_sentences = self.collect_source_sentences(triplets, entity_similarities)

        if not source_sentences:
            # This means triplet keys don't match - likely a data inconsistency
            print(f"WARNING: No source sentences found for {len(triplets)} triplets (key mismatch?)")
            # Return empty context rather than falling back to unfair triplet format
            return ""

        # Apply token budget limit
        context_parts = []
        current_tokens = 0

        for sentence_info in source_sentences:
            # Extract text from dict format
            sentence_text = sentence_info["text"]
            sentence_tokens = RetrieverConfig.estimate_tokens(sentence_text)

            if current_tokens + sentence_tokens > token_budget:
                break

            context_parts.append(sentence_text)
            current_tokens += sentence_tokens

        print(f"Using {len(context_parts)} source sentences as context (~{current_tokens} tokens)")
        return "\n\n".join(context_parts)

    def answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context

        Args:
            question: The question text
            context: Retrieved context (serialized graph)

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

    async def extract_query_entities_async(self, question: str) -> List[str]:
        """Async version: Extract entities from question using LLM

        Args:
            question: The question text

        Returns:
            List of extracted entity strings
        """
        try:
            messages = PromptTemplate.get_entity_messages(question)
            response = await self.async_llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            entities = json.loads(response.choices[0].message.content)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            return []

    async def answer_with_context_async(self, question: str, context: str) -> Optional[str]:
        """Async version: Generate answer using LLM with retrieved context

        Args:
            question: The question text
            context: Retrieved context (serialized graph)

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

    async def retrieve_seed_entities_async(self, question: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Async version: Retrieve top-k seed entities from entity index

        Args:
            question: The question text
            top_k: Number of seed entities to retrieve

        Returns:
            List of (entity_id, similarity) tuples
        """
        if top_k is None:
            top_k = RetrieverConfig.GRAPH_SEED_ENTITIES

        # Step 1: Extract entities from question using LLM (async)
        query_entities = await self.extract_query_entities_async(question)

        if not query_entities:
            query_entities = [question]

        # Step 2: For each extracted entity, find matching graph entities
        all_matches = {}

        for query_ent in query_entities:
            ent_embedding = self.embedding_processor.embed_texts([query_ent])
            ent_embedding = np.ascontiguousarray(ent_embedding, dtype=np.float32)
            faiss.normalize_L2(ent_embedding)

            distances, indices = self.entity_index.search(ent_embedding, top_k)

            for score, entity_idx in zip(distances[0], indices[0]):
                if entity_idx == -1:
                    continue

                similarity = float(score)

                if similarity <= RetrieverConfig.GRAPH_ENTITY_SIMILARITY_THRESHOLD:
                    continue

                entity_id = self.entity_id_list[entity_idx]

                if entity_id not in all_matches or similarity > all_matches[entity_id]:
                    all_matches[entity_id] = similarity

        seed_entities = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return seed_entities

    def process(self, dataset_name: str, resume: bool = True) -> List[Dict[str, Any]]:
        """Process all questions in the dataset using Graph RAG

        Args:
            dataset_name: Name of the dataset to process
            resume: If True, skip already processed questions

        Returns:
            List of results with format [{id, rag_answer}, ...]
        """
        print(f"Processing questions for dataset: {dataset_name}")
        print(f"Using Graph RAG with model: {self.model_name}")

        # Load questions
        questions = self.load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Initialize results tracking
        results = []
        processed_ids = set()

        # Load existing results if resuming
        if resume:
            from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
            existing_results = GraphRAGSaver.load_answers(self.model_name, dataset_name)

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

            # Step 1: Retrieve seed entities
            seed_entities = self.retrieve_seed_entities(question)

            # Build entity_name -> similarity mapping for sorting
            entity_similarities = {
                self.entity_id_to_text.get(entity_id, ""): similarity
                for entity_id, similarity in seed_entities
            }

            # Step 2: Use PPR on full graph to select relevant nodes (no BFS hop limit)
            filtered_nodes = self.retrieve_subgraph_by_ppr(seed_entities)

            # Step 3: Extract subgraph data
            subgraph_data = self.extract_subgraph_data(filtered_nodes, seed_entities)

            # Step 5: Collect source sentences (sorted by entity similarity)
            source_sentences = self.collect_source_sentences(subgraph_data["triplets"], entity_similarities)

            # Step 6: Serialize triplets to context (sorted by entity similarity)
            context = self.serialize_subgraph(subgraph_data["triplets"], entity_similarities)

            # Step 7: Generate answer with LLM
            answer = self.answer_with_context(question, context)

            if answer is not None:
                # Prepare answer result
                result = {"id": q_id, "rag_answer": answer}
                results.append(result)
                processed_ids.add(q_id)

                # Save incrementally
                if resume:
                    from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
                    GraphRAGSaver.save_answer(result, self.model_name, dataset_name)

                    # Save retrieval result (including source_sentences for debugging)
                    retrieval_result = {
                        "id": q_id,
                        "nodes": subgraph_data["nodes"],
                        "triplets": subgraph_data["triplets"],
                        "source_sentences": source_sentences  # Actual text used in prompt
                    }
                    GraphRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)
            else:
                # Record failure
                error_result = {"id": q_id, "rag_answer": None}
                results.append(error_result)

                if resume:
                    from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
                    GraphRAGSaver.save_answer(error_result, self.model_name, dataset_name)

                    # Save retrieval result (including source_sentences for debugging)
                    retrieval_result = {
                        "id": q_id,
                        "nodes": subgraph_data["nodes"],
                        "triplets": subgraph_data["triplets"],
                        "source_sentences": source_sentences  # Actual text used in prompt
                    }
                    GraphRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)

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
        print(f"Using Graph RAG with model: {self.model_name}")
        print(f"Max concurrent requests: {max_concurrent}")

        # Load questions
        questions = self.load_questions(dataset_name)
        total_questions = len(questions)
        print(f"Loaded {total_questions} questions")

        # Load existing results if resuming
        from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
        results = []
        processed_ids = set()

        if resume:
            existing_results = GraphRAGSaver.load_answers(self.model_name, dataset_name)
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

                # Step 1: Retrieve seed entities (async LLM call for entity extraction)
                seed_entities = await self.retrieve_seed_entities_async(question)

                # Build entity_name -> similarity mapping for sorting
                entity_similarities = {
                    self.entity_id_to_text.get(entity_id, ""): similarity
                    for entity_id, similarity in seed_entities
                }

                # Step 2: Use PPR on full graph to select relevant nodes
                filtered_nodes = self.retrieve_subgraph_by_ppr(seed_entities)

                # Step 3: Extract subgraph data
                subgraph_data = self.extract_subgraph_data(filtered_nodes, seed_entities)

                # Step 4: Collect source sentences
                source_sentences = self.collect_source_sentences(subgraph_data["triplets"], entity_similarities)

                # Step 5: Serialize to context
                context = self.serialize_subgraph(subgraph_data["triplets"], entity_similarities)

                # Step 6: Generate answer with LLM (async)
                answer = await self.answer_with_context_async(question, context)

                return {
                    "id": q_id,
                    "rag_answer": answer,
                    "nodes": subgraph_data["nodes"],
                    "triplets": subgraph_data["triplets"],
                    "source_sentences": source_sentences
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
                    GraphRAGSaver.save_answer(answer_result, self.model_name, dataset_name)
                    retrieval_result = {
                        "id": q_id,
                        "nodes": result["nodes"],
                        "triplets": result["triplets"],
                        "source_sentences": result["source_sentences"]
                    }
                    GraphRAGSaver.save_retrieval(retrieval_result, self.model_name, dataset_name)

        pbar.close()
        print(f"Processing Complete! Processed {len(processed_ids)}/{total_questions} questions")

        return results


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Graph RAG QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    processor = GraphRAGProcessor(dataset_name=args.dataset)
    results = processor.process(dataset_name=args.dataset, resume=not args.no_resume)

    # Save final results
    from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
    GraphRAGSaver.save_all(results, processor.model_name, args.dataset)

    print(f"\nResults saved to: {PathConfig.get_graph_rag_path(processor.model_name, args.dataset)}")
