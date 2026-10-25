import numpy as np
from typing import Dict, List
from tqdm import tqdm
from Config.EmbConfig import EmbConfig


class EmbeddingProcessor:
    """Generate embeddings for chunks using configured embedding model"""

    def __init__(self):
        self.provider = EmbConfig.PROVIDER
        self.batch_size = EmbConfig.BATCH_SIZE
        self.normalize = EmbConfig.NORMALIZE_EMBEDDINGS

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=EmbConfig.OPENAI_API_KEY)
            self.model = EmbConfig.OPENAI_MODEL

        elif self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            device = EmbConfig.DEVICE
            # MPS fallback: use cpu if mps causes issues
            try:
                import torch
                if device == "mps" and not torch.backends.mps.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
            self.st_model = SentenceTransformer(
                EmbConfig.HUGGINGFACE_MODEL, device=device
            )
            self.model = EmbConfig.HUGGINGFACE_MODEL

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "sentence-transformers":
            return self._embed_st(texts)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    def _embed_st(self, texts: List[str], pbar=None) -> np.ndarray:
        """Generate embeddings using sentence-transformers (local, no API key needed)"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_emb = self.st_model.encode(
                batch, convert_to_numpy=True, show_progress_bar=False
            )
            all_embeddings.append(batch_emb)
            if pbar:
                pbar.update(1)
        embeddings = np.vstack(all_embeddings)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1, norms)
        return embeddings

    def _embed_openai(self, texts: List[str], pbar=None) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            if pbar:
                pbar.update(1)
        embeddings = np.array(all_embeddings)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        return embeddings

    def process_chunks(self, chunks_by_doc: Dict[int, List[str]], dataset_name: str = None,
                      resume: bool = True) -> Dict[int, Dict[str, any]]:
        if resume and dataset_name:
            from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
            embeddings_by_doc = EmbeddingSaver.load_existing(dataset_name)
            if embeddings_by_doc:
                print(f"Resuming from checkpoint: {len(embeddings_by_doc)} documents already processed")
        else:
            embeddings_by_doc = {}

        total_batches = sum(
            (len(chunks) + self.batch_size - 1) // self.batch_size
            for doc_id, chunks in chunks_by_doc.items()
            if doc_id not in embeddings_by_doc
        )

        if total_batches == 0:
            print("All documents already processed")
            return embeddings_by_doc

        embed_fn = self._embed_st if self.provider == "sentence-transformers" else self._embed_openai

        with tqdm(total=total_batches, desc="Generating chunk embeddings", unit="batch") as pbar:
            for doc_id, chunks in chunks_by_doc.items():
                if doc_id in embeddings_by_doc:
                    continue
                embeddings = embed_fn(chunks, pbar=pbar)
                embeddings_by_doc[doc_id] = {"chunks": chunks, "embeddings": embeddings}
                if dataset_name:
                    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
                    EmbeddingSaver.save_incremental(doc_id, chunks, embeddings, dataset_name)

        return embeddings_by_doc
