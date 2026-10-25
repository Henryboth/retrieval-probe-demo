import json
import tiktoken
from typing import List, Dict, Any
from tqdm import tqdm
from Config.ChunkConfig import ChunkConfig


class ChunkProcessor:
    """Process corpus documents into chunks based on configuration"""

    def __init__(self):
        self.chunk_size = ChunkConfig.CHUNK_SIZE
        self.chunk_overlap = ChunkConfig.CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding(ChunkConfig.TOKENIZER_ENCODING)

    def load_corpus(self, corpus_path: str) -> List[Dict[str, Any]]:
        """Load corpus from JSON file"""
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(json.loads(line.strip()))
        return corpus

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += (self.chunk_size - self.chunk_overlap)

        return chunks

    def process_corpus(self, corpus: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """Process entire corpus into chunks

        Returns:
            Dict mapping doc_id to list of chunk texts
            Format: {doc_id: [chunk1_text, chunk2_text, ...]}
        """
        chunks_by_doc = {}

        # Add progress bar for chunking documents
        for doc in tqdm(corpus, desc="Chunking documents", unit="doc"):
            doc_id = doc["id"]
            title = doc["title"]
            context = doc["context"]

            # Combine title and context
            full_text = f"{title}\n\n{context}"

            # Chunk the document
            chunk_texts = self.chunk_text(full_text)
            chunks_by_doc[doc_id] = chunk_texts

        return chunks_by_doc


# Usage example
if __name__ == "__main__":
    from Config.PathConfig import PathConfig

    processor = ChunkProcessor()

    # Load and process corpus
    dataset_name = "hotpotqa"
    corpus_path = PathConfig.get_corpus_path(dataset_name)
    corpus = processor.load_corpus(corpus_path)

    print(f"Loaded {len(corpus)} documents")

    # Process into chunks
    chunks_by_doc = processor.process_corpus(corpus)

    total_chunks = sum(len(chunks) for chunks in chunks_by_doc.values())
    print(f"Created {total_chunks} chunks across {len(chunks_by_doc)} documents")

    # Show example
    first_doc_id = list(chunks_by_doc.keys())[0]
    print(f"\nDoc {first_doc_id} has {len(chunks_by_doc[first_doc_id])} chunks")
    print(f"First chunk: {chunks_by_doc[first_doc_id][0][:100]}...")
