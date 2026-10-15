"""
ingest.py — put an arbitrary corpus through the project's own pipeline.

Chunking, embedding, FAISS indexing and knowledge-graph construction all run
through RAGCore, with the same settings the experiments used. Nothing is
reimplemented and nothing is written into the research tree: a corpus added
here lives under demo/corpora/<name>/.

Roughly 65 seconds per megabyte of text on this machine, most of it spaCy
entity extraction for the graph.

Accepted input: plain text or Markdown files, each becoming one document, or a
JSON Lines file whose records carry a "context" or "text" field.
"""

from __future__ import annotations

import os
import sys

# see engine.py — must precede numpy/faiss/torch, and xgboost must precede faiss
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import xgboost  # noqa: F401

import json
import re
import shutil
from pathlib import Path
from typing import Iterator

from engine import CODE, USER_CORPORA, BENCHMARKS, point_paths_at, _require_cwd

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,38}$")


def valid_name(name: str) -> tuple[bool, str]:
    if not NAME_RE.match(name or ""):
        return False, ("Use 2–39 characters: lower-case letters, digits, "
                       "hyphen or underscore, starting with a letter or digit.")
    if name in BENCHMARKS:
        return False, f"'{name}' is a benchmark corpus name. Choose another."
    return True, ""


def documents_from_uploads(files: list[tuple[str, bytes]]) -> list[dict]:
    """Turn uploaded files into {title, context} records. A .jsonl/.json file
    is read as JSON Lines if its records carry a text field; any other file
    becomes a single document."""
    docs: list[dict] = []
    for filename, blob in files:
        try:
            raw = blob.decode("utf-8", errors="replace")
        except Exception:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            parsed = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    parsed = []
                    break
                if isinstance(rec, dict):
                    parsed.append(rec)
            if parsed and any(r.get("context") or r.get("text") for r in parsed):
                for i, rec in enumerate(parsed):
                    text = rec.get("context") or rec.get("text") or ""
                    if text.strip():
                        docs.append({"title": str(rec.get("title", f"{filename}#{i}")),
                                     "context": text})
                continue
        if raw.strip():
            docs.append({"title": Path(filename).stem, "context": raw})
    return docs


def write_corpus(name: str, docs: list[dict]) -> Path:
    """Write Corpus.json in the benchmark corpora's own format: one JSON object
    per line with title, context and an integer id. ChunkDo requires the id."""
    root = USER_CORPORA / name
    raw_dir = root / "RawData" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "Corpus.json"
    records = [
        {"title": d.get("title", f"document {i}"), "context": d.get("context", ""), "id": i}
        for i, d in enumerate(docs)
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    return path


def delete_corpus(name: str) -> bool:
    """Remove a corpus a user added. Benchmarks are refused."""
    if name in BENCHMARKS:
        return False
    root = USER_CORPORA / name
    if not root.exists() or USER_CORPORA not in root.resolve().parents:
        return False
    shutil.rmtree(root)
    return True


def ingest(name: str, docs: list[dict]) -> Iterator[tuple[str, str]]:
    """Run the pipeline, yielding (stage, message) as each stage completes."""
    _require_cwd()
    write_corpus(name, docs)
    point_paths_at(name)

    from Config.PathConfig import PathConfig
    from RAGCore.Chunk.ChunkDo import ChunkProcessor
    from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
    from RAGCore.Index.IndexDo import IndexProcessor
    from RAGCore.Index.IndexSave import IndexSaver
    from RAGCore.Graph.GraphDo import GraphProcessor
    from RAGCore.Graph.GraphSave import GraphSaver

    chars = sum(len(d.get("context", "")) for d in docs)
    yield "start", f"{len(docs)} documents, {chars:,} characters"

    chunker = ChunkProcessor()
    corpus = chunker.load_corpus(PathConfig.get_corpus_path(name))
    chunks = chunker.process_corpus(corpus)
    n_chunks = sum(len(v) for v in chunks.values())
    yield "chunk", f"{n_chunks:,} chunks at 180 tokens with 36 overlapping"

    embeddings = EmbeddingProcessor().process_chunks(chunks)
    EmbeddingSaver.save(embeddings, name)
    yield "embed", f"{n_chunks:,} chunk embeddings with all-MiniLM-L6-v2"

    IndexSaver.save(IndexProcessor().build_index(EmbeddingSaver.load(name)), name)
    yield "index", "FAISS index built"

    result = GraphProcessor().process(chunks, dataset_name=name, resume=False)
    GraphSaver.save(result, name)
    graph = result.get("graph")
    try:
        detail = f"{graph.number_of_nodes():,} entities, {graph.number_of_edges():,} edges"
    except Exception:
        detail = "knowledge graph built"
    yield "graph", detail

    yield "done", f"'{name}' is ready to query"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Add a corpus to the demo")
    ap.add_argument("name")
    ap.add_argument("paths", nargs="+", help="text, markdown or JSON Lines files")
    args = ap.parse_args()

    ok, why = valid_name(args.name)
    if not ok:
        raise SystemExit(why)

    uploads = []
    for p in args.paths:
        path = Path(p)
        for f in (sorted(path.rglob("*")) if path.is_dir() else [path]):
            if f.is_file():
                uploads.append((f.name, f.read_bytes()))
    docs = documents_from_uploads(uploads)
    if not docs:
        raise SystemExit("No readable documents found.")

    for stage, message in ingest(args.name, docs):
        print(f"[{stage:>5}] {message}", flush=True)


if __name__ == "__main__":
    main()
