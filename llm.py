"""
llm.py — the local generator, for the chat answer and for building calibration
reference questions.

Ollama with llama3.1:8b, which is what Config/LLMConfig.py already selects
(PROVIDER = "llama", LLAMA_BASE_URL = http://localhost:11434/v1). Nothing is
sent off the machine and no API key is involved.

The thesis evaluates retrieval only and makes no claim about generated text.
The generator is here so the demo reads like something a person would use; the
measured part is the retrieval beneath it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=3) as r:
            names = {m["name"] for m in json.loads(r.read()).get("models", [])}
        return MODEL in names
    except Exception:
        return False


def _post(payload: dict, timeout: int):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        raise LLMUnavailable(
            f"Could not reach Ollama at {HOST}. Start it with 'ollama serve'."
        ) from exc


def preload(keep_alive: str = "30m") -> bool:
    """Load the model into memory and hold it there, so the first answer does
    not pay for the load. Ollama unloads an idle model after five minutes by
    default, which is shorter than the gap between questions in a demo."""
    try:
        payload = {"model": MODEL, "prompt": "ready", "stream": False,
                   "keep_alive": keep_alive, "options": {"num_predict": 1}}
        with _post(payload, 180) as response:
            response.read()
        return True
    except Exception:
        return False


def complete(prompt: str, max_tokens: int = 220, temperature: float = 0.2,
             timeout: int = 240) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "keep_alive": "30m",
               "options": {"temperature": temperature, "num_predict": max_tokens}}
    with _post(payload, timeout) as response:
        return json.loads(response.read()).get("response", "").strip()


def stream(prompt: str, max_tokens: int = 400, temperature: float = 0.2,
           timeout: int = 300) -> Iterator[str]:
    payload = {"model": MODEL, "prompt": prompt, "stream": True,
               "keep_alive": "30m",
               "options": {"temperature": temperature, "num_predict": max_tokens}}
    with _post(payload, timeout) as response:
        for line in response:
            if not line.strip():
                continue
            try:
                piece = json.loads(line)
            except json.JSONDecodeError:
                continue
            if piece.get("response"):
                yield piece["response"]
            if piece.get("done"):
                break


ANSWER_PROMPT = """You are answering from a document collection. Use only the \
passages below. Cite them as [1], [2] and so on. If the passages do not contain \
the answer, say plainly that the collection does not appear to cover it rather \
than guessing.

Passages:
{context}

Question: {question}

Answer:"""

QUESTION_PROMPT = """Write one natural question that the passage below answers. \
It must read like a question a person would ask, not a restatement of the text. \
Output only the question.

Passage: {chunk}

Question:"""


def answer(question: str, passages: list, max_chars: int = 700) -> Iterator[str]:
    context = "\n\n".join(
        f"[{i + 1}] {' '.join((p.text or '').split())[:max_chars]}"
        for i, p in enumerate(passages)
    )
    return stream(ANSWER_PROMPT.format(context=context, question=question))


def make_question(chunk: str) -> str:
    text = " ".join(chunk.split())[:1200]
    out = complete(QUESTION_PROMPT.format(chunk=text), max_tokens=48,
                   temperature=0.8, timeout=120)
    return out.strip().strip('"').split("\n")[0].strip()
