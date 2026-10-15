"""
calibrate.py — turn the detector's uncalibrated score into a confidence band
that means something on the corpus actually being used.

Why this exists. The detector ranks well on a corpus it has never seen (AUC
0.62-0.86 leave-one-corpus-out) but its probabilities do not transfer: it will
report a high failure score for a question the corpus answers perfectly well.
Ranking is the part that survives, so the band is built from a ranking.

How. Two reference sets of questions are put through the probe once per corpus:

  answerable    questions written by the local model from the corpus's own
                chunks. The material answering them is present by construction,
                so these are questions the corpus can certainly serve.
  off-topic     questions written the same way from the other corpora. Nothing
                answering them is present, so these stand for questions the
                corpus cannot serve.

The questions are generated rather than taken verbatim. A fragment copied out
of a chunk matches that chunk almost exactly, which is a far easier retrieval
problem than a real question, and a reference built that way is so easy that
every genuine question falls below it. Generating questions costs about three
seconds each and is done once per corpus.

A new question's detector score is then placed against those two distributions.
Saying "this scores better than four in five questions the corpus can certainly
answer" is a claim the ranking supports; saying "87 per cent" is not.

This calibration is a device for the demo. It is not a method from the thesis
and no result in the thesis depends on it.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

from engine import BENCHMARKS, DEMO, corpus_root

CAL_DIR = DEMO / "data" / "calibration"
N_PER_SET = 60
MIN_CHUNK_WORDS = 60
SEED = 42


def _chunk_texts(name: str, limit: int = 4000) -> list[str]:
    """Chunk texts from a corpus's FAISS metadata, without loading a retriever."""
    meta = corpus_root(name) / "ProcessedData" / "Index" / name / "metadata.json"
    if not meta.exists():
        return []
    chunks = json.loads(meta.read_text())
    return [c.get("text", "") for c in chunks[:limit] if c.get("text")]


def _readable(text: str) -> bool:
    """Reject chunks that are mostly not prose — base64 images and the like."""
    if len(text.split()) < MIN_CHUNK_WORDS:
        return False
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / max(len(text), 1) > 0.80


def _sample_chunks(texts: list[str], n: int, rng: random.Random) -> list[str]:
    pool = [t for t in texts if _readable(t)]
    return rng.sample(pool, min(n, len(pool))) if pool else []


def reference_from_labelled(df, name: str) -> dict:
    """For a benchmark corpus, build the reference from its real questions and
    their relevance judgements. No generation and no guessing: 'answerable' is
    the recorded out-of-fold score for questions at least one paradigm served,
    'off_topic' the score for questions every paradigm failed. This is the
    reference the generated one is a substitute for."""
    rows = df[df.corpus == name]
    served = rows.loc[rows.all_failed == 0, "p_fail"].to_numpy()
    failed = rows.loc[rows.all_failed == 1, "p_fail"].to_numpy()
    rng = np.random.default_rng(SEED)
    cap = 4000

    def sample(values):
        if values.size > cap:
            values = rng.choice(values, size=cap, replace=False)
        return sorted(round(float(v), 5) for v in values)

    reference = {
        "corpus": name,
        "source": "labelled",
        "n_answerable": int(served.size),
        "n_off_topic": int(failed.size),
        "answerable": sample(served),
        "off_topic": sample(failed),
        "note": ("built from the corpus's own questions and relevance "
                 "judgements, using out-of-fold detector scores"),
    }
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    reference_path(name).write_text(json.dumps(reference))
    return reference


def reference_path(name: str) -> Path:
    return CAL_DIR / f"{name}.json"


def has_reference(name: str) -> bool:
    return reference_path(name).exists()


def build_reference(live_corpus, predictor, progress=None) -> dict:
    """Generate both reference sets, score them through the live probe, and
    cache the result. Roughly six minutes per corpus, done once."""
    import llm

    name = live_corpus.name
    rng = random.Random(SEED)

    own_chunks = _sample_chunks(_chunk_texts(name), N_PER_SET, rng)
    other_chunks: list[str] = []
    for other in [c for c in BENCHMARKS if c != name]:
        other_chunks += _sample_chunks(_chunk_texts(other), N_PER_SET // 2, rng)
    rng.shuffle(other_chunks)
    other_chunks = other_chunks[:N_PER_SET]

    scores, examples = {}, {}
    for label, chunks in (("answerable", own_chunks), ("off_topic", other_chunks)):
        values, questions = [], []
        for i, chunk in enumerate(chunks):
            if progress:
                progress(label, i, len(chunks))
            try:
                question = llm.make_question(chunk)
            except Exception:
                continue
            if len(question.split()) < 3 or "?" not in question:
                continue
            result = live_corpus.ask(question, run_all=False)
            values.append(predictor.predict(result["features"])["p_all_failed"])
            questions.append(question)
        scores[label] = values
        examples[label] = questions[:5]

    reference = {
        "corpus": name,
        "n_answerable": len(scores["answerable"]),
        "n_off_topic": len(scores["off_topic"]),
        "answerable": sorted(round(float(v), 5) for v in scores["answerable"]),
        "off_topic": sorted(round(float(v), 5) for v in scores["off_topic"]),
        "example_questions": examples,
        "seed": SEED,
        "generator": llm.MODEL,
        "note": "demo calibration from generated questions; not a thesis method",
    }
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    reference_path(name).write_text(json.dumps(reference, indent=1))
    return reference


def load_reference(name: str) -> dict | None:
    path = reference_path(name)
    return json.loads(path.read_text()) if path.exists() else None


def assess(score: float, reference: dict | None) -> dict:
    """Place a detector score against the two reference distributions."""
    if not reference or not reference.get("answerable"):
        return {
            "band": "unknown",
            "headline": "No reference for this corpus",
            "detail": "Build the confidence reference to interpret this score.",
            "better_than_answerable": None,
            "raw": score,
        }

    answerable = np.asarray(reference["answerable"], dtype=float)
    off_topic = np.asarray(reference["off_topic"], dtype=float) if reference.get("off_topic") else None

    # share of certainly-answerable questions this one scores better than
    better = float((answerable > score).mean())
    off_median = float(np.median(off_topic)) if off_topic is not None and off_topic.size else None

    if better >= 0.50:
        band = "high"
        headline = "Retrieval looks reliable"
    elif off_median is not None and score >= off_median:
        band = "low"
        headline = "Retrieval looks unreliable"
    elif better >= 0.20:
        band = "moderate"
        headline = "Retrieval looks usable, with reservations"
    else:
        band = "low"
        headline = "Retrieval looks unreliable"

    detail = (
        f"This question's failure score is lower than {better:.0%} of questions "
        f"the corpus can certainly answer"
    )
    if off_median is not None:
        side = "below" if score < off_median else "at or above"
        detail += f", and sits {side} the typical score for an off-topic question"
    return {
        "band": band,
        "headline": headline,
        "detail": detail + ".",
        "better_than_answerable": better,
        "off_topic_median": off_median,
        "raw": score,
    }
