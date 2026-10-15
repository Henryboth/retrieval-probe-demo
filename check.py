"""
check.py — is the demo ready to show?

Checks every part the demo needs and says what to do about anything missing.
Nothing here changes state except loading the local model into memory.

Run:  /opt/anaconda3/envs/ragDemo/bin/python demo/check.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))

OK, BAD, WARN = "  ok   ", " FAIL  ", " warn  "
problems: list[str] = []


def line(state: str, label: str, detail: str = "") -> None:
    print(f"[{state}] {label}" + (f" — {detail}" if detail else ""))


def check_data() -> None:
    print("\nData")
    table = DEMO / "data" / "demo.parquet"
    if table.exists():
        import pandas as pd
        df = pd.read_parquet(table, columns=["corpus"])
        line(OK, "recorded results", f"{len(df):,} rows, {table.stat().st_size/1e6:.1f} MB")
    else:
        line(BAD, "recorded results", "missing")
        problems.append("/opt/anaconda3/envs/ragBench/bin/python demo/build_data.py")

    for name in ["detector.joblib", "gate_pq.joblib"]:
        if (DEMO / "data" / "models" / name).exists():
            line(OK, f"model {name}")
        else:
            line(BAD, f"model {name}", "missing")
            problems.append("/opt/anaconda3/envs/ragBench/bin/python demo/train_models.py")


def check_llm() -> None:
    print("\nLocal model (answers in the chat)")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=4) as r:
            tags = json.loads(r.read())
    except Exception:
        line(BAD, "ollama server", "not reachable on port 11434")
        problems.append("ollama serve &")
        return
    names = {m["name"] for m in tags.get("models", [])}
    line(OK, "ollama server", "up")
    if "llama3.1:8b" in names:
        line(OK, "llama3.1:8b", "downloaded")
    else:
        line(BAD, "llama3.1:8b", f"not downloaded (have: {', '.join(names) or 'nothing'})")
        problems.append("ollama pull llama3.1:8b")
        return

    import llm
    print("       loading it into memory and generating one sentence…")
    try:
        reply = llm.complete("Reply with exactly: generation works", max_tokens=12, timeout=180)
        line(OK, "generation", f'replied "{reply[:60]}"')
    except Exception as exc:
        line(BAD, "generation", str(exc))
        problems.append("check 'ollama serve' is running and not out of memory")


def check_corpora() -> None:
    print("\nCollections")
    import engine as eng
    import calibrate as cal
    for name in eng.BENCHMARKS + eng.list_user_corpora():
        index = eng.corpus_root(name) / "ProcessedData" / "Index" / name / "index.faiss"
        graph = eng.corpus_root(name) / "ProcessedData" / "Graph" / name / "graph.gpickle"
        if not index.exists():
            line(BAD, name, "no FAISS index")
            continue
        bits = ["index"]
        bits.append("graph" if graph.exists() else "NO GRAPH")
        if cal.has_reference(name):
            bits.append("confidence reference")
        elif name in eng.BENCHMARKS:
            bits.append("reference builds itself from labels")
        else:
            bits.append("NO REFERENCE — build it on the Chat tab")
        line(OK if "NO GRAPH" not in bits else WARN, name, ", ".join(bits))


def check_retrieval() -> None:
    print("\nEnd to end (one real question)")
    import engine as eng
    import calibrate as cal
    try:
        corpus = eng.get_corpus("quality")
        result = corpus.ask("What happens during the chess tournament?", run_all=False)
        preds = eng.Predictor().predict(result["features"])
        import pandas as pd
        reference = cal.load_reference("quality") or cal.reference_from_labelled(
            pd.read_parquet(DEMO / "data" / "demo.parquet"), "quality")
        band = cal.assess(preds["p_all_failed"], reference)
        line(OK, "retrieval", f"{len(result['naive'])} passages, "
                              f"top score {result['naive'][0].score:.3f}")
        line(OK, "confidence", f"{band['band']} — {band['headline']}")
    except Exception as exc:
        line(BAD, "retrieval", f"{type(exc).__name__}: {exc}")
        problems.append("see the traceback above")


def main() -> None:
    print("=" * 62)
    print("  Demo health check")
    print("=" * 62)
    check_data()
    check_llm()
    check_corpora()
    check_retrieval()
    print("\n" + "=" * 62)
    if problems:
        print("  Not ready. Fix these:\n")
        for p in dict.fromkeys(problems):
            print(f"    {p}")
    else:
        print("  Everything is ready. Start the app with:\n")
        print("    /opt/anaconda3/envs/ragDemo/bin/streamlit run demo/app.py "
              "--server.address 127.0.0.1")
    print("=" * 62)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
