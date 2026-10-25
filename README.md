# Retrieval probe demo

Companion to a final-year project on routing RAG retrieval paradigms
(Stellenbosch University, Industrial Engineering, 2026).

The project asks whether one cheap retrieval tells you anything useful about
what it just found. It does not tell you which retrieval method to use. It does
tell you, fairly reliably, whether any method found anything at all — and that
is what this app demonstrates.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the sentence encoder (about 90 MB) from Hugging Face.

Answer generation is optional. With [Ollama](https://ollama.com) running
(`ollama serve`, `ollama pull llama3.1:8b`) the Chat tab also writes an answer
from the retrieved passages. Without it, everything else still works: the
probe, the twelve measurements, the confidence band, the escalation decision
and the passages themselves. The project evaluates retrieval and never
evaluates generated answers, so nothing it claims depends on the model being
there.

## What this copy can and cannot do

| | this copy | the full local setup |
|---|---|---|
| Recorded results — all four analysis tabs | yes, complete | yes |
| Chat, live retrieval | **QuALITY only** | QuALITY, Musique, UltraDomain Legal |
| Add your own corpus | yes | yes |
| Generated answers | needs Ollama | Ollama |

Only QuALITY travels because its index and graph are 43 MB. Musique's are
238 MB and UltraDomain Legal's 148 MB, which is why those two corpora appear in
the recorded tabs but not in Chat here.

Nothing else is reduced. The recorded tabs hold every one of the 118,920
query-condition pairs for all three corpora, and the live retrieval for QuALITY
runs the project's own retrievers and the project's own twelve measurements,
not a reimplementation.

## Tabs

- **Chat** — ask QuALITY in plain language. One cheap retrieval, twelve
  measurements, a confidence band for the evidence, escalation to hybrid
  retrieval when the gate says it is worth it, and the passages that were used.
- **Add a corpus** — put your own documents through chunk, embed, index and
  graph, with the project's settings.
- **Replay a query** — any of 118,920 recorded query-condition pairs, with the
  twelve measurements, what each paradigm scored, and the oracle.
- **Did it find anything?** — the failure detector against both controls.
- **Is retrieve-all worth it?** — the escalation-share curve and its controls.
- **Worth building?** — the routing opportunity score, labelled a proposal.

## Attribution

The retrieval, chunking, embedding, indexing and graph components under
`vendor/RAGCore/` and `vendor/Config/`, and the QuALITY corpus under
`vendor/Dataset/`, come from
[RAGRouter-Bench](https://github.com/ziqiwang0908/RAGRouter-Bench) (Wang et
al., 2026, arXiv:2602.00296), used and adapted under its MIT licence. The
underlying corpus is QuALITY (Pang et al., NAACL 2022).

Everything else — the app, the twelve per-query measurements in
`vendor/QualityCore/`, the degradation programme, the evaluation harness and
the routers in `vendor/Run/Quality/`, and the recorded results in `data/` — is
original work for this project. `NOTICE` sets out the split file by file.

## What is deliberately absent

Paradigm selection. The project finds the probe cannot do it, and the app says
so rather than quietly omitting it.

Every probability shown is out of fold, under
`StratifiedGroupKFold(5, shuffle, seed 42)` grouped by query id.
