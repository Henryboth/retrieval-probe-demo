# Retrieval probe demo — recorded results

Findings from a final-year project on routing RAG retrieval paradigms. This
bundle holds the recorded results of the experiments and the app that reads
them.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

## Tabs

- **Replay a query** — any of 118,920 recorded query-condition pairs, with the
  twelve probe measurements, what each paradigm scored, and the oracle.
- **Did it find anything?** — the failure detector against both controls.
- **Is retrieve-all worth it?** — the escalation-share curve and its controls.
- **Worth building?** — the routing opportunity score.

The **Chat** and **Add a corpus** tabs need live retrieval, which needs the
FAISS indices, the knowledge graphs and a local language model. They are not in
this bundle, and the app says so in those tabs rather than failing.

Every probability shown is out of fold, under
`StratifiedGroupKFold(5, shuffle, seed 42)` grouped by query id.
