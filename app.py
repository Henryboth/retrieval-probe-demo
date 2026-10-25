"""
app.py — RAGRouter-Bench demo: what a cheap retrieval probe can actually do.

Two components, both established in the thesis:
  1. the failure detector — judge whether the retrieval found relevant material
  2. the probe gate       — decide whether to pay for the retrieve-all selector

Paradigm selection is deliberately absent. The thesis finds the probe cannot tell
which paradigm will win; see the note in the sidebar.

Every probability shown is out of fold.

  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

DATA = Path(__file__).resolve().parent / "data"
CORPORA = {
    "quality": "QuALITY",
    "musique": "Musique",
    "ultraDomain_legal": "UltraDomain Legal",
}
OPERATOR_LABELS = {
    "text_truncation": "Text truncation",
    "entity_variation": "Entity name variation",
    "graph_edge_removal": "Graph edge removal",
    "document_dropout": "Document dropout",
    "combined": "Combined",
}
GATES = {"probe": "Probe", "static": "Static blocks", "qonly": "Query type", "random": "Pseudo-random"}

st.set_page_config(page_title="Retrieval probe demo", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet(DATA / "demo.parquet")
    meta = json.loads((DATA / "meta.json").read_text())
    detector = json.loads((DATA / "failure_detector.json").read_text())
    gate = json.loads((DATA / "gated_selector.json").read_text())
    omega = json.loads((DATA / "routing_uncertainty.json").read_text())
    return df, meta, detector, gate, omega


if not (DATA / "demo.parquet").exists():
    st.error("data/demo.parquet is missing from this copy.")
    st.stop()

df, meta, detector, gate_rec, omega = load()
feature_names = meta["feature_names"]

# Live retrieval needs faiss, torch and the corpus indices; the recorded tabs
# need none of them. Detecting this lets one copy of the app run both on this
# machine and on a host that has only the 7 MB of recorded results.
@st.cache_resource(show_spinner=False)
def live_available() -> tuple[bool, str]:
    try:
        import faiss, sentence_transformers  # noqa: F401
    except Exception as exc:
        return False, f"retrieval libraries are not installed here ({exc.__class__.__name__})"
    try:
        import engine as eng
        if not any((eng.corpus_root(c) / "ProcessedData" / "Index" / c / "index.faiss").exists()
                   for c in eng.BENCHMARKS):
            return False, "no corpus indices are present"
    except Exception as exc:
        return False, f"the retrieval engine could not start ({exc.__class__.__name__})"
    return True, ""


LIVE, LIVE_WHY = live_available()

# ───────────────────────────────── sidebar ──────────────────────────────────
with st.sidebar:
    st.title("Retrieval probe")
    st.caption("Can a cheap first search tell you whether to trust what it found?")

    corpus = st.selectbox(
        "Corpus for the analysis tabs", list(CORPORA),
        format_func=lambda c: CORPORA[c],
        help="Chat has its own picker. This one drives the four results tabs.",
    )

    if LIVE:
        import engine as _eng
        _status = _eng.status()
        _all = _eng.BENCHMARKS + _eng.list_user_corpora()
        _pending = [n for n in _all if _status.get(n) != "ready"]
        if _pending:
            st.caption(f"Loading {len(_pending)} of {len(_all)} collections…")
            for _name in _pending:
                st.caption(f"◐ {CORPORA.get(_name, _name)}")
            if st.button("Refresh", key="refresh_warm", use_container_width=True):
                st.rerun()
        else:
            st.caption(f"✓ {len(_all)} collections ready")

    with st.expander("What this is"):
        st.caption(
            "Every question starts with one cheap search. Twelve measurements "
            "are taken from what comes back, and those measurements drive two "
            "decisions: whether the results can be trusted, and whether a more "
            "expensive search is worth running."
        )
        st.caption(
            "Picking *which* search method to use is not shown, because the "
            "measurements cannot do it. They track how hard a question is, not "
            "which method answers it best. What is here is what survived that "
            "finding."
        )

    with st.expander("How the numbers were obtained"):
        st.caption(
            f"Every probability is out of fold — {meta['fold_scheme']}. No "
            "question is scored by a model trained on it."
        )
        st.caption(
            "Recorded results cover 118,920 question-condition pairs across "
            "three corpora, five degradation operators and eight severities."
        )

sub_all = df[df.corpus == corpus]
ds_meta = meta["datasets"][corpus]

(tab_chat, tab_add, tab_query, tab_detector, tab_gate, tab_omega) = st.tabs(
    ["Chat", "Add a corpus", "Replay a query",
     "Did it find anything?", "Is retrieve-all worth it?", "Worth building?"]
)

# ─── warm the corpora in the background so the first question is not slow ───
def _start_prewarm():
    if not LIVE:
        return
    import engine as eng
    eng.prewarm()


_start_prewarm()


# ───────────────────────────────── chat ─────────────────────────────────────
def live_corpus(name: str):
    """Shared with the background warmer, so a corpus already warmed is
    returned immediately rather than loaded again."""
    import engine as eng
    return eng.get_corpus(name)


@st.cache_resource(show_spinner=False)
def predictor():
    from engine import Predictor
    return Predictor()


BAND_STYLE = {
    "high":     ("#1a7f37", "Retrieval looks reliable"),
    "moderate": ("#9a6700", "Usable, with reservations"),
    "low":      ("#b42318", "Retrieval looks unreliable"),
    "unknown":  ("#57606a", "Not calibrated for this corpus"),
}


def confidence_badge(assessment: dict) -> str:
    colour, _ = BAND_STYLE.get(assessment["band"], BAND_STYLE["unknown"])
    pct = assessment.get("better_than_answerable")
    tail = f" · better than {pct:.0%} of answerable questions" if pct is not None else ""
    return (
        f"<div style='display:inline-block;padding:4px 12px;border-radius:14px;"
        f"background:{colour};color:#fff;font-size:0.84rem;font-weight:600'>"
        f"{assessment['headline']}{tail}</div>"
    )


with tab_chat:
    if not LIVE:
        st.info(
            "Chat needs live retrieval, which is not available in this copy because "
            + LIVE_WHY + ". The recorded results in the other tabs do not "
            "need it and are complete."
        )
    else:
        import engine as eng
        import calibrate as cal
        import llm

        user_corpora = eng.list_user_corpora()
        options = eng.available_benchmarks() + user_corpora

        head_l, head_r = st.columns([3, 2])
        with head_l:
            st.subheader("Ask the collection")
            st.caption(
                "Every message runs one cheap retrieval first. The twelve "
                "measurements taken from what came back decide how far to trust the "
                "evidence, and whether to spend more retrieval on it."
            )
        with head_r:
            chat_corpus = st.selectbox(
                "Collection", options,
                format_func=lambda c: CORPORA.get(c, f"{c} (added)"), key="chat_corpus",
            )

        if not llm.available():
            st.error(
                "The local model is not responding. Start it in a terminal with "
                "`ollama serve` — the demo uses llama3.1:8b on this machine, so "
                "nothing is sent anywhere."
            )

        reference = cal.load_reference(chat_corpus)
        if reference is None and chat_corpus in eng.BENCHMARKS:
            # this corpus has real questions and judgements, so no generation needed
            reference = cal.reference_from_labelled(df, chat_corpus)
        if reference is None:
            st.warning(
                "This collection has no confidence reference yet. Without one the "
                "detector's raw score cannot be read as anything, because its "
                "probabilities do not carry across collections — only its ranking "
                "does. Building the reference writes about 120 questions with the "
                "local model and scores them, which takes a few minutes and is done "
                "once."
            )
            if st.button("Build the confidence reference", type="primary", key="build_ref"):
                bar = st.progress(0.0, text="starting")

                def tick(label, i, n):
                    base = 0.0 if label == "answerable" else 0.5
                    bar.progress(min(base + 0.5 * (i / max(n, 1)), 1.0),
                                 text=f"{label.replace('_', ' ')} {i}/{n}")

                with st.spinner("Writing and scoring reference questions…"):
                    cal.build_reference(live_corpus(chat_corpus), predictor(), progress=tick)
                st.rerun()

        if "chat" not in st.session_state:
            st.session_state.chat = {}
        history = st.session_state.chat.setdefault(chat_corpus, [])

        c1, c2 = st.columns([1, 5])
        if c1.button("Clear", key="clear_chat"):
            st.session_state.chat[chat_corpus] = []
            st.rerun()
        c2.caption(
            "Answers are generated by llama3.1:8b on this machine from the "
            "retrieved passages. The thesis measures the retrieval, not the "
            "generated text."
        )

        for turn in history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(confidence_badge(turn["assessment"]), unsafe_allow_html=True)
                st.write(turn["answer"])
                with st.expander(f"Evidence · {turn['route']}"):
                    st.caption(turn["assessment"]["detail"])
                    for h in turn["passages"]:
                        body = " ".join((h["text"] or "").split())
                        st.markdown(f"**[{h['rank']}]** doc `{h['doc_id']}` · score {h['score']:.3f}")
                        st.caption(body[:320] + ("…" if len(body) > 320 else ""))
                    st.caption(
                        f"Detector score {turn['assessment']['raw']:.3f} · "
                        f"P(another paradigm beats naive) {turn['p_escalate']:.2f}"
                    )

        question = st.chat_input("Ask something this collection might answer")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving…"):
                    corpus_obj = live_corpus(chat_corpus)
                    probe = corpus_obj.ask(question, run_all=False)
                    preds = predictor().predict(probe["features"])
                    assessment = cal.assess(preds["p_all_failed"], reference)

                    # the gate decides whether this question is worth more retrieval
                    escalate = preds["p_escalate"] >= 0.5
                    if escalate and corpus_obj.hybrid is not None:
                        passages = corpus_obj.ask(question)["hybrid"]
                        route = "escalated to hybrid retrieval"
                    else:
                        passages = probe["naive"]
                        route = "answered from the first cheap retrieval"

                st.markdown(confidence_badge(assessment), unsafe_allow_html=True)
                try:
                    answer = st.write_stream(llm.answer(question, passages[:5]))
                except Exception as exc:
                    answer = f"Could not generate an answer: {exc}"
                    st.error(answer)

                with st.expander(f"Evidence · {route}"):
                    st.caption(assessment["detail"])
                    for h in passages[:5]:
                        body = " ".join((h.text or "").split())
                        st.markdown(f"**[{h.rank}]** doc `{h.doc_id}` · score {h.score:.3f}")
                        st.caption(body[:320] + ("…" if len(body) > 320 else ""))
                    st.caption(
                        f"Detector score {assessment['raw']:.3f} · "
                        f"P(another paradigm beats naive) {preds['p_escalate']:.2f}"
                    )
                    st.bar_chart(
                        pd.DataFrame(
                            {"value": [float(probe["features"].get(n, 0.0)) for n in feature_names]},
                            index=[n.replace("_", " ") for n in feature_names],
                        ),
                        horizontal=True, height=300,
                    )

            history.append({
                "question": question, "answer": answer, "assessment": assessment,
                "route": route, "p_escalate": preds["p_escalate"],
                "passages": [{"rank": h.rank, "doc_id": h.doc_id,
                              "score": h.score, "text": h.text} for h in passages[:5]],
            })

# ─────────────────────────── add a corpus ───────────────────────────────────
with tab_add:
    if not LIVE:
        st.info(
            "Adding a corpus needs live retrieval, which is not available in this copy because "
            + LIVE_WHY + ". The recorded results in the other tabs do not "
            "need it and are complete."
        )
    else:
        import engine as eng
        import ingest as ing

        st.subheader("Put your own corpus through the pipeline")
        st.caption(
            "Documents are chunked at 180 tokens with 36 overlapping, embedded with "
            "all-MiniLM-L6-v2, indexed in FAISS, and run through spaCy entity "
            "extraction to build a knowledge graph — the same settings the "
            "experiments used. Roughly a minute per megabyte, most of it the graph."
        )

        existing = eng.list_user_corpora()
        if existing:
            st.markdown("**Corpora you have added**")
            for name in existing:
                col_a, col_b = st.columns([5, 1])
                col_a.write(f"`{name}`")
                if col_b.button("Delete", key=f"del_{name}"):
                    ing.delete_corpus(name)
                    st.cache_resource.clear()
                    st.rerun()
            st.divider()

        new_name = st.text_input("Name it", placeholder="my-corpus", key="new_corpus_name")
        uploads = st.file_uploader(
            "Documents", accept_multiple_files=True,
            type=["txt", "md", "json", "jsonl"],
            help="Each text or Markdown file becomes one document. A JSON Lines file "
                 "whose records carry a 'context' or 'text' field becomes many.",
        )

        if uploads:
            total = sum(len(u.getvalue()) for u in uploads)
            st.caption(f"{len(uploads)} file(s), {total/1e6:.1f} MB — "
                       f"roughly {max(1, round(total/1e6 * 65 / 60))} minute(s) to ingest")

        ok_name, why = ing.valid_name(new_name) if new_name else (False, "")
        if new_name and not ok_name:
            st.error(why)

        if st.button("Build it", type="primary", disabled=not (uploads and ok_name)):
            docs = ing.documents_from_uploads([(u.name, u.getvalue()) for u in uploads])
            if not docs:
                st.error("No readable text found in those files.")
            else:
                progress = st.progress(0.0, text="starting")
                steps = {"start": 0.05, "chunk": 0.2, "embed": 0.55,
                         "index": 0.6, "graph": 0.97, "done": 1.0}
                log = st.empty()
                lines = []
                try:
                    for stage, message in ing.ingest(new_name, docs):
                        lines.append(f"**{stage}** — {message}")
                        log.markdown("  \n".join(lines))
                        progress.progress(steps.get(stage, 0.5), text=stage)
                    st.cache_resource.clear()
                    st.success(f"'{new_name}' is ready. Ask it something on the first tab.")
                except Exception as exc:  # surfaced rather than swallowed
                    st.error(f"Ingestion failed: {type(exc).__name__}: {exc}")

        st.caption(
            "Added corpora are written to corpora/ and never into the bundled "
            "data. Note that documents carrying embedded binary — base64 images in "
            "Markdown, for instance — produce chunks of noise that retrieval will "
            "happily return."
        )

# ──────────────────────────── A — one query ─────────────────────────────────
with tab_query:
    st.subheader("One query, end to end")
    st.caption(
        "The probe runs, twelve measurements come back, and two decisions follow. "
        "What actually happened stays hidden until you ask for it."
    )

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        operator = st.selectbox("Degradation operator", list(OPERATOR_LABELS),
                                format_func=lambda o: OPERATOR_LABELS[o])
    with c2:
        rate = st.select_slider("Severity r", options=sorted(sub_all.rate.unique()), value=0.0)
    with c3:
        s_single = st.select_slider(
            "Escalation budget s (share of queries allowed to escalate)",
            options=meta["s_grid"], value=0.5,
        )

    pool = sub_all[(sub_all.operator == operator) & (sub_all.rate == rate)].reset_index(drop=True)
    if pool.empty:
        st.warning("No rows for that condition.")
        st.stop()

    if "qidx" not in st.session_state:
        st.session_state.qidx = 0
    b1, b2, _ = st.columns([2, 2, 5])
    if b1.button("Random query", use_container_width=True):
        st.session_state.qidx = int(np.random.randint(len(pool)))
    if b2.button("Next query", use_container_width=True):
        st.session_state.qidx = (st.session_state.qidx + 1) % len(pool)
    row = pool.iloc[min(st.session_state.qidx, len(pool) - 1)]

    st.markdown(f"**Question** &nbsp; `{row.query_id}` &nbsp;·&nbsp; type: {row.query_type}")
    st.info(row.question if row.question else "(no question text available)")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**The twelve probe measurements**")
        st.caption("All normalised to [0,1]. This is everything the decisions below may use.")
        st.bar_chart(
            pd.DataFrame(
                {"value": [float(row[f"f_{n}"]) for n in feature_names]},
                index=[n.replace("_", " ") for n in feature_names],
            ),
            horizontal=True, height=330,
        )

    with right:
        theta = ds_meta["thresholds_by_fold"][str(int(row.fold))][str(s_single)]
        st.markdown("**Decision 1 — did retrieval find relevant material?**")
        conf = 1.0 - float(row.p_fail)
        st.progress(conf, text=f"Confidence relevant material was retrieved: {conf:.0%}")

        st.markdown("**Decision 2 — escalate to retrieve-all?**")
        escalate = float(row.p_escalate) >= theta
        st.progress(
            min(float(row.p_escalate) / max(theta, 1e-9), 1.0),
            text=f"P(another paradigm beats naive) = {row.p_escalate:.2f} "
                 f"vs threshold {theta:.2f} at s = {s_single}",
        )
        st.markdown(f"### {'Escalate' if escalate else 'Keep naive result'}")
        st.caption(
            "The threshold is the (1−s) quantile of out-of-fold probabilities inside "
            "this query's training folds, so it never sees held-out data."
        )

    with st.expander("Reveal what actually happened"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("naive nDCG@5", f"{row.ndcg_naive:.4f}")
        m2.metric("graph nDCG@5", f"{row.ndcg_graph:.4f}")
        m3.metric("hybrid nDCG@5", f"{row.ndcg_hybrid:.4f}")
        m4.metric("best available", f"{row.ndcg_oracle:.4f}", help=f"oracle paradigm: {row.oracle_label}")

        truth_failed = bool(row.all_failed)
        st.write(
            f"**Decision 1 was {'right' if (conf < 0.5) == truth_failed else 'wrong'}.** "
            + ("All three paradigms scored zero — nothing relevant was retrieved by any of them."
               if truth_failed else
               "At least one paradigm retrieved relevant material.")
        )
        headroom = float(row.ndcg_oracle - row.ndcg_naive)
        if bool(row.routing_matters):
            verdict = ("escalating was the right call" if escalate
                       else "escalating would have helped, and was skipped")
            st.write(f"**Decision 2:** another paradigm beat naive by {headroom:.4f} nDCG@5 — {verdict}.")
        else:
            verdict = ("escalating spent retrieval budget for nothing" if escalate
                       else "keeping the naive result was the right call")
            st.write(f"**Decision 2:** no paradigm beat naive here — {verdict}.")

# ─────────────────────────── B — the detector ───────────────────────────────
with tab_detector:
    st.subheader("Can the probe tell whether retrieval found anything relevant?")
    st.caption(
        "Target: all three paradigms score nDCG@5 = 0, so no choice among them "
        "could have helped. The detector sees the twelve probe measurements only."
    )

    rec = detector[corpus]["all_failed"]
    pooled, y, p = rec["pooled"], sub_all.all_failed.values, sub_all.p_fail.values
    auc = roc_auc_score(y, p)

    k1, k2, k3 = st.columns(3)
    k1.metric("AUC from the probe", f"{auc:.3f}",
              help="Reproduced out-of-fold from the demo table; matches the recorded value.")
    k2.metric("vs query type alone", f"{pooled['D-Q']['auc']:.3f}",
              delta=f"{auc - pooled['D-Q']['auc']:+.3f}")
    k3.metric("vs static corpus features", f"{pooled['D-S']['auc']:.3f}",
              delta=f"{auc - pooled['D-S']['auc']:+.3f}")
    st.caption(
        f"Prevalence on this corpus: {rec['prevalence']:.1%} of query-condition pairs "
        "are ones where every paradigm failed."
    )

    g1, g2 = st.columns(2)
    with g1:
        fpr, tpr, _ = roc_curve(y, p)
        st.markdown("**ROC**")
        st.line_chart(
            pd.DataFrame({"false positive rate": fpr, "detector": tpr, "chance": fpr}),
            x="false positive rate", y=["detector", "chance"], height=300,
        )
    with g2:
        prec, recl, _ = precision_recall_curve(y, p)
        st.markdown("**Precision against recall**")
        st.caption(f"Flat line is the base rate, {rec['prevalence']:.1%}.")
        st.line_chart(
            pd.DataFrame({"recall": recl[:-1], "precision": prec[:-1],
                          "base rate": np.full(len(recl) - 1, rec["prevalence"])}),
            x="recall", y=["precision", "base rate"], height=300,
        )

    st.markdown("**Ranking well is not the same as being usable**")
    st.caption(
        "A detector is only actionable if it can flag failures at a precision worth "
        "acting on. The pre-registered standard asked for recall of at least 0.30 at "
        "precision 0.80."
    )
    usable = pd.DataFrame([
        {
            "Corpus": CORPORA[c],
            "AUC": detector[c]["all_failed"]["pooled"]["D-PQ"]["auc"],
            "Recall at precision 0.80": detector[c]["all_failed"]["pooled"]["D-PQ"]["recall_at_precision0.80"],
            "Pre-registered gate": "pass" if detector["success_gate"][c]["pass"] else "fail",
        }
        for c in CORPORA
    ])
    st.dataframe(usable, hide_index=True, use_container_width=True)
    st.caption(
        "The probe ranks failures above successes on every corpus and beats both "
        "controls on every fold. It reaches usable precision on Musique only. That "
        "is the finding, not a presentation of it."
    )

# ───────────────────────────── C — the gate ─────────────────────────────────
with tab_gate:
    st.subheader("Should this query pay for retrieve-all?")
    st.caption(
        "The retrieve-all selector runs all three paradigms and then chooses. It "
        "improves retrieval on some corpora but costs the most. The gate uses the "
        "probe to spend that cost only where it is likely to pay."
    )

    rec = gate_rec[corpus]["pooled"]
    by_s = rec["by_s"]
    s_vals = sorted(by_s, key=float)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("always naive", f"{rec['CR_naive']:.4f}")
    k2.metric(f"best fixed ({rec['best_fixed']})", f"{rec['CR_fixed'][rec['best_fixed']]:.4f}")
    k3.metric("retrieve-all selector", f"{rec['CR_selector']:.4f}",
              delta=f"{rec['CR_selector'] - rec['CR_naive']:+.4f} vs naive")
    k4.metric("perfect gate", f"{rec['oracle']['CR']:.4f}",
              delta=f"{rec['oracle']['gain_over_best_fixed']:+.4f} vs best fixed")
    st.caption(
        f"The perfect gate escalates only the {rec['oracle']['e']:.0%} of queries where "
        f"escalating actually helps, at {rec['oracle']['cost_over_t_selector']:.0%} of the "
        "selector's cost. It is the ceiling for any gate, not something the probe reaches."
    )

    retained = pd.DataFrame({"escalation share s": [float(s) for s in s_vals]})
    for g in GATES:
        retained[GATES[g]] = [by_s[s][g]["retained_gain"] for s in s_vals]
    st.markdown("**Share of the selector's gain retained, against the share of queries escalated**")
    st.line_chart(retained, x="escalation share s", y=list(GATES.values()), height=320)
    st.caption(
        "The pseudo-random gate escalates the same number of queries as the probe "
        "gate, so any separation between them is the probe's contribution and not "
        "an effect of escalating more often."
    )

    s_pick = st.select_slider("Inspect an escalation share", options=[float(s) for s in s_vals], value=0.5)
    e = by_s[str(s_pick) if str(s_pick) in by_s else f"{s_pick:g}"]
    table = pd.DataFrame([
        {
            "Gate": GATES[g],
            "Queries escalated": f"{e[g]['e']:.1%}",
            "CR": round(e[g].get("CR", e[g].get("CR_mean", float("nan"))), 4),
            "Gain over best fixed": round(e[g]["gain_over_best_fixed"], 4),
            "Retained gain": f"{e[g]['retained_gain']:.1%}",
            "Cost (ms)": e.get("cost_ms", {}).get(g),
        }
        for g in GATES
    ])
    st.dataframe(table, hide_index=True, use_container_width=True)

    if corpus == "ultraDomain_legal" and "legal_success_gate_s0.5" in gate_rec:
        st.markdown("**Pre-registered standard for this corpus, at s = 0.5**")
        crit = gate_rec["legal_success_gate_s0.5"]
        rowsc = [
            {"Criterion": k.replace("_", " "),
             "Result": "pass" if v.get("pass") else "fail"}
            for k, v in crit.items() if isinstance(v, dict) and "pass" in v
        ]
        st.dataframe(pd.DataFrame(rowsc), hide_index=True, use_container_width=True)
        st.caption(
            "Two of the criteria are met and one is not: the gated selector clears "
            "hybrid on three of five folds where four were required. Reported as a "
            "fail, because that is what was pre-registered."
        )

# ───────────────────────────── D — is it worth it ───────────────────────────
with tab_omega:
    st.subheader("Is this worth building for a given corpus?")
    st.caption(
        "The routing opportunity score, Ω, is computed before any router is trained. "
        "It estimates how much a per-query decision has to gain on a corpus."
    )
    comp = pd.DataFrame([
        {
            "Corpus": CORPORA[c],
            "Naive variance ṽ": omega[c]["norm_variance"],
            "Oracle entropy h̃": omega[c]["norm_entropy"],
            "Quality sensitivity q̃": omega[c]["norm_quality_sensitivity"],
            "Discriminability s̃": omega[c]["norm_discriminability"],
            "Ω": omega[c]["routing_opportunity"],
        }
        for c in CORPORA
    ]).sort_values("Ω", ascending=False)
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.bar_chart(comp.set_index("Corpus")["Ω"], height=260)
    st.warning(
        "Ω is a proposal, not a validated instrument. It is characterised on three "
        "corpora, which is far too few to establish it as a general predictor. It "
        "ranks these three in the order of both their oracle gaps and the selector's "
        "realised gain, and that is the whole of the evidence for it."
    )
    st.caption(
        "Computing Ω still requires one evaluation pass of all three paradigms over "
        "a labelled sample, so it avoids the cost of training a router, not the cost "
        "of evaluating the paradigms."
    )
