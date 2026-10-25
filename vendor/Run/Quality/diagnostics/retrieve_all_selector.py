"""
Leak-free diagnostic (2026-09-17): a RETRIEVE-ALL SELECTOR. Run all three paradigms, then select per
query from the result-set features of all three (3 x 12 = 36) plus pairwise differences of the
semantic-fit features. GroupKFold by query id (5 folds). Baselines: always-naive, always-hybrid,
per-condition best fixed (chosen on training rows), and the oracle.
Also two zero-training rules: pick the paradigm whose own result set has the highest
semantic_coverage / top1_similarity.
"""
import json, glob
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

DATASETS = ["quality", "musique", "ultraDomain_legal"]
P = ["naive", "graph", "hybrid"]
FN = ['chunk_completeness','n_retrieved_ratio','score_mean','semantic_coverage','semantic_variance',
      'top1_similarity','entity_coverage','entity_density','score_gap','score_concentration',
      'query_length_norm','query_entity_count']

def load(ds):
    rows = []
    for f in sorted(glob.glob(f"Dataset/EvaluationData/RetrievalEval/{ds}/*_[0-9][0-9][0-9].json")):
        if "graph_probe" in os.path.basename(f): continue   # GP feature files share the directory (2026-09-22)
        d = json.load(open(f)); cond = (d["degradation_type"], float(d["rate"]))
        per = {p: {q["id"]: q for q in d["paradigms"][p]["per_query"]} for p in P}
        for qid in per["naive"]:
            if any(qid not in per[p] for p in P): continue
            rows.append(dict(qid=qid, cond=cond, rate=cond[1],
                nd={p: per[p][qid]["metrics"]["ndcg@5"] for p in P},
                ft={p: per[p][qid]["pq_features"] for p in P}))
    return rows

import os
PROBE_ONLY = os.environ.get("PROBE_ONLY", "0") == "1"   # naive probe features only (the cheap design point)
def X_of(rows):
    X = []
    for r in rows:
        v = []
        if PROBE_ONLY:
            v += [r["ft"]["naive"].get(n, 0.0) for n in FN]; X.append(v); continue
        for p in P: v += [r["ft"][p].get(n, 0.0) for n in FN]
        for n in ["semantic_coverage", "top1_similarity", "score_mean", "entity_coverage"]:
            v += [r["ft"]["naive"][n]-r["ft"]["graph"][n], r["ft"]["naive"][n]-r["ft"]["hybrid"][n], r["ft"]["graph"][n]-r["ft"]["hybrid"][n]]
        X.append(v)
    return np.array(X, dtype=np.float32)

def cr(rows, picks): return float(np.mean([r["nd"][p] for r, p in zip(rows, picks)]))

def run(ds):
    rows = load(ds); X = X_of(rows); qids = [r["qid"] for r in rows]
    y = np.array([max(P, key=lambda p: (r["nd"][p], -P.index(p))) for r in rows])   # tie -> cheapest
    yi = np.array([P.index(l) for l in y])
    out = {"n": len(rows), "always": {p: cr(rows, [p]*len(rows)) for p in P},
           "oracle": float(np.mean([max(r["nd"].values()) for r in rows]))}
    # zero-training rules
    for feat in ["semantic_coverage", "top1_similarity"]:
        picks = [max(P, key=lambda p: (r["ft"][p][feat], -P.index(p))) for r in rows]
        out[f"rule_max_{feat}"] = cr(rows, picks)
    gkf = GroupKFold(n_splits=5); idx = np.arange(len(rows))
    clf_picks = [None]*len(rows); reg_picks = [None]*len(rows); static_picks = [None]*len(rows); reg_thr_picks=[None]*len(rows)
    for tr, te in gkf.split(idx, groups=qids):
        trr = [rows[i] for i in tr]
        # per-condition best fixed from training rows
        bc = defaultdict(lambda: defaultdict(list))
        for r in trr:
            for p in P: bc[r["cond"]][p].append(r["nd"][p])
        bf = {c: max(P, key=lambda p: np.mean(v[p])) for c, v in bc.items()}
        for i in te: static_picks[i] = bf[rows[i]["cond"]]
        # classifier
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="mlogloss", random_state=42, verbosity=0)
        clf.fit(X[tr], yi[tr]); pr = clf.predict(X[te])
        for i, p in zip(te, pr): clf_picks[i] = P[p]
        # expected-value: regress each paradigm's nDCG, pick argmax; margin threshold chosen on training rows
        preds_tr = np.zeros((len(tr), 3)); preds_te = np.zeros((len(te), 3))
        for j, p in enumerate(P):
            reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
            reg.fit(X[tr], np.array([rows[i]["nd"][p] for i in tr])); preds_tr[:, j] = reg.predict(X[tr]); preds_te[:, j] = reg.predict(X[te])
        for i, pv in zip(te, preds_te): reg_picks[i] = P[int(np.argmax(pv))]
        # margin rule: escalate away from the training-best-fixed only if predicted margin > t (t chosen on train)
        best_t, best_c = 0.0, -1
        for t in np.arange(0.0, 0.31, 0.01):
            picks = []
            for k, i in enumerate(tr):
                base = P.index(bf[rows[i]["cond"]]); j = int(np.argmax(preds_tr[k]))
                picks.append(P[j] if preds_tr[k][j] - preds_tr[k][base] > t else P[base])
            c = cr(trr, picks)
            if c > best_c: best_c, best_t = c, t
        for k, i in enumerate(te):
            base = P.index(bf[rows[i]["cond"]]); j = int(np.argmax(preds_te[k]))
            reg_thr_picks[i] = P[j] if preds_te[k][j] - preds_te[k][base] > best_t else P[base]
    out["static_best_fixed"] = cr(rows, static_picks)
    out["selector_classifier"] = cr(rows, clf_picks)
    out["selector_expected_value"] = cr(rows, reg_picks)
    out["selector_expected_value_margin"] = cr(rows, reg_thr_picks)
    out["selector_pick_shares"] = {p: float(np.mean([x == p for x in reg_thr_picks])) for p in P}
    return out

if __name__ == "__main__":
    res = {}
    for ds in DATASETS:
        o = run(ds); res[ds] = o
        print(f"\n== {ds} n={o['n']}  always: " + "  ".join(f"{p}={o['always'][p]:.4f}" for p in P) + f"  oracle={o['oracle']:.4f}")
        sb = o["static_best_fixed"]
        print(f"  static best-fixed (per condition, train-chosen) : {sb:.4f}")
        for k in ["rule_max_semantic_coverage", "rule_max_top1_similarity", "selector_classifier", "selector_expected_value", "selector_expected_value_margin"]:
            print(f"  {k:36s}: {o[k]:.4f}  ({o[k]-sb:+.4f} vs static, {o[k]-o['always']['naive']:+.4f} vs naive)")
        print("  margin-selector pick shares:", {p: round(v, 3) for p, v in o["selector_pick_shares"].items()})
    json.dump(res, open(f"Dataset/EvaluationData/_backups/diagnostic_{'probe_first' if PROBE_ONLY else 'retrieve_all'}_selector.json", "w"), indent=1)
