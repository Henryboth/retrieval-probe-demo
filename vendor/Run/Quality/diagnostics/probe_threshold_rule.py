"""
Leak-free diagnostic (2026-09-17): can a ONE-FEATURE threshold rule on the naive probe's own
outcome realise routing gain on UNSEEN queries?

Rule: within each (operator, rate) condition, escalate the x% of queries with the LOWEST probe
feature value to a fixed target paradigm (hybrid or graph); route the rest to naive.
x (the escalation fraction) and the target are chosen on TRAINING queries only, per fold,
GroupKFold by query id (5 folds, seed 42). CR is reported on the held-out queries.
Also reported: an optimistic bound where x is chosen on the test queries (NOT a valid result,
shown only as the ceiling of this rule family), and the gain by severity band.
Pre-specified feature list; every feature is reported, none dropped.
"""
import json, os, glob, sys
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold

DATASETS = ["quality", "musique", "ultraDomain_legal"]
FEATURES = ["top1_similarity", "score_mean", "semantic_coverage", "chunk_completeness",
            "score_gap", "score_concentration", "semantic_variance", "entity_coverage"]
FRACTIONS = np.round(np.arange(0.0, 1.01, 0.02), 2)   # 0 = never escalate, 1 = always escalate
TARGETS = ["hybrid", "graph"]

def load(ds):
    rows = []
    for f in sorted(glob.glob(f"Dataset/EvaluationData/RetrievalEval/{ds}/*_[0-9][0-9][0-9].json")):
        if "graph_probe" in os.path.basename(f): continue   # GP feature files share the directory (2026-09-22)
        d = json.load(open(f)); cond = (d["degradation_type"], float(d["rate"]))
        per = {p: {q["id"]: q for q in d["paradigms"][p]["per_query"]} for p in ["naive", "graph", "hybrid"]}
        for qid, q in per["naive"].items():
            if qid not in per["graph"] or qid not in per["hybrid"]: continue
            rows.append(dict(qid=qid, cond=cond, rate=cond[1], feats=q["pq_features"],
                             n=q["metrics"]["ndcg@5"], g=per["graph"][qid]["metrics"]["ndcg@5"],
                             h=per["hybrid"][qid]["metrics"]["ndcg@5"]))
    return rows

def cr_of_rule(rows, feat, frac, target, thresholds=None):
    """Apply per-condition quantile threshold. thresholds: dict cond -> value (from training)."""
    scores = []
    for r in rows:
        thr = thresholds[r["cond"]] if thresholds is not None else None
        v = r["feats"].get(feat, 0.0)
        esc = (thr is not None) and (v <= thr)
        scores.append(r[target[0]] if esc else r["n"])
    return float(np.mean(scores))

def fit_thresholds(train_rows, feat, frac):
    by_cond = defaultdict(list)
    for r in train_rows: by_cond[r["cond"]].append(r["feats"].get(feat, 0.0))
    if frac == 0.0: return {c: -np.inf for c in by_cond}
    return {c: float(np.quantile(v, frac)) for c, v in by_cond.items()}

def run(ds):
    rows = load(ds); qids = [r["qid"] for r in rows]
    gkf = GroupKFold(n_splits=5)
    idx = np.arange(len(rows))
    out = {"n_rows": len(rows), "always_naive": float(np.mean([r["n"] for r in rows])),
           "always_hybrid": float(np.mean([r["h"] for r in rows])),
           "oracle": float(np.mean([max(r["n"], r["g"], r["h"]) for r in rows])), "features": {}}
    for feat in FEATURES:
        test_scores = []; chosen = []; ceiling_scores = []; static_scores = []
        band = defaultdict(list)  # severity band -> (rule, naive) pairs on test rows
        for tr, te in gkf.split(idx, groups=qids):
            trr = [rows[i] for i in tr]; ter = [rows[i] for i in te]
            # choose (frac, target) on TRAINING rows only
            best = (-1, None, None)
            for target in TARGETS:
                for frac in FRACTIONS:
                    thr = fit_thresholds(trr, feat, frac)
                    c = cr_of_rule(trr, feat, frac, target, thr)
                    if c > best[0]: best = (c, frac, target)
            _, frac, target = best
            thr = fit_thresholds(trr, feat, frac)          # thresholds from training rows
            # per-condition best fixed paradigm, chosen on TRAINING rows (what a static router can reach)
            bc = defaultdict(lambda: defaultdict(list))
            for r in trr:
                for p in "ngh": bc[r["cond"]][p].append(r[p])
            best_fixed = {c: max("ngh", key=lambda p: np.mean(v[p])) for c, v in bc.items()}
            for r in ter: static_scores.append(r[best_fixed[r["cond"]]])
            for r in ter:
                v = r["feats"].get(feat, 0.0); esc = v <= thr[r["cond"]]
                s = r[target[0]] if esc else r["n"]
                test_scores.append(s)
                b = "r=0" if r["rate"] == 0 else ("mild r<=0.3" if r["rate"] <= 0.3 else "severe r>=0.4")
                band[b].append((s, r["n"]))
            chosen.append((float(frac), target))
            # optimistic ceiling: choose frac/target on the TEST rows (invalid, for reference only)
            cb = max(cr_of_rule(ter, feat, f, t, fit_thresholds(ter, feat, f)) for t in TARGETS for f in FRACTIONS)
            ceiling_scores.append(cb)
        out["features"][feat] = {
            "test_cr": float(np.mean(test_scores)),
            "static_best_fixed_cr": float(np.mean(static_scores)),
            "gain_vs_static": float(np.mean(test_scores) - np.mean(static_scores)),
            "gain_vs_naive": float(np.mean(test_scores) - out["always_naive"]),
            "chosen_per_fold": chosen,
            "ceiling_cr_invalid": float(np.mean(ceiling_scores)),
            "gain_by_band": {b: float(np.mean([s for s, _ in v]) - np.mean([n for _, n in v])) for b, v in band.items()},
        }
    return out

if __name__ == "__main__":
    res = {}
    for ds in DATASETS:
        res[ds] = run(ds)
        o = res[ds]
        print(f"\n== {ds}  rows={o['n_rows']}  always-naive={o['always_naive']:.4f}  always-hybrid={o['always_hybrid']:.4f}  oracle={o['oracle']:.4f}")
        print(f"{'feature':20s} {'test CR':>8s} {'vs naive':>8s} {'vs static':>9s} {'ceiling*':>9s}  {'r=0':>7s} {'mild':>7s} {'severe':>7s}  chosen (frac,target) per fold")
        for feat, v in o["features"].items():
            gb = v["gain_by_band"]
            print(f"{feat:20s} {v['test_cr']:8.4f} {v['gain_vs_naive']:+8.4f} {v['gain_vs_static']:+9.4f} {v['ceiling_cr_invalid']-o['always_naive']:+9.4f}  "
                  f"{gb.get('r=0',0):+7.4f} {gb.get('mild r<=0.3',0):+7.4f} {gb.get('severe r>=0.4',0):+7.4f}  {v['chosen_per_fold']}")
    json.dump(res, open("Dataset/EvaluationData/RouterEval/probe_threshold_rule.json", "w"), indent=1)  # path of record (fixed 2026-09-22; was _backups/)
    print("\n* ceiling = fraction/target chosen on the test queries themselves; NOT a valid result, shown as the upper bound of this rule family.")
