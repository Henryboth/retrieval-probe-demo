"""
Paired bootstrap over queries for the two degradation-curve movements that Ch6 reports without a
mechanism: graph nDCG@5 under graph edge removal (r = 0.7 vs r = 0.0) on every corpus, and naive
nDCG@5 under entity variation on Legal (r = 0.5 and r = 0.7 vs r = 0.0). The question answered is
narrow: is each movement inside query-sampling noise? Per-query differences are resampled with
replacement (2 000 draws, seed 42); the mean difference, its standard error and the percentile 95 %
interval are written to RetrievalEval/degradation_rise_bootstrap.json.
"""
import json, os, numpy as np
EV = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Dataset", "EvaluationData", "RetrievalEval")
CASES = [("quality", "graph_edge_removal", "graph", 0.7), ("ultraDomain_legal", "graph_edge_removal", "graph", 0.7),
         ("musique", "graph_edge_removal", "graph", 0.7), ("ultraDomain_legal", "entity_variation", "naive", 0.5),
         ("ultraDomain_legal", "entity_variation", "naive", 0.7)]
def load(ds, op, rate):
    return json.load(open(os.path.join(EV, ds, f"{op}_{int(round(rate*100)):03d}.json")))
out = {}
for ds, op, par, r1 in CASES:
    a, b = load(ds, op, 0.0), load(ds, op, r1)
    pa = {q["id"]: q["metrics"]["ndcg@5"] for q in a["paradigms"][par]["per_query"]}
    pb = {q["id"]: q["metrics"]["ndcg@5"] for q in b["paradigms"][par]["per_query"]}
    ids = sorted(set(pa) & set(pb)); d = np.array([pb[i] - pa[i] for i in ids])
    rng = np.random.default_rng(42); boots = np.array([rng.choice(d, len(d)).mean() for _ in range(2000)])
    out[f"{ds}/{op}/{par}/r={r1}"] = dict(n=len(d), mean_r0=float(np.mean([pa[i] for i in ids])), mean_r1=float(np.mean([pb[i] for i in ids])),
        diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        share_up=float(np.mean(d > 0)), share_down=float(np.mean(d < 0)), resamples=2000, seed=42)
    print(f"{ds} {op} {par} r={r1}: diff={d.mean():+.4f} CI=[{out[list(out)[-1]]['ci95'][0]:+.4f},{out[list(out)[-1]]['ci95'][1]:+.4f}]")
json.dump(out, open(os.path.join(EV, "degradation_rise_bootstrap.json"), "w"), indent=1)
