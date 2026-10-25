"""
Confidence interval on the study's headline gain (2026-09-23).

Both blind reviewers asked for one: the RA selector's gain over the best fixed paradigm is
reported as +0.0045 / -0.0025 / +0.0084 with only a 5-of-5-fold sign count behind it.

Method, fixed before the result was read:
 - Reproduce the RA selector's OUT-OF-FOLD predictions exactly as run_router_training does:
   StratifiedGroupKFold(5, shuffle, seed 42) grouped by query id, expected-value rule,
   margin tau chosen on training rows only.
 - For every instance, pair the selected paradigm's nDCG@5 against the corpus's best fixed
   paradigm's nDCG@5 on the SAME instance.
 - CLUSTER bootstrap by query (not by instance): a query's 40 degradation-condition instances are
   correlated, so queries are the independent unit. 2000 resamples, seed 42.
 - Report the mean paired difference and the percentile 95 % interval.

Reads the data of record read-only. Writes only RouterEval/headline_gain_ci.json.
"""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.environ.setdefault("ROUTER_GROUPED", "1")
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from Run.Quality.run_router_training import (
    load_retrieval_results, build_dataset, build_X, make_splitter, split_iter,
    fit_expected_value, PARADIGMS,
)

DATASETS = ["quality", "musique", "ultraDomain_legal"]
CONFIG = "Q+C+Quality+RA"
N_BOOT, SEED = 2000, 42
OUT = "Dataset/EvaluationData/RouterEval/headline_gain_ci.json"


def best_fixed(rows):
    means = {p: float(np.mean([r["paradigm_scores"].get(p, {}).get("ndcg@5", 0.0) for r in rows]))
             for p in PARADIGMS}
    name = max(means, key=means.get)
    return name, means


def main():
    out = {}
    for ds in DATASETS:
        rows = build_dataset(load_retrieval_results(ds), ds)
        rows = rows if isinstance(rows, list) else rows.get("rows", rows)
        bf_name, bf_means = best_fixed(rows)
        X = build_X(rows, CONFIG)
        y = LabelEncoder().fit_transform([r["label"] for r in rows])
        sel = np.zeros(len(rows)); fix = np.zeros(len(rows)); covered = np.zeros(len(rows), bool)
        for tr, te in split_iter(make_splitter(), X, y, rows):
            preds, tau, _ = fit_expected_value(X[tr], [rows[i] for i in tr], X[te])
            for k, i in enumerate(te):
                sel[i] = rows[i]["paradigm_scores"].get(preds[k], {}).get("ndcg@5", 0.0)
                fix[i] = rows[i]["paradigm_scores"].get(bf_name, {}).get("ndcg@5", 0.0)
                covered[i] = True
        assert covered.all(), "some instances were never held out"
        diff = sel - fix
        by_q = defaultdict(list)
        for i, r in enumerate(rows):
            by_q[r["query_id"]].append(diff[i])
        qids = list(by_q); qmeans = np.array([np.mean(by_q[q]) for q in qids])
        qn = np.array([len(by_q[q]) for q in qids], float)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(qids), len(qids))
            boots.append(float(np.average(qmeans[idx], weights=qn[idx])))
        boots = np.array(boots)
        out[ds] = {
            "best_fixed": bf_name,
            "n_instances": int(len(rows)), "n_queries": int(len(qids)),
            "selector_cr_pooled": float(sel.mean()), "best_fixed_cr_pooled": float(fix.mean()),
            "mean_paired_diff": float(diff.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "excludes_zero": bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0),
            "resamples": N_BOOT, "seed": SEED, "cluster": "query_id",
        }
        o = out[ds]
        print(f"{ds:20s} bf={bf_name:6s} diff={o['mean_paired_diff']:+.4f} "
              f"CI=[{o['ci95'][0]:+.4f},{o['ci95'][1]:+.4f}] excl0={o['excludes_zero']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
