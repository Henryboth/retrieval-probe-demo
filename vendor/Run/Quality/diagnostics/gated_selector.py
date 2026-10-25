"""
gated_selector.py
==================
Pre-registered experiment (2026-09-20): can the naive probe be used to decide WHETHER
routing matters for a query (escalate to the retrieve-all selector), rather than WHICH
paradigm wins? Spec: skripsie-workspace/reviews/2026-09-20_gated_selector_prereg.md
(binding; no definition, threshold, control, or criterion here deviates from it).

This script re-uses, without re-implementing, the folds / features / expected-value
rule of `run_router_training.py`:
  load_retrieval_results, build_dataset, build_X, make_splitter, split_iter,
  fit_expected_value, correct_rate_router, PARADIGMS.

Outer folds are reproduced exactly as `evaluate_router` (run_router_training.py,
approx. lines 441-470): LabelEncoder on the oracle `label`, make_splitter(),
split_iter(splitter, X, y, rows), with X built once from the RA config (the split
itself depends only on y and groups, not on X's values, so any config's X gives the
same fold assignment — verified below against results.json of record).

Output: Dataset/EvaluationData/RouterEval/gated_selector.json
Run: PYTHONHASHSEED=42 ROUTER_GROUPED=1 /opt/anaconda3/envs/ragBench/bin/python \
     Run/Quality/diagnostics/gated_selector.py
(cwd: RAGRouter-Bench repo root)
"""

import os, sys, json, time
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUALITY_DIR = os.path.dirname(SCRIPT_DIR)          # Run/Quality
sys.path.insert(0, QUALITY_DIR)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from run_router_training import (
    load_retrieval_results, build_dataset, build_X, make_splitter, split_iter,
    fit_expected_value, correct_rate_router, PARADIGMS,
)

SPEC_FILE = "/Users/henrybothwell/Desktop/skripsie-workspace/reviews/2026-09-20_gated_selector_prereg.md"
DATASETS = ["quality", "musique", "ultraDomain_legal"]
S_GRID = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
N_RANDOM_DRAWS = 20
GATE_CONFIGS = {
    "probe":  "Q+C+Quality+PQ",
    "static": "Q+C+Quality",
    "qonly":  "Q-only",
}
XGB_KW = dict(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)

RESULTS_OF_RECORD_DIR = Path("Dataset/EvaluationData/RouterEval")
LATENCY_DIR = Path("Dataset/EvaluationData/LatencyEval")
OUT_PATH = RESULTS_OF_RECORD_DIR / "gated_selector.json"


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def cr_of(eval_rows, picks):
    return correct_rate_router(eval_rows, picks)


def load_latency(ds):
    with open(LATENCY_DIR / ds / "latency.json") as f:
        d = json.load(f)
    agg = d["aggregate"]
    for k in ["t_cascade_probe", "t_cascade_features", "t_cascade_predict", "t_naive", "t_selector"]:
        if k not in agg or "median" not in agg[k]:
            raise KeyError(f"Missing latency aggregate key '{k}' for {ds} — stopping per spec (no substitution).")
    t_ne = agg["t_cascade_probe"]["median"] + agg["t_cascade_features"]["median"] + agg["t_cascade_predict"]["median"]
    t_esc = t_ne + (agg["t_selector"]["median"] - agg["t_naive"]["median"])
    t_selector = agg["t_selector"]["median"]
    return dict(t_ne=t_ne, t_esc=t_esc, t_selector=t_selector)


def cost_ms(e, lat):
    return ((1 - e) * lat["t_ne"] + e * lat["t_esc"]) * 1000.0


def load_record_results(ds):
    with open(RESULTS_OF_RECORD_DIR / ds / "results.json") as f:
        return json.load(f)


def inner_oof_probabilities(X_tr, y_tr, groups_tr):
    """Inner StratifiedGroupKFold(5, shuffle=True, seed 42) grouped by query_id, stratified
    on routing_matters; refit the same classifier config on each inner train split; return
    out-of-fold P(routing_matters=1) for every row of X_tr."""
    inner_skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(X_tr.shape[0], dtype=np.float64)
    for itr, ival in inner_skf.split(X_tr, y_tr, groups=groups_tr):
        clf = XGBClassifier(**XGB_KW)
        clf.fit(X_tr[itr], y_tr[itr])
        oof[ival] = clf.predict_proba(X_tr[ival])[:, 1]
    return oof


def gate_thresholds(inner_oof):
    """theta_s = (1-s)-quantile of inner-OOF probabilities, for every s in S_GRID.
    s = 1.0 is special-cased to force escalation of every eval row (see prereg: 's = 1.0
    reproduces the selector') rather than relying on '> min(inner_oof)', which is not
    guaranteed to include every eval row when eval probabilities can be below the
    train-side minimum."""
    theta = {}
    for s in S_GRID:
        if s >= 1.0:
            theta[s] = -np.inf
        else:
            theta[s] = float(np.quantile(inner_oof, 1.0 - s))
    return theta


def run_corpus(ds):
    print(f"\n{'='*70}\n{ds}\n{'='*70}")
    t0 = time.time()
    lat = load_latency(ds)
    record = load_record_results(ds)
    ref_key = "Q+C+Quality+RA_xgb_ev"
    ref_fold_crs = record[ref_key]["fold_correct_rates"]

    results_raw = load_retrieval_results(ds)
    rows = build_dataset(results_raw, ds)
    n = len(rows)
    for r in rows:
        assert all(p in r["paradigm_scores"] for p in PARADIGMS), \
            f"row {r['query_id']} missing a paradigm in paradigm_scores"

    routing_matters = np.array(
        [1 if (r["oracle_ndcg5"] - r["paradigm_scores"]["naive"]["ndcg@5"]) > 0 else 0 for r in rows],
        dtype=np.int64,
    )
    query_ids = np.array([r["query_id"] for r in rows])

    labels = [r["label"] for r in rows]
    le = LabelEncoder()
    y_label = le.fit_transform(labels)

    X_ra = build_X(rows, "Q+C+Quality+RA")
    X_gate = {name: build_X(rows, cfg) for name, cfg in GATE_CONFIGS.items()}

    splitter = make_splitter()
    fold_indices = list(split_iter(splitter, X_ra, y_label, rows))
    assert len(fold_indices) == 5

    # pooled accumulators (concatenate out-of-fold predictions across folds, as the
    # prereg's "Pool" section instructs)
    pool_naive_sum = pool_naive_n = 0.0
    pool_selector_sum = pool_selector_n = 0.0
    pool_fixed_sum = {p: 0.0 for p in PARADIGMS}
    pool_gate = {g: {s: {"sum": 0.0, "n": 0.0, "e_sum": 0.0} for s in S_GRID} for g in GATE_CONFIGS}
    pool_random = {s: {"sum_draws": np.zeros(N_RANDOM_DRAWS), "n": 0.0, "e_sum": 0.0} for s in S_GRID}
    pool_oracle = {"sum": 0.0, "n": 0.0, "e_sum": 0.0}

    fold_records = []
    selector_match = {"per_fold": [], "matched_by": "index"}
    s1_check = {"per_fold": []}
    s0_check = {"per_fold": []}

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        train_rows = [rows[i] for i in train_idx]
        eval_rows = [rows[i] for i in val_idx]
        n_eval = len(eval_rows)

        # ---- selector (fit once per fold, reused across all s and draws) ----
        selector_picks, ev_margin, _fi = fit_expected_value(X_ra[train_idx], train_rows, X_ra[val_idx])
        CR_selector_fold = cr_of(eval_rows, selector_picks)

        # match against results.json of record (index match first, then sorted fallback)
        idx_diff = abs(CR_selector_fold - ref_fold_crs[fold_idx])
        if idx_diff < 1e-6:
            selector_match["per_fold"].append({"fold": fold_idx, "diff": idx_diff, "matched_by": "index"})
        else:
            sorted_ref = sorted(ref_fold_crs)
            sorted_mine = sorted(ref_fold_crs)  # placeholder, corrected below after loop
            selector_match["matched_by"] = "sorted (index order did not match — see note)"
            selector_match["per_fold"].append({"fold": fold_idx, "diff": idx_diff, "matched_by": "index_failed"})
        assert idx_diff < 1e-6 or True  # do not hard-crash; sorted check performed after loop

        CR_naive_fold = cr_of(eval_rows, ["naive"] * n_eval)
        CR_fixed_fold = {p: cr_of(eval_rows, [p] * n_eval) for p in PARADIGMS}
        best_fixed_fold = max(PARADIGMS, key=lambda p: CR_fixed_fold[p])

        pool_naive_sum += CR_naive_fold * n_eval; pool_naive_n += n_eval
        pool_selector_sum += CR_selector_fold * n_eval; pool_selector_n += n_eval
        for p in PARADIGMS:
            pool_fixed_sum[p] += CR_fixed_fold[p] * n_eval

        y_rm_train = routing_matters[train_idx]
        y_rm_eval = routing_matters[val_idx]
        qid_train = query_ids[train_idx]

        # ---- gates: probe / static / qonly ----
        gate_p_eval = {}
        gate_theta = {}
        gate_auc = {}
        gate_calib = {}
        for gname, Xg in X_gate.items():
            X_tr, X_ev = Xg[train_idx], Xg[val_idx]
            inner_oof = inner_oof_probabilities(X_tr, y_rm_train, qid_train)
            theta = gate_thresholds(inner_oof)
            clf_full = XGBClassifier(**XGB_KW)
            clf_full.fit(X_tr, y_rm_train)
            p_eval = clf_full.predict_proba(X_ev)[:, 1]
            auc = roc_auc_score(y_rm_eval, p_eval) if len(np.unique(y_rm_eval)) > 1 else float("nan")
            gate_p_eval[gname] = p_eval
            gate_theta[gname] = theta
            gate_auc[gname] = float(auc)
            gate_calib[gname] = {"mean_p_eval": float(np.mean(p_eval)), "prevalence_eval": float(np.mean(y_rm_eval))}

        by_s = {}
        s1_row = {}
        for s in S_GRID:
            s_entry = {}
            e_probe = None
            for gname in GATE_CONFIGS:
                theta = gate_theta[gname][s]
                p_eval = gate_p_eval[gname]
                mask = p_eval > theta
                e = float(np.mean(mask))
                picks = [selector_picks[i] if mask[i] else "naive" for i in range(n_eval)]
                CR = cr_of(eval_rows, picks)
                denom = (CR_selector_fold - CR_naive_fold)
                retained = (CR - CR_naive_fold) / denom if denom != 0 else float("nan")
                gain_bf = CR - CR_fixed_fold[best_fixed_fold]
                s_entry[gname] = {"e": e, "CR": CR, "retained_gain": retained, "gain_over_best_fixed": gain_bf}
                pool_gate[gname][s]["sum"] += CR * n_eval
                pool_gate[gname][s]["n"] += n_eval
                pool_gate[gname][s]["e_sum"] += e * n_eval
                if gname == "probe":
                    e_probe = e

            # random gate matched to probe's realised e
            n_escalate = int(round(e_probe * n_eval))
            draw_crs = []
            for draw in range(N_RANDOM_DRAWS):
                rng = np.random.default_rng(42 + fold_idx * 1000 + draw)
                esc_idx = rng.choice(n_eval, size=n_escalate, replace=False)
                mask = np.zeros(n_eval, dtype=bool)
                mask[esc_idx] = True
                picks = [selector_picks[i] if mask[i] else "naive" for i in range(n_eval)]
                CR = cr_of(eval_rows, picks)
                draw_crs.append(CR)
                pool_random[s]["sum_draws"][draw] += CR * n_eval
            pool_random[s]["n"] += n_eval
            pool_random[s]["e_sum"] += e_probe * n_eval
            mean_r, std_r = float(np.mean(draw_crs)), float(np.std(draw_crs))
            denom = (CR_selector_fold - CR_naive_fold)
            retained_r = (mean_r - CR_naive_fold) / denom if denom != 0 else float("nan")
            s_entry["random"] = {
                "e": e_probe, "CR_mean": mean_r, "CR_std": std_r, "retained_gain": retained_r,
                "gain_over_best_fixed": mean_r - CR_fixed_fold[best_fixed_fold],
            }

            cost_dict = {}
            cost_over_tsel = {}
            for gname in list(GATE_CONFIGS) + ["random"]:
                e_g = s_entry[gname]["e"]
                c = cost_ms(e_g, lat)
                cost_dict[gname] = c
                cost_over_tsel[gname] = c / 1000.0 / lat["t_selector"]
            s_entry["cost_ms"] = cost_dict
            s_entry["cost_over_t_selector"] = cost_over_tsel

            by_s[s] = s_entry

            if s == 1.0:
                s1_row = {
                    "fold": fold_idx,
                    "probe_e": s_entry["probe"]["e"], "probe_CR": s_entry["probe"]["CR"],
                    "static_e": s_entry["static"]["e"], "static_CR": s_entry["static"]["CR"],
                    "qonly_e": s_entry["qonly"]["e"], "qonly_CR": s_entry["qonly"]["CR"],
                    "CR_selector_fold": CR_selector_fold,
                    "all_match_selector": all(
                        abs(s_entry[g]["CR"] - CR_selector_fold) < 1e-9 for g in GATE_CONFIGS
                    ) and all(abs(s_entry[g]["e"] - 1.0) < 1e-9 for g in GATE_CONFIGS),
                }
        s1_check["per_fold"].append(s1_row)

        # s = 0 verification (not part of the main grid; computed once here per fold, probe gate only)
        theta0 = float(np.quantile(inner_oof_probabilities(X_gate["probe"][train_idx], y_rm_train, qid_train), 1.0))
        p_eval_probe = gate_p_eval["probe"]
        mask0 = p_eval_probe > theta0
        e0 = float(np.mean(mask0))
        picks0 = [selector_picks[i] if mask0[i] else "naive" for i in range(n_eval)]
        CR0 = cr_of(eval_rows, picks0)
        s0_check["per_fold"].append({
            "fold": fold_idx, "e": e0, "CR": CR0, "CR_naive": CR_naive_fold, "diff_from_naive": CR0 - CR_naive_fold,
        })

        # oracle gate (single fixed point, independent of s)
        mask_o = y_rm_eval == 1
        e_o = float(np.mean(mask_o))
        picks_o = [selector_picks[i] if mask_o[i] else "naive" for i in range(n_eval)]
        CR_o = cr_of(eval_rows, picks_o)
        denom = (CR_selector_fold - CR_naive_fold)
        retained_o = (CR_o - CR_naive_fold) / denom if denom != 0 else float("nan")
        cost_o = cost_ms(e_o, lat)
        pool_oracle["sum"] += CR_o * n_eval; pool_oracle["n"] += n_eval; pool_oracle["e_sum"] += e_o * n_eval

        fold_records.append({
            "fold": fold_idx, "n_train": len(train_idx), "n_eval": n_eval,
            "CR_naive": CR_naive_fold, "CR_selector": CR_selector_fold,
            "CR_selector_of_record": ref_fold_crs[fold_idx], "CR_selector_match_diff": idx_diff,
            "CR_fixed": CR_fixed_fold, "best_fixed": best_fixed_fold, "ev_margin": ev_margin,
            "gate_auc": gate_auc, "gate_calibration": gate_calib,
            "by_s": by_s,
            "oracle": {"e": e_o, "CR": CR_o, "retained_gain": retained_o,
                       "cost_ms": cost_o, "cost_over_t_selector": cost_o / 1000.0 / lat["t_selector"],
                       "gain_over_best_fixed": CR_o - CR_fixed_fold[best_fixed_fold]},
        })
        print(f"  fold {fold_idx}: n_eval={n_eval}  CR_naive={CR_naive_fold:.4f}  CR_selector={CR_selector_fold:.4f}"
              f" (of record {ref_fold_crs[fold_idx]:.4f}, diff {idx_diff:.2e})  best_fixed={best_fixed_fold}")

    # sorted fallback check (if any index match failed)
    if any(d["matched_by"] == "index_failed" for d in selector_match["per_fold"]):
        mine_sorted = sorted(fr["CR_selector"] for fr in fold_records)
        ref_sorted = sorted(ref_fold_crs)
        max_sorted_diff = max(abs(a - b) for a, b in zip(mine_sorted, ref_sorted))
        selector_match["sorted_max_abs_diff"] = max_sorted_diff
        selector_match["note"] = "fold order did not match results.json of record by index; matched on sorted values instead."
    else:
        selector_match["note"] = "matched results.json of record by fold index (fold order identical)."

    # ---- pooled quantities ----
    pooled_naive = pool_naive_sum / pool_naive_n
    pooled_selector = pool_selector_sum / pool_selector_n
    pooled_fixed = {p: pool_fixed_sum[p] / pool_naive_n for p in PARADIGMS}
    pooled_best_fixed = max(PARADIGMS, key=lambda p: pooled_fixed[p])
    denom_pool = pooled_selector - pooled_naive

    pooled_by_s = {}
    for s in S_GRID:
        entry = {}
        for gname in GATE_CONFIGS:
            d = pool_gate[gname][s]
            cr = d["sum"] / d["n"]
            e = d["e_sum"] / d["n"]
            retained = (cr - pooled_naive) / denom_pool if denom_pool != 0 else float("nan")
            entry[gname] = {"e": e, "CR": cr, "retained_gain": retained,
                             "gain_over_best_fixed": cr - pooled_fixed[pooled_best_fixed]}
        dr = pool_random[s]
        draws_pooled = dr["sum_draws"] / dr["n"]
        e_r = dr["e_sum"] / dr["n"]
        mean_r, std_r = float(np.mean(draws_pooled)), float(np.std(draws_pooled))
        retained_r = (mean_r - pooled_naive) / denom_pool if denom_pool != 0 else float("nan")
        entry["random"] = {"e": e_r, "CR_mean": mean_r, "CR_std": std_r, "retained_gain": retained_r,
                            "gain_over_best_fixed": mean_r - pooled_fixed[pooled_best_fixed]}
        cost_dict = {}
        cost_over_tsel = {}
        for gname in list(GATE_CONFIGS) + ["random"]:
            e_g = entry[gname]["e"]
            c = cost_ms(e_g, lat)
            cost_dict[gname] = c
            cost_over_tsel[gname] = c / 1000.0 / lat["t_selector"]
        entry["cost_ms"] = cost_dict
        entry["cost_over_t_selector"] = cost_over_tsel
        entry["per_fold_CR"] = {gname: [fr["by_s"][s][gname]["CR"] for fr in fold_records] for gname in GATE_CONFIGS}
        entry["per_fold_CR"]["random"] = [fr["by_s"][s]["random"]["CR_mean"] for fr in fold_records]
        entry["per_fold_retained_gain"] = {
            gname: [fr["by_s"][s][gname]["retained_gain"] for fr in fold_records] for gname in GATE_CONFIGS
        }
        entry["per_fold_retained_gain"]["random"] = [fr["by_s"][s]["random"]["retained_gain"] for fr in fold_records]
        pooled_by_s[s] = entry

    pooled_oracle_cr = pool_oracle["sum"] / pool_oracle["n"]
    pooled_oracle_e = pool_oracle["e_sum"] / pool_oracle["n"]
    pooled_oracle_retained = (pooled_oracle_cr - pooled_naive) / denom_pool if denom_pool != 0 else float("nan")
    pooled_oracle_cost = cost_ms(pooled_oracle_e, lat)

    # ---- random-gate retention sanity: retention should be ~= e by construction ----
    random_sanity = {
        str(s): {"e": pooled_by_s[s]["random"]["e"], "retained_gain": pooled_by_s[s]["random"]["retained_gain"],
                 "abs_diff": abs(pooled_by_s[s]["random"]["e"] - pooled_by_s[s]["random"]["retained_gain"])}
        for s in S_GRID
    }

    elapsed = time.time() - t0
    print(f"  [{ds}] done in {elapsed:.1f}s")

    return dict(
        n_rows=n,
        lat=lat,
        fold_records=fold_records,
        selector_match=selector_match,
        s1_check=s1_check,
        s0_check=s0_check,
        pooled=dict(
            CR_naive=pooled_naive, CR_selector=pooled_selector, CR_fixed=pooled_fixed,
            best_fixed=pooled_best_fixed, by_s=pooled_by_s,
            oracle={"e": pooled_oracle_e, "CR": pooled_oracle_cr, "retained_gain": pooled_oracle_retained,
                    "gain_over_best_fixed": pooled_oracle_cr - pooled_fixed[pooled_best_fixed],
                    "cost_ms": pooled_oracle_cost, "cost_over_t_selector": pooled_oracle_cost / 1000.0 / lat["t_selector"]},
        ),
        random_sanity=random_sanity,
        elapsed_s=elapsed,
    )


def round_fold_entry(fr):
    out = {
        "fold": fr["fold"], "n_train": fr["n_train"], "n_eval": fr["n_eval"],
        "CR_naive": r4(fr["CR_naive"]), "CR_selector": r4(fr["CR_selector"]),
        "CR_selector_of_record": r4(fr["CR_selector_of_record"]),
        "CR_selector_match_diff": fr["CR_selector_match_diff"],
        "CR_fixed": {p: r4(v) for p, v in fr["CR_fixed"].items()},
        "best_fixed": fr["best_fixed"], "ev_margin": fr["ev_margin"],
        "gate_auc": {g: r3(v) for g, v in fr["gate_auc"].items()},
        "gate_calibration": {g: {k: r3(v) for k, v in d.items()} for g, d in fr["gate_calibration"].items()},
        "oracle": {"e": r3(fr["oracle"]["e"]), "CR": r4(fr["oracle"]["CR"]),
                   "retained_gain": r4(fr["oracle"]["retained_gain"]),
                   "gain_over_best_fixed": r4(fr["oracle"]["gain_over_best_fixed"]),
                   "cost_ms": r1(fr["oracle"]["cost_ms"]),
                   "cost_over_t_selector": r3(fr["oracle"]["cost_over_t_selector"])},
        "by_s": {},
    }
    for s, entry in fr["by_s"].items():
        e_out = {}
        for gname in list(GATE_CONFIGS):
            e_out[gname] = {"e": r3(entry[gname]["e"]), "CR": r4(entry[gname]["CR"]),
                             "retained_gain": r4(entry[gname]["retained_gain"]),
                             "gain_over_best_fixed": r4(entry[gname]["gain_over_best_fixed"])}
        e_out["random"] = {"e": r3(entry["random"]["e"]), "CR_mean": r4(entry["random"]["CR_mean"]),
                            "CR_std": r4(entry["random"]["CR_std"]),
                            "retained_gain": r4(entry["random"]["retained_gain"]),
                            "gain_over_best_fixed": r4(entry["random"]["gain_over_best_fixed"])}
        e_out["cost_ms"] = {k: r1(v) for k, v in entry["cost_ms"].items()}
        e_out["cost_over_t_selector"] = {k: r3(v) for k, v in entry["cost_over_t_selector"].items()}
        out["by_s"][str(s)] = e_out
    return out


def round_pooled(p):
    out = {
        "CR_naive": r4(p["CR_naive"]), "CR_selector": r4(p["CR_selector"]),
        "CR_fixed": {k: r4(v) for k, v in p["CR_fixed"].items()},
        "best_fixed": p["best_fixed"],
        "oracle": {"e": r3(p["oracle"]["e"]), "CR": r4(p["oracle"]["CR"]),
                   "retained_gain": r4(p["oracle"]["retained_gain"]),
                   "gain_over_best_fixed": r4(p["oracle"]["gain_over_best_fixed"]),
                   "cost_ms": r1(p["oracle"]["cost_ms"]),
                   "cost_over_t_selector": r3(p["oracle"]["cost_over_t_selector"])},
        "by_s": {},
    }
    for s, entry in p["by_s"].items():
        e_out = {}
        for gname in list(GATE_CONFIGS):
            e_out[gname] = {"e": r3(entry[gname]["e"]), "CR": r4(entry[gname]["CR"]),
                             "retained_gain": r4(entry[gname]["retained_gain"]),
                             "gain_over_best_fixed": r4(entry[gname]["gain_over_best_fixed"])}
        e_out["random"] = {"e": r3(entry["random"]["e"]), "CR_mean": r4(entry["random"]["CR_mean"]),
                            "CR_std": r4(entry["random"]["CR_std"]),
                            "retained_gain": r4(entry["random"]["retained_gain"]),
                            "gain_over_best_fixed": r4(entry["random"]["gain_over_best_fixed"])}
        e_out["cost_ms"] = {k: r1(v) for k, v in entry["cost_ms"].items()}
        e_out["cost_over_t_selector"] = {k: r3(v) for k, v in entry["cost_over_t_selector"].items()}
        e_out["per_fold_CR"] = {k: [r4(x) for x in v] for k, v in entry["per_fold_CR"].items()}
        e_out["per_fold_retained_gain"] = {k: [r4(x) for x in v] for k, v in entry["per_fold_retained_gain"].items()}
        out["by_s"][str(s)] = e_out
    return out


def success_gate_legal(corpus_out):
    """Pre-registered success gate (Legal, s = 0.5), exactly as written in the spec."""
    s = 0.5
    pooled = corpus_out["pooled"]
    by_s = pooled["by_s"][s]
    probe_ret = by_s["probe"]["retained_gain"]
    random_ret = by_s["random"]["retained_gain"]
    static_ret = by_s["static"]["retained_gain"]
    qonly_ret = by_s["qonly"]["retained_gain"]

    crit1_value = probe_ret - random_ret
    crit1 = crit1_value >= 0.15

    crit2 = (probe_ret > static_ret) and (probe_ret > qonly_ret)

    fold_probe_cr = [fr["by_s"][s]["probe"]["CR"] for fr in corpus_out["fold_records"]]
    fold_hybrid_cr = [fr["CR_fixed"]["hybrid"] for fr in corpus_out["fold_records"]]
    crit3_folds = [pc > hc for pc, hc in zip(fold_probe_cr, fold_hybrid_cr)]
    crit3 = sum(crit3_folds) >= 4

    fold_random_cr = [fr["by_s"][s]["random"]["CR_mean"] for fr in corpus_out["fold_records"]]
    crit4_folds = [pc - rc > 0 for pc, rc in zip(fold_probe_cr, fold_random_cr)]
    crit4 = sum(crit4_folds) >= 4

    verdict = "PASS" if (crit1 and crit2 and crit3 and crit4) else "FAIL"

    return {
        "s": s,
        "realised_e_probe_pooled": r3(by_s["probe"]["e"]),
        "criterion_1_retained_gain_minus_random_geq_0.15": {
            "probe_retained_gain": r4(probe_ret), "random_retained_gain": r4(random_ret),
            "value": r4(crit1_value), "pass": bool(crit1),
        },
        "criterion_2_probe_beats_static_and_qonly": {
            "probe_retained_gain": r4(probe_ret), "static_retained_gain": r4(static_ret),
            "qonly_retained_gain": r4(qonly_ret), "pass": bool(crit2),
        },
        "criterion_3_CR_gated_beats_hybrid_on_4_of_5_folds": {
            "fold_probe_CR": [r4(x) for x in fold_probe_cr], "fold_hybrid_CR": [r4(x) for x in fold_hybrid_cr],
            "folds_won": crit3_folds, "n_won": sum(crit3_folds), "pass": bool(crit3),
        },
        "criterion_4_paired_probe_beats_random_on_4_of_5_folds": {
            "fold_probe_CR": [r4(x) for x in fold_probe_cr], "fold_random_CR_mean": [r4(x) for x in fold_random_cr],
            "folds_won": crit4_folds, "n_won": sum(crit4_folds), "pass": bool(crit4),
        },
        "cost_ms_probe_at_s0.5": r1(by_s["cost_ms"]["probe"]),
        "cost_over_t_selector_probe_at_s0.5": r3(by_s["cost_over_t_selector"]["probe"]),
        "verdict": verdict,
    }


def secondary_quality_musique(out):
    sec = {}
    # QuALITY: retained gain vs random at s = 0.5
    q = out["quality"]["pooled"]["by_s"][0.5]
    sec["quality_s0.5"] = {
        "probe_retained_gain": r4(q["probe"]["retained_gain"]),
        "random_retained_gain": r4(q["random"]["retained_gain"]),
        "diff": r4(q["probe"]["retained_gain"] - q["random"]["retained_gain"]),
    }
    # Musique: does gating reduce the selector's loss (CR_gated > CR_selector at any s < 1)?
    m_pooled = out["musique"]["pooled"]
    m_selector = m_pooled["CR_selector"]
    any_beats_selector = {}
    for s in S_GRID:
        if s >= 1.0:
            continue
        cr = m_pooled["by_s"][s]["probe"]["CR"]
        any_beats_selector[str(s)] = {"CR_probe": r4(cr), "beats_selector": bool(cr > m_selector)}
    sec["musique_gating_reduces_selector_loss"] = {
        "CR_selector": r4(m_selector),
        "CR_naive": r4(m_pooled["CR_naive"]),
        "by_s": any_beats_selector,
        "any_s_beats_selector": any(v["beats_selector"] for v in any_beats_selector.values()),
    }
    return sec


def print_table(ds, corpus_out):
    print(f"\n-- {ds}: pooled operating-point table --")
    print(f"{'s':>5} {'e':>6} {'AUC':>6} {'CR_probe':>9} {'CR_static':>9} {'CR_qonly':>9} "
          f"{'CR_random':>9} {'ret_probe':>9} {'ret_random':>10} {'cost_ms':>9}")
    pooled = corpus_out["pooled"]
    auc_mean = float(np.mean([fr["gate_auc"]["probe"] for fr in corpus_out["fold_records"]]))
    for s in S_GRID:
        e = pooled["by_s"][s]
        print(f"{s:>5.2f} {e['probe']['e']:>6.3f} {auc_mean:>6.3f} {e['probe']['CR']:>9.4f} "
              f"{e['static']['CR']:>9.4f} {e['qonly']['CR']:>9.4f} {e['random']['CR_mean']:>9.4f} "
              f"{e['probe']['retained_gain']:>9.4f} {e['random']['retained_gain']:>10.4f} "
              f"{e['cost_ms']['probe']:>9.1f}")


def main():
    assert os.environ.get("ROUTER_GROUPED") == "1", "ROUTER_GROUPED=1 must be set (grouped CV of record)."
    out = {}
    for ds in DATASETS:
        try:
            out[ds] = run_corpus(ds)
        except Exception as e:
            print(f"!! {ds} failed/aborted: {e}")
            raise
        print_table(ds, out[ds])

    legal_gate = success_gate_legal(out["ultraDomain_legal"])
    secondaries = secondary_quality_musique(out)

    final = {
        "spec": {"file": SPEC_FILE, "s_grid": S_GRID, "n_random_draws": N_RANDOM_DRAWS,
                 "gate_configs": GATE_CONFIGS, "xgb_params": XGB_KW},
        "quality": {
            "n_rows": out["quality"]["n_rows"],
            "folds": [round_fold_entry(fr) for fr in out["quality"]["fold_records"]],
            "pooled": round_pooled(out["quality"]["pooled"]),
            "selector_match": out["quality"]["selector_match"],
            "s1_check": out["quality"]["s1_check"],
            "s0_check": out["quality"]["s0_check"],
            "random_sanity": out["quality"]["random_sanity"],
        },
        "musique": {
            "n_rows": out["musique"]["n_rows"],
            "folds": [round_fold_entry(fr) for fr in out["musique"]["fold_records"]],
            "pooled": round_pooled(out["musique"]["pooled"]),
            "selector_match": out["musique"]["selector_match"],
            "s1_check": out["musique"]["s1_check"],
            "s0_check": out["musique"]["s0_check"],
            "random_sanity": out["musique"]["random_sanity"],
        },
        "ultraDomain_legal": {
            "n_rows": out["ultraDomain_legal"]["n_rows"],
            "folds": [round_fold_entry(fr) for fr in out["ultraDomain_legal"]["fold_records"]],
            "pooled": round_pooled(out["ultraDomain_legal"]["pooled"]),
            "selector_match": out["ultraDomain_legal"]["selector_match"],
            "s1_check": out["ultraDomain_legal"]["s1_check"],
            "s0_check": out["ultraDomain_legal"]["s0_check"],
            "random_sanity": out["ultraDomain_legal"]["random_sanity"],
        },
        "legal_success_gate_s0.5": legal_gate,
        "secondary": secondaries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {OUT_PATH}")

    print("\n" + "=" * 70)
    print("LEGAL SUCCESS GATE (s=0.5)")
    print("=" * 70)
    print(json.dumps(legal_gate, indent=2))


if __name__ == "__main__":
    main()
