#!/usr/bin/env bash
#
# rerun_dropout_fix.sh
# =====================
# Hand-over script for the two evaluation-defect fixes applied to
# Run/Quality/run_retrieval_eval.py on 2026-09-15:
#
#   Defect 1 — GraphRetriever could return doc_ids of documents dropped by
#              DocumentDropout because the graph itself is never rebuilt.
#              Fixed with GraphRetriever(allowed_doc_ids=...).
#   Defect 2 — Legal's graph_edge_removal reused the persisted disk index
#              (built with an unconditional "title\n\ncontext" chunk text),
#              which diverges from the in-process rebuild path used by every
#              other operator when title is empty. Fixed by rebuilding the
#              naive index in-process for graph_edge_removal too.
#
# Only document_dropout, combined (all three corpora) and, for
# ultraDomain_legal only, graph_edge_removal are affected by these fixes —
# see the operator lists below. text_truncation and entity_variation are
# untouched by either fix and are NOT rerun here.
#
# Run with:
#   nohup bash Run/Quality/rerun_dropout_fix.sh \
#     > Dataset/EvaluationData/_backups/rerun_dropout_fix_2026-09-15.log 2>&1 &
#
# Safety notes encoded below, per /Users/henrybothwell/.claude/commands/rerun-safe.md
# and PROJECT_RECORD.md section 8:
#   - never --dataset all (silently drops ultraDomain_legal in three scripts)
#   - musique retrieval always runs with --limit 500
#   - a single-operator retrieval rerun overwrites summary.json wholesale;
#     it is regenerated from the 40 per-rate files after all reruns finish
#   - OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE avoid the faiss/xgboost
#     duplicate-OpenMP segfault/deadlock documented in PROJECT_RECORD.md §8

set -euo pipefail

# ── 1. Environment ────────────────────────────────────────────────────────
REPO_ROOT="/Users/henrybothwell/Desktop/RAGRouter-Bench"
cd "$REPO_ROOT"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ragBench

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=42

STAMP="2026-09-15"
RETRIEVAL_BACKUP="Dataset/EvaluationData/_backups/RetrievalEval_pre_dropoutfix_${STAMP}"
ROUTER_BACKUP="Dataset/EvaluationData/_backups/RouterEval_pre_dropoutfix_${STAMP}"
CORPORA=(quality musique ultraDomain_legal)

SECONDS=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "RERUN START — repo root: $REPO_ROOT"

# ── 2. Backups (idempotent — skip if already present) ─────────────────────
log "STEP 2: backups"

mkdir -p "$RETRIEVAL_BACKUP" "$ROUTER_BACKUP"

for c in "${CORPORA[@]}"; do
    if [ -d "$RETRIEVAL_BACKUP/$c" ]; then
        log "  RetrievalEval/$c already backed up, skipping"
    else
        log "  backing up RetrievalEval/$c"
        cp -R "Dataset/EvaluationData/RetrievalEval/$c" "$RETRIEVAL_BACKUP/$c"
    fi

    if [ -d "$ROUTER_BACKUP/$c" ]; then
        log "  RouterEval/$c already backed up, skipping"
    elif [ -d "Dataset/EvaluationData/RouterEval/$c" ]; then
        log "  backing up RouterEval/$c"
        cp -R "Dataset/EvaluationData/RouterEval/$c" "$ROUTER_BACKUP/$c"
    fi
done

for pair in cross_quality_to_musique cross_quality_to_ultraDomain_legal \
            cross_musique_to_quality cross_musique_to_ultraDomain_legal \
            cross_ultraDomain_legal_to_quality cross_ultraDomain_legal_to_musique; do
    if [ -d "$ROUTER_BACKUP/$pair" ]; then
        log "  RouterEval/$pair already backed up, skipping"
    elif [ -d "Dataset/EvaluationData/RouterEval/$pair" ]; then
        log "  backing up RouterEval/$pair"
        cp -R "Dataset/EvaluationData/RouterEval/$pair" "$ROUTER_BACKUP/$pair"
    fi
done

log "STEP 2 done"

# ── 3. Retrieval reruns — CONSISTENCY PASS (2026-09-16 afternoon) ─────────
# The text_truncation, entity_variation and graph_edge_removal graph/hybrid
# results still date from August code without the sorted node_to_docs
# determinism fix and without PYTHONHASHSEED; document_dropout/combined were
# regenerated with current code. r=0 graph values therefore differ across
# operators (Legal 0.142 vs 0.193). Rerun graph+hybrid for the three remaining
# operators on all corpora so every condition comes from one code version.
log "STEP 3: retrieval reruns (consistency pass)"
for ds in quality musique ultraDomain_legal; do
  LIM=""; [ "$ds" = "musique" ] && LIM="--limit 500"
  for op in text_truncation entity_variation graph_edge_removal; do
    log "  $ds / $op — start"
    python Run/Quality/run_retrieval_eval.py --dataset $ds $LIM --degradation $op --paradigms graph,hybrid
    log "  $ds / $op — done"
  done
done
log "STEP 3 done"

# ── 4. Regenerate summary.json per corpus ──────────────────────────────
# A single-operator retrieval rerun overwrites summary.json wholesale with
# only the operators run this session (see run_dataset() in
# run_retrieval_eval.py). Regenerate it from the 40 per-rate files instead
# of trusting the post-rerun summary.json, mirroring the exact structure
# the runner writes: {deg_name: {rate: {rate, overall_quality, completeness,
# consistency, paradigms: {paradigm: aggregate}}}}.
log "STEP 4: regenerating summary.json per corpus"

for c in "${CORPORA[@]}"; do
    log "  regenerating summary.json for $c"
    python3 - "$c" <<'PYEOF'
import json, sys
from pathlib import Path

corpus = sys.argv[1]
base = Path(f"Dataset/EvaluationData/RetrievalEval/{corpus}")
summary = {}

for f in sorted(base.glob("*.json")):
    if f.name == "summary.json":
        continue
    data = json.load(open(f))
    deg_name = data["degradation_type"]
    rate = data["rate"]
    qf = data["quality_features"]
    summary.setdefault(deg_name, {})[str(rate)] = {
        "rate": rate,
        "overall_quality": qf["overall"],
        "completeness": qf["completeness"],
        "consistency": qf["consistency"],
        "paradigms": {
            p: data["paradigms"][p]["aggregate"] for p in data["paradigms"]
        },
    }

with open(base / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"    {corpus}: {sum(len(v) for v in summary.values())} rate entries "
      f"across {len(summary)} operators")
PYEOF
done

log "STEP 4 done"

# ── 5. File-count check ─────────────────────────────────────────────────
# Expect exactly 40 condition files (5 operators x 8 rates) + summary.json
# per corpus. STOP (exit 1) if the count differs.
log "STEP 5: file-count check"

for c in "${CORPORA[@]}"; do
    n_cond=$(find "Dataset/EvaluationData/RetrievalEval/$c" -maxdepth 1 -type f -name "*.json" ! -name "summary.json" | wc -l | tr -d ' ')
    has_summary=$([ -f "Dataset/EvaluationData/RetrievalEval/$c/summary.json" ] && echo yes || echo no)
    log "  $c: $n_cond condition files, summary.json present: $has_summary"
    if [ "$n_cond" -ne 40 ] || [ "$has_summary" != "yes" ]; then
        log "STOP: $c does not have exactly 40 condition files + summary.json"
        exit 1
    fi
done

log "STEP 5 done — all three corpora have 40 condition files + summary.json"

# ── 6. Downstream, in rerun-safe order ──────────────────────────────────
# 1) router training per corpus, 2) --cross, 3) ablation per corpus,
# 4) routing_uncertainty + its sample-size check.
# run_chunk_quality_routing.py is deliberately NOT included: its output
# (chunk_quality_experiment.json) is not in the set of files the thesis
# consumes (RouterEval/*/results.json, oracle_gap_decomposition.json,
# ablation.json, feature_importance.json, routing_uncertainty.json,
# omega_sample_size_stability.json) per the hand-over spec.
log "STEP 6: downstream — router training"

for c in "${CORPORA[@]}"; do
    log "  router training / $c — start"
    python Run/Quality/run_router_training.py --dataset "$c"
    log "  router training / $c — done"
done

log "  router training / --cross — start"
python Run/Quality/run_router_training.py --cross
log "  router training / --cross — done"

log "STEP 6a done — router training + cross"

for c in "${CORPORA[@]}"; do
    log "  ablation / $c — start"
    python Run/Quality/run_ablation.py --dataset "$c"
    log "  ablation / $c — done"
done

log "STEP 6b done — ablation"

log "  routing_uncertainty — start"
python Run/Quality/run_routing_uncertainty.py
log "  routing_uncertainty — done"

log "  routing_uncertainty --sample-size-check — start"
python Run/Quality/run_routing_uncertainty.py --sample-size-check
log "  routing_uncertainty --sample-size-check — done"

log "STEP 6 done"

# ── 7. n_instances check ────────────────────────────────────────────────
# STOP if any of the three does not match — this is the signature of
# directory contamination from a misplaced backup (rerun-safe.md).
log "STEP 7: n_instances check"

expected_for() { case "$1" in quality) echo 47920;; musique) echo 19920;; ultraDomain_legal) echo 51080;; esac; }

for c in "${CORPORA[@]}"; do
    actual=$(python3 - "$c" <<'PYEOF'
import json, sys
c = sys.argv[1]
data = json.load(open(f"Dataset/EvaluationData/RouterEval/{c}/results.json"))
# results.json is a dict keyed by "{config}_{architecture}", each value a
# dict carrying n_instances (same value across every config/architecture).
vals = {int(v["n_instances"]) for v in data.values() if isinstance(v, dict) and "n_instances" in v}
print(sorted(vals)[-1] if len(vals) == 1 else (-1 if not vals else -2))
PYEOF
)
    expected="$(expected_for "$c")"
    log "  $c: n_instances=$actual (expected $expected)"
    if [ "$actual" != "$expected" ]; then
        log "STOP: $c results.json n_instances ($actual) != expected ($expected)"
        exit 1
    fi
done

log "STEP 7 done — n_instances matches expected for all three corpora"

# ── 8. Done ────────────────────────────────────────────────────────────
log "RERUN COMPLETE — wall-clock: ${SECONDS}s ($((SECONDS/60)) min)"
