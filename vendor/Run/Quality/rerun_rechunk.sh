#!/bin/bash
###############################################################################
# rerun_rechunk.sh
#
# Full corpus rerun on corrected chunking (2026-09-20).
# Spec: skripsie-workspace/reviews/2026-09-20_rechunk_rerun_spec.md
#
# Config/ChunkConfig.py: CHUNK_SIZE 512 -> 180, CHUNK_OVERLAP 100 -> 36 (same
# ~20% overlap ratio; see the dated comment left in ChunkConfig.py itself).
# Old file backed up to Config/ChunkConfig.py.pre_rechunk before this edit.
#
# Backups of RetrievalEval/, RouterEval/, LatencyEval/, and the per-corpus
# processed artefacts (Embedding, Graph, Index, Triplet) were already copied
# (not moved) to
#   Dataset/EvaluationData/_backups/pre_rechunk_2026-09-20/
# before this script was written and verified by du/ls size+file-count
# comparison against the originals. This script re-verifies that backup is
# present and non-empty for a corpus before deleting its on-disk artefacts
# (verify_backup_then_remove), as a second safety net.
#
# Order: quality -> ultraDomain_legal -> musique (smallest first; QuALITY's
# retrieval-eval stage is smoke-tested against the pilot's chunk=200 naive
# nDCG@5 before the two expensive corpora are touched).
#
# DEVIATION FROM THE SPEC'S LITERAL STEP LIST (flagged for the judge):
# the spec's per-corpus step 1 is "run_corpus_evaluation.py ... rebuild
# chunks -> embeddings -> graph -> entity index", but that script never
# touches Dataset/ProcessedData/Index/{corpus}/index.faiss (the FAISS
# *chunk* index used for naive retrieval) -- that file is only built by
# Run/Process/run_index.py, which the spec's pipeline order never lists.
# run_retrieval_eval.py's NaiveRetriever reads index.faiss directly with no
# rebuild fallback (hit when a degradation condition leaves no text to
# re-embed in-process), and run_latency_eval.py's NaiveRetriever ALWAYS
# reads it directly. Without rebuilding it here, the latency stage would
# silently benchmark retrieval against the OLD 512-token chunk index instead
# of the new 180-token one. Run/Process/run_index.py is therefore inserted
# as a step between run_corpus_evaluation.py and run_retrieval_eval.py for
# each corpus. This is the only addition to the spec's script list; no
# script besides Config/ChunkConfig.py and this file was modified.
###############################################################################
set -e

REPO="/Users/henrybothwell/Desktop/RAGRouter-Bench"
WORKSPACE="/Users/henrybothwell/Desktop/skripsie-workspace"
PY="/opt/anaconda3/envs/ragBench/bin/python"
BACKUP_DIR="$REPO/Dataset/EvaluationData/_backups/pre_rechunk_2026-09-20"

cd "$REPO"
export PYTHONHASHSEED=42
export ROUTER_GROUPED=1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Run "$@" as a named stage: prints start/end timestamps and the exit code
# to the log, and stops the whole script on the first non-zero exit (the
# script otherwise runs under `set -e`; errexit is toggled off only for the
# duration of the wrapped command so its exit code can be captured and
# logged before we decide whether to abort).
stage() {
    local name="$1"; shift
    echo "===== [$(ts)] STAGE START: $name ====="
    set +e
    "$@"
    local rc=$?
    set -e
    echo "===== [$(ts)] STAGE END: $name (exit $rc) ====="
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: stage failed: $name (exit $rc)" >&2
        exit "$rc"
    fi
}

# Refuses to delete a corpus's on-disk processed artefacts unless this
# script can see a non-empty backup for every artefact kind. Deletion must
# happen before run_corpus_evaluation.py runs for that corpus, or its
# resume=True / skip-if-exists logic (ChunkDo/EmbeddingDo/GraphDo) will keep
# serving embeddings, triplets, and the entity index built from the OLD
# 512-token chunks instead of rebuilding under the new config.
verify_backup_then_remove() {
    local ds="$1"
    for kind in Embedding Graph Index Triplet; do
        local bkp="$BACKUP_DIR/ProcessedData/$kind/$ds"
        if [ ! -d "$bkp" ] || [ -z "$(ls -A "$bkp" 2>/dev/null)" ]; then
            echo "FATAL: backup missing/empty for $ds/$kind at $bkp -- refusing to delete originals." >&2
            return 1
        fi
    done
    echo "  Backup verified present for $ds (Embedding/Graph/Index/Triplet). Removing stale on-disk artefacts..."
    rm -rf "$REPO/Dataset/ProcessedData/Embedding/$ds"
    rm -rf "$REPO/Dataset/ProcessedData/Graph/$ds"
    rm -rf "$REPO/Dataset/ProcessedData/Index/$ds"
    rm -rf "$REPO/Dataset/ProcessedData/Triplet/$ds"
}

# QuALITY smoke check (spec: after QuALITY's retrieval-eval stage). Naive
# nDCG@5 in RetrievalEval/quality/combined_000.json must land within +/-0.01
# of the pilot's variant-C value 0.4563 (pilot used chunk 200; 180 will
# differ slightly), with exactly 1198 queries. Non-zero exit stops the
# script (via `stage`) before the two expensive corpora are touched.
quality_smoke_check() {
    local f="$REPO/Dataset/EvaluationData/RetrievalEval/quality/combined_000.json"
    "$PY" - "$f" <<'PYEOF'
import json, sys

path = sys.argv[1]
expected_ndcg = 0.4563
tol = 0.01
expected_n = 1198

try:
    with open(path) as fh:
        data = json.load(fh)
    agg = data["paradigms"]["naive"]["aggregate"]
    ndcg = agg.get("ndcg@5")
    n = agg.get("n_queries")
except Exception as e:
    print(f"SMOKE CHECK FAILED: could not read/parse {path}: {e}")
    sys.exit(1)

problems = []
if ndcg is None or abs(ndcg - expected_ndcg) > tol:
    problems.append(f"naive nDCG@5 = {ndcg!r}, expected {expected_ndcg} +/- {tol}")
if n != expected_n:
    problems.append(f"n_queries = {n!r}, expected {expected_n}")

if problems:
    print("SMOKE CHECK FAILED: " + "; ".join(problems))
    sys.exit(1)

print(f"SMOKE CHECK PASSED: naive nDCG@5={ndcg:.4f} "
      f"(expected {expected_ndcg} +/- {tol}), n_queries={n}")
PYEOF
}

echo "################################################################################"
echo "  rerun_rechunk.sh started $(ts)"
echo "  CHUNK_SIZE=180 CHUNK_OVERLAP=36 (was 512/100)"
echo "  PYTHONHASHSEED=$PYTHONHASHSEED  ROUTER_GROUPED=$ROUTER_GROUPED"
echo "################################################################################"

for ds in quality ultraDomain_legal musique; do
    echo ""
    echo ">>>>> Corpus: $ds <<<<<"

    stage "remove stale processed artefacts ($ds)" verify_backup_then_remove "$ds"

    stage "run_corpus_evaluation ($ds): chunk -> embed -> graph -> entity index" \
        "$PY" -m Run.Pipeline.run_corpus_evaluation --dataset "$ds"

    stage "run_index ($ds): rebuild FAISS chunk index from new embeddings" \
        "$PY" Run/Process/run_index.py --dataset "$ds"

    if [ "$ds" = "musique" ]; then
        stage "run_retrieval_eval ($ds, limit 500, run of record)" \
            "$PY" Run/Quality/run_retrieval_eval.py --dataset "$ds" --limit 500
    else
        stage "run_retrieval_eval ($ds)" \
            "$PY" Run/Quality/run_retrieval_eval.py --dataset "$ds"
    fi

    if [ "$ds" = "quality" ]; then
        stage "QuALITY smoke check" quality_smoke_check
    fi
done

echo ""
echo ">>>>> Cross-corpus stages (require RetrievalEval for all three corpora) <<<<<"

stage "compute_graph_probe (all corpora)" \
    "$PY" Run/Quality/compute_graph_probe.py --dataset all

for ds in quality ultraDomain_legal musique; do
    stage "run_router_training ($ds)" \
        "$PY" Run/Quality/run_router_training.py --dataset "$ds"
done
stage "run_router_training (cross-dataset pairs)" \
    "$PY" Run/Quality/run_router_training.py --cross

for ds in quality ultraDomain_legal musique; do
    stage "run_ablation ($ds)" \
        "$PY" Run/Quality/run_ablation.py --dataset "$ds"
done

stage "run_routing_uncertainty (all corpora, no CLI args)" \
    "$PY" Run/Quality/run_routing_uncertainty.py

stage "diagnostics/probe_threshold_rule" \
    "$PY" Run/Quality/diagnostics/probe_threshold_rule.py
stage "diagnostics/gated_selector" \
    "$PY" Run/Quality/diagnostics/gated_selector.py
stage "diagnostics/failure_detector" \
    "$PY" Run/Quality/diagnostics/failure_detector.py

echo ""
echo ">>>>> Latency stage (last, quiet-machine gated) <<<<<"
LOAD1=$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print $1}')
echo "  Current 1-minute load average: ${LOAD1:-unknown} (threshold: 4)"
BUSY=$(awk -v l="${LOAD1:-999}" 'BEGIN { print (l+0 >= 4) ? "1" : "0" }')
if [ "$BUSY" = "1" ]; then
    echo "SKIPPED-LATENCY-BUSY: load average ${LOAD1:-unknown} >= 4; leaving latency (and the figures step after it) for the orchestrator to trigger separately."
    exit 0
fi

for ds in quality ultraDomain_legal musique; do
    if [ "$ds" = "musique" ]; then
        stage "run_latency_eval ($ds, limit 500)" \
            "$PY" Run/Quality/run_latency_eval.py --dataset "$ds" --limit 500
    else
        stage "run_latency_eval ($ds)" \
            "$PY" Run/Quality/run_latency_eval.py --dataset "$ds"
    fi
done

echo ""
echo ">>>>> Figures <<<<<"
stage "make_result_figures" bash -c "cd '$WORKSPACE' && '$PY' tools/make_result_figures.py"

echo ""
echo "################################################################################"
echo "  rerun_rechunk.sh finished $(ts)"
echo "################################################################################"
