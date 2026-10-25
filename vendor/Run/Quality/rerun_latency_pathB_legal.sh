#!/bin/bash
# Path B latency, Legal only, stratified seeded sample of 300 queries (full 1 277 at ~17 s/query = 6 h; the
# selector's result-set features embed whole contracts). QUIET MACHINE ONLY.
set -uo pipefail
cd /Users/henrybothwell/Desktop/RAGRouter-Bench
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ragBench
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=42
echo "[$(date '+%F %T')] latency ultraDomain_legal --limit 300 — start"
python Run/Quality/run_latency_eval.py --dataset ultraDomain_legal --limit 300 || echo "[$(date '+%F %T')] FAILED"
echo "[$(date '+%F %T')] latency ultraDomain_legal — done"
echo "[$(date '+%F %T')] LATENCY PATHB COMPLETE"
