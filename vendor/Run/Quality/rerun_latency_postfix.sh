#!/bin/bash
set -uo pipefail
cd /Users/henrybothwell/Desktop/RAGRouter-Bench
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragBench
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=42
for args in "quality" "musique --limit 500" "ultraDomain_legal"; do
  echo "[$(date '+%F %T')] latency $args — start"
  python Run/Quality/run_latency_eval.py --dataset $args || echo "[$(date '+%F %T')] latency $args — FAILED"
  echo "[$(date '+%F %T')] latency $args — done"
done
echo "[$(date '+%F %T')] LATENCY RERUN COMPLETE"
