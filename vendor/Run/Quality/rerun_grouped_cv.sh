#!/bin/bash
# Grouped-CV leakage check (2026-09-17). Writes ONLY to Dataset/EvaluationData/RouterEval_grouped/.
# Step 0 re-runs the ungrouped script for quality into RouterEval_regress/ to prove the patch is a no-op by default.
set -u
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ragBench
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=42
LOG=Dataset/EvaluationData/_backups/grouped_cv_$(date +%Y%m%d_%H%M).log
echo "start $(date)" | tee "$LOG"
ROUTER_GROUPED=0 ROUTER_OUT_ROOT=RouterEval_regress python3 Run/Quality/run_router_training.py --dataset quality >> "$LOG" 2>&1
echo "regression run done $(date)" | tee -a "$LOG"
for d in quality musique ultraDomain_legal; do
  ROUTER_GROUPED=1 ROUTER_OUT_ROOT=RouterEval_grouped python3 Run/Quality/run_router_training.py --dataset $d >> "$LOG" 2>&1
  echo "grouped router $d done $(date)" | tee -a "$LOG"
  ROUTER_GROUPED=1 ROUTER_OUT_ROOT=RouterEval_grouped python3 Run/Quality/run_ablation.py --dataset $d >> "$LOG" 2>&1
  echo "grouped ablation $d done $(date)" | tee -a "$LOG"
done
echo "ALL DONE $(date)" | tee -a "$LOG"
