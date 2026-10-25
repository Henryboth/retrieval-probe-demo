#!/bin/bash
# Path B (2026-09-17): grouped-by-query router of record — training, ablation, transfer, Ω.
set -u
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ragBench
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=42
LOG=Dataset/EvaluationData/_backups/pathB_router_$(date +%Y%m%d_%H%M).log
echo "start $(date)" | tee "$LOG"
for d in quality musique ultraDomain_legal; do
  python Run/Quality/run_router_training.py --dataset $d >> "$LOG" 2>&1 && echo "router $d done $(date)" | tee -a "$LOG"
  python Run/Quality/run_ablation.py --dataset $d >> "$LOG" 2>&1 && echo "ablation $d done $(date)" | tee -a "$LOG"
done
python Run/Quality/run_router_training.py --cross >> "$LOG" 2>&1 && echo "cross done $(date)" | tee -a "$LOG"
python Run/Quality/run_routing_uncertainty.py >> "$LOG" 2>&1 && echo "omega done $(date)" | tee -a "$LOG"
echo "ALL DONE $(date)" | tee -a "$LOG"
