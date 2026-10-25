#!/bin/bash
# Resume of rerun_rechunk.sh from the downstream stages, after compute_graph_probe failed on a transient
# HuggingFace Hub HTTP error ("client has been closed") on 2026-09-21 23:16. Retrieval sweeps for all three
# corpora completed successfully (exit 0). HF_HUB_OFFLINE=1 avoids hub calls; the model is cached locally.
set -e
cd /Users/henrybothwell/Desktop/RAGRouter-Bench
export PYTHONHASHSEED=42 ROUTER_GROUPED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/opt/anaconda3/envs/ragBench/bin/python
WORKSPACE=/Users/henrybothwell/Desktop/skripsie-workspace
stage() { local name="$1"; shift; echo "===== [$(date '+%F %T')] STAGE START: $name ====="; "$@"; local rc=$?; echo "===== [$(date '+%F %T')] STAGE END: $name (exit $rc) ====="; return $rc; }
stage "compute_graph_probe (all corpora)" "$PY" Run/Quality/compute_graph_probe.py --dataset all
for ds in quality ultraDomain_legal musique; do stage "run_router_training ($ds)" "$PY" Run/Quality/run_router_training.py --dataset "$ds"; done
stage "run_router_training (cross-dataset pairs)" "$PY" Run/Quality/run_router_training.py --cross
for ds in quality ultraDomain_legal musique; do stage "run_ablation ($ds)" "$PY" Run/Quality/run_ablation.py --dataset "$ds"; done
stage "run_routing_uncertainty" "$PY" Run/Quality/run_routing_uncertainty.py
stage "diagnostics/probe_threshold_rule" "$PY" Run/Quality/diagnostics/probe_threshold_rule.py
stage "diagnostics/gated_selector" "$PY" Run/Quality/diagnostics/gated_selector.py
stage "diagnostics/failure_detector" "$PY" Run/Quality/diagnostics/failure_detector.py
LOAD1=$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print $1}')
echo "1-minute load: $LOAD1"
for ds in quality ultraDomain_legal; do stage "run_latency_eval ($ds)" "$PY" Run/Quality/run_latency_eval.py --dataset "$ds"; done
stage "run_latency_eval (musique, limit 500)" "$PY" Run/Quality/run_latency_eval.py --dataset musique --limit 500
stage "make_result_figures" bash -c "cd '$WORKSPACE' && '$PY' tools/make_result_figures.py"
echo "ALL DONE $(date '+%F %T')"
