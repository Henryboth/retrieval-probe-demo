#!/bin/bash
set -e
cd /Users/henrybothwell/Desktop/RAGRouter-Bench
export PYTHONHASHSEED=42 ROUTER_GROUPED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/opt/anaconda3/envs/ragBench/bin/python; WORKSPACE=/Users/henrybothwell/Desktop/skripsie-workspace
stage() { local name="$1"; shift; echo "===== [$(date '+%F %T')] STAGE START: $name ====="; "$@"; local rc=$?; echo "===== [$(date '+%F %T')] STAGE END: $name (exit $rc) ====="; return $rc; }
stage "diagnostics/probe_threshold_rule" "$PY" Run/Quality/diagnostics/probe_threshold_rule.py
stage "diagnostics/gated_selector" "$PY" Run/Quality/diagnostics/gated_selector.py
stage "diagnostics/failure_detector" "$PY" Run/Quality/diagnostics/failure_detector.py
for ds in quality ultraDomain_legal; do stage "run_latency_eval ($ds)" "$PY" Run/Quality/run_latency_eval.py --dataset "$ds"; done
stage "run_latency_eval (musique, limit 500)" "$PY" Run/Quality/run_latency_eval.py --dataset musique --limit 500
stage "make_result_figures" bash -c "cd '$WORKSPACE' && '$PY' tools/make_result_figures.py"
echo "ALL DONE $(date '+%F %T')"
