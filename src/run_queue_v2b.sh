#!/usr/bin/env bash
# Compute queue (v2), part B: runs after run_queue_v2.sh has finished.
set -u
PY=/home/tempuser/miniforge3/envs/bsk/bin/python3.10
cd "$(dirname "$0")"
until grep -q QUEUE_V2_DONE out_queue_v2.log 2>/dev/null; do sleep 60; done
$PY exp_convergence_v2.py
$PY exp_nl_sensitivity_v2.py
$PY exp_variable_ablation_v2.py
$PY exp_inner_loop_v2.py
$PY exp_odc_sweep_v2.py
MOO_VIOLATION_MODE=normalized MOO_CONSTRAINT_HANDLING=deb $PY exp_bo_baseline.py --outdir out_bo_v2 --runs IDS_linear IDS_nonlinear
$PY exp_lin_vs_nl_v2.py
$PY analyze_v2.py
$PY make_tables_v2.py
$PY make_figures_v2.py
echo QUEUE_V2B_DONE
