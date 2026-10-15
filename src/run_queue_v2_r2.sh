#!/usr/bin/env bash
# Second pass: rerun everything affected by the mission-lifetime-dose radiation
# scale (IDS, SUNCATCHER, LEO_OOS), the ROMAN sweeps, then analysis/tables/figures/reports.
set -u
PY=/home/tempuser/miniforge3/envs/bsk/bin/python3.10
cd "$(dirname "$0")"
$PY run_benchmark_v2.py --suite main    --outdir out_benchmark_v2
$PY run_benchmark_v2.py --suite archive --outdir out_archive_v2 --workloads SUNCATCHER IDS JWST
$PY exp_convergence_v2.py
$PY exp_nl_sensitivity_v2.py
$PY exp_variable_ablation_v2.py
$PY exp_inner_loop_v2.py
MOO_VIOLATION_MODE=normalized MOO_CONSTRAINT_HANDLING=deb $PY exp_bo_baseline.py --outdir out_bo_v2 --runs IDS_linear IDS_nonlinear
$PY exp_arrival_sweep_v2.py
$PY exp_lin_vs_nl_v2.py
$PY exp_roman_sweeps_v2.py
$PY analyze_v2.py
$PY make_tables_v2.py
$PY make_figures_v2.py
$PY -m unittest tests/test_v2.py
echo QUEUE_R2_DONE
