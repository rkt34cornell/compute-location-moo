#!/bin/bash
# Runs every experiment that uses the nonlinear evaluator (a destination at
# utilization rho_j >= 1 makes a policy infeasible, stable queues use the exact
# M/M/1 wait), then the analysis, tables, figures, and unit tests. Linear-only
# experiments (E2-E6, E8 ablation, E14) are run by run_queue_v2.sh.
set -u
PY=/home/tempuser/miniforge3/envs/bsk/bin/python3.10
cd "$(dirname "$0")"
step() { echo "=== $(date '+%F %T') $*"; }
step E1;            $PY exp_feasibility_v2.py
step E9;            $PY run_benchmark_v2.py --suite roman --outdir out_roman_v2
step E10;           $PY exp_odc_sweep_v2.py
step E11;           $PY exp_roman_sweeps_v2.py
step E12-validity;  $PY exp_lin_vs_nl_v2.py
step E12-innerloop; $PY exp_inner_loop_v2.py
step E13;           $PY exp_nl_sensitivity_v2.py
step E13-arrival;   $PY exp_arrival_sweep_v2.py
step E7;            MOO_VIOLATION_MODE=normalized MOO_CONSTRAINT_HANDLING=deb $PY exp_bo_baseline.py --outdir out_bo_v2 --runs IDS_linear IDS_nonlinear
step analyze;       $PY analyze_v2.py
step tables;        $PY make_tables_v2.py && $PY make_quality_summary_v2.py
step figures;       $PY make_figures_v2.py
step tests;         $PY -m unittest tests/test_v2.py
step QUEUE_STABILITY_DONE
