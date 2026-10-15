#!/usr/bin/env bash
# Compute queue (v2), part C: IDS arrival sweep after part B, then regenerate tables/figures.
set -u
PY=/home/tempuser/miniforge3/envs/bsk/bin/python3.10
cd "$(dirname "$0")"
until grep -q QUEUE_V2B_DONE out_queue_v2b.log 2>/dev/null; do sleep 60; done
$PY exp_arrival_sweep_v2.py
$PY make_tables_v2.py
$PY make_figures_v2.py
echo QUEUE_V2C_DONE
