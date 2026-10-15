#!/usr/bin/env bash
# Compute queue (v2), part D: data-driven reports after everything else.
set -u
PY=/home/tempuser/miniforge3/envs/bsk/bin/python3.10
cd "$(dirname "$0")"
until grep -q QUEUE_V2C_DONE out_queue_v2c.log 2>/dev/null; do sleep 60; done
$PY -m unittest tests/test_v2.py
echo QUEUE_V2D_DONE
