#!/usr/bin/env bash
# Compute queue (v2). Outputs only to *_v2 directories.
set -u
PY=/home/tempuser/miniforge3/envs/bsk/bin/python3.10   # env that reproduces legacy runs bit-for-bit
cd "$(dirname "$0")"
$PY run_benchmark_v2.py --suite main    --outdir out_benchmark_v2
$PY run_benchmark_v2.py --suite roman   --outdir out_roman_v2
$PY run_benchmark_v2.py --suite archive --outdir out_archive_v2 --workloads SUNCATCHER IDS JWST
echo QUEUE_V2_DONE
