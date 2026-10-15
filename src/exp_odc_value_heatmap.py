#!/usr/bin/env python3
"""
exp_odc_value_heatmap.py — The "value of an L2 companion ODC" experiment.
2D sweep over ODC throughput × ODC TID hardness. For each cell, run MOPSO
on ROMAN_TRANSIENT under the nonlinear evaluator with ODC at TRL=7
(Variant B). Record:
  - mean p_ODC across the Pareto front (i.e. how much routing the optimizer
    gives to ODC)
  - HV achieved
  - whether ODC is "used" (mean p_ODC > 0.05)

The bright region in the heatmap is the answer to "when does building an
L2 companion ODC start to pay off."

Output: out_odc_value/heatmap.csv, heatmap.json
"""
from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from moo_core import (
    Direction,
    Workload,
    WORKLOAD_NONLINEAR_PARAMS,
    aggregate_metrics_nonlinear,
    apply_odc_variant,
    feasibility_and_violation,
    load_workload,
)
import optimizers_methods
import algorithms_extended
from optimizers_methods import mopso


WL_NAME = "ROMAN_TRANSIENT"
LK = "Latency p99 (ms)"
CK = "Cost per task (USD)"
THK = "Throughput (GFLOPS)"
RADK = "Rad dose margin (krad)"
HV_REF = np.array([1.0e8, 0.12])

# Sweep grid
THROUGHPUTS = [100, 300, 1000, 3000, 10000]      # ODC GFLOPS
TID_MARGINS = [10, 25, 50, 100, 200]              # ODC krad lifetime margin

# Reference points for industry programs (rough; for annotation only)
INDUSTRY_REFS = {
    "Project Suncatcher (target)": (3000, 100),   # ML-class GPU + L2-tolerant
    "Starcloud (target)":          (1000, 50),
}


def install_nl(params):
    nl_eval = lambda w, p: aggregate_metrics_nonlinear(w, p, params)
    saved_o = optimizers_methods.aggregate_metrics_linear
    saved_a = algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = nl_eval
    algorithms_extended.aggregate_metrics_linear = nl_eval
    def restore():
        optimizers_methods.aggregate_metrics_linear = saved_o
        algorithms_extended.aggregate_metrics_linear = saved_a
    return restore


def override_odc(wl: Workload, throughput: float, tid: float) -> Workload:
    new_tv = {t: dict(wl.tier_values[t]) for t in wl.tiers}
    new_tv["ODC"][THK]  = float(throughput)
    new_tv["ODC"][RADK] = float(tid)
    return Workload(name=wl.name + f"_T{throughput}_R{tid}",
                    tiers=list(wl.tiers), metrics=wl.metrics,
                    tier_values=new_tv, thresholds=wl.thresholds)


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def main():
    outdir = Path("out_odc_value"); outdir.mkdir(exist_ok=True)
    wl_base = load_workload(Path("data.json"), WL_NAME)
    # Use Variant B as the framing: ODC is assumed mature TRL=7
    wl_B = apply_odc_variant(wl_base, "B")
    mk = [m.key for m in wl_B.metrics]
    kd = {m.key: m.direction for m in wl_B.metrics}
    params = WORKLOAD_NONLINEAR_PARAMS[WL_NAME]

    odc_idx = wl_B.tiers.index("ODC")
    n_thr = len(THROUGHPUTS); n_tid = len(TID_MARGINS)

    p_odc_grid     = np.zeros((n_thr, n_tid))
    hv_grid        = np.zeros((n_thr, n_tid))
    used_grid      = np.zeros((n_thr, n_tid), dtype=int)
    rows = [["throughput_GFLOPS", "tid_krad", "mean_p_odc",
             "max_p_odc", "hv", "n_pareto", "odc_used",
             "best_cost", "rt_s"]]
    print(f"ODC value heatmap: {n_thr}x{n_tid} = {n_thr*n_tid} cells "
          f"on {WL_NAME} (Variant B)")

    t_all = time.perf_counter()
    for i, thr in enumerate(THROUGHPUTS):
        for j, tid in enumerate(TID_MARGINS):
            wl = override_odc(wl_B, thr, tid)
            restore = install_nl(params)
            try:
                t0 = time.perf_counter()
                P, V = mopso(workload=wl, metric_keys=mk,
                             particles=50, iters=60, archive_max=150, seed=31)
                rt = time.perf_counter() - t0

                if P.shape[0] == 0:
                    p_odc_mean = 0.0; p_odc_max = 0.0
                else:
                    p_odc_mean = float(P[:, odc_idx].mean())
                    p_odc_max  = float(P[:, odc_idx].max())

                li, ci = mk.index(LK), mk.index(CK)
                pts = V[:, [li, ci]].copy()
                if kd.get(LK) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
                if kd.get(CK) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
                hv = hv2d(pts, HV_REF)

                best_cost = float(pts[:, 1].min()) if pts.shape[0] else float("nan")
            finally:
                restore()

            used = int(p_odc_mean > 0.05)
            p_odc_grid[i, j] = p_odc_mean
            hv_grid[i, j]    = hv
            used_grid[i, j]  = used
            rows.append([thr, tid, round(p_odc_mean, 4), round(p_odc_max, 4),
                         round(hv, 3), int(P.shape[0]), used,
                         round(best_cost, 4), round(rt, 1)])
            print(f"  thr={thr:>6d} GFLOPS  tid={tid:>4d} krad  "
                  f"p_ODC_mean={p_odc_mean:.3f}  HV={hv:>8.2f}  "
                  f"used={'Y' if used else 'N'}  t={rt:.1f}s")

    elapsed = time.perf_counter() - t_all
    print(f"\nTotal wall time: {elapsed/60:.1f} min")

    np.savez(outdir / "heatmap_grids.npz",
             throughputs=np.array(THROUGHPUTS), tids=np.array(TID_MARGINS),
             p_odc=p_odc_grid, hv=hv_grid, used=used_grid)
    with open(outdir / "heatmap.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "heatmap.json").write_text(json.dumps({
        "throughputs_gflops": THROUGHPUTS,
        "tid_krad":           TID_MARGINS,
        "p_odc_mean_grid":    p_odc_grid.tolist(),
        "hv_grid":            hv_grid.tolist(),
        "used_grid":          used_grid.tolist(),
        "industry_refs":      INDUSTRY_REFS,
        "wall_time_min":      elapsed / 60.0,
    }, indent=2))

    # Frontier analysis: min throughput at any TID s.t. p_ODC>0.1
    print("\nODC viability frontier:")
    for j, tid in enumerate(TID_MARGINS):
        col = p_odc_grid[:, j]
        viable = [(THROUGHPUTS[i], v) for i, v in enumerate(col) if v > 0.10]
        if viable:
            min_thr = min(t for t, _ in viable)
            print(f"  TID={tid:>4d} krad  →  min ODC throughput for p_ODC>0.10: "
                  f"{min_thr} GFLOPS")
        else:
            print(f"  TID={tid:>4d} krad  →  ODC never used (>0.10) at any throughput")

    print(f"\nWrote {outdir}/heatmap.csv, heatmap.json, heatmap_grids.npz")


if __name__ == "__main__":
    main()
