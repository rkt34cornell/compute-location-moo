#!/usr/bin/env python3
"""
exp_roman_arrival_sweep.py — Sweep arrival rate λ on ROMAN_TRANSIENT for all
three ODC variants under the nonlinear evaluator. Reveals how the optimal
policy shifts as transient detection load increases (e.g., supernova-rich
weeks). Expected: TDC-dominant at low λ → mixed → FC/ODC-dominant at burst.

Output: out_roman_arrival/sweep.csv, sweep.json (figures in make_roman_figures.py).
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
VARIANTS = ["A", "B", "C"]
LK = "Latency p99 (ms)"
CK = "Cost per task (USD)"
HV_REF = np.array([1.0e8, 0.12])

# λ sweep: spans low load (1/day) to extreme burst (~1 task/2s)
LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


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


def main():
    outdir = Path("out_roman_arrival"); outdir.mkdir(exist_ok=True)
    wl_base = load_workload(Path("data.json"), WL_NAME)
    mk = [m.key for m in wl_base.metrics]
    kd = {m.key: m.direction for m in wl_base.metrics}
    base_params = WORKLOAD_NONLINEAR_PARAMS[WL_NAME]

    rows = [["variant", "lambda_tasks_per_s", "hv", "best_lat", "best_cost",
             "FC_frac", "ODC_frac", "GSE_frac", "TDC_frac",
             "n_pareto", "feas_frac_random1k", "rt_s"]]
    sweep = []
    print(f"Sweep λ for {WL_NAME} across variants {VARIANTS}")

    for variant in VARIANTS:
        wl = apply_odc_variant(wl_base, variant)
        for lam in LAMBDAS:
            params = deepcopy(base_params); params.lambda_tasks_per_s = float(lam)
            restore = install_nl(params)
            try:
                t0 = time.perf_counter()
                P, V = mopso(workload=wl, metric_keys=mk,
                             particles=50, iters=70, archive_max=180, seed=21)
                rt = time.perf_counter() - t0

                li, ci = mk.index(LK), mk.index(CK)
                pts = V[:, [li, ci]].copy()
                if kd.get(LK) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
                if kd.get(CK) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
                hv = hv2d(pts, HV_REF)

                # Best singleton (min cost, feasible by latency)
                best_i = -1; best_cost = float("inf")
                for i in range(V.shape[0]):
                    if pts[i, 0] < HV_REF[0] and pts[i, 1] < best_cost:
                        best_cost = float(pts[i, 1]); best_i = i
                if best_i >= 0:
                    p_best = P[best_i]; best_lat = float(pts[best_i, 0])
                else:
                    p_best = P[0]; best_lat = float("nan"); best_cost = float("nan")

                rng = np.random.default_rng(42)
                Ps = rng.dirichlet(np.ones(len(wl.tiers)), size=1000)
                feas = sum(1 for p in Ps
                           if feasibility_and_violation(wl,
                              aggregate_metrics_nonlinear(wl, p, params))[0]) / 1000
            finally:
                restore()

            fracs = {t: float(p_best[i]) for i, t in enumerate(wl.tiers)}
            rows.append([variant, lam, round(hv, 3),
                         round(best_lat, 3), round(best_cost, 4),
                         round(fracs.get("FC", 0), 4), round(fracs.get("ODC", 0), 4),
                         round(fracs.get("GSE", 0), 4), round(fracs.get("TDC", 0), 4),
                         int(V.shape[0]), round(feas, 4), round(rt, 1)])
            sweep.append({"variant": variant, "lambda": lam, "fracs": fracs,
                          "hv": hv, "best_lat": best_lat, "best_cost": best_cost,
                          "feas": feas, "rt_s": rt})

            print(f"  var={variant}  λ={lam:>7.3f}  HV={hv:>8.2f}  "
                  f"FC={fracs.get('FC',0):.0%} ODC={fracs.get('ODC',0):.0%} "
                  f"GSE={fracs.get('GSE',0):.0%} TDC={fracs.get('TDC',0):.0%}  "
                  f"feas={feas:.1%}")

    with open(outdir / "sweep.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "sweep.json").write_text(json.dumps({"data": sweep,
                                                    "lambdas": LAMBDAS,
                                                    "variants": VARIANTS}, indent=2))
    print(f"\nWrote {outdir}/sweep.csv, sweep.json")


if __name__ == "__main__":
    main()
