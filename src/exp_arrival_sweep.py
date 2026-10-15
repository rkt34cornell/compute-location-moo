#!/usr/bin/env python3
"""
exp_arrival_sweep.py — Sweep arrival rate λ on IDS from
light load (~10% utilization at FC) to near-saturation (~90%), under the
nonlinear evaluator. For each λ, run MOPSO and report how the optimal
policy structure shifts toward off-spacecraft tiers as load rises.

Validates that the M/M/1 queuing term in aggregate_metrics_nonlinear actually
drives a policy response, not just a numerical wobble.

Output: out_arrival_sweep/sweep.csv, sweep_report.md, sweep_figure.png
"""
from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moo_core import (
    Direction,
    NonlinearParams,
    WORKLOAD_NONLINEAR_PARAMS,
    aggregate_metrics_nonlinear,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
)
import optimizers_methods
import algorithms_extended
from optimizers_methods import mopso


WL_NAME = "IDS"
LK = "Latency p99 (ms)"
CK = "Cost per task (USD)"
REF = np.array([260.0, 3.2])

# Sweep these λ values (tasks/s). The baseline IDS λ is 500/s.
LAMBDAS = [50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def install_nl_evaluator(wl, params):
    """Monkey-patch both optimizer modules' aggregate_metrics_linear so they
    actually call the nonlinear evaluator. Returns a restore function."""
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
    outdir = Path("out_arrival_sweep"); outdir.mkdir(exist_ok=True)
    wl = load_workload(Path("data.json"), WL_NAME)
    mk = [m.key for m in wl.metrics]
    kd = {m.key: m.direction for m in wl.metrics}
    base_params = WORKLOAD_NONLINEAR_PARAMS[WL_NAME]

    rows = [["lambda_tasks_per_s", "best_hv", "best_lat", "best_cost",
             "FC_frac", "ODC_frac", "GSE_frac", "TDC_frac",
             "n_pareto", "feas_frac_random1k", "runtime_s"]]
    sweep_data = []

    print(f"Sweep on {WL_NAME}: λ ∈ {LAMBDAS}")
    for lam in LAMBDAS:
        params = deepcopy(base_params)
        params.lambda_tasks_per_s = float(lam)

        restore = install_nl_evaluator(wl, params)
        try:
            t0 = time.perf_counter()
            P, V = mopso(workload=wl, metric_keys=mk,
                         particles=60, iters=80, archive_max=200, seed=11)
            rt = time.perf_counter() - t0

            # Re-evaluate under full NL (P was optimised against patched aggregator
            # which already calls NL, so V are already nonlinear values)
            li, ci = mk.index(LK), mk.index(CK)
            pts2 = V[:, [li, ci]].copy()
            if kd.get(LK) == Direction.MAXIMIZE: pts2[:, 0] = -pts2[:, 0]
            if kd.get(CK) == Direction.MAXIMIZE: pts2[:, 1] = -pts2[:, 1]
            hv = hv2d(pts2, REF)

            # Best singleton (min cost feasible)
            best_i = -1; best_cost = float("inf")
            for i in range(V.shape[0]):
                if pts2[i, 1] < best_cost and pts2[i, 0] < REF[0]:
                    best_cost = float(pts2[i, 1]); best_i = i

            if best_i >= 0:
                p_best = P[best_i]
                best_lat = float(pts2[best_i, 0])
            else:
                p_best = P[0]; best_lat = float("nan"); best_cost = float("nan")

            # Random-survey feasibility under this λ
            rng = np.random.default_rng(42)
            Psurv = rng.dirichlet(np.ones(len(wl.tiers)), size=1000)
            feas = sum(1 for p in Psurv if feasibility_and_violation(wl,
                       aggregate_metrics_nonlinear(wl, p, params))[0]) / 1000
        finally:
            restore()

        fracs = {t: float(p_best[i]) for i, t in enumerate(wl.tiers)}
        rows.append([lam, round(hv, 3), round(best_lat, 3), round(best_cost, 4),
                     round(fracs.get("FC", 0), 4), round(fracs.get("ODC", 0), 4),
                     round(fracs.get("GSE", 0), 4), round(fracs.get("TDC", 0), 4),
                     int(V.shape[0]), round(feas, 4), round(rt, 1)])
        sweep_data.append({"lambda": lam, "fracs": fracs, "hv": hv,
                           "best_lat": best_lat, "best_cost": best_cost,
                           "feas": feas})

        print(f"  λ={lam:>7.1f}  HV={hv:>7.2f}  best=(lat={best_lat:.2f},cost={best_cost:.3f})  "
              f"FC={fracs.get('FC',0):.0%} ODC={fracs.get('ODC',0):.0%} "
              f"GSE={fracs.get('GSE',0):.0%} TDC={fracs.get('TDC',0):.0%}  feas={feas:.1%}")

    with open(outdir / "sweep.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "sweep.json").write_text(json.dumps(sweep_data, indent=2))

    # Stacked-area figure: tier fractions vs λ
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    lams = [d["lambda"] for d in sweep_data]
    F = np.array([[d["fracs"].get(t, 0) for d in sweep_data] for t in wl.tiers])
    axes[0].stackplot(lams, F, labels=wl.tiers, alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Best-policy tier fraction")
    axes[0].set_title(f"{WL_NAME}: optimal policy structure vs arrival rate λ")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lams, [d["best_lat"] for d in sweep_data], "o-", label="best lat (ms)")
    axes[1].plot(lams, [d["best_cost"]*100 for d in sweep_data], "s-",
                 label="best cost (USD × 100)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("λ (tasks/s)")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_figure.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {outdir}/sweep.csv, sweep.json, sweep_figure.png")


if __name__ == "__main__":
    main()
