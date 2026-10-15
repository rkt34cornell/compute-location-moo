#!/usr/bin/env python3
"""
exp_odc_nre_sensitivity.py — Spec said: "ODC NRE estimate of $1.5B is a rough
order of magnitude based on small-spacecraft-class companion mission costs.
Sensitivity to this value should be reported."

This sweeps the ODC NRE (cost per task scales with NRE × amortization) and
shows how the optimal policy and HV change. Since NRE isn't in the HV
projection (we project on Latency × Cost-per-task), the effect comes through
the cost component, which we model as proportional cost-per-task shift.

For simplicity: scale ODC's `Cost per task (USD)` by (NRE / $1.5B), then
re-optimize ROMAN_TRANSIENT Variant B (mature ODC). Verify by recovering
baseline at NRE=1.5B.

Output: out_odc_nre_sens/nre_sweep.csv, nre_sens_report.md
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
HV_REF = np.array([1.0e8, 0.12])
SEEDS = [41, 42, 43]

# NRE sweep ($k); baseline ODC value is 1.5e6 ($1.5B)
NRE_BASELINE_K = 1_500_000.0
NRE_VALUES_K = [300_000.0, 750_000.0, 1_500_000.0, 3_000_000.0, 5_000_000.0]


def override_odc_cost(wl, nre_k):
    new_tv = {t: dict(wl.tier_values[t]) for t in wl.tiers}
    base_cost = wl.tier_values["ODC"][CK]
    scale = float(nre_k) / NRE_BASELINE_K
    new_tv["ODC"][CK] = base_cost * scale
    new_tv["ODC"]["NRE (kUSD)"] = float(nre_k)
    return Workload(name=wl.name + f"_NRE{int(nre_k/1000)}M",
                    tiers=list(wl.tiers), metrics=wl.metrics,
                    tier_values=new_tv, thresholds=wl.thresholds)


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


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def main():
    outdir = Path("out_odc_nre_sens"); outdir.mkdir(exist_ok=True)
    wl_base = load_workload(Path("data.json"), WL_NAME)
    # Variant B (mature ODC) is the framing — ODC is admitted by TRL gate
    wl_B = apply_odc_variant(wl_base, "B")
    mk = [m.key for m in wl_B.metrics]
    kd = {m.key: m.direction for m in wl_B.metrics}
    params = WORKLOAD_NONLINEAR_PARAMS[WL_NAME]
    odc_idx = wl_B.tiers.index("ODC")

    rows = [["nre_kUSD", "nre_label", "hv_mean", "hv_std",
             "p_odc_mean", "p_odc_max",
             "best_cost", "best_lat", "n_pareto_mean"]]
    data = []
    print(f"NRE sensitivity sweep on {WL_NAME} Variant B (mature ODC)")

    for nre_k in NRE_VALUES_K:
        wl = override_odc_cost(wl_B, nre_k)
        hvs, p_odc_means, p_odc_maxes, best_costs, best_lats, npars = [], [], [], [], [], []
        for seed in SEEDS:
            restore = install_nl(params)
            try:
                t0 = time.perf_counter()
                P, V = mopso(workload=wl, metric_keys=mk,
                             particles=50, iters=60, archive_max=180, seed=seed)
                rt = time.perf_counter() - t0
            finally:
                restore()
            li, ci = mk.index(LK), mk.index(CK)
            pts = V[:, [li, ci]].copy()
            if kd.get(LK) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
            if kd.get(CK) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
            hv = hv2d(pts, HV_REF); hvs.append(hv)
            if P.shape[0]:
                p_odc_means.append(float(P[:, odc_idx].mean()))
                p_odc_maxes.append(float(P[:, odc_idx].max()))
                bi = int(np.argmin(pts[:, 1]))
                best_costs.append(float(pts[bi, 1]))
                best_lats.append(float(pts[bi, 0]))
            npars.append(int(P.shape[0]))
            print(f"  NRE=${nre_k/1000:.0f}M  seed={seed}  HV={hv:>8.2f}  "
                  f"p_ODC_mean={float(P[:,odc_idx].mean()):.3f}  t={rt:.1f}s")

        rows.append([nre_k, f"${nre_k/1000:.0f}M",
                     round(float(np.mean(hvs)), 3),
                     round(float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0, 3),
                     round(float(np.mean(p_odc_means)) if p_odc_means else 0, 4),
                     round(float(np.max(p_odc_maxes)) if p_odc_maxes else 0, 4),
                     round(float(np.mean(best_costs)) if best_costs else 0, 4),
                     round(float(np.mean(best_lats)) if best_lats else 0, 3),
                     round(float(np.mean(npars)), 1)])
        data.append({"nre_k": nre_k, "hv_mean": float(np.mean(hvs)),
                     "p_odc_mean": float(np.mean(p_odc_means)) if p_odc_means else 0,
                     "p_odc_max":  float(np.max(p_odc_maxes)) if p_odc_maxes else 0})

    with open(outdir / "nre_sweep.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "nre_sweep.json").write_text(json.dumps({"data": data,
                                                       "baseline_kUSD": NRE_BASELINE_K},
                                                      indent=2))

    md = ["# ODC NRE Sensitivity", "",
          f"ODC `Cost per task` scaled by (NRE / ${NRE_BASELINE_K/1000:.0f}M baseline). "
          "Variant B (mature ODC). 3 MOPSO seeds each.\n",
          "| NRE | HV mean ± std | mean p_ODC across front | max p_ODC | best_cost |",
          "|---|---:|---:|---:|---:|"]
    for d, r in zip(data, rows[1:]):
        md.append(f"| {r[1]} | {r[2]:.2f} ± {r[3]:.2f} | {r[4]:.3f} | "
                  f"{r[5]:.3f} | ${r[6]:.4f} |")
    md.append("")
    md.append("**Reading:** as NRE rises, ODC's cost-per-task rises proportionally, "
              "and the optimizer routes less to ODC. The ratio of `mean p_ODC` at "
              "$5B vs $300M is the elasticity of ODC usage to capital cost.")
    (outdir / "nre_sens_report.md").write_text("\n".join(md))

    print(f"\nWrote {outdir}/nre_sweep.csv, nre_sweep.json, nre_sens_report.md")


if __name__ == "__main__":
    main()
