#!/usr/bin/env python3
"""
exp_nonlinear_sensitivity.py — Perturb each of the
hard-coded nonlinear coefficients by +25% and -25% and measure the change
in best-policy HV under MOPSO. If the Pareto front is unstable to small
perturbations of these constants, the nonlinear model is brittle.

Coefficients perturbed (defined on NonlinearParams in moo_core.py):
  lambda_tasks_per_s        — arrival rate
  compute_gflops_per_task   — compute intensity
  queue_max_utilization     — ρ cap
  contact_sat_exponent      — latency blowup exponent
  link_setup_energy_fraction— fixed setup overhead fraction
  thermal_power_exponent    — superlinear heat exponent
  thermal_coupling_coeff    — heat-to-margin coupling
  link_congestion_exponent  — availability falloff
  sdc_compound_factor       — per-tier SDC multiplier

Output: out_nl_sensitivity/sensitivity.csv, sensitivity_report.md, tornado.png
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
)
import optimizers_methods
import algorithms_extended
from optimizers_methods import mopso


WL_NAME = "IDS"          # primary workload for the sensitivity
LK = "Latency p99 (ms)"
CK = "Cost per task (USD)"
REF = np.array([260.0, 3.2])

PERTURB_FRAC = 0.25
COEFFS = [
    "lambda_tasks_per_s",
    "compute_gflops_per_task",
    "queue_max_utilization",
    "contact_sat_exponent",
    "link_setup_energy_fraction",
    "thermal_power_exponent",
    "thermal_coupling_coeff",
    "link_congestion_exponent",
    "sdc_compound_factor",
]


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def install_nl(wl, params):
    nl_eval = lambda w, p: aggregate_metrics_nonlinear(w, p, params)
    saved_o = optimizers_methods.aggregate_metrics_linear
    saved_a = algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = nl_eval
    algorithms_extended.aggregate_metrics_linear = nl_eval
    def restore():
        optimizers_methods.aggregate_metrics_linear = saved_o
        algorithms_extended.aggregate_metrics_linear = saved_a
    return restore


def run_mopso_hv(wl, mk, params, kd, seed=11):
    restore = install_nl(wl, params)
    try:
        P, V = mopso(workload=wl, metric_keys=mk,
                     particles=60, iters=60, archive_max=200, seed=seed)
        li, ci = mk.index(LK), mk.index(CK)
        pts = V[:, [li, ci]].copy()
        if kd.get(LK) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
        if kd.get(CK) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
        return hv2d(pts, REF), int(P.shape[0])
    finally:
        restore()


def main():
    outdir = Path("out_nl_sensitivity"); outdir.mkdir(exist_ok=True)
    wl = load_workload(Path("data.json"), WL_NAME)
    mk = [m.key for m in wl.metrics]
    kd = {m.key: m.direction for m in wl.metrics}
    base_params = WORKLOAD_NONLINEAR_PARAMS[WL_NAME]

    # Baseline
    print(f"Baseline MOPSO under nonlinear evaluator ({WL_NAME})...")
    t0 = time.perf_counter()
    base_hv, base_n = run_mopso_hv(wl, mk, base_params, kd, seed=11)
    print(f"  baseline HV={base_hv:.3f}  n_pareto={base_n}  t={time.perf_counter()-t0:.1f}s")

    rows = [["coefficient", "baseline_val", "perturbed_val", "direction",
             "perturbed_hv", "delta_hv", "delta_pct"]]
    summary = {"baseline_hv": base_hv, "perturbations": []}

    for coef in COEFFS:
        base_val = getattr(base_params, coef)
        for direction, factor in [("-25%", 1.0 - PERTURB_FRAC),
                                  ("+25%", 1.0 + PERTURB_FRAC)]:
            params = deepcopy(base_params)
            new_val = float(base_val) * factor
            setattr(params, coef, new_val)
            t0 = time.perf_counter()
            hv, n = run_mopso_hv(wl, mk, params, kd, seed=11)
            delta = hv - base_hv
            pct = 100.0 * delta / base_hv if base_hv > 0 else 0.0
            print(f"  {coef:<28s} {direction:>5s}  HV={hv:>7.3f}  Δ={delta:>+7.3f} "
                  f"({pct:>+6.2f}%)  t={time.perf_counter()-t0:.0f}s")
            rows.append([coef, base_val, new_val, direction,
                         round(hv, 4), round(delta, 4), round(pct, 3)])
            summary["perturbations"].append({
                "coefficient": coef, "direction": direction,
                "baseline_val": float(base_val), "perturbed_val": new_val,
                "hv": hv, "delta_hv": delta, "delta_pct": pct,
            })

    with open(outdir / "sensitivity.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "sensitivity.json").write_text(json.dumps(summary, indent=2))

    # Tornado plot: for each coefficient, show -25% and +25% delta-HV
    coef_to_d = {c: {} for c in COEFFS}
    for p in summary["perturbations"]:
        coef_to_d[p["coefficient"]][p["direction"]] = p["delta_pct"]

    # Sort by max |delta|
    order = sorted(COEFFS,
                   key=lambda c: max(abs(coef_to_d[c].get("-25%", 0)),
                                     abs(coef_to_d[c].get("+25%", 0))),
                   reverse=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(order))
    minus = [coef_to_d[c].get("-25%", 0) for c in order]
    plus  = [coef_to_d[c].get("+25%", 0) for c in order]
    ax.barh(y_pos - 0.18, minus, height=0.36, color="#e76f51", label="-25%")
    ax.barh(y_pos + 0.18, plus,  height=0.36, color="#2a9d8f", label="+25%")
    ax.set_yticks(y_pos); ax.set_yticklabels(order)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Δ HV vs baseline (%)")
    ax.set_title(f"{WL_NAME}: nonlinear coefficient sensitivity (±25%)\n"
                 f"baseline HV = {base_hv:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(outdir / "tornado.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {outdir}/sensitivity.csv, sensitivity.json, tornado.png")


if __name__ == "__main__":
    main()
