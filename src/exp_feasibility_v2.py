#!/usr/bin/env python3
"""
exp_feasibility_v2.py — random-policy feasibility density.

For every workload (and ROMAN ODC variant) draws N policies uniformly from the
simplex (Dirichlet(1,...,1), seed 12345) and reports the fraction that satisfy
every hard threshold under the linear and the nonlinear evaluator, with Wilson
95% confidence intervals. Workloads are then sorted by linear feasibility.

Output: out_feasibility_v2/{feasibility.csv, feasibility.json}
"""
from __future__ import annotations

import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moo_core as mc

N = 100_000
CASES = [(w, None) for w in ["SUNCATCHER", "LEO_OOS", "IDS", "DEEP_SPACE_CRUISE", "JWST", "RST"]] + \
        [(w, v) for w in ["ROMAN_TRANSIENT", "ROMAN_TRANSIENT_FOLLOWUP"] for v in ["A", "B", "C"]]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def run(case):
    wl_name, var = case
    wl = mc.load_workload(Path("data.json"), wl_name)
    if var:
        wl = mc.apply_odc_variant(wl, var)
    params = mc.WORKLOAD_NONLINEAR_PARAMS.get(wl_name, mc.NonlinearParams())
    Ps = np.random.default_rng(12345).dirichlet(np.ones(len(wl.tiers)), N)
    k_lin = sum(mc.feasibility_and_violation(wl, mc.aggregate_metrics_linear(wl, p))[0] for p in Ps)
    k_nl = sum(mc.feasibility_and_violation(wl, mc.aggregate_metrics_nonlinear(wl, p, params))[0] for p in Ps)
    lo1, hi1 = wilson(k_lin, N); lo2, hi2 = wilson(k_nl, N)
    return {"workload": wl_name, "variant": var, "J": len(wl.tiers), "n": N,
            "linear_pct": 100 * k_lin / N, "linear_ci95": [100 * lo1, 100 * hi1],
            "nonlinear_pct": 100 * k_nl / N, "nonlinear_ci95": [100 * lo2, 100 * hi2]}


def main():
    out = Path("out_feasibility_v2"); out.mkdir(exist_ok=True)
    with Pool(8) as pool:
        res = pool.map(run, CASES)
    res.sort(key=lambda r: r["linear_pct"])
    with open(out / "feasibility.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_linear", "workload", "variant", "J", "n_samples", "linear_pct", "linear_ci_lo",
                    "linear_ci_hi", "nonlinear_pct", "nonlinear_ci_lo", "nonlinear_ci_hi"])
        for i, r in enumerate(res, 1):
            w.writerow([i, r["workload"], r["variant"] or "", r["J"], r["n"], round(r["linear_pct"], 4),
                        round(r["linear_ci95"][0], 4), round(r["linear_ci95"][1], 4), round(r["nonlinear_pct"], 4),
                        round(r["nonlinear_ci95"][0], 4), round(r["nonlinear_ci95"][1], 4)])
    (out / "feasibility.json").write_text(json.dumps(res, indent=2))
    for r in res:
        print(f"{r['workload']:<26s}{(r['variant'] or ''):<3s} lin {r['linear_pct']:7.3f}%  nl {r['nonlinear_pct']:7.3f}%")


if __name__ == "__main__":
    main()
