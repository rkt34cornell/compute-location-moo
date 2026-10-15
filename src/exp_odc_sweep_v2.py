#!/usr/bin/env python3
"""
exp_odc_sweep_v2.py — multi-seed ROMAN ODC throughput x radiation-margin sweep.

Same grid and per-run configuration as exp_odc_value_heatmap.py
(ROMAN_TRANSIENT, Variant B, nonlinear evaluator, MOPSO 50 particles x 60
iterations, archive 150). Note the swept quantity is the ODC tier's
"Rad dose margin (krad)" (a margin, not a TID tolerance). Differences: 10 seeds
per cell (31-40) instead of seed 31 only, full fronts saved, current code and
the constraint modes given on the command line.

Per cell it reports the distribution of p_ODC over the pooled Pareto points of
all seeds (mean, SD, median, P5, P25, P75, P95, min, max, share of points in
[0.20, 0.30], share < 0.05) and the across-seed SD of the per-seed mean. Two
reference distributions test whether p_ODC is concentrated near 0.25:
  * uniform Dirichlet(1,1,1,1) marginal = Beta(1,3) (mean 0.25, SD 0.194);
  * p_ODC of uniformly sampled policies that are FEASIBLE in that cell
    (20,000 Dirichlet draws filtered by the nonlinear constraints).
Two-sample Kolmogorov-Smirnov statistics against the feasible-uniform reference
are reported.

Output: out_odc_sweep_v2/{fronts/*.npy, cells.csv, summary.json}
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moo_core as mc
import optimizers_methods
import algorithms_extended
from exp_odc_value_heatmap import THROUGHPUTS, TID_MARGINS, override_odc, WL_NAME, HV_REF
from run_multiseed_benchmark import hv2d, to_min_2d

SEEDS = list(range(31, 41))  # default; --seeds N uses 31..31+N-1


def base_workload():
    return mc.apply_odc_variant(mc.load_workload(Path("data.json"), WL_NAME), "B")


def run(job):
    thr, tid, seed, out, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    wl = override_odc(base_workload(), thr, tid)
    mk = [m.key for m in wl.metrics]
    params = mc.WORKLOAD_NONLINEAR_PARAMS[WL_NAME]
    ev = lambda w, p: mc.aggregate_metrics_nonlinear(w, p, params)
    saved = optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = ev
    algorithms_extended.aggregate_metrics_linear = ev
    try:
        P, V = optimizers_methods.mopso(workload=wl, metric_keys=mk, particles=50, iters=60,
                                        archive_max=150, seed=seed)
    finally:
        optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear = saved
    feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
    P, V = P[feas], V[feas]
    np.save(Path(out) / "fronts" / f"P_T{thr}_R{tid}_seed{seed}.npy", P)
    hv = hv2d(to_min_2d(V, wl, mk, "Latency p99 (ms)", "Cost per task (USD)"), HV_REF) if V.shape[0] else 0.0
    return thr, tid, seed, float(hv), int(P.shape[0])


def feasible_uniform_reference(thr, tid, n=20000, seed=0):
    wl = override_odc(base_workload(), thr, tid)
    params = mc.WORKLOAD_NONLINEAR_PARAMS[WL_NAME]
    rng = np.random.default_rng(seed)
    Ps = rng.dirichlet(np.ones(len(wl.tiers)), n)
    keep = [p for p in Ps if mc.feasibility_and_violation(wl, mc.aggregate_metrics_nonlinear(wl, p, params))[0]]
    return np.array(keep)[:, wl.tiers.index("ODC")]


def describe(x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(q[2]),
            "p5": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p95": float(q[4]),
            "min": float(x.min()), "max": float(x.max()),
            "share_in_0.20_0.30": float(np.mean((x >= 0.2) & (x <= 0.3))),
            "share_below_0.05": float(np.mean(x < 0.05))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_odc_sweep_v2")
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--grid", choices=["full", "smoke"], default="full")
    args = ap.parse_args()
    global SEEDS, THROUGHPUTS, TID_MARGINS
    SEEDS = list(range(31, 31 + args.seeds))
    if args.grid == "smoke":
        THROUGHPUTS, TID_MARGINS = THROUGHPUTS[:1], TID_MARGINS[:2]
    out = Path(args.outdir); (out / "fronts").mkdir(parents=True, exist_ok=True)
    jobs = [(t, r, s, str(out), args.violation, args.handling) for t in THROUGHPUTS for r in TID_MARGINS for s in SEEDS
            if not (out / "fronts" / f"P_T{t}_R{r}_seed{s}.npy").exists()]
    with Pool(args.jobs) as pool:
        runs = list(pool.imap_unordered(run, jobs))
    from scipy.stats import ks_2samp, beta
    odc = base_workload().tiers.index("ODC")
    rows, cells = [], {}
    pooled_all, ref_all = [], []
    for t in THROUGHPUTS:
        for r in TID_MARGINS:
            per_seed = [np.load(out / "fronts" / f"P_T{t}_R{r}_seed{s}.npy")[:, odc] for s in SEEDS]
            x = np.concatenate(per_seed)
            ref = feasible_uniform_reference(t, r)
            seed_means = np.array([p.mean() for p in per_seed if p.size])
            d = describe(x)
            d.update({"seed_mean_mean": float(seed_means.mean()), "seed_mean_sd": float(seed_means.std(ddof=1)),
                      "hv_mean": float(np.mean([h for (tt, rr, s, h, n) in runs if (tt, rr) == (t, r)])),
                      "feasible_uniform_ref": describe(ref),
                      "ks_vs_feasible_uniform": float(ks_2samp(x, ref).statistic),
                      "ks_vs_beta13": float(ks_2samp(x, beta(1, 3).rvs(20000, random_state=0)).statistic)})
            cells[f"T{t}_R{r}"] = d
            pooled_all.append(x); ref_all.append(ref)
            rows.append([t, r, d["n"], d["mean"], d["sd"], d["median"], d["p5"], d["p25"], d["p75"], d["p95"],
                         d["min"], d["max"], d["share_in_0.20_0.30"], d["share_below_0.05"],
                         d["seed_mean_sd"], d["feasible_uniform_ref"]["mean"], d["feasible_uniform_ref"]["sd"],
                         d["ks_vs_feasible_uniform"], d["hv_mean"]])
    pooled = np.concatenate(pooled_all)
    summary = {"cells": cells, "pooled": describe(pooled), "pooled_feasible_uniform": describe(np.concatenate(ref_all)),
               "beta13": {"mean": 0.25, "sd": float(np.sqrt(3 / 80)), "median": float(1 - 0.5 ** (1 / 3)),
                          "p5": float(1 - 0.95 ** (1 / 3)), "p95": float(1 - 0.05 ** (1 / 3))},
               "range_of_cell_means": [float(min(c["mean"] for c in cells.values())),
                                       float(max(c["mean"] for c in cells.values()))],
               "seeds": SEEDS}
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["throughput_gflops", "rad_margin_krad", "n_points", "mean", "sd", "median", "p5", "p25", "p75",
                    "p95", "min", "max", "share_0.20_0.30", "share_below_0.05", "sd_of_seed_means",
                    "feasible_uniform_mean", "feasible_uniform_sd", "ks_vs_feasible_uniform", "hv_mean"])
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("odc sweep v2 complete")


if __name__ == "__main__":
    main()
