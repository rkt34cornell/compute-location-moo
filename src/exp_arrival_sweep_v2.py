#!/usr/bin/env python3
"""
exp_arrival_sweep_v2.py — IDS arrival-rate sweep.

Same lambda grid and MOPSO configuration as exp_arrival_sweep.py (IDS, nonlinear
evaluator, 60 particles x 80 iterations, archive 200). Differences: current
data.json (FC latency 6 ms / jitter 2 ms), 10 seeds (11-20) instead of one,
feasibility re-checked before HV, constraint modes from the command line, and
random-policy feasibility from 20,000 samples.

Output: out_arrival_sweep_v2/{runs.csv, summary.json}
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moo_core as mc
import optimizers_methods
import algorithms_extended
from exp_arrival_sweep import LAMBDAS
from run_multiseed_benchmark import WCFG, hv2d, to_min_2d


def run(job):
    lam, seed, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    wl = mc.load_workload(Path("data.json"), "IDS")
    mk = [m.key for m in wl.metrics]
    params = deepcopy(mc.WORKLOAD_NONLINEAR_PARAMS["IDS"]); params.lambda_tasks_per_s = float(lam)
    ev = lambda w, p: mc.aggregate_metrics_nonlinear(w, p, params)
    saved = optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = ev
    algorithms_extended.aggregate_metrics_linear = ev
    try:
        P, V = optimizers_methods.mopso(workload=wl, metric_keys=mk, particles=60, iters=80, archive_max=200, seed=seed)
    finally:
        optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear = saved
    feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
    lk, ck, ref = WCFG["IDS"]
    P, V = P[feas], V[feas]
    if V.shape[0] == 0:
        return dict(lam=lam, seed=seed, hv=0.0, n_front=0, best_cost=np.nan, best_lat=np.nan,
                    FC=np.nan, ODC=np.nan, GSE=np.nan, TDC=np.nan)
    pts = to_min_2d(V, wl, mk, lk, ck)
    i = int(np.argmin(pts[:, 1]))
    return dict(lam=lam, seed=seed, hv=float(hv2d(pts, ref)), n_front=int(V.shape[0]),
                best_cost=float(pts[i, 1]), best_lat=float(pts[i, 0]),
                **{t: float(P[i, j]) for j, t in enumerate(wl.tiers)})


def feas_rate(lam, n=20000):
    wl = mc.load_workload(Path("data.json"), "IDS")
    params = deepcopy(mc.WORKLOAD_NONLINEAR_PARAMS["IDS"]); params.lambda_tasks_per_s = float(lam)
    Ps = np.random.default_rng(42).dirichlet(np.ones(4), n)
    return float(np.mean([mc.feasibility_and_violation(wl, mc.aggregate_metrics_nonlinear(wl, p, params))[0] for p in Ps]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_arrival_sweep_v2")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(exist_ok=True)
    jobs = [(l, s, args.violation, args.handling) for l in LAMBDAS for s in range(11, 11 + args.seeds)]
    with Pool(8) as pool:
        rows = sorted(pool.map(run, jobs), key=lambda r: (r["lam"], r["seed"]))
        fr = dict(zip(LAMBDAS, pool.map(feas_rate, LAMBDAS)))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = []
    for l in LAMBDAS:
        R = [r for r in rows if r["lam"] == l]
        h = np.array([r["hv"] for r in R])
        summ.append({"lambda": l, "rho_FC_at_pFC_1": l / 400.0, "hv_mean": float(h.mean()), "hv_sd": float(h.std(ddof=1)),
                     "random_feasible_pct": 100 * fr[l],
                     **{f"{t}_frac_of_min_cost_policy_mean": float(np.nanmean([r[t] for r in R])) for t in ["FC", "ODC", "GSE", "TDC"]}})
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print("arrival sweep v2 complete")


if __name__ == "__main__":
    main()
