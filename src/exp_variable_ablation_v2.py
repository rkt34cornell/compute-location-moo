#!/usr/bin/env python3
"""
exp_variable_ablation_v2.py — full vs reduced variable set.

Reuses the legacy reduction (exp_variable_ablation.reduce_workload with
preserve_constraints=True: the original-8 metrics plus every metric that
carries a threshold). Differences from legacy:
  * equal budget of ~4,860 evaluations for every method (legacy: random 8,000,
    MOPSO/NSGA-II 4,860, MOEA/D 9,720);
  * all six main workloads, 20 seeds (11-30), SPEA2 added;
  * feasibility re-checked before HV (infeasible fallback scores 0);
  * current data.json and the constraint modes given on the command line.

Output: out_var_ablation_v2/{runs.csv, summary.json}
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
from exp_variable_ablation import reduce_workload
from run_multiseed_benchmark import WCFG, hv2d, to_min_2d

BUDGET = 60 * 81  # 4,860
CFG = {
    "random": dict(n=BUDGET),
    "mopso": dict(particles=60, iters=80, archive_max=200),
    "nsga2": dict(pop_size=60, generations=80),
    "spea2": dict(pop_size=60, archive_size=60, generations=80),
    "moead": dict(H=8, T=5, generations=BUDGET // 120 - 1),  # 120 x (1 + 39) = 4,800
}


def run(job):
    wl_name, vset, method, seed, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    full = mc.load_workload(Path("data.json"), wl_name)
    wl = full if vset == "full" else reduce_workload(full, preserve_constraints=True)
    mk = [m.key for m in wl.metrics]
    if method == "random":
        rng = np.random.default_rng(seed)
        J = len(wl.tiers)
        Ps = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), BUDGET - J)])
        V = np.array([[mc.aggregate_metrics_linear(wl, p)[k] for k in mk] for p in Ps])
    elif method == "mopso":
        _, V = optimizers_methods.mopso(workload=wl, metric_keys=mk, seed=seed, **CFG[method])
    else:
        _, V = getattr(algorithms_extended, method)(workload=wl, metric_keys=mk, seed=seed, **CFG[method])
    feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
    Vf = V[feas]
    lk, ck, ref = WCFG[wl_name]
    if Vf.shape[0]:
        par = mc.pareto_front_indices(Vf, wl.metrics, mk)
        hv = hv2d(to_min_2d(Vf[par], wl, mk, lk, ck), ref); n = int(len(par))
    else:
        hv, n = 0.0, 0
    return dict(workload=wl_name, variable_set=vset, n_vars=len(mk), method=method, seed=seed,
                hv=float(hv), n_front=n, kept_keys="|".join(mk) if vset == "reduced" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_var_ablation_v2")
    ap.add_argument("--workloads", nargs="+", default=["IDS", "SUNCATCHER", "LEO_OOS", "DEEP_SPACE_CRUISE", "JWST", "RST"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(exist_ok=True)
    jobs = [(w, vs, m, s, args.violation, args.handling) for w in args.workloads
            for vs in ["full", "reduced"] for m in CFG for s in range(11, 11 + args.seeds)]
    jobs.sort(key=lambda j: j[2] != "mopso")
    rows = []
    with Pool(args.jobs) as pool:
        for r in pool.imap_unordered(run, jobs):
            rows.append(r)
            if len(rows) % 50 == 0:
                print(f"{len(rows)}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["workload"], r["variable_set"], r["method"], r["seed"]))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from scipy.stats import wilcoxon
    summ = {}
    for wl in args.workloads:
        summ[wl] = {}
        for m in CFG:
            a = np.array([r["hv"] for r in rows if r["workload"] == wl and r["method"] == m and r["variable_set"] == "full"])
            b = np.array([r["hv"] for r in rows if r["workload"] == wl and r["method"] == m and r["variable_set"] == "reduced"])
            try:
                p = float(wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0
            except ValueError:
                p = 1.0
            nv = {vs: next(r["n_vars"] for r in rows if r["workload"] == wl and r["variable_set"] == vs) for vs in ["full", "reduced"]}
            summ[wl][m] = {"full_hv_mean": float(a.mean()), "full_hv_std": float(a.std(ddof=1)),
                           "reduced_hv_mean": float(b.mean()), "reduced_hv_std": float(b.std(ddof=1)),
                           "delta_hv_mean": float((b - a).mean()), "wilcoxon_p": p,
                           "n_vars_full": nv["full"], "n_vars_reduced": nv["reduced"]}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(args), "budget": BUDGET, "cfg": CFG}, indent=2))
    print("ablation v2 complete")


if __name__ == "__main__":
    main()
