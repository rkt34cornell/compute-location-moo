#!/usr/bin/env python3
"""
exp_inner_loop_v2.py — inner-loop nonlinear optimization versus the vertex-proxy.

inner_loop : the optimizer calls aggregate_metrics_nonlinear on every candidate.
proxy      : the optimizer runs on a workload whose tier values are the
             nonlinear values at the pure-tier vertices (linear interpolation
             in between; run_nonlinear_benchmark._make_nl_workload_proxy); its
             returned front is re-evaluated with the full nonlinear evaluator,
             filtered for feasibility and Pareto-filtered.

Both modes use the SAME seeds (11-20) and the same configuration
(60 x 81 = 4,860 evaluations). The legacy table instead compared against a
single-seed (seed 7) Phase-2 proxy run with a different budget.

Evaluator variants: "eight_effect" (as defined in moo_core) and
"seven_effect" (Effect 8 eclipse gating disabled: eclipse_duty_cycle = {}).

Output: out_inner_loop_v2/{runs.csv, summary.json}
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
from run_multiseed_benchmark import WCFG, hv2d, to_min_2d
from run_nonlinear_benchmark import _make_nl_workload_proxy
from exp_inner_loop_nonlinear import HV_KEYS as INNER_HV_KEYS  # legacy table's reference points

CFG = {"mopso": dict(particles=60, iters=80, archive_max=200),
       "nsga2": dict(pop_size=60, generations=80)}


def score(wl, mk, P, params):
    """Nonlinear re-evaluation, feasibility filter, Pareto filter, 2-D HV."""
    # same reference points as the legacy inner-loop table (SUNCATCHER: 5e6 ms, 15 USD,
    # because nonlinear SUNCATCHER latencies are orders of magnitude above the 110 ms
    # benchmark reference)
    lk, ck, ref = INNER_HV_KEYS[wl.name]
    V = np.array([[mc.aggregate_metrics_nonlinear(wl, p, params)[k] for k in mk] for p in P])
    feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
    if not feas.any():
        return 0.0, 0
    Vf = V[feas]
    par = mc.pareto_front_indices(Vf, wl.metrics, mk)
    return hv2d(to_min_2d(Vf[par], wl, mk, lk, ck), ref), int(len(par))


def run(job):
    wl_name, effects, mode, method, seed, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    wl = mc.load_workload(Path("data.json"), wl_name)
    mk = [m.key for m in wl.metrics]
    params = deepcopy(mc.WORKLOAD_NONLINEAR_PARAMS[wl_name])
    if effects == "seven_effect":
        params.eclipse_duty_cycle = {}
    fn = optimizers_methods.mopso if method == "mopso" else algorithms_extended.nsga2
    if mode == "inner_loop":
        ev = lambda w, p: mc.aggregate_metrics_nonlinear(w, p, params)
        saved = optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear
        optimizers_methods.aggregate_metrics_linear = ev
        algorithms_extended.aggregate_metrics_linear = ev
        try:
            P, _ = fn(workload=wl, metric_keys=mk, seed=seed, **CFG[method])
        finally:
            optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear = saved
    else:
        P, _ = fn(workload=_make_nl_workload_proxy(wl, params), metric_keys=mk, seed=seed, **CFG[method])
    hv, n = score(wl, mk, P, params)
    return dict(workload=wl_name, effects=effects, mode=mode, method=method, seed=seed, hv=float(hv), n_front=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_inner_loop_v2")
    ap.add_argument("--workloads", nargs="+", default=["IDS", "SUNCATCHER"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(exist_ok=True)
    jobs = [(w, e, mo, m, s, args.violation, args.handling) for w in args.workloads
            for e in ["eight_effect", "seven_effect"] for mo in ["inner_loop", "proxy"]
            for m in CFG for s in range(11, 11 + args.seeds)]
    jobs.sort(key=lambda j: j[3] != "mopso")
    with Pool(args.jobs) as pool:
        rows = list(pool.imap_unordered(run, jobs))
    rows.sort(key=lambda r: (r["workload"], r["effects"], r["method"], r["mode"], r["seed"]))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from scipy.stats import wilcoxon
    summ = {}
    for r in rows:
        summ.setdefault(r["workload"], {}).setdefault(r["effects"], {}).setdefault(r["method"], {})
    for wl, d1 in summ.items():
        for e, d2 in d1.items():
            for m in d2:
                hv = {mo: np.array([r["hv"] for r in rows if (r["workload"], r["effects"], r["method"], r["mode"]) == (wl, e, m, mo)])
                      for mo in ["inner_loop", "proxy"]}
                a, b = hv["inner_loop"], hv["proxy"]
                try:
                    p = float(wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0
                except ValueError:
                    p = 1.0
                d2[m] = {"inner_hv_mean": float(a.mean()), "inner_hv_std": float(a.std(ddof=1)),
                         "proxy_hv_mean": float(b.mean()), "proxy_hv_std": float(b.std(ddof=1)),
                         "delta_pct": float(100 * (a.mean() - b.mean()) / b.mean()) if b.mean() else float("nan"),
                         "wilcoxon_p": p, "n_seeds": int(len(a)),
                         "hv_ref": [float(x) for x in INNER_HV_KEYS[wl][2]]}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(args), "cfg": CFG}, indent=2))
    print("inner loop v2 complete")


if __name__ == "__main__":
    main()
