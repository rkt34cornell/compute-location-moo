#!/usr/bin/env python3
"""
exp_nl_sensitivity_v2.py — coefficient +/-25% sensitivity across algorithms.

Same nine coefficients and +/-25% perturbations as exp_nonlinear_sensitivity.py
(IDS, nonlinear evaluator, current data.json). Differences: three algorithms
(MOPSO 60x61 / archive 200 = legacy config; NSGA-II pop 60 x 60 generations,
3,660 evals; Random Dirichlet 3,660 samples incl. vertices), 10 seeds (11-20)
instead of one, feasibility re-checked before HV (an infeasible fallback scores
HV = 0), and the constraint modes given on the command line.

Caveat preserved from legacy: +25% on queue_max_utilization gives a cap of
1.15 > 1, where the M/M/1 term is evaluated with (1 - rho) floored at 1e-6.

Output: out_nl_sensitivity_v2/{runs.csv, summary.json}
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
from exp_nonlinear_sensitivity import COEFFS, PERTURB_FRAC
from run_multiseed_benchmark import WCFG, hv2d, to_min_2d

WL = "IDS"
BUDGET = 60 * 61


def settings():
    out = [("baseline", None, 1.0)]
    for c in COEFFS:
        out += [(c, "-25%", 1 - PERTURB_FRAC), (c, "+25%", 1 + PERTURB_FRAC)]
    return out


def run(job):
    coef, direction, factor, method, seed, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    wl = mc.load_workload(Path("data.json"), WL)
    mk = [m.key for m in wl.metrics]
    params = deepcopy(mc.WORKLOAD_NONLINEAR_PARAMS[WL])
    if coef != "baseline":
        setattr(params, coef, float(getattr(params, coef)) * factor)
    ev = lambda w, p: mc.aggregate_metrics_nonlinear(w, p, params)
    saved = optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = ev
    algorithms_extended.aggregate_metrics_linear = ev
    try:
        if method == "mopso":
            P, V = optimizers_methods.mopso(workload=wl, metric_keys=mk, particles=60, iters=60,
                                            archive_max=200, seed=seed)
        elif method == "nsga2":
            P, V = algorithms_extended.nsga2(workload=wl, metric_keys=mk, pop_size=60,
                                             generations=60, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            J = len(wl.tiers)
            Ps = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), BUDGET - J)])
            V = np.array([[ev(wl, p)[k] for k in mk] for p in Ps])
    finally:
        optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear = saved
    feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
    Vf = V[feas]
    lk, ck, ref = WCFG[WL]
    if Vf.shape[0]:
        par = mc.pareto_front_indices(Vf, wl.metrics, mk)
        hv = hv2d(to_min_2d(Vf[par], wl, mk, lk, ck), ref)
    else:
        hv = 0.0
    return dict(coefficient=coef, direction=direction or "baseline", factor=factor,
                method=method, seed=seed, hv=float(hv), n_feasible=int(feas.sum()))


def summarize(rows, out: Path):
    from scipy.stats import wilcoxon, kendalltau, spearmanr
    res = {}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        base = {r["seed"]: r["hv"] for r in R if r["coefficient"] == "baseline"}
        b = np.array([base[s] for s in sorted(base)])
        res[m] = {"baseline_hv_mean": float(b.mean()), "baseline_hv_std": float(b.std(ddof=1)), "coeffs": {}}
        for c in COEFFS:
            res[m]["coeffs"][c] = {}
            for d in ["-25%", "+25%"]:
                x = {r["seed"]: r["hv"] for r in R if r["coefficient"] == c and r["direction"] == d}
                xv = np.array([x[s] for s in sorted(base)])
                delta = 100 * (xv - b) / np.where(b > 0, b, np.nan)
                try:
                    p = float(wilcoxon(xv, b).pvalue) if not np.allclose(xv, b) else 1.0
                except ValueError:
                    p = 1.0
                res[m]["coeffs"][c][d] = {"hv_mean": float(xv.mean()), "hv_std": float(xv.std(ddof=1)),
                                          "delta_pct_mean": float(np.nanmean(delta)),
                                          "delta_pct_std": float(np.nanstd(delta, ddof=1)),
                                          "wilcoxon_p_vs_baseline": p}
            res[m]["coeffs"][c]["importance"] = max(abs(res[m]["coeffs"][c][d]["delta_pct_mean"])
                                                    for d in ["-25%", "+25%"])
        order = sorted(COEFFS, key=lambda c: -res[m]["coeffs"][c]["importance"])
        res[m]["importance_order"] = order
    agree = {}
    for i, a in enumerate(methods):
        for bm in methods[i + 1:]:
            ia = [res[a]["coeffs"][c]["importance"] for c in COEFFS]
            ib = [res[bm]["coeffs"][c]["importance"] for c in COEFFS]
            agree[f"{a}_vs_{bm}"] = {"kendall_tau": float(kendalltau(ia, ib).statistic),
                                     "kendall_p": float(kendalltau(ia, ib).pvalue),
                                     "spearman_rho": float(spearmanr(ia, ib).statistic),
                                     "top3_a": res[a]["importance_order"][:3],
                                     "top3_b": res[bm]["importance_order"][:3]}
    res["ordering_agreement"] = agree
    (out / "summary.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_nl_sensitivity_v2")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=["mopso", "nsga2", "random"])
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(exist_ok=True)
    jobs = [(c, d, f, m, s, args.violation, args.handling) for (c, d, f) in settings()
            for m in args.methods for s in range(11, 11 + args.seeds)]
    jobs.sort(key=lambda j: j[3] != "mopso")
    with Pool(args.jobs) as pool:
        rows = []
        for r in pool.imap_unordered(run, jobs):
            rows.append(r)
            if len(rows) % 25 == 0:
                print(f"{len(rows)}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["method"], r["coefficient"], r["direction"], r["seed"]))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summarize(rows, out)
    (out / "config.json").write_text(json.dumps({**vars(args), "workload": WL, "budget": BUDGET,
                                                 "coefficients": COEFFS}, indent=2))
    print("sensitivity v2 complete")


if __name__ == "__main__":
    main()
