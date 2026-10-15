#!/usr/bin/env python3
"""
run_roman_multiseed.py — Multi-seed equal-budget benchmark for ROMAN_TRANSIENT.

Cross-product:
  workloads:  ROMAN_TRANSIENT, ROMAN_TRANSIENT_FOLLOWUP   (2 constraint sets)
  variants:   A (TRL=3), B (TRL=7), C (no ODC)            (3 ODC scenarios)
  evaluators: linear, nonlinear                            (2 aggregators)
  methods:    random, mopso, nsga2, spea2, moead, epsilon  (6 algorithms)
  seeds:      configurable (default 20)

Output (under --outdir, default out_roman_multiseed/):
  <WL>_<VAR>_<EV>_<METHOD>_seed<S>_{P,V}.npy
  summary.json with mean ± std HV, Wilcoxon pairwise tests
  summary.csv flat table
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

from moo_core import (
    Direction,
    NonlinearParams,
    WORKLOAD_NONLINEAR_PARAMS,
    aggregate_metrics_linear,
    aggregate_metrics_nonlinear,
    apply_odc_variant,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
)
import optimizers_methods
import algorithms_extended
from optimizers_methods import mopso
from algorithms_extended import nsga2, moead, spea2, epsilon_constraint


WORKLOADS = ["ROMAN_TRANSIENT", "ROMAN_TRANSIENT_FOLLOWUP"]
VARIANTS = ["A", "B", "C"]
EVALUATORS = ["linear", "nonlinear"]
METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon"]
LK = "Latency p99 (ms)"
CK = "Cost per task (USD)"
# Higher ref points than IDS because ROMAN latency is in (10^4, 10^6) ms.
HV_REF = np.array([1.0e8, 0.12])


def configs_for_budget(budget: int) -> Dict[str, dict]:
    pop = 60
    gens_pop = max(1, budget // pop - 1)
    moead_N = 120
    gens_moead = max(1, budget // moead_N - 1)
    eps_levels = max(1, (budget - 2000) // 500)
    return {
        "random":  dict(n=budget),
        "mopso":   dict(particles=pop, iters=gens_pop, archive_max=250),
        "nsga2":   dict(pop_size=pop, generations=gens_pop),
        "spea2":   dict(pop_size=pop, archive_size=pop, generations=gens_pop),
        "moead":   dict(H=8, T=5, generations=gens_moead),
        "epsilon": dict(n_levels=eps_levels, samples_per_level=500),
    }


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def to_min_2d(V, mk, kd):
    li, ci = mk.index(LK), mk.index(CK)
    pts = V[:, [li, ci]].copy()
    if kd.get(LK) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
    if kd.get(CK) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
    return pts


def install_nl(params):
    """Monkey-patch optimizer-module references so they call nonlinear."""
    nl_eval = lambda w, p: aggregate_metrics_nonlinear(w, p, params)
    saved_o = optimizers_methods.aggregate_metrics_linear
    saved_a = algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = nl_eval
    algorithms_extended.aggregate_metrics_linear = nl_eval
    def restore():
        optimizers_methods.aggregate_metrics_linear = saved_o
        algorithms_extended.aggregate_metrics_linear = saved_a
    return restore


def random_baseline(wl, mk, n, seed, evaluator, params):
    rng = np.random.default_rng(seed)
    J = len(wl.tiers)
    P = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), size=n)])
    raw, fs = [], []
    for p in P:
        pm = (aggregate_metrics_linear(wl, p) if evaluator == "linear"
              else aggregate_metrics_nonlinear(wl, p, params))
        f, _, _ = feasibility_and_violation(wl, pm)
        raw.append([pm[k] for k in mk]); fs.append(f)
    V = np.array(raw); fs = np.array(fs); fi = np.where(fs)[0]
    if fi.size == 0:
        return P[[0]], V[[0]]
    Vf = V[fi]; par = pareto_front_indices(Vf, wl.metrics, mk)
    return P[fi[par]], Vf[par]


def run_one(method, wl, mk, cfg, seed, evaluator, params):
    if method == "random":
        return random_baseline(wl, mk, n=cfg["n"], seed=seed,
                               evaluator=evaluator, params=params)
    fn_map = {"mopso": mopso, "nsga2": nsga2, "spea2": spea2,
              "moead": moead, "epsilon": epsilon_constraint}
    fn = fn_map[method]
    if evaluator == "nonlinear":
        restore = install_nl(params)
        try:
            return fn(workload=wl, metric_keys=mk, seed=seed, **cfg)
        finally:
            restore()
    return fn(workload=wl, metric_keys=mk, seed=seed, **cfg)


def pairwise_wilcoxon(hv_by_method: Dict[str, np.ndarray]) -> dict:
    out = {}
    methods = list(hv_by_method.keys())
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            a, b = hv_by_method[m1], hv_by_method[m2]
            if len(a) != len(b) or len(a) < 5:
                stat, p = float("nan"), float("nan")
            elif np.allclose(a, b):
                stat, p = 0.0, 1.0
            else:
                try:
                    r = wilcoxon(a, b)
                    stat, p = float(r.statistic), float(r.pvalue)
                except Exception:
                    stat, p = float("nan"), float("nan")
            out[f"{m1}_vs_{m2}"] = {
                "wilcoxon_stat": stat, "p_value": p,
                "mean_diff": float(a.mean() - b.mean()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="data.json")
    ap.add_argument("--outdir",    default="out_roman_multiseed")
    ap.add_argument("--seeds",     type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=2000)
    ap.add_argument("--eval-budget", type=int, default=12000)
    ap.add_argument("--workloads", nargs="+", default=WORKLOADS)
    ap.add_argument("--variants",  nargs="+", default=VARIANTS)
    ap.add_argument("--evaluators", nargs="+", default=EVALUATORS)
    ap.add_argument("--methods",   nargs="+", default=METHODS)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfgs = configs_for_budget(args.eval_budget)
    print(f"Budget {args.eval_budget:,} evals/method  ({args.seeds} seeds × "
          f"{len(args.methods)} methods × {len(args.variants)} variants × "
          f"{len(args.evaluators)} evaluators × {len(args.workloads)} workloads)")

    t_all = time.perf_counter()
    results: Dict[Tuple[str, str, str], Dict[str, List[dict]]] = {}

    for wl_name in args.workloads:
        wl_base = load_workload(Path(args.data), wl_name)
        mk = [m.key for m in wl_base.metrics]
        kd = {m.key: m.direction for m in wl_base.metrics}
        params = WORKLOAD_NONLINEAR_PARAMS[wl_name]

        for variant in args.variants:
            wl = apply_odc_variant(wl_base, variant)
            for evaluator in args.evaluators:
                key = (wl_name, variant, evaluator)
                results[key] = {}
                print(f"\n=== {wl_name} / Variant {variant} / {evaluator} ===")
                print(f"   tiers={wl.tiers}")

                for method in args.methods:
                    runs = []
                    for s_i in range(args.seeds):
                        seed = args.seed_offset + s_i
                        t0 = time.perf_counter()
                        P, V = run_one(method, wl, mk, cfgs[method], seed,
                                       evaluator, params)
                        rt = time.perf_counter() - t0
                        pts2 = to_min_2d(V, mk, kd) if (LK in mk and CK in mk) else np.zeros((0, 2))
                        hv = hv2d(pts2, HV_REF) if pts2.shape[0] else 0.0
                        runs.append({"seed": seed, "hv": float(hv),
                                     "n_pareto": int(P.shape[0]),
                                     "rt_s": float(rt)})
                        stem = f"{wl_name}_{variant}_{evaluator}_{method}_seed{seed}"
                        np.save(str(outdir / f"{stem}_P.npy"), P)
                        np.save(str(outdir / f"{stem}_V.npy"), V)
                        print(f"   {method:<8s} s={seed} hv={hv:>10.2f} "
                              f"n={P.shape[0]:>4d} t={rt:>6.1f}s")
                    results[key][method] = runs

    elapsed = time.perf_counter() - t_all
    print(f"\nTotal wall time: {elapsed/60:.1f} min")

    # ── Summarize ────────────────────────────────────────────────────────────
    summary = {}
    pairwise = {}
    for (wl_name, variant, evaluator), methods in results.items():
        skey = f"{wl_name}|{variant}|{evaluator}"
        summary[skey] = {}
        hv_by_method: Dict[str, np.ndarray] = {}
        for method, runs in methods.items():
            hvs = np.array([r["hv"] for r in runs])
            ns  = np.array([r["n_pareto"] for r in runs])
            rts = np.array([r["rt_s"] for r in runs])
            summary[skey][method] = {
                "n_seeds":          int(len(hvs)),
                "hv_mean":          float(hvs.mean()),
                "hv_std":           float(hvs.std(ddof=1)) if len(hvs) > 1 else 0.0,
                "hv_median":        float(np.median(hvs)),
                "pareto_size_mean": float(ns.mean()),
                "runtime_mean_s":   float(rts.mean()),
            }
            hv_by_method[method] = hvs
        pairwise[skey] = pairwise_wilcoxon(hv_by_method)

    payload = {
        "config": {
            "seeds":       args.seeds,
            "seed_offset": args.seed_offset,
            "eval_budget": args.eval_budget,
            "workloads":   list(args.workloads),
            "variants":    list(args.variants),
            "evaluators":  list(args.evaluators),
            "methods":     list(args.methods),
            "hv_ref":      HV_REF.tolist(),
            "configs":     cfgs,
        },
        "summary":           summary,
        "pairwise_wilcoxon": pairwise,
        "raw":               {f"{wl}|{v}|{e}":
                              {m: results[(wl, v, e)][m]
                               for m in results[(wl, v, e)]}
                              for (wl, v, e) in results},
        "wall_time_min":     elapsed / 60.0,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))

    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "variant", "evaluator", "method",
                    "n_seeds", "hv_mean", "hv_std", "hv_median",
                    "pareto_size_mean", "runtime_mean_s"])
        for skey, mdict in summary.items():
            wl_name, variant, evaluator = skey.split("|")
            for method, s in mdict.items():
                w.writerow([wl_name, variant, evaluator, method,
                            s["n_seeds"], f"{s['hv_mean']:.4f}",
                            f"{s['hv_std']:.4f}", f"{s['hv_median']:.4f}",
                            f"{s['pareto_size_mean']:.2f}",
                            f"{s['runtime_mean_s']:.2f}"])

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  Multi-seed Roman summary (HV mean±std over {args.seeds} seeds)")
    print("=" * 100)
    print(f"  {'WL':<24}{'Var':<4}{'Eval':<10}{'Method':<10}"
          f"{'HV mean':>12}{'± std':>10}{'n_par':>8}{'t/run(s)':>10}")
    for skey, mdict in summary.items():
        wl_name, variant, evaluator = skey.split("|")
        wl_short = wl_name.replace("ROMAN_TRANSIENT", "ROMAN_T")
        for method, s in mdict.items():
            print(f"  {wl_short:<24}{variant:<4}{evaluator:<10}{method:<10}"
                  f"{s['hv_mean']:>12.2f}{s['hv_std']:>10.2f}"
                  f"{s['pareto_size_mean']:>8.0f}{s['runtime_mean_s']:>10.2f}")
    print(f"\nSaved {outdir}/summary.json, summary.csv, *_P.npy, *_V.npy")


if __name__ == "__main__":
    main()
