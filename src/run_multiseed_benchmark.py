#!/usr/bin/env python3
"""
run_multiseed_benchmark.py
==========================
Multi-seed, equal-budget benchmark for statistical significance testing.

Addresses two methodological requirements:

  (1) Single-seed results are not defensible. Each stochastic algorithm is
      run for N independent seeds. We report mean +/- std and a paired
      Wilcoxon signed-rank test on hypervolume across seeds.

  (2) run_full_benchmark.py was not equal-budget. MOPSO had ~16k function
      evaluations while NSGA-II/SPEA2 had ~4.8k. This harness sizes every
      algorithm to a common target budget so the comparison isolates
      algorithmic quality from compute spent.

Outputs (under --outdir):
  <WL>_<METHOD>_seed<S>_{P,V}.npy   per-seed Pareto policies/metrics
  summary.json                       per-(WL, method) HV statistics
  pairwise.json                      Wilcoxon paired tests per workload
  summary.csv                        flat table for the paper

Usage:
  python run_multiseed_benchmark.py --seeds 20 --eval-budget 16000
  python run_multiseed_benchmark.py --seeds 5 --workloads IDS --eval-budget 8000
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

from moo_core import (
    Direction,
    aggregate_metrics_linear,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
)
from optimizers_methods import mopso
from algorithms_extended import nsga2, moead, spea2, epsilon_constraint


# ── Per-workload 2D HV reference points (Latency x Cost) ──────────────────────
WCFG = {
    "IDS":               ("Latency p99 (ms)", "Cost per task (USD)", np.array([260.0, 3.2])),
    "SUNCATCHER":        ("Latency p99 (ms)", "Cost per unit (USD)", np.array([110.0, 14.5])),
    "JWST":              ("Latency (hrs)",    "Cost per obs (USD)",  np.array([50.0, 20.0])),
    "RST":               ("Latency (hrs)",    "Cost per obs (USD)",  np.array([26.0, 25.0])),
    "LEO_OOS":           ("Latency p99 (ms)", "Cost per task (USD)", np.array([320.0, 2.5])),
    "DEEP_SPACE_CRUISE": ("Latency (hrs)",    "Cost per obs (USD)",  np.array([5.0, 15.0])),
}


# ── Equal-budget configurations ───────────────────────────────────────────────
# Each algorithm's hyperparameters are sized so total function evaluations
# approximately equal `budget`. Calls per algorithm (approximate):
#   random:  budget Dirichlet samples
#   mopso:   particles * (1 + iters)
#   nsga2:   pop_size * (1 + generations)         (1 init + 1 offspring per gen)
#   spea2:   pop_size * (1 + generations)
#   moead:   N * (1 + generations)   where N = 120 (fixed by _uniform_weight_vectors)
#   epsilon: 2000 (range survey) + n_levels * samples_per_level

def configs_for_budget(budget: int) -> Dict[str, dict]:
    pop = 80
    gens_pop = max(1, budget // pop - 1)
    moead_N = 120  # hardcoded in algorithms_extended._uniform_weight_vectors
    gens_moead = max(1, budget // moead_N - 1)
    eps_levels = max(1, (budget - 2000) // 500)
    return {
        "random":  dict(n=budget),
        "mopso":   dict(particles=pop, iters=gens_pop, archive_max=350),
        "nsga2":   dict(pop_size=pop, generations=gens_pop),
        "spea2":   dict(pop_size=pop, archive_size=pop, generations=gens_pop),
        "moead":   dict(H=8, T=5, generations=gens_moead),
        "epsilon": dict(n_levels=eps_levels, samples_per_level=500),
    }


def expected_evals(method: str, cfg: dict) -> int:
    """Estimated function-evaluation count for a given (method, cfg)."""
    if method == "random":
        return int(cfg["n"])
    if method == "mopso":
        return int(cfg["particles"] * (1 + cfg["iters"]))
    if method == "nsga2" or method == "spea2":
        return int(cfg["pop_size"] * (1 + cfg["generations"]))
    if method == "moead":
        return int(120 * (1 + cfg["generations"]))
    if method == "epsilon":
        return int(2000 + cfg["n_levels"] * cfg["samples_per_level"])
    return -1


# ── Metric helpers ────────────────────────────────────────────────────────────

def hv2d(pts: np.ndarray, ref: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0] - x) * (pv - y)
            pv = y
    return hv


def spacing(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    dm = [np.delete(np.abs(pts - pts[i]).sum(1), i).min() for i in range(len(pts))]
    return float(np.std(dm))


def to_min_2d(V: np.ndarray, wl, mk: List[str], lk: str, ck: str) -> np.ndarray:
    kd = {m.key: m.direction for m in wl.metrics}
    li, ci = mk.index(lk), mk.index(ck)
    pts = V[:, [li, ci]].copy()
    if kd.get(lk) == Direction.MAXIMIZE:
        pts[:, 0] = -pts[:, 0]
    if kd.get(ck) == Direction.MAXIMIZE:
        pts[:, 1] = -pts[:, 1]
    return pts


def random_baseline(wl, mk, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    J = len(wl.tiers)
    P = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), size=n)])
    raw, fs = [], []
    for p in P:
        pm = aggregate_metrics_linear(wl, p)
        f, _, _ = feasibility_and_violation(wl, pm)
        raw.append([pm[k] for k in mk])
        fs.append(f)
    V = np.array(raw); fs = np.array(fs); fi = np.where(fs)[0]
    if fi.size == 0:
        viols = [feasibility_and_violation(wl, aggregate_metrics_linear(wl, P[i]))[1]
                 for i in range(len(P))]
        best = int(np.argmin(viols))
        return P[[best]], V[[best]]
    Vf = V[fi]
    par = pareto_front_indices(Vf, wl.metrics, mk)
    return P[fi[par]], Vf[par]


def run_one(method: str, wl, mk, cfg: dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if method == "random":
        return random_baseline(wl, mk, n=cfg["n"], seed=seed)
    fn_map = {"mopso": mopso, "nsga2": nsga2, "spea2": spea2,
              "moead": moead, "epsilon": epsilon_constraint}
    return fn_map[method](workload=wl, metric_keys=mk, seed=seed, **cfg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--outdir", default="out_multiseed")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--eval-budget", type=int, default=16000)
    ap.add_argument("--workloads", nargs="+",
                    default=["IDS", "SUNCATCHER", "JWST", "RST"])
    ap.add_argument("--methods", nargs="+",
                    default=["random", "mopso", "nsga2", "spea2", "moead", "epsilon"])
    ap.add_argument("--seed-offset", type=int, default=1000,
                    help="First seed value (so seed-0 doesn't collide with paper seed=7)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfgs = configs_for_budget(args.eval_budget)

    print("Equal-budget configs (target evals = {:,}):".format(args.eval_budget))
    for m in args.methods:
        print(f"  {m:<8s} expected_evals = {expected_evals(m, cfgs[m]):>6,}    "
              f"cfg = {cfgs[m]}")

    results: Dict[str, Dict[str, List[dict]]] = {}
    t_start_all = time.perf_counter()

    for wl_name in args.workloads:
        wl = load_workload(Path(args.data), wl_name)
        mk = [m.key for m in wl.metrics]
        lk, ck, ref = WCFG[wl_name]
        results[wl_name] = {}
        print(f"\n{'='*70}\n  {wl_name}\n{'='*70}")

        for method in args.methods:
            runs = []
            for s_i in range(args.seeds):
                seed = args.seed_offset + s_i
                t0 = time.perf_counter()
                P, V = run_one(method, wl, mk, cfgs[method], seed)
                rt = time.perf_counter() - t0

                pts2 = to_min_2d(V, wl, mk, lk, ck) if (lk in mk and ck in mk) else np.zeros((0, 2))
                hv = hv2d(pts2, ref) if pts2.shape[0] else 0.0
                sp = spacing(pts2) if pts2.shape[0] >= 2 else 0.0
                runs.append({"seed": seed, "hv": float(hv), "spacing": float(sp),
                             "n_pareto": int(P.shape[0]), "rt_s": float(rt)})

                np.save(str(outdir / f"{wl_name}_{method}_seed{seed}_P.npy"), P)
                np.save(str(outdir / f"{wl_name}_{method}_seed{seed}_V.npy"), V)
                print(f"  {method:<8s} seed={seed:<5d} hv={hv:>8.2f} "
                      f"n={P.shape[0]:>4d} t={rt:>6.1f}s")
            results[wl_name][method] = runs

    elapsed_all = time.perf_counter() - t_start_all
    print(f"\nTotal wall time: {elapsed_all/60:.1f} min")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    summary: Dict[str, Dict[str, dict]] = {}
    for wl_name, methods in results.items():
        summary[wl_name] = {}
        for method, runs in methods.items():
            hvs = np.array([r["hv"] for r in runs])
            ns  = np.array([r["n_pareto"] for r in runs])
            rts = np.array([r["rt_s"] for r in runs])
            summary[wl_name][method] = {
                "n_seeds":          int(len(hvs)),
                "hv_mean":          float(hvs.mean()),
                "hv_std":           float(hvs.std(ddof=1)) if len(hvs) > 1 else 0.0,
                "hv_median":        float(np.median(hvs)),
                "hv_q25":           float(np.quantile(hvs, 0.25)),
                "hv_q75":           float(np.quantile(hvs, 0.75)),
                "hv_min":           float(hvs.min()),
                "hv_max":           float(hvs.max()),
                "pareto_size_mean": float(ns.mean()),
                "pareto_size_std":  float(ns.std(ddof=1)) if len(ns) > 1 else 0.0,
                "runtime_mean_s":   float(rts.mean()),
            }

    # ── Pairwise Wilcoxon signed-rank on HV (paired by seed) ─────────────────
    pairwise: Dict[str, Dict[str, dict]] = {}
    for wl_name in args.workloads:
        pairwise[wl_name] = {}
        method_list = list(results[wl_name].keys())
        for i, m1 in enumerate(method_list):
            for m2 in method_list[i+1:]:
                hv1 = np.array([r["hv"] for r in results[wl_name][m1]])
                hv2 = np.array([r["hv"] for r in results[wl_name][m2]])
                if len(hv1) != len(hv2) or len(hv1) < 5:
                    stat, pval = float("nan"), float("nan")
                elif np.allclose(hv1, hv2):
                    stat, pval = 0.0, 1.0
                else:
                    try:
                        res = wilcoxon(hv1, hv2)
                        stat, pval = float(res.statistic), float(res.pvalue)
                    except Exception:
                        stat, pval = float("nan"), float("nan")
                pairwise[wl_name][f"{m1}_vs_{m2}"] = {
                    "wilcoxon_stat": stat,
                    "p_value":       pval,
                    "mean_diff":     float(hv1.mean() - hv2.mean()),
                    "median_diff":   float(np.median(hv1) - np.median(hv2)),
                }

    # ── Persist ───────────────────────────────────────────────────────────────
    payload = {
        "config": {
            "seeds":         args.seeds,
            "seed_offset":   args.seed_offset,
            "eval_budget":   args.eval_budget,
            "workloads":     list(args.workloads),
            "methods":       list(args.methods),
            "per_method_cfg": {m: {**cfgs[m], "expected_evals": expected_evals(m, cfgs[m])}
                              for m in args.methods},
        },
        "summary":            summary,
        "pairwise_wilcoxon":  pairwise,
        "raw":                results,
        "wall_time_min":      elapsed_all / 60.0,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))

    # ── Flat CSV ─────────────────────────────────────────────────────────────
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "method", "n_seeds", "hv_mean", "hv_std",
                    "hv_median", "hv_q25", "hv_q75", "pareto_size_mean",
                    "runtime_mean_s"])
        for wl_name in args.workloads:
            for m in args.methods:
                s = summary[wl_name][m]
                w.writerow([wl_name, m, s["n_seeds"],
                            f"{s['hv_mean']:.4f}", f"{s['hv_std']:.4f}",
                            f"{s['hv_median']:.4f}", f"{s['hv_q25']:.4f}",
                            f"{s['hv_q75']:.4f}",
                            f"{s['pareto_size_mean']:.2f}",
                            f"{s['runtime_mean_s']:.2f}"])

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print(f"  Summary (HV mean +/- std over {args.seeds} seeds; "
          f"budget={args.eval_budget:,} evals)")
    print("=" * 82)
    print(f"  {'Workload':<14}{'Method':<10}{'HV mean':>10}{'+/- std':>10}"
          f"{'median':>10}{'n_par':>8}{'t/run(s)':>10}")
    print("  " + "-" * 76)
    for wl_name in args.workloads:
        for m in args.methods:
            s = summary[wl_name][m]
            print(f"  {wl_name:<14}{m:<10}{s['hv_mean']:>10.2f}"
                  f"{s['hv_std']:>10.2f}{s['hv_median']:>10.2f}"
                  f"{s['pareto_size_mean']:>8.0f}{s['runtime_mean_s']:>10.2f}")
        print()

    # ── Significant differences (p<0.05) ─────────────────────────────────────
    print("=" * 82)
    print("  Significant pairwise differences (Wilcoxon paired, p < 0.05)")
    print("=" * 82)
    for wl_name in args.workloads:
        sig = [(pair, d) for pair, d in pairwise[wl_name].items()
               if not np.isnan(d["p_value"]) and d["p_value"] < 0.05]
        if not sig:
            print(f"  {wl_name}: no significant pairwise differences at p<0.05")
            continue
        print(f"  {wl_name}:")
        for pair, d in sorted(sig, key=lambda x: x[1]["p_value"]):
            winner = pair.split("_vs_")[0 if d["mean_diff"] > 0 else 1]
            print(f"    {pair:<32s} p={d['p_value']:.4f}  "
                  f"diff={d['mean_diff']:+8.2f}  winner: {winner}")

    print(f"\nSaved: {outdir}/summary.json, summary.csv, *_P.npy, *_V.npy")


if __name__ == "__main__":
    main()
