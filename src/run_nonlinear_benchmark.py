#!/usr/bin/env python3
"""
run_nonlinear_benchmark.py
==========================
Runs the full 4-workload, 4-algorithm benchmark under the nonlinear evaluator
and saves results for figure generation.

Outputs (in --outdir, default out_nonlinear/):
  <WL>_<method>_P.npy   - policy vectors
  <WL>_<method>_V.npy   - raw metric vectors
  <WL>_<method>_Vlin.npy - linear metric vectors for same policies (comparison)
  benchmark_nonlinear.json  - summary table

Algorithms: random, nsga2, moead, mopso
Workloads:  IDS, SUNCATCHER, JWST, RST

Nonlinear effects active:
  1. M/M/1 queuing on latency
  2. Contact window saturation on latency
  3. Fixed link setup cost on energy
  4. Security protocol energy overhead
  5. Thermal margin degradation (superlinear)
  6. Link availability congestion
  7. SDC compound error rate

Usage:
  python run_nonlinear_benchmark.py
  python run_nonlinear_benchmark.py --workloads IDS JWST --skip-mopso
  python run_nonlinear_benchmark.py --data data.json --outdir out_nonlinear
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from moo_core import (
    Direction,
    NonlinearParams,
    Workload,
    WORKLOAD_NONLINEAR_PARAMS,
    aggregate_metrics_linear,
    aggregate_metrics_nonlinear,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
    project_to_simplex,
)
from optimizers_methods import mopso
from algorithms_extended import nsga2, moead, spea2, epsilon_constraint


# ── Hypervolume reference points (Latency × Cost projection) ──────────────────
# These are set beyond the worst feasible point for each workload under the
# nonlinear evaluator. SUNCATCHER has no latency constraint and nonlinear
# effects inflate latency to millions of ms, so the ref point is much larger
# than the linear case.
HV_REF = {
    "IDS":        np.array([260.0, 3.2]),
    "SUNCATCHER": np.array([5_000_000.0, 15.0]),
    "JWST":       np.array([50.0, 20.0]),
    "RST":        np.array([26.0, 25.0]),
}

LATENCY_KEYS = {
    "IDS":        "Latency p99 (ms)",
    "SUNCATCHER": "Latency p99 (ms)",
    "JWST":       "Latency (hrs)",
    "RST":        "Latency (hrs)",
}
COST_KEYS = {
    "IDS":        "Cost per task (USD)",
    "SUNCATCHER": "Cost per unit (USD)",
    "JWST":       "Cost per obs (USD)",
    "RST":        "Cost per obs (USD)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def hv2d(pts: np.ndarray, ref: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    pv = ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0] - x) * (pv - y)
            pv = y
    return hv


def to_min_2d(V: np.ndarray, wl: Workload, mk: List[str], lk: str, ck: str) -> np.ndarray:
    """Return 2D projection in minimization space (latency, cost both already min)."""
    kd = {m.key: m.direction for m in wl.metrics}
    li, ci = mk.index(lk), mk.index(ck)
    pts = V[:, [li, ci]].copy()
    if kd.get(lk) == Direction.MAXIMIZE:
        pts[:, 0] = -pts[:, 0]
    if kd.get(ck) == Direction.MAXIMIZE:
        pts[:, 1] = -pts[:, 1]
    return pts


def eval_population_nonlinear(
    wl: Workload,
    mk: List[str],
    P: np.ndarray,
    params: NonlinearParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate P under nonlinear evaluator. Returns (V_nonlinear, V_linear)."""
    n = P.shape[0]
    d = len(mk)
    V_nl = np.zeros((n, d))
    V_lin = np.zeros((n, d))
    for i in range(n):
        pm_nl  = aggregate_metrics_nonlinear(wl, P[i], params)
        pm_lin = aggregate_metrics_linear(wl, P[i])
        V_nl[i]  = [pm_nl.get(k, 0.0) for k in mk]
        V_lin[i] = [pm_lin.get(k, 0.0) for k in mk]
    return V_nl, V_lin


def random_baseline_nonlinear(
    wl: Workload, mk: List[str], params: NonlinearParams,
    n: int = 3000, seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Random Dirichlet sampling under nonlinear evaluator.
    Returns (P_pareto, V_nl_pareto, V_lin_pareto, feas_rate)."""
    rng = np.random.default_rng(seed)
    J = len(wl.tiers)
    P = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), size=n)])

    V_nl, V_lin = eval_population_nonlinear(wl, mk, P, params)

    feas_mask = np.array([
        feasibility_and_violation(wl, {k: V_nl[i, j] for j, k in enumerate(mk)})[0]
        for i in range(len(P))
    ])
    fi = np.where(feas_mask)[0]
    feas_rate = fi.size / len(P)

    if fi.size == 0:
        best = int(np.argmin([
            feasibility_and_violation(wl, {k: V_nl[i, j] for j, k in enumerate(mk)})[1]
            for i in range(len(P))
        ]))
        return P[[best]], V_nl[[best]], V_lin[[best]], 0.0

    Vf = V_nl[fi]
    par = pareto_front_indices(Vf, wl.metrics, mk)
    return P[fi[par]], Vf[par], V_lin[fi[par]], feas_rate


# ── Nonlinear-aware wrappers for population-based optimizers ─────────────────
# The optimizers in optimizers_methods.py / algorithms_extended.py call
# aggregate_metrics_linear internally. We monkey-patch the workload evaluator
# by wrapping the optimizer evaluation function to use nonlinear instead.
# Cleanest approach: pass a custom eval_fn to each algorithm.

def _make_nl_workload_proxy(wl: Workload, params: NonlinearParams) -> Workload:
    """
    Returns a Workload-like object whose tier_values are pre-computed under the
    nonlinear model at pure-tier policies.

    This is an approximation: it captures the nonlinear effects AT the vertices
    but not in the interior. The optimizers then interpolate linearly between
    these nonlinear vertex values — which is better than pure linear but not
    the full nonlinear treatment.

    For full nonlinear treatment, see the post-hoc re-evaluation below.
    """
    import copy
    wl_proxy = copy.deepcopy(wl)
    J = len(wl.tiers)
    for j, t in enumerate(wl.tiers):
        p_pure = np.zeros(J)
        p_pure[j] = 1.0
        pm_nl = aggregate_metrics_nonlinear(wl, p_pure, params)
        for k, v in pm_nl.items():
            wl_proxy.tier_values[t][k] = v
    return wl_proxy


def run_method_nonlinear(
    method: str,
    wl: Workload,
    mk: List[str],
    params: NonlinearParams,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run one algorithm under the nonlinear evaluator.

    Strategy:
    - Build a nonlinear proxy workload (tier values = nonlinear values at
      pure-tier policies). Feed this to the optimizer so it sees nonlinear
      vertex values and interpolates between them.
    - After optimization, re-evaluate every Pareto solution under the FULL
      nonlinear evaluator (which captures interior nonlinearities from queuing,
      setup costs, etc.) and recompute the Pareto front.

    This two-pass approach (proxy optimization + full re-evaluation) gives
    the optimizer a nonlinear signal while keeping the optimizer API unchanged.

    Returns
    -------
    P_pareto : (n, J)   - policy vectors
    V_nl_pareto : (n, d) - nonlinear metric values
    V_lin_pareto : (n, d) - linear metric values at same policies
    feas_rate : float
    """
    wl_proxy = _make_nl_workload_proxy(wl, params)
    J = len(wl.tiers)

    if method == "random":
        return random_baseline_nonlinear(wl, mk, params, n=3000, seed=seed)

    elif method == "nsga2":
        aP, _ = nsga2(workload=wl_proxy, metric_keys=mk,
                      pop_size=60, generations=80, seed=seed)
    elif method == "moead":
        aP, _ = moead(workload=wl_proxy, metric_keys=mk,
                      H=8, T=5, generations=120, seed=seed)
    elif method == "mopso":
        aP, _ = mopso(workload=wl_proxy, metric_keys=mk,
                      particles=60, iters=150, archive_max=250, seed=seed)
    elif method == "spea2":
        aP, _ = spea2(workload=wl_proxy, metric_keys=mk,
                      pop_size=60, archive_size=60, generations=80, seed=seed)
    elif method == "epsilon":
        aP, _ = epsilon_constraint(workload=wl_proxy, metric_keys=mk,
                                   n_levels=30, samples_per_level=500, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Full nonlinear re-evaluation of all candidate policies
    V_nl, V_lin = eval_population_nonlinear(wl, mk, aP, params)

    # Recompute feasibility under full nonlinear
    feas_mask = np.array([
        feasibility_and_violation(wl, {k: V_nl[i, j] for j, k in enumerate(mk)})[0]
        for i in range(len(aP))
    ])
    fi = np.where(feas_mask)[0]

    # True simplex feasibility via random survey (comparable across methods)
    rng_feas = np.random.default_rng(seed + 999)
    J = len(wl.tiers)
    P_survey = rng_feas.dirichlet(np.ones(J), size=1000)
    feas_count = sum(1 for p in P_survey
                     if feasibility_and_violation(wl, aggregate_metrics_nonlinear(wl, p, params))[0])
    feas_rate = feas_count / 1000

    if fi.size == 0:
        best = int(np.argmin([
            feasibility_and_violation(wl, {k: V_nl[i, j] for j, k in enumerate(mk)})[1]
            for i in range(len(aP))
        ]))
        return aP[[best]], V_nl[[best]], V_lin[[best]], feas_rate

    Vf = V_nl[fi]
    par = pareto_front_indices(Vf, wl.metrics, mk)
    return aP[fi[par]], Vf[par], V_lin[fi[par]], feas_rate


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_workload(
    wl_name: str,
    data_path: Path,
    outdir: Path,
    methods: List[str],
    seed: int = 7,
) -> Dict:
    wl = load_workload(data_path, wl_name)
    mk = [m.key for m in wl.metrics]
    lk = LATENCY_KEYS[wl_name]
    ck = COST_KEYS[wl_name]
    ref = HV_REF[wl_name]
    params = WORKLOAD_NONLINEAR_PARAMS.get(wl_name, NonlinearParams())

    results = {}
    for method in methods:
        print(f"  [{wl_name}] {method} (nonlinear)...", end=" ", flush=True)
        t0 = time.perf_counter()

        aP, aV_nl, aV_lin, feas_r = run_method_nonlinear(method, wl, mk, params, seed)
        rt = time.perf_counter() - t0

        # 2D HV in minimisation space
        pts = to_min_2d(aV_nl, wl, mk, lk, ck)
        hv = hv2d(pts, ref)

        np.save(str(outdir / f"{wl_name}_{method}_P.npy"),    aP)
        np.save(str(outdir / f"{wl_name}_{method}_V.npy"),    aV_nl)
        np.save(str(outdir / f"{wl_name}_{method}_Vlin.npy"), aV_lin)

        results[method] = {
            "pareto_size":   int(aP.shape[0]),
            "hv_2d":         round(hv, 3),
            "feasible_rate": round(feas_r, 4),
            "runtime_s":     round(rt, 2),
        }
        print(f"n={aP.shape[0]:4d}  hv={hv:7.2f}  feas={feas_r:.0%}  t={rt:.0f}s")

    return results


def sensitivity_analysis_nonlinear(data_path: Path, outdir: Path) -> Dict:
    """Sweep orbital compute radiation tolerance under the nonlinear evaluator."""
    import copy

    print("\n[Sensitivity] Radiation tolerance sweep (nonlinear)...")

    tolerances = [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
    IDS_LIFETIME = 50.0
    SUN_LIFETIME = 100.0

    results = {}

    for wl_name, lifetime_dose, orbital_tiers in [
        ("IDS",        IDS_LIFETIME, ["ODC"]),
        ("SUNCATCHER", SUN_LIFETIME, ["LEO_TPU_CLUSTER", "HYBRID_SPLIT"]),
    ]:
        wl = load_workload(data_path, wl_name)
        mk = [m.key for m in wl.metrics]
        J = len(wl.tiers)
        rad_key = "Rad dose margin (krad)"
        params = WORKLOAD_NONLINEAR_PARAMS.get(wl_name, NonlinearParams())

        wl_results = []

        for tol in tolerances:
            new_margin = tol - lifetime_dose

            patched_wl = copy.deepcopy(wl)
            for t in orbital_tiers:
                if t in patched_wl.tier_values:
                    patched_wl.tier_values[t][rad_key] = new_margin

            rng = np.random.default_rng(42)
            P = rng.dirichlet(np.ones(J), size=1000)
            feas_count = 0
            for p in P:
                pm = aggregate_metrics_nonlinear(patched_wl, p, params)
                f, _, _ = feasibility_and_violation(patched_wl, pm)
                if f:
                    feas_count += 1
            feas_frac = feas_count / 1000

            wl_results.append({
                "tolerance_krad": tol,
                "margin_krad": round(new_margin, 1),
                "feasible_fraction": round(feas_frac, 4),
            })
            print(f"  {wl_name} tol={tol:4d} krad -> margin={new_margin:+.0f} krad -> feas={feas_frac:.1%}")

        results[wl_name] = wl_results

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="data.json")
    ap.add_argument("--outdir",    default="out_nonlinear")
    ap.add_argument("--workloads", nargs="+",
                    default=["IDS", "SUNCATCHER", "JWST", "RST"])
    ap.add_argument("--seed",      type=int, default=7)
    ap.add_argument("--skip-mopso", action="store_true",
                    help="Skip MOPSO (slow) for quick testing")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = ["random", "nsga2", "moead", "spea2", "epsilon"]
    if not args.skip_mopso:
        methods.append("mopso")

    all_results = {}
    for wl in args.workloads:
        print(f"\n{'='*60}\n  {wl} (nonlinear evaluator)\n{'='*60}")
        all_results[wl] = run_workload(wl, Path(args.data), outdir, methods, args.seed)

    (outdir / "benchmark_nonlinear.json").write_text(
        json.dumps(all_results, indent=2))
    print(f"\nSaved {outdir}/benchmark_nonlinear.json")

    # Nonlinear sensitivity analysis
    sens_nl = sensitivity_analysis_nonlinear(Path(args.data), outdir)
    (outdir / "sensitivity_nonlinear.json").write_text(json.dumps(sens_nl, indent=2))
    print(f"Saved {outdir}/sensitivity_nonlinear.json")

    # Summary table
    print("\n" + "="*75)
    print(f"  {'Workload':<14} {'Method':<10} {'n':>5} {'HV':>8} {'Feas':>7} {'t(s)':>6}")
    print("  " + "-"*73)
    for wl, methods_r in all_results.items():
        for m, r in methods_r.items():
            print(f"  {wl:<14} {m:<10} {r['pareto_size']:>5} "
                  f"{r['hv_2d']:>8.2f} {r['feasible_rate']:>6.1%} {r['runtime_s']:>6.1f}")
        print()


if __name__ == "__main__":
    main()