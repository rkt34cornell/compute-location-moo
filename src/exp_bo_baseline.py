#!/usr/bin/env python3
"""
exp_bo_baseline.py — Bayesian optimization baseline (qEHVI from BoTorch).

Baseline: Bayesian optimization (qEHVI from BoTorch) on
one workload. We use qExpectedHypervolumeImprovement (qEHVI) with feasibility
constraint encoded as an outcome.

Search-space handling on the simplex:
  - BoTorch optimizes on a box. We search on [0,1]^J and project each candidate
    to Delta^{J-1} via moo_core.project_to_simplex before evaluating the
    objectives + constraint.
  - The projection means many points in the box collapse to the same simplex
    policy. BoTorch's GP still finds promising regions, just with some wasted
    points near the box boundary.

Workloads: IDS (linear+nonlinear) and ROMAN_TRANSIENT Variant B
           (linear+nonlinear) — the headline cases.
Budget:   16 Sobol init + 64 BO iterations = 80 total evaluations.
          (Reminder: evolutionary baselines use ~12-16k.)
Seeds:    5 per (workload, evaluator) — BO is more sample-efficient so 5
          gives a stable HV estimate.

Output: out_bo_baseline/<WL>_<EV>_seed<S>_{P,V}.npy
        bo_summary.json with mean/std HV vs evolutionary baseline.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
warnings.filterwarnings("ignore")

from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

from moo_core import (
    Direction,
    WORKLOAD_NONLINEAR_PARAMS,
    aggregate_metrics_linear,
    aggregate_metrics_nonlinear,
    apply_odc_variant,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
    project_to_simplex,
)

torch.set_default_dtype(torch.double)

# ── Run plan ─────────────────────────────────────────────────────────────────
# (workload_name, variant_or_None, evaluator)
RUNS = [
    ("IDS",             None, "linear"),
    ("IDS",             None, "nonlinear"),
    ("ROMAN_TRANSIENT", "B",  "linear"),
    ("ROMAN_TRANSIENT", "B",  "nonlinear"),
]
SEEDS = [11, 12, 13, 14, 15]
N_INIT = 16
N_BO_ITER = 64
BATCH_Q = 1   # single-point acquisition per BO iteration

HV_KEYS = {
    "IDS":             ("Latency p99 (ms)",  "Cost per task (USD)",  np.array([260.0, 3.2])),
    "ROMAN_TRANSIENT": ("Latency p99 (ms)",  "Cost per task (USD)",  np.array([1.0e8, 0.12])),
}


def hv2d(pts, ref):
    """Min-space 2D hypervolume (same convention as run_multiseed_benchmark)."""
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0] - x) * (pv - y); pv = y
    return hv


def to_min_proj(V, mk, lk, ck, kd):
    li, ci = mk.index(lk), mk.index(ck)
    pts = V[:, [li, ci]].copy()
    if kd.get(lk) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
    if kd.get(ck) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
    return pts


def make_evaluator(wl, mk, lk, ck, kd, evaluator: str, params):
    """Closure: x_box (n, J) tensor in [0,1]^J -> (Y, C, P, V) tensors.

    Y has shape (n, 2): -latency, -cost (botorch maximizes; we minimize).
    C has shape (n, 1): -viol so that C[i] >= 0 == feasible.
    P numpy   (n, J):   policies on simplex.
    V numpy   (n, d):   full metric vectors (for downstream analysis).
    """
    li, ci = mk.index(lk), mk.index(ck)
    sign_lk = 1.0 if kd.get(lk) == Direction.MINIMIZE else -1.0
    sign_ck = 1.0 if kd.get(ck) == Direction.MINIMIZE else -1.0

    def evaluate(x_box: torch.Tensor):
        x_np = x_box.detach().numpy()
        n = x_np.shape[0]
        P = np.stack([project_to_simplex(x_np[i]) for i in range(n)], axis=0)
        Y = np.zeros((n, 2))
        C = np.zeros((n, 1))
        V = np.zeros((n, len(mk)))
        for i in range(n):
            if evaluator == "linear":
                pm = aggregate_metrics_linear(wl, P[i])
            else:
                pm = aggregate_metrics_nonlinear(wl, P[i], params)
            V[i] = [pm[k] for k in mk]
            feas, viol, _ = feasibility_and_violation(wl, pm)
            # qEHVI maximises Y; we want to MIN latency and cost.
            lat  = pm.get(lk, np.nan); cost = pm.get(ck, np.nan)
            Y[i, 0] = -(sign_lk * lat)
            Y[i, 1] = -(sign_ck * cost)
            # Outcome >= 0 means feasible
            C[i, 0] = -float(viol)
        return (torch.tensor(Y), torch.tensor(C), P, V)
    return evaluate


def bo_run(wl, mk, lk, ck, kd, evaluator: str, params, ref_point: np.ndarray,
            seed: int):
    """One full BO run. Returns (P_pareto, V_pareto, hv_curve, runtime_s)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    J = len(wl.tiers)

    bounds = torch.stack([torch.zeros(J), torch.ones(J)])  # (2, J)

    evaluate = make_evaluator(wl, mk, lk, ck, kd, evaluator, params)

    # ── Init: Sobol + Dirichlet to cover the simplex vertices ───────────
    t0 = time.perf_counter()
    sobol_x = draw_sobol_samples(bounds=bounds, n=N_INIT, q=1, seed=seed).squeeze(1)
    # Replace some inits with Dirichlet to ensure interior coverage
    diri = torch.tensor(rng.dirichlet(np.ones(J), size=4), dtype=torch.double)
    x_train = torch.cat([sobol_x, diri], dim=0)
    Y_train, C_train, P_train, V_train = evaluate(x_train)

    # qEHVI ref point in maximization form
    ref_torch = torch.tensor([-ref_point[0], -ref_point[1]], dtype=torch.double)

    hv_curve = []
    for step in range(N_BO_ITER):
        # Standardize Y for GP fitting; C kept raw
        try:
            models = [
                SingleTaskGP(x_train, Y_train[:, i:i+1],
                              input_transform=Normalize(d=J),
                              outcome_transform=Standardize(m=1))
                for i in range(2)
            ]
            models.append(SingleTaskGP(x_train, C_train,
                                        input_transform=Normalize(d=J),
                                        outcome_transform=Standardize(m=1)))
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except Exception as e:
            # GP fit can fail rarely on duplicate points; fall back to random
            x_new = torch.tensor(rng.dirichlet(np.ones(J), size=1), dtype=torch.double)
            Y_new, C_new, P_new, V_new = evaluate(x_new)
            x_train = torch.cat([x_train, x_new]); Y_train = torch.cat([Y_train, Y_new])
            C_train = torch.cat([C_train, C_new]); P_train = np.vstack([P_train, P_new])
            V_train = np.vstack([V_train, V_new])
            continue

        # Constraint callable: feasible iff outcome[..., 2] >= 0
        constraints = [lambda Z: -Z[..., 2]]

        # Partitioning over feasible Pareto front (use observed; cheap heuristic)
        feas_mask = (C_train[:, 0] >= 0)
        if feas_mask.sum() >= 1:
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_torch, Y=Y_train[feas_mask, :2]
            )
        else:
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_torch, Y=Y_train[:, :2]
            )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=seed + step)
        acq = qExpectedHypervolumeImprovement(
            model=model, ref_point=ref_torch,
            partitioning=partitioning, sampler=sampler,
            constraints=constraints,
        )

        try:
            cand, _ = optimize_acqf(
                acq_function=acq, bounds=bounds, q=BATCH_Q,
                num_restarts=4, raw_samples=128,
            )
        except Exception:
            cand = torch.tensor(rng.dirichlet(np.ones(J), size=1), dtype=torch.double)

        Y_new, C_new, P_new, V_new = evaluate(cand)
        x_train = torch.cat([x_train, cand])
        Y_train = torch.cat([Y_train, Y_new])
        C_train = torch.cat([C_train, C_new])
        P_train = np.vstack([P_train, P_new])
        V_train = np.vstack([V_train, V_new])

        # Record running best-feasible HV
        feas_now = (C_train[:, 0] >= 0).numpy()
        Vf = V_train[feas_now]
        if Vf.shape[0]:
            pts = to_min_proj(Vf, mk, lk, ck, kd)
            par = pareto_front_indices(Vf, wl.metrics, mk)
            hv_curve.append(float(hv2d(pts[par], ref_point)))
        else:
            hv_curve.append(0.0)

    rt = time.perf_counter() - t0

    # Final Pareto-front extraction
    feas_final = (C_train[:, 0] >= 0).numpy()
    if feas_final.sum() == 0:
        # No feasible: return single least-violation
        best = int(np.argmin(-(C_train[:, 0]).numpy()))
        return (P_train[[best]], V_train[[best]],
                np.array(hv_curve), rt)
    Pf = P_train[feas_final]; Vf = V_train[feas_final]
    par_idx = pareto_front_indices(Vf, wl.metrics, mk)
    return (Pf[par_idx], Vf[par_idx], np.array(hv_curve), rt)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_bo_baseline")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="subset of run labels, e.g. IDS_linear IDS_nonlinear")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    summary = {}

    for wl_name, variant, evaluator in RUNS:
        label = f"{wl_name}" + (f"_V{variant}" if variant else "") + f"_{evaluator}"
        if args.runs and label not in args.runs:
            continue
        wl_base = load_workload(Path("data.json"), wl_name)
        wl = apply_odc_variant(wl_base, variant) if variant else wl_base
        mk = [m.key for m in wl.metrics]
        kd = {m.key: m.direction for m in wl.metrics}
        lk, ck, ref = HV_KEYS[wl_name]
        params = WORKLOAD_NONLINEAR_PARAMS.get(wl_name)

        label = f"{wl_name}" + (f"_V{variant}" if variant else "") + f"_{evaluator}"
        print(f"\n=== BO run: {label} ===  (J={len(wl.tiers)}, budget={N_INIT+N_BO_ITER})")
        hv_seeds = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            P, V, hv_curve, rt = bo_run(wl, mk, lk, ck, kd, evaluator, params, ref, seed)
            li, ci = mk.index(lk), mk.index(ck)
            # v2 scoring rule: an infeasible least-violation fallback scores 0
            feas_rows = np.array([feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], dtype=bool)
            Vs = V[feas_rows]
            pts = to_min_proj(Vs, mk, lk, ck, kd) if Vs.shape[0] else np.zeros((0, 2))
            hv = hv2d(pts, ref) if pts.shape[0] else 0.0
            hv_seeds.append(hv)
            stem = f"{label}_seed{seed}"
            np.save(str(outdir / f"{stem}_P.npy"), P)
            np.save(str(outdir / f"{stem}_V.npy"), V)
            np.save(str(outdir / f"{stem}_hvcurve.npy"), hv_curve)
            print(f"  seed={seed}  HV={hv:>10.2f}  n_pareto={P.shape[0]:>3d}  "
                  f"t={rt:.1f}s  final_curve_HV={hv_curve[-1] if len(hv_curve) else 0:.2f}")

        summary[label] = {
            "hv_seeds":  [float(h) for h in hv_seeds],
            "hv_mean":   float(np.mean(hv_seeds)),
            "hv_std":    float(np.std(hv_seeds, ddof=1)) if len(hv_seeds) > 1 else 0.0,
            "n_seeds":   len(hv_seeds),
            "n_eval_budget": N_INIT + N_BO_ITER,
            "n_evals_actual": N_INIT + 4 + N_BO_ITER,  # 16 Sobol + 4 Dirichlet + 64 BO steps
            "violation_mode": __import__("moo_core").VIOLATION_MODE,
        }

    (outdir / "bo_summary.json").write_text(json.dumps(summary, indent=2))

    # ── Side-by-side comparison to evolutionary HV (multiseed) ───────────
    print("\n" + "=" * 90)
    print("  BO vs evolutionary baselines (multiseed HV, all at higher budget)")
    print("=" * 90)
    for label, s in summary.items():
        print(f"  {label:<32s}  BO: {s['hv_mean']:>10.2f} ± {s['hv_std']:.2f} "
              f"({s['n_eval_budget']} evals × {s['n_seeds']} seeds)")
        wl_name = label.split("_")[0]
        # Try to load evolutionary HV for the same workload+evaluator
        if "_V" in label:
            # ROMAN variant
            variant = label.split("_V")[1].split("_")[0]
            ev = label.split("_")[-1]
            ms_key = f"{wl_name}_TRANSIENT|{variant}|{ev}" if wl_name == "ROMAN" else None
            ms_path = Path("out_roman_multiseed") / "summary.json"
            if ms_path.exists() and ms_key:
                ms = json.loads(ms_path.read_text())["summary"].get(ms_key, {})
                for m, v in ms.items():
                    print(f"      vs {m:<10s}  HV: {v['hv_mean']:>10.2f} ± {v['hv_std']:.2f} (~16k evals)")
        else:
            ms_path = Path(f"out_multiseed_{wl_name}") / "summary.json"
            if ms_path.exists():
                ms = json.loads(ms_path.read_text())
                # multiseed summary structure: summary[wl_name][method] under "summary"
                wm = ms.get("summary", {}).get(wl_name, {})
                for m, v in wm.items():
                    print(f"      vs {m:<10s}  HV: {v['hv_mean']:>10.2f} ± {v['hv_std']:.2f} (~16k evals)")
    print(f"\nWrote {outdir}/bo_summary.json + per-seed P/V/hvcurve .npy")


if __name__ == "__main__":
    main()
