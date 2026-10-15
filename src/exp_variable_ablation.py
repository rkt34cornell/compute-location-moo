#!/usr/bin/env python3
"""
exp_variable_ablation.py — Strip data.json down to the
original 8 variables from Thummala & Falco (2025), then rerun the
multi-method benchmark. Compare the 8-var Pareto fronts to the 42-var
fronts (from out_multiseed_<WL>/) to show the variable expansion actually
shifts the optimization, not just the bookkeeping.

The original 8 variables (per the paper) cover: latency, success prob,
cost, energy, throughput, link availability, security, radiation margin.
Concrete keys used (best match in current data.json):
   Latency p99 (ms)            (or Latency (hrs))
   Success probability
   Cost per task (USD)         (or Cost per unit / Cost per obs)
   Energy per task (J)         (or per unit / per obs)
   Throughput (GFLOPS)
   Link availability
   Security protocol level
   Rad dose margin (krad)

Only constraints over the retained metrics survive; others are dropped.

Output: out_var_ablation/ablation.csv, ablation_report.md, comparison.png
"""
from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moo_core import (
    Direction,
    Metric,
    Threshold,
    Workload,
    aggregate_metrics_linear,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
)
from optimizers_methods import mopso
from algorithms_extended import nsga2, moead


WORKLOADS = ["IDS", "SUNCATCHER", "JWST", "RST"]
METHODS = [("random", None, None), ("mopso", mopso, dict(particles=60, iters=80, archive_max=200)),
           ("nsga2", nsga2, dict(pop_size=60, generations=80)),
           ("moead", moead, dict(H=8, T=5, generations=80))]
SEEDS = [11, 12, 13]  # 3 seeds for the ablation; full 20 seeds would double runtime

# Original-8 keys, with fallbacks per workload (latency/cost/energy unit differs)
ORIG_8_PATTERNS = [
    ["Latency p99 (ms)", "Latency (hrs)"],
    ["Success probability"],
    ["Cost per task (USD)", "Cost per unit (USD)", "Cost per obs (USD)"],
    ["Energy per task (J)", "Energy per unit (J)", "Energy per obs (kJ)"],
    ["Throughput (GFLOPS)"],
    ["Link availability"],
    ["Security protocol level"],
    ["Rad dose margin (krad)"],
]

HV_KEYS = {
    "IDS":        ("Latency p99 (ms)",  "Cost per task (USD)",  np.array([260.0, 3.2])),
    "SUNCATCHER": ("Latency p99 (ms)",  "Cost per unit (USD)",  np.array([110.0, 14.5])),
    "JWST":       ("Latency (hrs)",     "Cost per obs (USD)",   np.array([50.0, 20.0])),
    "RST":        ("Latency (hrs)",     "Cost per obs (USD)",   np.array([26.0, 25.0])),
}


def pick_keys(metric_keys):
    chosen = []
    for pat in ORIG_8_PATTERNS:
        for cand in pat:
            if cand in metric_keys:
                chosen.append(cand)
                break
    return chosen


def reduce_workload(wl: Workload, preserve_constraints: bool = True) -> Workload:
    """Reduce a workload's metric set to the original-8 keys.

    `preserve_constraints` (default True): if a constraint references a metric
    NOT in the kept set, we still keep that metric (so the constraint
    binds the same way as in the full workload). This disentangles the
    "fewer variables" effect from the "fewer constraints" effect.

    Setting preserve_constraints=False reproduces the original (confounded)
    behavior where reducing metrics also dropped constraints.
    """
    keep_keys = set(pick_keys([m.key for m in wl.metrics]))
    if preserve_constraints:
        # Add back any metric that's referenced by a threshold
        keep_keys = keep_keys | set(wl.thresholds.keys())
    new_metrics = [m for m in wl.metrics if m.key in keep_keys]
    new_tier_values = {t: {k: v for k, v in vs.items() if k in keep_keys}
                       for t, vs in wl.tier_values.items()}
    new_thresholds = {k: thr for k, thr in wl.thresholds.items() if k in keep_keys}
    return Workload(
        name=wl.name + ("_orig8plusCons" if preserve_constraints else "_orig8"),
        tiers=list(wl.tiers),
        metrics=new_metrics,
        tier_values=new_tier_values,
        thresholds=new_thresholds,
        mission_life_years=wl.mission_life_years,
    )


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def random_baseline(wl, mk, n, seed):
    rng = np.random.default_rng(seed)
    J = len(wl.tiers)
    P = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), size=n)])
    raw, fs = [], []
    for p in P:
        pm = aggregate_metrics_linear(wl, p)
        f, _, _ = feasibility_and_violation(wl, pm)
        raw.append([pm[k] for k in mk]); fs.append(f)
    V = np.array(raw); fs = np.array(fs); fi = np.where(fs)[0]
    if fi.size == 0:
        return P[[0]], V[[0]]
    Vf = V[fi]; par = pareto_front_indices(Vf, wl.metrics, mk)
    return P[fi[par]], Vf[par]


def run_method(method, fn, kw, wl, mk, seed):
    if method == "random":
        return random_baseline(wl, mk, n=8000, seed=seed)
    return fn(workload=wl, metric_keys=mk, seed=seed, **kw)


def proj_hv(V, wl, mk, lk, ck, ref):
    if lk not in mk or ck not in mk: return 0.0, 0
    kd = {m.key: m.direction for m in wl.metrics}
    li, ci = mk.index(lk), mk.index(ck)
    pts = V[:, [li, ci]].copy()
    if kd.get(lk) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
    if kd.get(ck) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
    return hv2d(pts, ref), pts.shape[0]


def main():
    outdir = Path("out_var_ablation"); outdir.mkdir(exist_ok=True)
    data_path = Path("data.json")

    rows = [["workload", "method", "n_vars", "seed", "hv_2d", "n_pareto", "runtime_s"]]
    summary = {}

    for wl_name in WORKLOADS:
        wl_full = load_workload(data_path, wl_name)
        wl_8 = reduce_workload(wl_full)
        kept = [m.key for m in wl_8.metrics]
        print(f"\n=== {wl_name} ===  reduced to {len(kept)} vars: {kept}")
        lk, ck, ref = HV_KEYS[wl_name]
        summary[wl_name] = {"kept_keys": kept,
                            "full":  {m[0]: [] for m in METHODS},
                            "orig8": {m[0]: [] for m in METHODS}}

        for label, wl in [("full", wl_full), ("orig8", wl_8)]:
            mk = [m.key for m in wl.metrics]
            n_vars = len(mk)
            for method, fn, kw in METHODS:
                for seed in SEEDS:
                    t0 = time.perf_counter()
                    P, V = run_method(method, fn, kw, wl, mk, seed)
                    rt = time.perf_counter() - t0
                    hv, n_par = proj_hv(V, wl, mk, lk, ck, ref)
                    rows.append([wl_name + ("_orig8" if label == "orig8" else ""),
                                 method, n_vars, seed, round(hv, 3), n_par, round(rt, 1)])
                    summary[wl_name][label][method].append(hv)
                    print(f"  {label:<5s} {method:<8s} seed={seed} "
                          f"HV={hv:>7.2f} n={n_par:>4d} t={rt:.1f}s")

    with open(outdir / "ablation.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "ablation.json").write_text(json.dumps(summary, indent=2))

    # Comparison figure: mean HV per (workload, method) for full vs orig8
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    method_names = [m[0] for m in METHODS]
    x = np.arange(len(method_names))
    width = 0.36
    for ax, wl_name in zip(axes, WORKLOADS):
        full_means = [np.mean(summary[wl_name]["full"][m]) for m in method_names]
        o8_means   = [np.mean(summary[wl_name]["orig8"][m]) for m in method_names]
        ax.bar(x - width/2, full_means, width, label="full (42 vars)", color="#264653")
        ax.bar(x + width/2, o8_means,   width, label="orig8 (8 vars)", color="#e9c46a")
        ax.set_xticks(x); ax.set_xticklabels(method_names, rotation=15)
        ax.set_title(wl_name)
        ax.set_ylabel("HV (mean across seeds)")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].legend()
    plt.suptitle("Variable ablation: 42-var vs original 8-var (Thummala & Falco) on HV",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {outdir}/ablation.csv, ablation.json, comparison.png")


if __name__ == "__main__":
    main()
