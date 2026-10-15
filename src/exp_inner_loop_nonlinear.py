#!/usr/bin/env python3
"""
exp_inner_loop_nonlinear.py — The Phase-2 nonlinear
benchmark currently runs each optimizer against a *proxy* workload whose
tier values were precomputed at pure-tier policies, then re-evaluates the
final Pareto set under the full nonlinear evaluator. This biases the
optimizer because interior nonlinear effects (queueing, fixed setup,
SDC compounding) never enter the search.

This experiment installs the full nonlinear aggregator directly inside the
optimizer's evaluation loop (via monkey-patching the names that
optimizers_methods and algorithms_extended look up). Compares HV of the
true inner-loop run vs the proxy-based run from out_nonlinear/ on IDS and
SUNCATCHER, using MOPSO and NSGA-II.

Output: out_inner_loop/inner_loop.csv, inner_loop_report.md
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from moo_core import (
    Direction,
    WORKLOAD_NONLINEAR_PARAMS,
    aggregate_metrics_linear,
    aggregate_metrics_nonlinear,
    feasibility_and_violation,
    load_workload,
    pareto_front_indices,
)
import optimizers_methods
import algorithms_extended
from optimizers_methods import mopso
from algorithms_extended import nsga2


WORKLOADS = ["IDS", "SUNCATCHER"]
SEEDS = [11, 12, 13]

HV_KEYS = {
    "IDS":        ("Latency p99 (ms)",  "Cost per task (USD)",  np.array([260.0, 3.2])),
    "SUNCATCHER": ("Latency p99 (ms)",  "Cost per unit (USD)",  np.array([5_000_000.0, 15.0])),
}


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


def install_nl(params):
    nl_eval = lambda w, p: aggregate_metrics_nonlinear(w, p, params)
    saved_o = optimizers_methods.aggregate_metrics_linear
    saved_a = algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = nl_eval
    algorithms_extended.aggregate_metrics_linear = nl_eval
    def restore():
        optimizers_methods.aggregate_metrics_linear = saved_o
        algorithms_extended.aggregate_metrics_linear = saved_a
    return restore


def proj_pts(V, mk, lk, ck, kd):
    li, ci = mk.index(lk), mk.index(ck)
    pts = V[:, [li, ci]].copy()
    if kd.get(lk) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
    if kd.get(ck) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
    return pts


def load_proxy_result(wl_name, method):
    """Try to load Phase-2 proxy result for HV comparison; returns None if missing."""
    p = Path("out_nonlinear") / f"{wl_name}_{method}_V.npy"
    return np.load(str(p)) if p.exists() else None


def main():
    outdir = Path("out_inner_loop"); outdir.mkdir(exist_ok=True)
    data_path = Path("data.json")

    rows = [["workload", "method", "mode", "seed", "hv_2d", "n_pareto", "runtime_s"]]
    summary = {}

    for wl_name in WORKLOADS:
        wl = load_workload(data_path, wl_name)
        mk = [m.key for m in wl.metrics]
        kd = {m.key: m.direction for m in wl.metrics}
        params = WORKLOAD_NONLINEAR_PARAMS[wl_name]
        lk, ck, ref = HV_KEYS[wl_name]

        summary[wl_name] = {"inner_loop": {}, "proxy_loaded": {}}

        for method, fn, kw in [
            ("mopso", mopso, dict(particles=60, iters=80, archive_max=200)),
            ("nsga2", nsga2, dict(pop_size=60, generations=80)),
        ]:
            hvs = []
            for seed in SEEDS:
                restore = install_nl(params)
                try:
                    t0 = time.perf_counter()
                    P, V = fn(workload=wl, metric_keys=mk, seed=seed, **kw)
                    rt = time.perf_counter() - t0
                finally:
                    restore()
                pts = proj_pts(V, mk, lk, ck, kd)
                hv = hv2d(pts, ref)
                hvs.append(hv)
                rows.append([wl_name, method, "inner_loop", seed,
                             round(hv, 3), int(P.shape[0]), round(rt, 1)])
                print(f"  {wl_name:<11s} {method:<6s} inner seed={seed} "
                      f"HV={hv:>10.2f}  n={P.shape[0]:>4d}  t={rt:.1f}s")

            summary[wl_name]["inner_loop"][method] = {
                "hv_mean":   float(np.mean(hvs)),
                "hv_std":    float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
                "seeds":     SEEDS,
                "hv_values": [float(x) for x in hvs],
            }

            # Compare to proxy-loop result if available
            V_proxy = load_proxy_result(wl_name, method)
            if V_proxy is not None and V_proxy.shape[0] > 0:
                pts_p = proj_pts(V_proxy, mk, lk, ck, kd)
                hv_p = hv2d(pts_p, ref)
                summary[wl_name]["proxy_loaded"][method] = {
                    "hv_proxy_phase2": hv_p,
                    "n_pareto_proxy": int(V_proxy.shape[0]),
                }
                rows.append([wl_name, method, "proxy_phase2", "n/a",
                             round(hv_p, 3), int(V_proxy.shape[0]), 0.0])

    with open(outdir / "inner_loop.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "inner_loop.json").write_text(json.dumps(summary, indent=2))

    md = ["# Inner-loop vs proxy nonlinear evaluator", "",
          "Mean HV across 3 seeds: full nonlinear evaluator installed inside the "
          "optimizer's evaluation loop, vs the Phase-2 proxy approach "
          "(precomputed vertex values + post-hoc re-evaluation).", ""]
    for wl_name in WORKLOADS:
        md.append(f"## {wl_name}")
        md.append("")
        md.append("| Method | Inner-loop HV (mean±std) | Proxy HV (Phase 2) | Δ (inner − proxy) |")
        md.append("|---|---:|---:|---:|")
        for method in ["mopso", "nsga2"]:
            inner = summary[wl_name]["inner_loop"].get(method, {})
            proxy = summary[wl_name]["proxy_loaded"].get(method, {})
            ih = inner.get("hv_mean", float("nan"))
            isd = inner.get("hv_std", float("nan"))
            ph = proxy.get("hv_proxy_phase2", float("nan"))
            delta = (ih - ph) if not (np.isnan(ih) or np.isnan(ph)) else float("nan")
            md.append(f"| {method} | {ih:.2f} ± {isd:.2f} | "
                      f"{ph:.2f} | {delta:+.2f} |")
        md.append("")
    (outdir / "inner_loop_report.md").write_text("\n".join(md))

    print(f"\nWrote {outdir}/inner_loop.csv, inner_loop.json, inner_loop_report.md")


if __name__ == "__main__":
    main()
