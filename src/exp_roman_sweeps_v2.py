#!/usr/bin/env python3
"""
exp_roman_sweeps_v2.py — the four ROMAN sweeps under the v2 methodology: arrival-rate sweep, burst case, quality-threshold sweep,
ODC NRE (cost) sweep.

Each sweep keeps the grid, MOPSO configuration, evaluator (nonlinear) and
workload transform of its legacy script:
  arrival  exp_roman_arrival_sweep: lambda in LAMBDAS, variants A/B/C, MOPSO 50x70 / archive 180, seed 21
  burst    exp_roman_burst:         lambda in {baseline 0.01, burst 0.1}, variants A/B/C, MOPSO 50x70 / 180, seeds 31-33
  quality  exp_roman_quality_sweep: Quality >= q, q in QUALITY_THRESHOLDS, variants A/B/C, MOPSO 50x70 / 180, seed 22
  nre      exp_odc_nre_sensitivity: ODC cost x NRE/1.5B, Variant B, MOPSO 50x60 / 180, seeds 41-43
Change from legacy: every condition is run with 10 seeds that INCLUDE the legacy
seed(s) (arrival 21-30, quality 22-31, burst 31-40, nre 41-50), because 1-3
seeds cannot support mean +/- SD or distribution statements; plus normalized
violation, Deb handling and feasibility-checked scoring (an infeasible fallback
scores HV = 0).

Reported per condition: HV (saturated 1e8 ms / 0.12 USD reference used by the
legacy sweeps, and the tight nonlinear reference 2e6 ms / 0.06 USD of the
manuscript's ROMAN figure), front size, routing of the minimum-cost Pareto
policy, distribution of p_ODC over the pooled Pareto points, and random-policy
feasibility (5,000 Dirichlet samples).

Output: out_roman_sweeps_v2/<sweep>/{runs.csv, summary.json}
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
from run_multiseed_benchmark import hv2d
from exp_roman_arrival_sweep import LAMBDAS
from exp_roman_quality_sweep import QUALITY_THRESHOLDS, with_quality_threshold
from exp_odc_nre_sensitivity import NRE_VALUES_K, override_odc_cost

WL = "ROMAN_TRANSIENT"
LK, CK = "Latency p99 (ms)", "Cost per task (USD)"
REF_SAT = np.array([1.0e8, 0.12])
REF_TIGHT = np.array([2.0e6, 0.06])
SWEEPS = {
    "arrival": dict(seeds=range(21, 31), legacy_seeds=[21], iters=70),
    "burst":   dict(seeds=range(31, 41), legacy_seeds=[31, 32, 33], iters=70),
    "quality": dict(seeds=range(22, 32), legacy_seeds=[22], iters=70),
    "nre":     dict(seeds=range(41, 51), legacy_seeds=[41, 42, 43], iters=60),
}


def conditions(sweep):
    base = mc.WORKLOAD_NONLINEAR_PARAMS[WL]
    if sweep == "arrival":
        return [(v, f"lambda={l}", dict(lam=l)) for v in "ABC" for l in LAMBDAS]
    if sweep == "burst":
        return [(v, r, dict(lam=l)) for v in "ABC" for r, l in
                [("baseline", base.lambda_tasks_per_s), ("burst", base.lambda_burst)]]
    if sweep == "quality":
        return [(v, f"Q>={q}", dict(q=q)) for v in "ABC" for q in QUALITY_THRESHOLDS]
    return [("B", f"NRE={int(n / 1000)}M", dict(nre=n)) for n in NRE_VALUES_K]


def build(variant, spec):
    wl = mc.apply_odc_variant(mc.load_workload(Path("data.json"), WL), variant)
    if "q" in spec:
        wl = with_quality_threshold(wl, spec["q"])
    if "nre" in spec:
        wl = override_odc_cost(wl, spec["nre"])
    params = deepcopy(mc.WORKLOAD_NONLINEAR_PARAMS[WL])
    if "lam" in spec:
        params.lambda_tasks_per_s = float(spec["lam"])
    return wl, params


def pts2(V, mk):
    return V[:, [mk.index(LK), mk.index(CK)]].copy()  # both minimized


def run(job):
    sweep, variant, label, spec, seed, iters, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    wl, params = build(variant, spec)
    mk = [m.key for m in wl.metrics]
    ev = lambda w, p: mc.aggregate_metrics_nonlinear(w, p, params)
    saved = optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = ev
    algorithms_extended.aggregate_metrics_linear = ev
    try:
        P, V = optimizers_methods.mopso(workload=wl, metric_keys=mk, particles=50, iters=iters,
                                        archive_max=180, seed=seed)
    finally:
        optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear = saved
    feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
    P, V = P[feas], V[feas]
    rec = dict(sweep=sweep, variant=variant, condition=label, seed=seed, n_front=int(P.shape[0]),
               hv_sat=0.0, hv_tight=0.0, p_odc=[], minc=None)
    if P.shape[0]:
        X = pts2(V, mk)
        rec["hv_sat"] = float(hv2d(X, REF_SAT)); rec["hv_tight"] = float(hv2d(X, REF_TIGHT))
        i = int(np.argmin(X[:, 1]))
        rec["minc"] = {"cost": float(X[i, 1]), "latency_ms": float(X[i, 0]),
                       **{t: float(P[i, j]) for j, t in enumerate(wl.tiers)}}
        if "ODC" in wl.tiers:
            rec["p_odc"] = P[:, wl.tiers.index("ODC")].tolist()
    return rec


def feas_rate(args):
    variant, spec = args
    wl, params = build(variant, spec)
    Ps = np.random.default_rng(42).dirichlet(np.ones(len(wl.tiers)), 5000)
    return float(np.mean([mc.feasibility_and_violation(wl, mc.aggregate_metrics_nonlinear(wl, p, params))[0] for p in Ps]))


def summarize(sweep, recs, conds, fr):
    out = []
    for (variant, label, spec), f in zip(conds, fr):
        R = [r for r in recs if r["variant"] == variant and r["condition"] == label]
        hs, ht = np.array([r["hv_sat"] for r in R]), np.array([r["hv_tight"] for r in R])
        po = np.concatenate([np.array(r["p_odc"]) for r in R]) if any(r["p_odc"] for r in R) else np.array([])
        mins = [r["minc"] for r in R if r["minc"]]
        tiers = [k for k in (mins[0] if mins else {}) if k not in ("cost", "latency_ms")]
        row = {"sweep": sweep, "variant": variant, "condition": label, **{k: v for k, v in spec.items()},
               "n_seeds": len(R), "seeds_feasible": int(sum(r["n_front"] > 0 for r in R)),
               "hv_sat_mean": float(hs.mean()), "hv_sat_sd": float(hs.std(ddof=1)),
               "hv_tight_mean": float(ht.mean()), "hv_tight_sd": float(ht.std(ddof=1)),
               "front_mean": float(np.mean([r["n_front"] for r in R])),
               "random_feasible_pct": 100 * f,
               "min_cost_mean": float(np.mean([m["cost"] for m in mins])) if mins else None,
               "min_cost_latency_ms_mean": float(np.mean([m["latency_ms"] for m in mins])) if mins else None}
        for t in tiers:
            row[f"min_cost_policy_{t}_mean"] = float(np.mean([m[t] for m in mins]))
        if po.size:
            q = np.percentile(po, [5, 50, 95])
            row.update({"p_odc_mean": float(po.mean()), "p_odc_sd": float(po.std(ddof=1)), "p_odc_median": float(q[1]),
                        "p_odc_p5": float(q[0]), "p_odc_p95": float(q[2]), "p_odc_min": float(po.min()),
                        "p_odc_max": float(po.max())})
        legacy = [r for r in R if r["seed"] in SWEEPS[sweep]["legacy_seeds"]]
        row["hv_sat_mean_legacy_seeds_only"] = float(np.mean([r["hv_sat"] for r in legacy])) if legacy else None
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS))
    ap.add_argument("--outdir", default="out_roman_sweeps_v2")
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    for sweep in args.sweeps:
        out = Path(args.outdir) / sweep; out.mkdir(parents=True, exist_ok=True)
        cfg = SWEEPS[sweep]; conds = conditions(sweep)
        jobs = [(sweep, v, lab, spec, s, cfg["iters"], args.violation, args.handling)
                for (v, lab, spec) in conds for s in cfg["seeds"]]
        with Pool(args.jobs) as pool:
            recs = pool.map(run, jobs)
            fr = pool.map(feas_rate, [(v, spec) for (v, lab, spec) in conds])
        rows = summarize(sweep, recs, conds, fr)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        with open(out / "runs.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["variant", "condition", "seed", "n_front", "hv_sat", "hv_tight", "p_odc_mean"])
            for r in recs:
                w.writerow([r["variant"], r["condition"], r["seed"], r["n_front"], r["hv_sat"], r["hv_tight"],
                            np.mean(r["p_odc"]) if r["p_odc"] else ""])
        (out / "summary.json").write_text(json.dumps({"config": {**{k: list(v) if isinstance(v, range) else v
                                                                     for k, v in cfg.items()},
                                                                  "mopso": dict(particles=50, iters=cfg["iters"], archive_max=180),
                                                                  "evaluator": "nonlinear", "workload": WL,
                                                                  "violation": args.violation, "handling": args.handling},
                                                       "rows": rows}, indent=2))
        print(f"{sweep}: {len(jobs)} runs -> {out}", flush=True)
    print("roman sweeps v2 complete")


if __name__ == "__main__":
    main()
