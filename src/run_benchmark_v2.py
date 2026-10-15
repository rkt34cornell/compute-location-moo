#!/usr/bin/env python3
"""
run_benchmark_v2.py — v2 benchmark harness.

Differences from run_multiseed_benchmark.py / run_roman_multiseed.py
(legacy outputs are never touched; everything goes to --outdir):

  * Constraint modes are explicit: --violation {raw,normalized} and
    --handling {legacy_penalty,deb} (see moo_core.set_constraint_modes).
  * Every returned front is re-checked for feasibility. A run that retains no
    feasible solution (the algorithm returned its least-violation fallback) is
    scored as "no feasible front": hv = 0, n_front = 0, feasible_found = False.
    Legacy scoring computed HV on such infeasible fallback points.
  * Function evaluations and feasible evaluations are counted per run.
  * Resumable: one JSON record per run in <outdir>/runs/.

Suites
  main      6 workloads x {random, mopso, nsga2, spea2, moead, epsilon,
            epsilon2d} x 20 seeds, linear evaluator, 16,000-eval budget.
  roman     ROMAN_TRANSIENT{,_FOLLOWUP} x variants A/B/C x {linear, nonlinear}
            x the same methods x 20 seeds, 12,000-eval budget.
  archive   MOPSO archive-cap study: caps 50/100/250/350/500 on selected
            workloads, 16,000-eval budget.

epsilon is the legacy formulation (all non-primary metrics epsilon-
constrained at a common level index); epsilon2d is the supplementary
bi-objective form (primary = latency, epsilon on the HV-plane cost metric).

Usage
  python run_benchmark_v2.py --suite main --outdir out_benchmark_v2
  python run_benchmark_v2.py --suite main --violation raw --handling legacy_penalty \
      --workloads IDS --outdir out_benchmark_v2_ids_legacy
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import moo_core as mc
import optimizers_methods
import algorithms_extended
from run_multiseed_benchmark import WCFG, configs_for_budget as main_cfgs, hv2d, to_min_2d
from run_roman_multiseed import configs_for_budget as roman_cfgs, HV_REF as ROMAN_HV_REF

MAIN_WORKLOADS = ["SUNCATCHER", "LEO_OOS", "IDS", "DEEP_SPACE_CRUISE", "JWST", "RST"]
MAIN_SEED0 = {"IDS": 1000, "SUNCATCHER": 1000, "JWST": 1000, "RST": 1000,
              "LEO_OOS": 3000, "DEEP_SPACE_CRUISE": 3000}
ROMAN_WORKLOADS = ["ROMAN_TRANSIENT", "ROMAN_TRANSIENT_FOLLOWUP"]
METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon", "epsilon2d"]
ARCHIVE_CAPS = [50, 100, 250, 350, 500]


# ── job description ───────────────────────────────────────────────────────────

def job_id(j: dict) -> str:
    parts = [j["workload"], j["variant"] or "-", j["evaluator"], j["method"]]
    if j.get("archive_max") is not None:
        parts.append(f"arch{j['archive_max']}")
    parts.append(f"seed{j['seed']}")
    return "_".join(parts)


def hv_ref_keys(wl_name: str) -> Tuple[str, str, np.ndarray]:
    if wl_name in WCFG:
        return WCFG[wl_name]
    return ("Latency p99 (ms)", "Cost per task (USD)", ROMAN_HV_REF)


# ── evaluation counting ───────────────────────────────────────────────────────

class _Counter:
    def __init__(self):
        self.n = 0; self.n_feas = 0

    def wrap(self, fn):
        def inner(workload, pm):
            r = fn(workload, pm)
            self.n += 1
            self.n_feas += int(bool(r[0]))
            return r
        return inner


def _random_baseline(wl, mk, n, seed, evaluate, counter):
    """Legacy random baseline (vertices + n Dirichlet draws) with the
    documented fallback: least violation among all evaluated candidates."""
    rng = np.random.default_rng(seed)
    J = len(wl.tiers)
    P = np.vstack([np.eye(J), rng.dirichlet(np.ones(J), size=n)])
    raw, fs, vs = [], [], []
    for p in P:
        pm = evaluate(wl, p)
        f, v, _ = mc.feasibility_and_violation(wl, pm)
        counter.n += 1; counter.n_feas += int(f)
        raw.append([pm[k] for k in mk]); fs.append(f); vs.append(v)
    V = np.array(raw); fs = np.array(fs); fi = np.where(fs)[0]
    if fi.size == 0:
        b = int(np.argmin(vs))
        return P[[b]], V[[b]]
    Vf = V[fi]
    par = mc.pareto_front_indices(Vf, wl.metrics, mk)
    return P[fi[par]], Vf[par]


def run_job(j: dict) -> dict:
    mc.set_constraint_modes(j["violation"], j["handling"])
    wl = mc.load_workload(Path(j["data"]), j["workload"])
    if j["variant"]:
        wl = mc.apply_odc_variant(wl, j["variant"])
    mk = [m.key for m in wl.metrics]
    params = mc.WORKLOAD_NONLINEAR_PARAMS.get(j["workload"], mc.NonlinearParams())
    if j["evaluator"] == "nonlinear":
        evaluate = lambda w, p: mc.aggregate_metrics_nonlinear(w, p, params)
    else:
        evaluate = mc.aggregate_metrics_linear

    cfg = dict(j["cfg"])
    counter = _Counter()
    saved = (optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear,
             optimizers_methods.feasibility_and_violation, algorithms_extended.feasibility_and_violation)
    optimizers_methods.aggregate_metrics_linear = evaluate
    algorithms_extended.aggregate_metrics_linear = evaluate
    optimizers_methods.feasibility_and_violation = counter.wrap(mc.feasibility_and_violation)
    algorithms_extended.feasibility_and_violation = counter.wrap(mc.feasibility_and_violation)
    eps_diag = None
    t0 = time.perf_counter()
    try:
        m = j["method"]
        if m == "random":
            P, V = _random_baseline(wl, mk, cfg["n"], j["seed"], evaluate, counter)
        elif m == "mopso":
            P, V = optimizers_methods.mopso(workload=wl, metric_keys=mk, seed=j["seed"], **cfg)
        elif m in ("nsga2", "spea2", "moead"):
            P, V = getattr(algorithms_extended, m)(workload=wl, metric_keys=mk, seed=j["seed"], **cfg)
        elif m in ("epsilon", "epsilon2d"):
            lk, ck, _ = hv_ref_keys(j["workload"])
            sec = [ck] if m == "epsilon2d" else None
            P, V = algorithms_extended.epsilon_constraint(
                workload=wl, metric_keys=mk, primary_key=lk if m == "epsilon2d" else None,
                secondary_keys=sec, seed=j["seed"], **cfg)
            eps_diag = dict(algorithms_extended.EPSILON_LAST_RUN)
        else:
            raise ValueError(m)
    finally:
        (optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear,
         optimizers_methods.feasibility_and_violation, algorithms_extended.feasibility_and_violation) = saved
    rt = time.perf_counter() - t0

    # Re-check feasibility of every returned point
    feas_rows = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, row)))[0] for row in V], dtype=bool)
    Vf = V[feas_rows]
    lk, ck, ref = hv_ref_keys(j["workload"])
    pts = to_min_2d(Vf, wl, mk, lk, ck) if Vf.shape[0] else np.zeros((0, 2))
    hv = hv2d(pts, ref) if pts.shape[0] else 0.0

    out = Path(j["outdir"])
    stem = job_id(j)
    np.save(out / "fronts" / f"{stem}_P.npy", P)
    np.save(out / "fronts" / f"{stem}_V.npy", V)
    rec = {**{k: v for k, v in j.items() if k != "outdir"},
           "job_id": stem, "hv": float(hv), "n_returned": int(V.shape[0]),
           "n_front": int(feas_rows.sum()), "feasible_found": bool(feas_rows.all() and V.shape[0] > 0),
           "n_evals": counter.n, "n_feasible_evals": counter.n_feas,
           "search_feasible_rate": counter.n_feas / counter.n if counter.n else float("nan"),
           "runtime_s": rt, "epsilon_diag": eps_diag,
           "violation_mode": mc.VIOLATION_MODE, "handling_mode": mc.CONSTRAINT_HANDLING}
    tmp = out / "runs" / f"{stem}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, out / "runs" / f"{stem}.json")
    return rec


# ── suite construction ────────────────────────────────────────────────────────

def method_cfg(method: str, cfgs: Dict[str, dict]) -> dict:
    return dict(cfgs["epsilon"] if method == "epsilon2d" else cfgs[method])


def build_jobs(args) -> List[dict]:
    base = dict(violation=args.violation, handling=args.handling, data=args.data, outdir=args.outdir)
    jobs = []
    if args.suite == "main":
        cfgs = main_cfgs(16000)
        for wl in args.workloads or MAIN_WORKLOADS:
            for m in args.methods:
                for s in range(args.seeds):
                    jobs.append(dict(base, suite="main", workload=wl, variant=None, evaluator="linear",
                                     method=m, seed=MAIN_SEED0[wl] + s, cfg=method_cfg(m, cfgs),
                                     budget=16000))
    elif args.suite == "roman":
        cfgs = roman_cfgs(12000)
        for wl in args.workloads or ROMAN_WORKLOADS:
            for var in ["A", "B", "C"]:
                for ev in ["linear", "nonlinear"]:
                    for m in args.methods:
                        for s in range(args.seeds):
                            jobs.append(dict(base, suite="roman", workload=wl, variant=var, evaluator=ev,
                                             method=m, seed=2000 + s, cfg=method_cfg(m, cfgs),
                                             budget=12000))
    elif args.suite == "archive":
        cfgs = main_cfgs(16000)
        for wl in args.workloads:
            for cap in ARCHIVE_CAPS:
                for s in range(args.seeds):
                    cfg = dict(cfgs["mopso"], archive_max=cap)
                    jobs.append(dict(base, suite="archive", workload=wl, variant=None, evaluator="linear",
                                     method="mopso", seed=MAIN_SEED0[wl] + s, cfg=cfg,
                                     archive_max=cap, budget=16000))
    # longest jobs first for load balance
    jobs.sort(key=lambda j: (j["method"] != "mopso", j["method"] != "random"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["main", "roman", "archive"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--violation", default="normalized", choices=mc.VIOLATION_MODES)
    ap.add_argument("--handling", default="deb", choices=mc.CONSTRAINT_HANDLING_MODES)
    ap.add_argument("--workloads", nargs="+")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.suite == "archive" and not args.workloads:
        ap.error("--suite archive needs --workloads")

    out = Path(args.outdir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "fronts").mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    todo = [j for j in jobs if not (out / "runs" / f"{job_id(j)}.json").exists()]
    (out / "suite_config.json").write_text(json.dumps({
        "suite": args.suite, "violation": args.violation, "handling": args.handling,
        "data": args.data, "data_mtime": os.path.getmtime(args.data),
        "n_jobs": len(jobs), "methods": args.methods, "seeds": args.seeds,
        "workloads": args.workloads, "numpy": np.__version__,
        "python": os.sys.version}, indent=2))
    print(f"{args.suite}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    done = 0
    with Pool(args.jobs) as pool:
        for rec in pool.imap_unordered(run_job, todo):
            done += 1
            print(f"[{done}/{len(todo)}] {rec['job_id']} hv={rec['hv']:.4g} n={rec['n_front']} "
                  f"feas={rec['feasible_found']} evals={rec['n_evals']} t={rec['runtime_s']:.0f}s", flush=True)
    print("suite complete", flush=True)


if __name__ == "__main__":
    main()
