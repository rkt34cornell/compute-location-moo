#!/usr/bin/env python3
"""
exp_convergence_v2.py — 20-seed repeat of exp_convergence.py.

Same per-method configuration and the same HV definition as the legacy
experiment: at each checkpoint, the 2-D (latency, cost) hypervolume of the
nondominated set of ALL feasible candidates evaluated so far (an "anytime"
archive HV, not the HV of the algorithm's returned front). Checkpoints every 50
evaluations. Differences from legacy: 20 seeds (11-30) instead of 3, current
data.json, the constraint modes given on the command line, all six main
workloads (IDS is the manuscript figure), and a staircase HV computation that
is mathematically identical to the legacy O(N^2) filter.

Note: MOEA/D's legacy configuration (N=120 weight vectors x 121 generations)
uses 14,520 evaluations versus 7,260 for the other methods; the summary reports
every method at the common 7,260-evaluation checkpoint as well.

Output: out_convergence_v2/{traces/*.npz, convergence_summary.json, convergence_checkpoints.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moo_core as mc
import optimizers_methods
import algorithms_extended
from run_multiseed_benchmark import WCFG, hv2d

METHODS = {
    "mopso": ("mopso", dict(particles=60, iters=120, archive_max=200)),
    "nsga2": ("nsga2", dict(pop_size=60, generations=120)),
    "spea2": ("spea2", dict(pop_size=60, archive_size=60, generations=120)),
    "moead": ("moead", dict(H=8, T=5, generations=120)),
}
K_EVERY = 50
COMMON_BUDGET = 60 * 121  # 7,260


def staircase_hv_curve(V2: np.ndarray, F: np.ndarray, ref: np.ndarray, k_every: int):
    """Running HV of the feasible nondominated set (2-D, minimization)."""
    N = V2.shape[0]
    steps = list(range(k_every, N + 1, k_every))
    if not steps or steps[-1] != N:
        steps.append(N)
    out = []
    for s in steps:
        Vf = V2[:s][F[:s]]
        if Vf.shape[0] == 0:
            out.append(0.0); continue
        o = np.lexsort((Vf[:, 1], Vf[:, 0]))
        Vs = Vf[o]
        keep = Vs[:, 1] < np.minimum.accumulate(np.concatenate([[np.inf], Vs[:-1, 1]]))
        out.append(hv2d(Vs[keep], ref))
    return np.array(steps), np.array(out)


def run(job):
    wl_name, method, seed, outdir, violation, handling = job
    mc.set_constraint_modes(violation, handling)
    wl = mc.load_workload(Path("data.json"), wl_name)
    mk = [m.key for m in wl.metrics]
    kd = {m.key: m.direction for m in wl.metrics}
    lk, ck, ref = WCFG[wl_name]
    trace_v, trace_f = [], []

    def tracer(w, p):
        pm = mc.aggregate_metrics_linear(w, p)
        trace_f.append(bool(mc.feasibility_and_violation(w, pm)[0]))
        v = [pm[lk], pm[ck]]
        if kd[lk] == mc.Direction.MAXIMIZE: v[0] = -v[0]
        if kd[ck] == mc.Direction.MAXIMIZE: v[1] = -v[1]
        trace_v.append(v)
        return pm

    fn_name, kw = METHODS[method]
    fn = getattr(optimizers_methods if fn_name == "mopso" else algorithms_extended, fn_name)
    saved = optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear
    optimizers_methods.aggregate_metrics_linear = tracer
    algorithms_extended.aggregate_metrics_linear = tracer
    t0 = time.perf_counter()
    try:
        fn(workload=wl, metric_keys=mk, seed=seed, **kw)
    finally:
        optimizers_methods.aggregate_metrics_linear, algorithms_extended.aggregate_metrics_linear = saved
    rt = time.perf_counter() - t0
    steps, hv = staircase_hv_curve(np.array(trace_v, float), np.array(trace_f, bool), ref, K_EVERY)
    np.savez(Path(outdir) / "traces" / f"{wl_name}_{method}_seed{seed}.npz", steps=steps, hv=hv)
    return wl_name, method, seed, len(trace_v), rt


def summarize(outdir: Path, workloads, seeds):
    """Per checkpoint: HV mean, SD and fraction of the method's final mean HV.
    Per method: evaluations to 95% / 98% of final; plateau test (mean gain over
    the last 1,000 evaluations < 0.1% of final HV); ordering changes across
    checkpoints up to the common 7,260-evaluation budget."""
    summary, rows = {}, [["workload", "method", "evals", "n_seeds", "hv_mean", "hv_std", "fraction_of_final"]]
    for wl in workloads:
        summary[wl] = {}
        curves = {}
        for m in METHODS:
            C = [np.load(outdir / "traces" / f"{wl}_{m}_seed{s}.npz") for s in seeds]
            steps = C[0]["steps"]
            H = np.array([c["hv"] for c in C])
            curves[m] = (steps, H.mean(0))
            mean, sd = H.mean(0), H.std(0, ddof=1)
            final = float(mean[-1])
            for st, mu, s in zip(steps, mean, sd):
                rows.append([wl, m, int(st), len(seeds), float(mu), float(s), float(mu / final) if final else ""])
            i_common = int(np.searchsorted(steps, COMMON_BUDGET, side="right") - 1)
            reach = lambda frac: int(steps[np.argmax(mean >= frac * final)]) if final > 0 else None
            last = steps >= steps[-1] - 2000
            slope = float(np.polyfit(steps[last], mean[last], 1)[0] * 1000) if last.sum() > 2 else None
            j = int(np.searchsorted(steps, steps[-1] - 1000))
            gain_last_1000 = float(mean[-1] - mean[j])
            summary[wl][m] = {
                "evals_total": int(steps[-1]), "final_hv_mean": final, "final_hv_std": float(sd[-1]),
                "hv_at_common_budget_mean": float(mean[i_common]), "hv_at_common_budget_std": float(sd[i_common]),
                "common_budget": COMMON_BUDGET, "evals_to_95pct_of_final": reach(0.95),
                "evals_to_98pct_of_final": reach(0.98), "slope_last_2000_evals_per_1000": slope,
                "gain_last_1000_evals": gain_last_1000,
                "gain_last_1000_pct_of_final": 100 * gain_last_1000 / final if final else None,
                "plateaued": bool(final and 100 * gain_last_1000 / final < 0.1),
            }
        # ordering over common checkpoints (all methods share the 50-eval grid up to 7,260)
        grid = [st for st in curves["mopso"][0] if st <= COMMON_BUDGET and all(st in set(curves[m][0]) for m in METHODS)]
        def order_at(st):
            vals = {m: curves[m][1][list(curves[m][0]).index(st)] for m in METHODS}
            return ">".join(sorted(vals, key=lambda m: -vals[m]))
        orders = [(int(st), order_at(st)) for st in grid]
        final_order = orders[-1][1]
        changes, prev = [], None
        for st, o in orders:
            if o != prev:
                changes.append({"from_evals": st, "order": o})
                prev = o
        summary[wl]["_ordering"] = {"order_at_common_budget": final_order, "order_changes": changes,
                                    "last_change_at_evals": changes[-1]["from_evals"]}
    (outdir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    with open(outdir / "convergence_checkpoints.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_convergence_v2")
    ap.add_argument("--workloads", nargs="+", default=["IDS", "SUNCATCHER", "LEO_OOS", "DEEP_SPACE_CRUISE", "JWST", "RST"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--violation", default="normalized")
    ap.add_argument("--handling", default="deb")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.outdir); (out / "traces").mkdir(parents=True, exist_ok=True)
    seeds = list(range(11, 11 + args.seeds))
    jobs = [(w, m, s, str(out), args.violation, args.handling) for w in args.workloads
            for m in METHODS for s in seeds
            if not (out / "traces" / f"{w}_{m}_seed{s}.npz").exists()]
    jobs.sort(key=lambda j: j[1] != "mopso")
    with Pool(args.jobs) as pool:
        for r in pool.imap_unordered(run, jobs):
            print("done", *r, flush=True)
    summarize(out, args.workloads, seeds)
    (out / "config.json").write_text(json.dumps({**vars(args), "seeds_list": seeds,
                                                 "methods": {k: v[1] for k, v in METHODS.items()},
                                                 "k_every": K_EVERY}, indent=2))
    print("convergence v2 complete")


if __name__ == "__main__":
    main()
