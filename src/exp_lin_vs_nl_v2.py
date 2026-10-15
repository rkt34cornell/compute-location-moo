#!/usr/bin/env python3
"""
exp_lin_vs_nl_v2.py — linear-optimal policies re-scored under the nonlinear
evaluator.

Same pick rule as exp_lin_vs_nl_validity.py: for each (method, seed) run of the
linear benchmark, the front point with the largest single-point 2-D HV box is
re-evaluated with aggregate_metrics_nonlinear and checked for feasibility.
Differences: reads the v2 fronts (out_benchmark_v2, out_roman_v2), uses only
points that are feasible under the linear evaluator (legacy picks could be
infeasible fallback points), covers all six main workloads plus the ROMAN
variants, and also reports the fraction of ALL linear-front points that stay
feasible.

Output: out_lin_vs_nl_v2/{validity.csv, validity.json}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import moo_core as mc
from run_benchmark_v2 import hv_ref_keys
from run_multiseed_benchmark import to_min_2d

METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon"]


def picks_for(fronts_dir: Path, wl, wl_name, variant):
    mk = [m.key for m in wl.metrics]
    lk, ck, ref = hv_ref_keys(wl_name)
    picks, all_pts = [], []
    tag = variant or "-"
    for m in METHODS:
        for vp in sorted(fronts_dir.glob(f"{wl_name}_{tag}_linear_{m}_seed*_V.npy")):
            V = np.load(vp); P = np.load(str(vp).replace("_V.npy", "_P.npy"))
            feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V], bool)
            V, P = V[feas], P[feas]
            if V.shape[0] == 0:
                continue
            pts = to_min_2d(V, wl, mk, lk, ck)
            box = np.where((pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1]),
                           (ref[0] - pts[:, 0]) * (ref[1] - pts[:, 1]), -1.0)
            i = int(np.argmax(box))
            if box[i] >= 0:
                picks.append({"method": m, "seed": int(vp.stem.split("seed")[1].split("_")[0]), "p": P[i]})
            all_pts.extend(list(P))
    return picks, all_pts


def main():
    out = Path("out_lin_vs_nl_v2"); out.mkdir(exist_ok=True)
    mc.set_constraint_modes("normalized", "deb")
    cases = [(w, None, Path("out_benchmark_v2/fronts")) for w in
             ["SUNCATCHER", "LEO_OOS", "IDS", "DEEP_SPACE_CRUISE", "JWST", "RST"]]
    cases += [(w, v, Path("out_roman_v2/fronts")) for w in ["ROMAN_TRANSIENT", "ROMAN_TRANSIENT_FOLLOWUP"]
              for v in ["A", "B", "C"]]
    rows, js = [], {}
    for wl_name, var, d in cases:
        wl = mc.load_workload(Path("data.json"), wl_name)
        if var:
            wl = mc.apply_odc_variant(wl, var)
        params = mc.WORKLOAD_NONLINEAR_PARAMS.get(wl_name, mc.NonlinearParams())
        picks, all_p = picks_for(d, wl, wl_name, var)
        lk, ck, _ = hv_ref_keys(wl_name)

        def nl_feas(p):
            return mc.feasibility_and_violation(wl, mc.aggregate_metrics_nonlinear(wl, p, params))[0]

        pf = [nl_feas(pk["p"]) for pk in picks]
        nl_lat = [mc.aggregate_metrics_nonlinear(wl, pk["p"], params)[lk] for pk in picks]
        allf = [nl_feas(p) for p in all_p]
        label = wl_name + (f"|{var}" if var else "")
        by_m = {m: float(np.mean([f for f, pk in zip(pf, picks) if pk["method"] == m]))
                for m in METHODS if any(pk["method"] == m for pk in picks)}
        rec = {"label": label, "n_picks": len(picks), "n_picks_nl_feasible": int(sum(pf)),
               "pct_picks_nl_feasible": 100.0 * float(np.mean(pf)) if pf else float("nan"),
               "pct_by_method": by_m, "n_front_points": len(all_p),
               "pct_all_front_points_nl_feasible": 100.0 * float(np.mean(allf)) if allf else float("nan"),
               "nl_latency_median_of_picks": float(np.median(nl_lat)) if nl_lat else float("nan")}
        js[label] = rec
        rows.append([label, rec["n_picks"], rec["n_picks_nl_feasible"], round(rec["pct_picks_nl_feasible"], 2),
                     rec["n_front_points"], round(rec["pct_all_front_points_nl_feasible"], 2),
                     rec["nl_latency_median_of_picks"]])
        print(label, rec["pct_picks_nl_feasible"], rec["pct_all_front_points_nl_feasible"], flush=True)
    with open(out / "validity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_picks", "n_picks_nl_feasible", "pct_picks_nl_feasible",
                    "n_front_points", "pct_all_front_points_nl_feasible", "nl_latency_median_of_picks"])
        w.writerows(rows)
    (out / "validity.json").write_text(json.dumps(js, indent=2))


if __name__ == "__main__":
    main()
