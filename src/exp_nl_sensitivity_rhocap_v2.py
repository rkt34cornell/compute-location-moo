#!/usr/bin/env python3
"""
exp_nl_sensitivity_rhocap_v2.py — E13 coefficient sensitivity with an
admissible upper perturbation of the utilization cap.

Scaling queue_max_utilization (0.92) by +25% gives 1.15, outside the stable
M/M/1 domain (rho < 1). This driver keeps every run of out_nl_sensitivity_v2
except that arm, reruns it with the cap set to RHO_UPPER = 0.99 (the same
methods, seeds, budget, and constraint modes), and recomputes the summary.
All other coefficients keep their -25% / +25% perturbations.

Directions are labelled "lower" and "upper"; the actual perturbed value of
every coefficient is recorded in summary.json.

Output: out_nl_sensitivity_rhocap_v2/{runs.csv, summary.json, config.json}
"""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moo_core as mc
from exp_nonlinear_sensitivity import COEFFS, PERTURB_FRAC
from exp_nl_sensitivity_v2 import run, WL, BUDGET

SRC = Path("out_nl_sensitivity_v2")
OUT = Path("out_nl_sensitivity_rhocap_v2")
RHO = "queue_max_utilization"
RHO_UPPER = 0.99
DIR_MAP = {"baseline": "baseline", "-25%": "lower", "+25%": "upper"}


def summarize(rows, base_params):
    from scipy.stats import wilcoxon, kendalltau, spearmanr
    res = {"perturbed_values": {}}
    for c in COEFFS:
        b0 = float(getattr(base_params, c))
        lo = {r["factor"] for r in rows if r["coefficient"] == c and r["direction"] == "lower"}
        up = {r["factor"] for r in rows if r["coefficient"] == c and r["direction"] == "upper"}
        res["perturbed_values"][c] = {"baseline": b0, "lower": b0 * lo.pop(), "upper": b0 * up.pop()}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        base = {r["seed"]: r["hv"] for r in R if r["coefficient"] == "baseline"}
        b = np.array([base[s] for s in sorted(base)])
        res[m] = {"baseline_hv_mean": float(b.mean()), "baseline_hv_std": float(b.std(ddof=1)), "coeffs": {}}
        for c in COEFFS:
            res[m]["coeffs"][c] = {}
            for d in ["lower", "upper"]:
                x = {r["seed"]: r["hv"] for r in R if r["coefficient"] == c and r["direction"] == d}
                xv = np.array([x[s] for s in sorted(base)])
                delta = 100 * (xv - b) / np.where(b > 0, b, np.nan)
                try:
                    p = float(wilcoxon(xv, b).pvalue) if not np.allclose(xv, b) else 1.0
                except ValueError:
                    p = 1.0
                res[m]["coeffs"][c][d] = {"hv_mean": float(xv.mean()), "hv_std": float(xv.std(ddof=1)),
                                          "delta_pct_mean": float(np.nanmean(delta)),
                                          "delta_pct_std": float(np.nanstd(delta, ddof=1)),
                                          "wilcoxon_p_vs_baseline": p}
            res[m]["coeffs"][c]["importance"] = max(abs(res[m]["coeffs"][c][d]["delta_pct_mean"])
                                                    for d in ["lower", "upper"])
        res[m]["importance_order"] = sorted(COEFFS, key=lambda c: -res[m]["coeffs"][c]["importance"])
    agree = {}
    for i, a in enumerate(methods):
        for bm in methods[i + 1:]:
            ia = [res[a]["coeffs"][c]["importance"] for c in COEFFS]
            ib = [res[bm]["coeffs"][c]["importance"] for c in COEFFS]
            agree[f"{a}_vs_{bm}"] = {"kendall_tau": float(kendalltau(ia, ib).statistic),
                                     "kendall_p": float(kendalltau(ia, ib).pvalue),
                                     "spearman_rho": float(spearmanr(ia, ib).statistic),
                                     "top3_a": res[a]["importance_order"][:3],
                                     "top3_b": res[bm]["importance_order"][:3]}
    res["ordering_agreement"] = agree
    return res


def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())
    base_params = deepcopy(mc.WORKLOAD_NONLINEAR_PARAMS[WL])
    rows = []
    with open(SRC / "runs.csv") as f:
        for r in csv.DictReader(f):
            if r["coefficient"] == RHO and r["direction"] == "+25%":
                continue
            rows.append(dict(coefficient=r["coefficient"], direction=DIR_MAP[r["direction"]],
                             factor=float(r["factor"]), method=r["method"], seed=int(r["seed"]),
                             hv=float(r["hv"]), n_feasible=int(r["n_feasible"])))
    factor = RHO_UPPER / float(getattr(base_params, RHO))
    jobs = [(RHO, "upper", factor, m, s, cfg["violation"], cfg["handling"])
            for m in cfg["methods"] for s in range(11, 11 + cfg["seeds"])]
    with Pool(cfg["jobs"]) as pool:
        rows += list(pool.imap_unordered(run, jobs))
    rows.sort(key=lambda r: (r["method"], r["coefficient"], r["direction"], r["seed"]))
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summarize(rows, base_params), indent=2))
    (OUT / "config.json").write_text(json.dumps({**cfg, "outdir": str(OUT), "source_runs": str(SRC),
                                                 "rho_max_upper": RHO_UPPER,
                                                 "perturb_frac_other": PERTURB_FRAC,
                                                 "budget": BUDGET}, indent=2))


if __name__ == "__main__":
    main()
