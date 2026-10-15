#!/usr/bin/env python3
"""
exp_mission_duration_sweep.py — For SUNCATCHER, sweep
mission duration from 5 to 20 years. Cumulative TID is roughly linear in
years (10 krad/yr in this analysis). For each duration, compute the minimum
orbital-tier radiation tolerance such that feasible_fraction > 0.5.

Produces the "Minimum TID hardness vs mission years" chart.

Output: out_duration_sweep/sweep.csv, sweep_figure.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moo_core import (
    aggregate_metrics_linear,
    feasibility_and_violation,
    load_workload,
)


WL_NAME = "SUNCATCHER"
ORBITAL_TIERS = ["LEO_TPU_CLUSTER", "HYBRID_SPLIT"]
KRAD_PER_YEAR = 10.0
RAD_KEY = "Rad dose margin (krad)"

# Mission durations to sweep (years)
DURATIONS = [3, 5, 7, 10, 12, 15, 20, 25, 30]
# Tolerance sweep per duration (krad)
TOLERANCES = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

TARGET_FEAS_FRAC = 0.5  # threshold defining "viable"


def patched_workload(wl, overrides):
    class Patched:
        pass
    p = Patched()
    p.__dict__.update(wl.__dict__)
    p.tier_values = {
        t: {**wl.tier_values[t],
            **({RAD_KEY: overrides[t]} if t in overrides else {})}
        for t in wl.tiers
    }
    p.metrics = wl.metrics
    p.thresholds = wl.thresholds
    p.tiers = wl.tiers
    p.name = wl.name
    return p


def feasibility_fraction(wl, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(len(wl.tiers)), size=n)
    c = 0
    for p in P:
        f, _, _ = feasibility_and_violation(wl, aggregate_metrics_linear(wl, p))
        if f:
            c += 1
    return c / n


def main():
    outdir = Path("out_duration_sweep"); outdir.mkdir(exist_ok=True)
    wl = load_workload(Path("data.json"), WL_NAME)

    rows = [["duration_yrs", "lifetime_dose_krad", "min_tol_for_feas_gt_50pct", "feas_at_min"]]
    grid = {}  # grid[duration] = list of (tol, feas)
    min_tol = []

    print(f"Sweep mission duration on {WL_NAME}")
    for d in DURATIONS:
        dose = d * KRAD_PER_YEAR
        per_tol = []
        chosen = None
        for tol in TOLERANCES:
            margin = tol - dose
            wlp = patched_workload(wl, {t: margin for t in ORBITAL_TIERS})
            feas = feasibility_fraction(wlp)
            per_tol.append((tol, feas))
            if chosen is None and feas >= TARGET_FEAS_FRAC:
                chosen = (tol, feas)
        grid[d] = per_tol
        if chosen:
            min_tol.append((d, chosen[0], chosen[1]))
            rows.append([d, dose, chosen[0], round(chosen[1], 3)])
            print(f"  duration={d:>3d}yr  dose={dose:>5.0f}krad  "
                  f"min_tol={chosen[0]:>5}krad  feas_at_min={chosen[1]:.1%}")
        else:
            min_tol.append((d, float("inf"), 0.0))
            rows.append([d, dose, "INF", 0.0])
            print(f"  duration={d:>3d}yr  dose={dose:>5.0f}krad  "
                  f"NO viable tolerance up to {TOLERANCES[-1]} krad")

    with open(outdir / "sweep.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (outdir / "sweep_grid.json").write_text(json.dumps({
        "durations": DURATIONS, "tolerances": TOLERANCES,
        "krad_per_year": KRAD_PER_YEAR, "target_feas": TARGET_FEAS_FRAC,
        "grid": {str(d): [(t, f) for t, f in v] for d, v in grid.items()},
        "min_tol": [{"duration_yr": d, "min_tol_krad": t, "feas_at_min": f}
                    for d, t, f in min_tol],
    }, indent=2))

    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: min tolerance vs duration
    valid = [(d, t) for d, t, _ in min_tol if t != float("inf")]
    if valid:
        ds, ts = zip(*valid)
        axes[0].plot(ds, ts, "o-", color="#2a6e8f", linewidth=2, markersize=8)
        axes[0].set_xlabel("Mission duration (years)")
        axes[0].set_ylabel(f"Min orbital TID tolerance (krad)\nfor feasibility ≥ {TARGET_FEAS_FRAC:.0%}")
        axes[0].set_title(f"{WL_NAME}: required radiation hardening")
        axes[0].grid(True, alpha=0.3)
        for d, t in zip(ds, ts):
            axes[0].annotate(f"{t}", (d, t), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=9)

    # Right: heatmap of (duration, tolerance) -> feas
    F = np.zeros((len(DURATIONS), len(TOLERANCES)))
    for i, d in enumerate(DURATIONS):
        for j, (tol, fr) in enumerate(grid[d]):
            F[i, j] = fr
    im = axes[1].imshow(F, aspect="auto", origin="lower", cmap="RdYlGn",
                        vmin=0, vmax=1,
                        extent=[TOLERANCES[0], TOLERANCES[-1], DURATIONS[0], DURATIONS[-1]])
    axes[1].set_xlabel("Orbital TID tolerance (krad)")
    axes[1].set_ylabel("Mission duration (years)")
    axes[1].set_title(f"Feasible fraction over random policies")
    cb = plt.colorbar(im, ax=axes[1]); cb.set_label("Feasible fraction")

    plt.tight_layout()
    plt.savefig(outdir / "sweep_figure.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {outdir}/sweep.csv, sweep_grid.json, sweep_figure.png")


if __name__ == "__main__":
    main()
