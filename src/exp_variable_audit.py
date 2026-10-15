#!/usr/bin/env python3
"""
exp_variable_audit.py — Static audit of which dataset variables actually enter
the optimization. Answers the question: "are all 41 variables
actually used in the empirical results?"

For each workload, classifies each metric key as:
  HV       — drives the 2D HV (Latency or Cost) — directly visible in results
  CONS     — appears in constraint thresholds (gates feasibility)
  NL       — used by the nonlinear evaluator (M/M/1, contact, energy, etc.)
  FC_S1    — used by Stage 1 FC processor selection
  LIN_ONLY — only enters via the linear weighted-mean; no direct effect on
             HV, constraints, NL, or Stage 1. These are "dead weight" in
             practice — they shift the linear average, and an ablation
             experiment tests whether they matter.

Output: out_audit/audit.md, audit.csv
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from moo_core import (
    load_workload,
    FC_STAGE1_WEIGHTS,
)

# Nonlinear effects touch these specific keys (from moo_core.aggregate_metrics_nonlinear):
NL_KEYS_CANDIDATES = {
    "Latency p99 (ms)", "Latency (hrs)",
    "Energy per task (J)", "Energy per unit (J)", "Energy per obs (kJ)",
    "Thermal margin (C)", "Thermal margin (◦C)",
    "Link availability", "Contact duty cycle",
    "Throughput (GFLOPS)", "Peak power (W)",
    "Security protocol level", "SDC rate (per hr)",
}

# 2D HV uses one latency key + one cost key per workload
HV_KEYS_PER_WORKLOAD = {
    "IDS":               {"Latency p99 (ms)", "Cost per task (USD)"},
    "SUNCATCHER":        {"Latency p99 (ms)", "Cost per unit (USD)"},
    "JWST":              {"Latency (hrs)",    "Cost per obs (USD)"},
    "RST":               {"Latency (hrs)",    "Cost per obs (USD)"},
    "LEO_OOS":           {"Latency p99 (ms)", "Cost per task (USD)"},
    "DEEP_SPACE_CRUISE": {"Latency (hrs)",    "Cost per obs (USD)"},
}

# FC_STAGE1_WEIGHTS uses "throughput_gflops" internally, but the workload metric
# key is "Throughput (GFLOPS)". Alias so the FC Stage-1 role assignment matches
# the actual metric key present on the workload.
_FC_STAGE1_KEY_ALIASES = {"throughput_gflops": "Throughput (GFLOPS)"}
FC_STAGE1_KEYS = {_FC_STAGE1_KEY_ALIASES.get(k, k) for k in FC_STAGE1_WEIGHTS.keys()}


def classify(workload, hv_keys):
    """Return {metric_key: set(roles)} for one workload."""
    all_keys = [m.key for m in workload.metrics]
    cons_keys = set(workload.thresholds.keys())
    roles = defaultdict(set)
    for k in all_keys:
        if k in hv_keys:
            roles[k].add("HV")
        if k in cons_keys:
            roles[k].add("CONS")
        if k in NL_KEYS_CANDIDATES:
            roles[k].add("NL")
        if k in FC_STAGE1_KEYS:
            roles[k].add("FC_S1")
        if not roles[k]:
            roles[k].add("LIN_ONLY")
    return all_keys, roles


def main():
    data_path = Path("data.json")
    outdir = Path("out_audit"); outdir.mkdir(exist_ok=True)

    per_wl_summary = {}
    lines_md = ["# Variable usage audit", "",
                "Roles: **HV** = drives 2D HV; **CONS** = constraint threshold; "
                "**NL** = used by a nonlinear effect; **FC_S1** = Stage-1 FC selection; "
                "**LIN_ONLY** = only enters via linear weighted mean.", ""]
    csv_rows = [["workload", "metric", "roles"]]

    for wl_name in ["IDS", "SUNCATCHER", "JWST", "RST", "LEO_OOS", "DEEP_SPACE_CRUISE"]:
        wl = load_workload(data_path, wl_name)
        hv_keys = HV_KEYS_PER_WORKLOAD[wl_name]
        all_keys, roles = classify(wl, hv_keys)

        counts = defaultdict(int)
        for k, rs in roles.items():
            for r in rs:
                counts[r] += 1

        per_wl_summary[wl_name] = {
            "n_metrics": len(all_keys),
            "by_role": dict(counts),
            "lin_only_metrics": sorted([k for k, rs in roles.items() if rs == {"LIN_ONLY"}]),
        }

        lines_md.append(f"## {wl_name} ({len(all_keys)} metrics)")
        lines_md.append("")
        lines_md.append(f"- HV-driving: **{counts['HV']}**  |  Constraint: **{counts['CONS']}**  "
                        f"|  Nonlinear: **{counts['NL']}**  |  FC Stage-1: **{counts['FC_S1']}**  "
                        f"|  Linear-only: **{counts['LIN_ONLY']}**")
        lines_md.append("")
        lines_md.append("| Metric | Roles |")
        lines_md.append("|---|---|")
        for k in all_keys:
            rs = ",".join(sorted(roles[k]))
            lines_md.append(f"| {k} | {rs} |")
            csv_rows.append([wl_name, k, rs])
        lines_md.append("")
        lines_md.append(f"**Linear-only metrics (`{counts['LIN_ONLY']}` of `{len(all_keys)}`)** — "
                        f"only enter the optimization through the weighted mean; never gate "
                        f"feasibility, never appear in the HV projection, never trigger a "
                        f"nonlinear effect. An ablation tests whether they matter.")
        lines_md.append("")
        lines_md.append("```")
        for k in per_wl_summary[wl_name]["lin_only_metrics"]:
            lines_md.append(f"  {k}")
        lines_md.append("```")
        lines_md.append("")

    (outdir / "audit.md").write_text("\n".join(lines_md))
    with open(outdir / "audit.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    (outdir / "audit_summary.json").write_text(json.dumps(per_wl_summary, indent=2))

    print("=" * 70)
    print("Variable audit summary")
    print("=" * 70)
    for wl_name, s in per_wl_summary.items():
        print(f"{wl_name:<12s} n={s['n_metrics']:>3d}  "
              f"HV={s['by_role'].get('HV', 0):>2d}  "
              f"CONS={s['by_role'].get('CONS', 0):>2d}  "
              f"NL={s['by_role'].get('NL', 0):>2d}  "
              f"FC_S1={s['by_role'].get('FC_S1', 0):>2d}  "
              f"LIN_ONLY={s['by_role'].get('LIN_ONLY', 0):>2d}")
    print(f"\nWrote {outdir}/audit.md, audit.csv, audit_summary.json")


if __name__ == "__main__":
    main()
