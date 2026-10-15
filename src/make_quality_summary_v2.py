#!/usr/bin/env python3
"""
make_quality_summary_v2.py — one consolidated quality-indicator table for main_R1.tex.

Reads the per-workload tables_v2/results_metrics_<W>.csv written by make_tables_v2.py and
writes tables_v2/results_metrics_all.{csv,json,tex}: HV, spacing, GD (RMS) and IGD for every
method on the six non-ROMAN workloads (mean +- SD over 20 seeds). No numbers are recomputed.
"""
import csv

import make_tables_v2 as mt  # formatting helpers and writer only

WORKLOADS = ["SUNCATCHER", "LEO_OOS", "DEEP_SPACE_CRUISE", "IDS", "JWST", "RST"]  # by feasibility
NAME = {"SUNCATCHER": "SUNCATCHER", "LEO_OOS": r"LEO\_\allowbreak{}OOS",
        "DEEP_SPACE_CRUISE": r"DEEP\_\allowbreak{}SPACE\_\allowbreak{}CRUISE",
        "IDS": "IDS", "JWST": "JWST", "RST": "RST"}


def main():
    rows, trows = [], []
    for w in WORKLOADS:
        with open(mt.OUT / f"results_metrics_{w}.csv") as f:
            recs = list(csv.DictReader(f))
        for i, r in enumerate(recs):
            g = lambda k: float(r[k]) if r[k] not in ("", "nan") else float("nan")
            rows.append([w, r["method"], r["hv_mean"], r["hv_sd"], r["spacing_norm_mean"], r["spacing_norm_sd"],
                         r["gd_rms_mean"], r["gd_rms_sd"], r["igd_mean"], r["igd_sd"]])
            trows.append([NAME[w] if i == 0 else "", mt.METHOD_LABEL[r["method"]],
                          mt.pm(g("hv_mean"), g("hv_sd")), mt.pm(g("spacing_norm_mean"), g("spacing_norm_sd"), 3),
                          mt.pm(g("gd_rms_mean"), g("gd_rms_sd"), 3), mt.pm(g("igd_mean"), g("igd_sd"), 3)])
    mt.write("results_metrics_all",
             ["workload", "method", "hv_mean", "hv_sd", "spacing_norm_mean", "spacing_norm_sd",
              "gd_rms_mean", "gd_rms_sd", "igd_mean", "igd_sd"], rows,
             "Quality indicators for the six non-ROMAN workloads (mean $\\pm$ SD over 20 seeds). HV is in physical "
             "units against the fixed reference point. Spacing, GD (RMS form), and IGD are computed on the "
             "min-max-normalized latency--cost projection against the nondominated union of all runs. Spacing is "
             "undefined for the single-point all-metric $\\varepsilon$-constraint fronts.",
             colspec=r">{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}p{2.4cm}"
                     + r">{\centering\arraybackslash}p{2.45cm}" * 4, tex_rows=trows,
             tex_header=["Workload", "Method", "HV", "Spacing (norm.)", "GD (RMS)", "IGD"])
    # separate workload groups with a rule (the shared writer has no grouping option)
    tex = mt.OUT / "results_metrics_all.tex"
    lines = tex.read_text().split("\n")
    starts = tuple(NAME[w] + " &" for w in WORKLOADS[1:])
    out = "\n".join((r"\midrule" + "\n" + l) if l.startswith(starts) else l for l in lines)
    tex.write_text(out.replace("{\\scriptsize\n", "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n", 1))
    print("  results_metrics_all.{csv,json,tex}")


if __name__ == "__main__":
    main()
