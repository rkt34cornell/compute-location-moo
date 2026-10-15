#!/usr/bin/env python3
"""
make_typology_figure_v2.py — metric-role figure for main_R1.tex.

Same data (out_audit/audit_summary.json), same plotting code and style as
make_paper_figures.fig4_variable_typology; only two labels differ:
  title  "Variable-role memberships by workload" -> "Metric-role memberships by workload"
  legend "LIN_ONLY"                              -> "Documentation only"

Output: ../figures/fig_variable_typology_v2.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import make_paper_figures as mpf  # applies the legacy rcParams; does not regenerate anything
plt = mpf.plt

OUT = Path("../figures")
LEGEND_LABEL = {"LIN_ONLY": "Documentation only"}
WL_LABEL = {"SUNCATCHER": "Suncatcher", "RST": "Roman Survey (RST)", "LEO_OOS": "LEO OOS",
            "DEEP_SPACE_CRUISE": "Deep-Space Cruise"}


def main():
    a = json.loads(Path("out_audit/audit_summary.json").read_text())
    roles = ["HV", "CONS", "NL", "FC_S1", "LIN_ONLY"]
    role_color = {"HV": "#e63946", "CONS": "#f4a261", "NL": "#2a9d8f", "FC_S1": "#264653", "LIN_ONLY": "#9b9b9b"}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    wls = mpf.WORKLOADS
    names = [WL_LABEL.get(w, w) for w in wls]
    bottom = np.zeros(len(wls))
    for r in roles:
        vals = np.array([a[w]["by_role"].get(r, 0) for w in wls])
        ax.bar(names, vals, bottom=bottom, label=LEGEND_LABEL.get(r, r), color=role_color[r], alpha=0.85, edgecolor="white")
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if v > 0:
                ax.text(i, b + v/2, str(int(v)), ha="center", va="center",
                        fontsize=10, color="white" if r != "LIN_ONLY" else "black", fontweight="bold")
        bottom += vals
    ax.set_ylabel("Number of role assignments")
    ax.set_title("Metric-role memberships by workload")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(names, rotation=25, ha="right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"fig_variable_typology_v2.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_variable_typology_v2.{png,pdf}")


if __name__ == "__main__":
    main()
