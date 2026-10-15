#!/usr/bin/env python3
"""
make_paper_figures.py — Produces all paper-ready figures into
../figures/. Reads from the various out_*/ output directories.
Each figure is referenced by the LaTeX Results section.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from moo_core import Direction, load_workload, apply_odc_variant


# Font sizing balances AIAA readability (8 pt minimum at 3.25 in single-column)
# with layout clarity in the current conference format. At figsize width ~7 in
# and journal downscale to 3.25 in, a 12 pt source ends at ~5.6 pt, so the
# journal AIAA production will typically upscale slightly rather than shrink.
# The values below give a comfortable in-paper display without crowding.
plt.rcParams.update({
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.titlesize":  14,
    "axes.linewidth":    1.0,
})


OUTDIR = Path("../figures")
OUTDIR.mkdir(exist_ok=True)

METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon"]
METHOD_COLOR = {"random":"#9b9b9b","mopso":"#e76f51","nsga2":"#f4a261",
                "spea2":"#264653","moead":"#06d6a0","epsilon":"#e9c46a"}
METHOD_LABEL = {"random":"Random","mopso":"MOPSO","nsga2":"NSGA-II",
                "spea2":"SPEA2","moead":"MOEA/D","epsilon":r"$\varepsilon$-constraint"}

WORKLOADS = ["IDS","SUNCATCHER","JWST","RST","LEO_OOS","DEEP_SPACE_CRUISE"]
WORKLOADS_4 = ["IDS","SUNCATCHER","JWST","RST"]  # subset for figures keyed to the original 4
ROMAN_VARIANTS = ["A","B","C"]
VARIANT_LABEL = {"A":"Variant A (TRL=3)","B":"Variant B (TRL=7)","C":"Variant C (no ODC)"}
VARIANT_COLOR = {"A":"#e76f51","B":"#2a9d8f","C":"#264653"}

TIER_COLOR_ROMAN = {"FC":"#e63946","ODC":"#f4a261","GSE":"#2a9d8f","TDC":"#264653"}

HV_KEYS = {
    "IDS":               ("Latency p99 (ms)", "Cost per task (USD)", np.array([260., 3.2])),
    "SUNCATCHER":        ("Latency p99 (ms)", "Cost per unit (USD)", np.array([110., 14.5])),
    "JWST":              ("Latency (hrs)",    "Cost per obs (USD)",  np.array([50., 20.])),
    "RST":               ("Latency (hrs)",    "Cost per obs (USD)",  np.array([26., 25.])),
    "LEO_OOS":           ("Latency p99 (ms)", "Cost per task (USD)", np.array([320., 2.5])),
    "DEEP_SPACE_CRUISE": ("Latency (hrs)",    "Cost per obs (USD)",  np.array([5., 15.])),
}


def hv2d(pts, ref):
    if len(pts) == 0: return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, pv = 0.0, ref[1]
    for x, y in pts:
        if y < pv and x < ref[0]:
            hv += (ref[0]-x)*(pv-y); pv = y
    return hv


# ───── Figure 1: HV distributions per workload ────────────────────────────────
def fig1_hv_distributions():
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.5), sharey=False)
    fig.suptitle("Hypervolume distribution across 20 seeds (equal budget = 16,000 evals)",
                 fontsize=13, fontweight="bold")
    # Order workloads by random-policy feasibility density to visualize the regime ordering
    ordered = ["SUNCATCHER", "LEO_OOS", "IDS", "DEEP_SPACE_CRUISE", "JWST", "RST"]
    for ax, wl in zip(axes.flat, ordered):
        path = Path(f"out_multiseed_{wl}/summary.json")
        if not path.exists():
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center"); continue
        s = json.loads(path.read_text())
        raw = s["raw"][wl]
        hv_by_method = {m: np.array([r["hv"] for r in raw[m]]) for m in METHODS if m in raw}
        positions = list(range(1, len(hv_by_method)+1))
        bp = ax.boxplot(list(hv_by_method.values()), positions=positions, widths=0.55,
                         patch_artist=True, showmeans=True,
                         meanprops={"marker":"D","markerfacecolor":"white","markeredgecolor":"black","markersize":5},
                         medianprops={"color":"black","linewidth":1.5})
        for patch, m in zip(bp["boxes"], hv_by_method.keys()):
            patch.set_facecolor(METHOD_COLOR[m]); patch.set_alpha(0.75)
        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_LABEL[m] for m in hv_by_method.keys()],
                            rotation=20, ha="right")
        ax.set_ylabel("Hypervolume")
        ax.set_title(wl)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_hv_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_hv_distributions.png")


# ───── Figure 2: Convergence on IDS ────────────────────────────────────────────
def fig2_convergence():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in ["mopso","nsga2","spea2","moead"]:
        seed_curves = []
        for seed in [11, 12, 13]:
            p = Path(f"out_convergence/IDS_{method}_seed{seed}_trace.npz")
            if not p.exists(): continue
            d = np.load(p); seed_curves.append((d["steps"], d["hv"]))
        if not seed_curves: continue
        steps_min = min(c[0][-1] for c in seed_curves)
        common_x = np.linspace(seed_curves[0][0][0], steps_min, 50)
        ys = []
        for s, h in seed_curves:
            ys.append(np.interp(common_x, s, h))
        Y = np.array(ys)
        mean = Y.mean(axis=0); std = Y.std(axis=0)
        ax.plot(common_x, mean, color=METHOD_COLOR[method], label=METHOD_LABEL[method], linewidth=2)
        ax.fill_between(common_x, mean-std, mean+std, color=METHOD_COLOR[method], alpha=0.18)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best-feasible Pareto-front HV (Latency × Cost)")
    ax.set_title("Convergence diagnostics on IDS — HV vs evaluations (3-seed mean ± std)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_convergence_IDS.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_convergence_IDS.png")


# ───── Figure 3: Pareto fronts per workload ────────────────────────────────────
def fig3_pareto_fronts():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    fig.suptitle("Pareto fronts (Latency × Cost projection) — all 6 algorithms, 1 representative seed",
                 fontsize=12, fontweight="bold")
    for ax, wl in zip(axes.flat, WORKLOADS):
        wlobj = load_workload(Path("data.json"), wl)
        mk = [m.key for m in wlobj.metrics]
        kd = {m.key: m.direction for m in wlobj.metrics}
        lk, ck, ref = HV_KEYS[wl]
        li, ci = mk.index(lk), mk.index(ck)
        for method in METHODS:
            p = Path(f"out_multiseed_{wl}/{wl}_{method}_seed1000_V.npy")
            if not p.exists(): continue
            V = np.load(p)
            if V.shape[0] == 0: continue
            pts = V[:, [li, ci]].copy()
            if kd.get(lk) == Direction.MAXIMIZE: pts[:, 0] = -pts[:, 0]
            if kd.get(ck) == Direction.MAXIMIZE: pts[:, 1] = -pts[:, 1]
            sz = 6 if method == "random" else 30
            ax.scatter(pts[:, 0], pts[:, 1], s=sz, alpha=0.55,
                       color=METHOD_COLOR[method], label=METHOD_LABEL[method], edgecolors="none")
        if "Latency p99" in lk:
            ax.set_xlabel("Latency p99 (ms)")
        else:
            ax.set_xlabel("Latency (hrs)")
        ax.set_ylabel(ck)
        ax.axvline(ref[0], color="red", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(ref[1], color="red", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_title(wl); ax.grid(True, alpha=0.3)
    # Shared legend at the bottom to keep all subplot y-axis labels visible.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTDIR/"fig_pareto_fronts.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_pareto_fronts.png")


# ───── Figure 4: Variable typology stacked bar ────────────────────────────────
def fig4_variable_typology():
    a = json.loads(Path("out_audit/audit_summary.json").read_text())
    roles = ["HV","CONS","NL","FC_S1","LIN_ONLY"]
    role_color = {"HV":"#e63946","CONS":"#f4a261","NL":"#2a9d8f","FC_S1":"#264653","LIN_ONLY":"#9b9b9b"}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    wls = WORKLOADS
    bottom = np.zeros(len(wls))
    for r in roles:
        vals = np.array([a[w]["by_role"].get(r, 0) for w in wls])
        ax.bar(wls, vals, bottom=bottom, label=r, color=role_color[r], alpha=0.85, edgecolor="white")
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if v > 0:
                ax.text(i, b + v/2, str(int(v)), ha="center", va="center",
                        fontsize=10, color="white" if r != "LIN_ONLY" else "black", fontweight="bold")
        bottom += vals
    ax.set_ylabel("Number of role assignments")
    ax.set_title("Variable-role memberships by workload")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(wls, rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_variable_typology.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_variable_typology.png")


# ───── Figure 5: Variable ablation (corrected) ────────────────────────────────
def fig5_variable_ablation():
    s = json.loads(Path("out_var_ablation/ablation.json").read_text())
    method_names = ["random","mopso","nsga2","moead"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharey=False)
    fig.suptitle("Variable ablation (constraint-preserved): full vs reduced 8-variable model",
                 fontsize=12, fontweight="bold")
    for ax, wl in zip(axes.flat, WORKLOADS):
        x = np.arange(len(method_names)); width = 0.36
        full_means = np.array([np.mean(s[wl]["full"][m]) for m in method_names])
        full_stds  = np.array([np.std(s[wl]["full"][m], ddof=1) if len(s[wl]["full"][m])>1 else 0 for m in method_names])
        o8_means   = np.array([np.mean(s[wl]["orig8"][m]) for m in method_names])
        o8_stds    = np.array([np.std(s[wl]["orig8"][m], ddof=1) if len(s[wl]["orig8"][m])>1 else 0 for m in method_names])
        ax.bar(x - width/2, full_means, width, yerr=full_stds, capsize=3,
                label="42 vars", color="#264653", alpha=0.85)
        ax.bar(x + width/2, o8_means, width, yerr=o8_stds, capsize=3,
                label="8 vars + constraints", color="#e9c46a", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in method_names], rotation=15)
        ax.set_title(wl); ax.set_ylabel("HV mean (3 seeds)" if wl=="IDS" else "")
        ax.grid(True, alpha=0.3, axis="y")
        if wl == "IDS": ax.legend(loc="best", fontsize=10)
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_variable_ablation.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_variable_ablation.png")


# ───── Figure 6: NL coefficient sensitivity tornado ────────────────────────────
def fig6_nl_sensitivity():
    s = json.loads(Path("out_nl_sensitivity/sensitivity.json").read_text())
    base = s["baseline_hv"]
    by_coef = {}
    for p in s["perturbations"]:
        by_coef.setdefault(p["coefficient"], {})[p["direction"]] = p["delta_pct"]
    coefs = sorted(by_coef.keys(),
                   key=lambda c: max(abs(by_coef[c].get("-25%",0)), abs(by_coef[c].get("+25%",0))))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    y = np.arange(len(coefs))
    minus = [by_coef[c].get("-25%",0) for c in coefs]
    plus  = [by_coef[c].get("+25%",0) for c in coefs]
    ax.barh(y - 0.18, minus, height=0.36, color="#e76f51", label="-25% perturbation")
    ax.barh(y + 0.18, plus,  height=0.36, color="#2a9d8f", label="+25% perturbation")
    ax.set_yticks(y); ax.set_yticklabels(coefs)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Delta HV vs baseline (%)")
    ax.set_title(f"Nonlinear coefficient sensitivity (IDS, MOPSO; baseline HV = {base:.1f})")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_nl_sensitivity_tornado.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_nl_sensitivity_tornado.png")


# ───── Figure 7: ROMAN variant comparison ─────────────────────────────────────
def fig7_roman_variants():
    s = json.loads(Path("out_roman_hv_recompute/recomputed_summary.json").read_text())
    summary = s["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.8))
    for ax, ev in zip(axes, ["linear","nonlinear"]):
        x = np.arange(len(METHODS)); width = 0.25
        for vi, v in enumerate(ROMAN_VARIANTS):
            means = [summary.get(f"ROMAN_TRANSIENT|{v}|{ev}",{}).get(m,{}).get("hv_mean",0) for m in METHODS]
            stds  = [summary.get(f"ROMAN_TRANSIENT|{v}|{ev}",{}).get(m,{}).get("hv_std",0)  for m in METHODS]
            ax.bar(x + (vi-1)*width, means, width, yerr=stds, capsize=3,
                   label=VARIANT_LABEL[v], color=VARIANT_COLOR[v], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], rotation=35, ha="right")
        ax.set_title(f"{ev.capitalize()} evaluator")
        ax.set_ylabel("HV mean $\\pm$ std" if ev == "linear" else "")
        ax.grid(True, alpha=0.3, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTDIR/"fig_roman_variant_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_roman_variant_comparison.png")


# ───── Figure 8: ODC throughput × TID heatmap ─────────────────────────────────
def fig8_odc_heatmap():
    g = np.load("out_odc_value/heatmap_grids.npz")
    j = json.loads(Path("out_odc_value/heatmap.json").read_text())
    thrs, tids = g["throughputs"], g["tids"]
    p_odc, hv = g["p_odc"], g["hv"]
    industry = j.get("industry_refs", {})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5))
    im = axes[0].imshow(p_odc, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(tids))); axes[0].set_xticklabels([str(t) for t in tids], rotation=30)
    axes[0].set_yticks(range(len(thrs))); axes[0].set_yticklabels([str(t) for t in thrs])
    axes[0].set_xlabel("ODC TID (krad)")
    axes[0].set_ylabel("ODC throughput (GFLOPS)")
    axes[0].set_title("Mean $p_{\\mathrm{ODC}}$ (Variant B)")
    plt.colorbar(im, ax=axes[0]).set_label("Routing fraction")

    im2 = axes[1].imshow(hv, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_xticks(range(len(tids))); axes[1].set_xticklabels([str(t) for t in tids], rotation=30)
    axes[1].set_yticks(range(len(thrs))); axes[1].set_yticklabels([str(t) for t in thrs])
    axes[1].set_xlabel("ODC TID (krad)")
    axes[1].set_title("Hypervolume achieved")
    plt.colorbar(im2, ax=axes[1]).set_label("HV")

    plt.tight_layout(); plt.savefig(OUTDIR/"fig_odc_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_odc_heatmap.png")


# ───── Figure 9: BO vs evolutionary sample efficiency ─────────────────────────
def fig9_bo_efficiency():
    bo_summary = json.loads(Path("out_bo_baseline/bo_summary.json").read_text())
    ms_summary = json.loads(Path("out_multiseed_IDS/summary.json").read_text())["summary"]["IDS"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    # Evolutionary points: HV at 16k evals
    ev_methods = ["mopso","nsga2","spea2","moead","epsilon","random"]
    for m in ev_methods:
        if m not in ms_summary: continue
        ax.errorbar([16000], [ms_summary[m]["hv_mean"]], yerr=[ms_summary[m]["hv_std"]],
                     marker="o", markersize=10, color=METHOD_COLOR[m],
                     capsize=4, label=METHOD_LABEL[m])
    bo = bo_summary["IDS_linear"]
    ax.errorbar([80], [bo["hv_mean"]], yerr=[bo["hv_std"]],
                marker="*", markersize=18, color="#9c27b0",
                capsize=4, label="qEHVI (Bayesian opt)")
    ax.set_xscale("log")
    ax.set_xlabel("Function evaluation budget (log scale)")
    ax.set_ylabel("HV on IDS (mean ± std)")
    ax.set_title("Sample efficiency: q-EHVI vs evolutionary")
    ax.axvspan(50, 200, alpha=0.10, color="purple")
    ax.axvspan(10000, 20000, alpha=0.08, color="gray")
    ax.text(110, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            "BO budget", ha="center", color="purple", fontsize=10)
    ax.text(16000, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            "Evolutionary budget", ha="center", color="gray", fontsize=10)
    ax.legend(loc="best", fontsize=10, ncol=2); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_bo_vs_evolutionary.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_bo_vs_evolutionary.png")


# ───── Figure 10: Mission duration TID requirement (SUNCATCHER) ───────────────
def fig10_mission_duration():
    g = json.loads(Path("out_duration_sweep/sweep_grid.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5))
    valid = [(d["duration_yr"], d["min_tol_krad"]) for d in g["min_tol"]
             if d["min_tol_krad"] != float("inf") and isinstance(d["min_tol_krad"], (int, float))]
    if valid:
        ds, ts = zip(*valid)
        axes[0].plot(ds, ts, "o-", color="#2a6e8f", linewidth=2.5, markersize=7)
        axes[0].set_xlabel("Mission duration (yr)")
        axes[0].set_ylabel("Min TID (krad) for ≥50% feasibility")
        axes[0].set_title("Required TID vs lifetime")
        axes[0].grid(True, alpha=0.3)
        # Headroom for the top annotation.
        axes[0].set_ylim(0, max(ts) * 1.20)
        # Annotate the first occurrence of each TID value with a comfortable
        # above-center offset so labels sit clearly above the line and
        # markers without touching the axis.
        last_t = None
        for d, t in zip(ds, ts):
            if t != last_t:
                axes[0].annotate(f"{t}", (d, t), textcoords="offset points",
                                  xytext=(0, 12), ha="center", va="bottom",
                                  fontsize=10)
                last_t = t

    durations = g["durations"]; tolerances = g["tolerances"]
    F = np.zeros((len(durations), len(tolerances)))
    for i, d in enumerate(durations):
        for jj, (tol, fr) in enumerate(g["grid"][str(d)]):
            F[i, jj] = fr
    im = axes[1].imshow(F, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    # Sparsify: label every other tick so 45deg labels don't overlap at 3.25 in scale.
    axes[1].set_xticks(range(len(tolerances)))
    tick_labels = [str(t) if i % 2 == 0 else "" for i, t in enumerate(tolerances)]
    axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[1].set_yticks(range(len(durations)))
    axes[1].set_yticklabels([str(d) for d in durations])
    axes[1].set_xlabel("TID tolerance (krad)")
    axes[1].set_ylabel("Duration (yr)")
    axes[1].set_title("Feasibility surface")
    plt.colorbar(im, ax=axes[1]).set_label("Feasible fraction")
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_mission_duration_TID.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_mission_duration_TID.png")


# ───── Figure 11: Linear-vs-NL policy validity ─────────────────────────────────
def fig11_lin_vs_nl():
    import csv
    rows = list(csv.DictReader(open("out_lin_vs_nl/validity.csv")))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = []; fracs = []; colors = []
    main_wls = ["IDS","SUNCATCHER","JWST","RST"]
    roman_v = ["ROMAN_TRANSIENT|A","ROMAN_TRANSIENT|B","ROMAN_TRANSIENT|C"]
    keep = main_wls + roman_v
    for r in rows:
        if r["label"] in keep:
            labels.append(r["label"].replace("ROMAN_TRANSIENT","ROMAN_T"))
            fracs.append(100.0 * float(r["frac_feas_nl"]))
            colors.append("#e63946" if 100.0 * float(r["frac_feas_nl"]) < 90 else "#2a9d8f")
    ax.bar(labels, fracs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.axhline(100, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_ylabel("% feasible under NL")
    ax.set_title("Policy validity by workload")
    ax.set_ylim(0, 118); ax.grid(True, alpha=0.3, axis="y")
    for i, (l, f) in enumerate(zip(labels, fracs)):
        ax.text(i, f + 2, f"{f:.0f}%", ha="center", fontsize=10, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(OUTDIR/"fig_lin_vs_nl_validity.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  fig_lin_vs_nl_validity.png")


if __name__ == "__main__":
    print("Generating paper figures into ../figures/...")
    fig1_hv_distributions()
    fig2_convergence()
    fig3_pareto_fronts()
    fig4_variable_typology()
    fig5_variable_ablation()
    fig6_nl_sensitivity()
    fig7_roman_variants()
    fig8_odc_heatmap()
    fig9_bo_efficiency()
    fig10_mission_duration()
    fig11_lin_vs_nl()
    print(f"\nDone. {len(list(OUTDIR.glob('*.png')))} figures in {OUTDIR.resolve()}")
