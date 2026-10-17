#!/usr/bin/env python3
"""
make_figures_v2.py — figures, generated only from v2 outputs.
Output: ../figures/<name>.{png,pdf}. Every figure has a CSV twin in
../tables_v2 (make_tables_v2.py) for exact values.

Palette: reference categorical slots (validated: CVD/normal-vision pass; three
hues < 3:1 contrast on white, so every figure names methods on an axis or in a
legend and is backed by a table). Colors follow the method, never its rank.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import moo_core as mc
from run_benchmark_v2 import hv_ref_keys
from run_multiseed_benchmark import to_min_2d

OUT = Path("../figures")
A = Path("out_analysis_v2")
METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon", "epsilon2d"]
LABEL = {"random": "Random", "mopso": "MOPSO", "nsga2": "NSGA-II", "spea2": "SPEA2", "moead": "MOEA/D",
         "epsilon": "ε-constr. (all-metric)", "epsilon2d": "ε-constr. (bi-obj., suppl.)"}
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
MARK = dict(zip(METHODS, ["o", "s", "^", "D", "v", "P", "X"]))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
# publication-facing workload names for figures; code identifiers stay in the data and tables
WL_SHORT = {"SUNCATCHER": "Suncatcher", "LEO_OOS": "LEO OOS", "IDS": "IDS", "DEEP_SPACE_CRUISE": "Deep-Space Cruise",
            "JWST": "JWST", "RST": "Roman Survey (RST)", "ROMAN_TRANSIENT": "Roman Transient",
            "ROMAN_TRANSIENT_FOLLOWUP": "Roman Follow-Up", "ROMAN_T_FOLLOWUP": "Roman Follow-Up"}

COEFF_SYMBOL = {"queue_max_utilization": r"$\rho_{\max}$", "compute_gflops_per_task": r"$\theta$",
                "lambda_tasks_per_s": r"$\lambda$", "thermal_coupling_coeff": r"$\kappa$",
                "sdc_compound_factor": r"$\xi$", "link_setup_energy_fraction": r"$\eta$",
                "link_congestion_exponent": r"$\gamma_c$", "contact_sat_exponent": r"$\alpha_c$",
                "thermal_power_exponent": r"$\beta$"}
P_ODC = r"$p_{\mathrm{ODC}}$"


def pretty_workload(label):
    """'ROMAN_TRANSIENT|A' or 'ROMAN_TRANSIENT | A' -> 'Roman Transient (Variant A)'; publication-facing names."""
    name, _, var = [x.strip() for x in label.partition("|")]
    name = WL_SHORT.get(name, name)
    return f"{name} (Variant {var})" if var else name


plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "lines.linewidth": 2.0})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def read_csv(p):
    p = Path(p)
    return list(csv.DictReader(open(p))) if p.exists() else []


def feas_order():
    fj = Path("out_feasibility_v2/feasibility.json")
    F = [r for r in json.loads(fj.read_text()) if not r["variant"]]
    F.sort(key=lambda r: r["linear_pct"])
    return [r["workload"] for r in F], {r["workload"]: r["linear_pct"] for r in F}


def runs(d):
    return [json.loads(p.read_text()) for p in sorted((Path(d) / "runs").glob("*.json"))]


# ── figures ───────────────────────────────────────────────────────────────────

def fig_hv_distributions():
    order, f = feas_order()
    R = runs("out_benchmark_v2")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, w in zip(axes.flat, order):
        data = [[r["hv"] for r in R if r["workload"] == w and r["method"] == m] for m in METHODS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set(facecolor=COLOR[m], alpha=0.35, edgecolor=COLOR[m])
        rng = np.random.default_rng(0)
        for i, (m, y) in enumerate(zip(METHODS, data), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=9, color=COLOR[m], edgecolor="white",
                       linewidth=0.4, zorder=3)
            ax.scatter([i], [np.mean(y)], marker="D", s=22, color="white", edgecolor=INK, linewidth=0.9, zorder=4)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=40, ha="right", fontsize=6.5)
        ax.set_title(f"{WL_SHORT[w]} — random-policy feasibility {f[w]:.2f}%", fontsize=8, color=INK)
        ax.set_ylabel("HV (latency × cost)")
    fig.suptitle("Hypervolume over 20 seeds, six workloads ordered by random-policy feasibility (◇ = mean, line = median)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig_hv_distributions_v2")


def fig_pareto_fronts():
    order, f = feas_order()
    R = runs("out_benchmark_v2")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, w in zip(axes.flat, order):
        wl = mc.load_workload(Path("data.json"), w); mk = [m.key for m in wl.metrics]
        lk, ck, ref = hv_ref_keys(w)
        seed0 = min(r["seed"] for r in R if r["workload"] == w)
        # draw order: densest fronts first so the sparse fronts stay visible on top
        for z, m in enumerate(["random", "mopso", "moead", "nsga2", "spea2", "epsilon2d", "epsilon"], start=1):
            r = next(x for x in R if x["workload"] == w and x["method"] == m and x["seed"] == seed0)
            V = np.load(Path("out_benchmark_v2/fronts") / f"{r['job_id']}_V.npy")
            feas = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, v)))[0] for v in V], bool)
            if not feas.any():
                continue
            P = to_min_2d(V[feas], wl, mk, lk, ck)
            if m == "random":
                ax.scatter(P[:, 0], P[:, 1], s=3, color="#bdbcb6", zorder=z, rasterized=True)
            elif m == "mopso":
                ax.scatter(P[:, 0], P[:, 1], s=5, marker=MARK[m], color=COLOR[m], alpha=0.45, linewidth=0, zorder=z)
            else:
                ax.scatter(P[:, 0], P[:, 1], s=20, marker=MARK[m], color=COLOR[m], edgecolor="white", linewidth=0.5,
                           zorder=z)
        for k, axis in [(lk, "x"), (ck, "y")]:
            t = wl.thresholds.get(k)
            if t is not None and t.max_value is not None:
                (ax.axvline if axis == "x" else ax.axhline)(t.max_value, color=INK, linestyle="--", linewidth=1)
                ax.annotate("constraint", xy=(t.max_value, 0) if axis == "x" else (0, t.max_value),
                            xycoords=("data", "axes fraction") if axis == "x" else ("axes fraction", "data"),
                            fontsize=6, color=INK, xytext=(2, 2), textcoords="offset points")
        ax.scatter([ref[0]], [ref[1]], marker="x", color=INK, s=30, zorder=4)
        ax.annotate("HV ref.", (ref[0], ref[1]), fontsize=6, color=INK, xytext=(5, -3), textcoords="offset points",
                    ha="left", va="top", annotation_clip=False)
        ax.set_title(f"{WL_SHORT[w]} (seed {seed0})", fontsize=8, color=INK)
        ax.set_xlabel(lk); ax.set_ylabel(ck)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], linestyle="none", marker="o", ms=3, color="#bdbcb6", label="Random (front points)")]
    h += [Line2D([], [], linestyle="none", marker=MARK[m], ms=5.5, color=COLOR[m], markeredgecolor="white",
                 markeredgewidth=0.5, label=LABEL[m]) for m in METHODS if m != "random"]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=7)
    fig.suptitle("Latency–cost projection of the returned feasible fronts (one seed). Dashed = hard constraint; × = HV reference point",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_pareto_fronts_v2")


def fig_convergence():
    d = Path("out_convergence_v2/traces")
    if not d.exists():
        return
    for w in ["IDS"] + [x for x in ["SUNCATCHER", "LEO_OOS", "DEEP_SPACE_CRUISE", "JWST", "RST"]]:
        files = list(d.glob(f"{w}_*_seed*.npz"))
        if not files:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for m in ["mopso", "nsga2", "spea2", "moead"]:
            C = [np.load(p) for p in sorted(d.glob(f"{w}_{m}_seed*.npz"))]
            st = C[0]["steps"]; H = np.array([c["hv"] for c in C])
            mu, sd = H.mean(0), H.std(0, ddof=1)
            ax.plot(st, mu, color=COLOR[m], label=f"{LABEL[m]} (n={len(C)})")
            ax.fill_between(st, mu - sd, mu + sd, color=COLOR[m], alpha=0.15, linewidth=0)
            ax.annotate(LABEL[m], (st[-1], mu[-1]), fontsize=7, color=INK, xytext=(3, 0), textcoords="offset points",
                        va="center")
        ax.axvline(7260, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate("common budget 7,260", (7260, ax.get_ylim()[0]), fontsize=6.5, color=MUTED, xytext=(3, 4),
                    textcoords="offset points")
        ax.set_xlabel("Function evaluations"); ax.set_ylabel("Anytime HV (all feasible evaluated points)")
        ax.set_title(f"{WL_SHORT.get(w, w)} convergence, mean ± SD over 20 seeds (MOEA/D configuration uses 14,520 evals)",
                     fontsize=8, color=INK)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        save(fig, f"fig_convergence_{w}_v2")


def fig_ablation():
    p = Path("out_var_ablation_v2/summary.json")
    if not p.exists():
        return
    S = json.loads(p.read_text()); order, _ = feas_order()
    ms = ["random", "mopso", "nsga2", "spea2", "moead"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, w in zip(axes.flat, order):
        x = np.arange(len(ms))
        for j, (vs, col) in enumerate([("full", "#2a78d6"), ("reduced", "#eb6834")]):
            mu = [S[w][m][f"{vs}_hv_mean"] for m in ms]; sd = [S[w][m][f"{vs}_hv_std"] for m in ms]
            nv = S[w][ms[0]][f"n_vars_{vs}"]
            ax.bar(x + (j - 0.5) * 0.38, mu, 0.36, yerr=sd, color=col, edgecolor="white", linewidth=1,
                   error_kw=dict(ecolor=MUTED, lw=0.8), label=f"{vs} ({nv} metrics)")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in ms], rotation=30, ha="right", fontsize=7)
        ax.set_title(WL_SHORT[w], fontsize=8, color=INK); ax.set_ylabel("HV"); ax.legend(fontsize=6.5)
    fig.suptitle("Full vs reduced metric set (constraints preserved; equal 4,860-evaluation budget; 20 seeds; mean ± SD)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig_variable_ablation_v2")


def fig_lin_vs_nl():
    rows = read_csv("out_lin_vs_nl_v2/validity.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    y = np.arange(len(rows))
    a = [float(r["pct_picks_nl_feasible"]) for r in rows]; b = [float(r["pct_all_front_points_nl_feasible"]) for r in rows]
    ax.barh(y - 0.2, a, 0.38, color="#2a78d6", label="Best-hypervolume policy of each run")
    ax.barh(y + 0.2, b, 0.38, color="#eb6834", label="All linear front points")
    for yi, v in zip(y, a):
        ax.annotate(f"{v:.0f}%", (v, yi - 0.2), fontsize=6.5, color=INK, xytext=(2, 0), textcoords="offset points", va="center")
    ax.set_yticks(y); ax.set_yticklabels([pretty_workload(r["label"]) for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("% still feasible under the nonlinear evaluator"); ax.set_xlim(0, 110)
    ax.set_title("Linear-optimal policies re-scored with the nonlinear evaluator", fontsize=8, color=INK)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=2)   # below the axes: never over a bar annotation
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_lin_vs_nl_validity_v2")


def fig_sensitivity():
    # admissible-cap rerun: rho_max upper arm is 0.99 instead of 0.92 x 1.25 = 1.15
    p = Path("out_nl_sensitivity_rhocap_v2/summary.json")
    if not p.exists():
        return
    S = json.loads(p.read_text())
    rho_up = S["perturbed_values"]["queue_max_utilization"]["upper"]
    ms = [m for m in ["mopso", "nsga2", "random"] if m in S]
    coeffs = S[ms[0]]["importance_order"]
    fig, axes = plt.subplots(1, len(ms), figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, ms):
        y = np.arange(len(coeffs))
        for d, lab, col, off in [("lower", "\u221225%", "#2a78d6", -0.2),
                                 ("upper", f"+25% ($\\rho_{{\\max}}$: {rho_up:.2f})", "#eb6834", 0.2)]:
            mu = [S[m]["coeffs"][c][d]["delta_pct_mean"] for c in coeffs]
            sd = [S[m]["coeffs"][c][d]["delta_pct_std"] for c in coeffs]
            ax.barh(y + off, mu, 0.38, xerr=sd, color=col, error_kw=dict(ecolor=MUTED, lw=0.8),
                    label=lab)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y); ax.set_yticklabels([COEFF_SYMBOL.get(c, c) for c in coeffs], fontsize=9); ax.invert_yaxis()
        ax.set_title(f"{LABEL[m]} (baseline HV {S[m]['baseline_hv_mean']:.1f})", fontsize=8, color=INK)
        ax.set_xlabel("ΔHV (%) vs baseline, mean ± SD (10 seeds)")
    axes[0].legend(fontsize=7)
    fig.suptitle("IDS nonlinear-coefficient sensitivity across algorithms", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig_nl_sensitivity_v2")


def fig_bo():
    p = Path("out_bo_v2/bo_summary.json")
    if not p.exists():
        return
    S = json.loads(p.read_text())
    M = {r["method"]: r for r in read_csv(A / "main_metrics.csv") if r["config"] == "IDS"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, r in M.items():
        ax.errorbar(float(r["evals_mean"]), float(r["hv_mean"]), yerr=float(r["hv_sd"]), fmt=MARK[m], color=COLOR[m],
                    ms=7, capsize=2, label=f"{LABEL[m]} (20 seeds)")
    if "IDS_linear" in S:
        b = S["IDS_linear"]
        ax.errorbar(b.get("n_evals_actual", 84), b["hv_mean"], yerr=b["hv_std"], fmt="*", color=INK, ms=11, capsize=2,
                    label=f"q-EHVI ({b['n_seeds']} seeds)")
    ax.set_xscale("log"); ax.set_xlabel("Function evaluations (log)"); ax.set_ylabel("HV (IDS, linear)")
    ax.legend(fontsize=6.5, ncol=2); ax.set_title("IDS: q-EHVI at 84 evaluations vs population methods at 16,000",
                                                   fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, "fig_bo_vs_evolutionary_v2")


def fig_odc():
    p = Path("out_odc_sweep_v2/summary.json")
    if not p.exists():
        return
    S = json.loads(p.read_text())
    from exp_odc_value_heatmap import THROUGHPUTS, TID_MARGINS
    G = np.array([[S["cells"][f"T{t}_R{r}"]["mean"] for r in TID_MARGINS] for t in THROUGHPUTS])
    SD = np.array([[S["cells"][f"T{t}_R{r}"]["sd"] for r in TID_MARGINS] for t in THROUGHPUTS])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    vmin, vmax = float(G.min()), float(G.max())   # colour scale spans the observed range of cell means
    im = a1.imshow(G, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(THROUGHPUTS)):
        for j in range(len(TID_MARGINS)):
            a1.text(j, i, f"{G[i, j]:.3f}\n±{SD[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if (G[i, j] - vmin) / max(vmax - vmin, 1e-12) > 0.5 else INK)
    a1.set_xticks(range(len(TID_MARGINS))); a1.set_xticklabels(TID_MARGINS)
    a1.set_yticks(range(len(THROUGHPUTS))); a1.set_yticklabels(THROUGHPUTS)
    a1.set_xlabel("ODC radiation dose margin (krad)"); a1.set_ylabel("ODC throughput (GFLOPS)")
    a1.set_title(f"Mean ± SD of {P_ODC} over Pareto points (10 seeds/cell); colour scale = observed range", fontsize=8, color=INK)
    a1.grid(False)
    cb = plt.colorbar(im, ax=a1); cb.set_label(f"mean {P_ODC}"); cb.formatter.set_useOffset(False); cb.update_ticks()
    pooled = np.concatenate([np.load(f)[:, 1] for f in sorted(Path("out_odc_sweep_v2/fronts").glob("P_*.npy"))])
    bins = np.linspace(0, 1, 41)
    a2.hist(pooled, bins=bins, density=True, color="#2a78d6", alpha=0.75, label=f"Pareto points (n={pooled.size})")
    xs = np.linspace(0, 1, 200)
    a2.plot(xs, 3 * (1 - xs) ** 2, color="#eb6834", label="uniform simplex, Beta(1,3)")
    ref = S["pooled_feasible_uniform"]
    a2.axvline(0.25, color=MUTED, linestyle=":", linewidth=1)
    a2.annotate("0.25", (0.25, a2.get_ylim()[1] * 0.9), fontsize=6.5, color=MUTED, xytext=(3, 0), textcoords="offset points")
    a2.set_xlabel(P_ODC); a2.set_ylabel("density")
    a2.set_title(f"Pooled distribution: mean {S['pooled']['mean']:.3f}, SD {S['pooled']['sd']:.3f}, median "
                 f"{S['pooled']['median']:.3f}; feasible-uniform mean {ref['mean']:.3f}", fontsize=7.5, color=INK)
    a2.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_odc_sweep_v2")


def fig_roman():
    R = read_csv(A / "roman_metrics.csv")
    if not R:
        return
    ms = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon", "epsilon2d"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    vcol = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    vlab = {"A": "A: ODC TRL 3", "B": "B: ODC TRL 7", "C": "C: no ODC"}
    for ax, ev in zip(axes, ["linear", "nonlinear"]):
        x = np.arange(len(ms))
        for j, v in enumerate("ABC"):
            g = {r["method"]: r for r in R if r["config"] == f"ROMAN_TRANSIENT|{v}|{ev}"}
            mu = [float(g[m]["hv_tight_mean"]) for m in ms]; sd = [float(g[m]["hv_tight_sd"]) for m in ms]
            ax.bar(x + (j - 1) * 0.27, mu, 0.25, yerr=sd, color=vcol[v], edgecolor="white", linewidth=1,
                   error_kw=dict(ecolor=MUTED, lw=0.8), label=vlab[v])
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in ms], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"Roman Transient, {ev} evaluator (tight HV reference)", fontsize=8, color=INK)
        ax.set_ylabel("HV, mean ± SD (20 seeds)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, frameon=False)  # above the panels, clear of all bars
    save(fig, "fig_roman_variant_comparison_v2")


def fig_pointcount():
    P = read_csv(A / "pointcount_hv_main.csv")
    if not P:
        return
    order, _ = feas_order()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, w in zip(axes.flat, order):
        # Primary E3 figure: all-metric epsilon-constraint, as in the E2 benchmark and pointcount_rankings.
        for m in ["random", "mopso", "nsga2", "spea2", "moead", "epsilon"]:
            rr = sorted([r for r in P if r["config"] == w and r["method"] == m], key=lambda r: int(r["k"]))
            k = [int(r["k"]) for r in rr]
            ax.plot(k, [float(r["random_subsample_hv_mean"]) for r in rr], color=COLOR[m], marker=MARK[m], ms=4,
                    label=r"$\varepsilon$-constraint" if m == "epsilon" else LABEL[m])
            ax.plot(k, [float(r["greedy_best_k_hv_mean"]) for r in rr], color=COLOR[m], linestyle=":", linewidth=1.2)
        ax.set_xscale("log"); ax.set_xlabel("retained points k"); ax.set_ylabel("HV")
        ax.set_title(WL_SHORT[w], fontsize=8, color=INK)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7)
    fig.suptitle("HV at fixed front cardinality: solid = random subsample (100 repeats), dotted = greedy best-k",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_pointcount_hv_v2")


def fig_archive():
    R = read_csv(A / "archive_cap.csv")
    if not R:
        return
    ws = sorted({r["workload"] for r in R})
    fig, axes = plt.subplots(1, len(ws), figsize=(3.4 * len(ws), 3.2))
    for ax, w in zip(np.atleast_1d(axes), ws):
        rr = sorted([r for r in R if r["workload"] == w], key=lambda r: int(r["archive_max"]))
        caps = [int(r["archive_max"]) for r in rr]
        ax.errorbar(caps, [float(r["hv_mean"]) for r in rr], yerr=[float(r["hv_sd"]) for r in rr], color=COLOR["mopso"],
                    marker="s", capsize=2, label="MOPSO")
        bo = float(rr[0]["best_other_hv_mean"])
        ax.axhline(bo, color=INK, linestyle="--", linewidth=1)
        ax.annotate(f"best other: {LABEL[rr[0]['best_other_method']]}", (caps[0], bo), fontsize=6.5, color=INK,
                    xytext=(0, 4), textcoords="offset points")
        ax.set_xlabel("archive cap"); ax.set_ylabel("HV (20 seeds)"); ax.set_title(WL_SHORT.get(w, w), fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, "fig_archive_cap_v2")


def fig_selection():
    p = A / "summary.json"
    if not p.exists():
        return
    S = json.loads(p.read_text())
    keys = [k for k in ["selection_main", "selection_roman", "selection_all"] if k in S]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), keys):
        pol = S[k]["policies"]
        names = list(pol)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [pol[n]["mean_regret_pct"] for n in names], 0.38, color="#2a78d6", label="Mean")
        ax.bar(x + 0.2, [pol[n]["worst_regret_pct"] for n in names], 0.38, color="#eb6834", label="Worst case")
        pol_label = {"rule_nominal(t1=1%,t2=10%)": "Threshold rule (1%, 10%)", "always_mopso": "Always MOPSO",
                     "always_random": "Always Random", "always_nsga2": "Always NSGA-II", "oracle": "Best per workload"}
        panel = {"selection_main": "Non-Roman workloads", "selection_roman": "Roman configurations",
                 "selection_all": "All workloads"}
        ax.set_xticks(x); ax.set_xticklabels([pol_label.get(n, n) for n in names], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("HV regret vs best per workload (%)"); ax.set_title(panel.get(k, k), fontsize=8, color=INK)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_selection_rule_regret_v2")


def fig_feasibility():
    fj = Path("out_feasibility_v2/feasibility.json")
    if not fj.exists():
        return
    F = sorted(json.loads(fj.read_text()), key=lambda r: r["linear_pct"])
    lab = [pretty_workload(r["workload"] + (f"|{r['variant']}" if r["variant"] else "")) for r in F]
    y = np.arange(len(F))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ev, col, off in [("linear", "#2a78d6", -0.2), ("nonlinear", "#eb6834", 0.2)]:
        v = np.array([max(r[f"{ev}_pct"], 1e-3) for r in F])
        ax.barh(y + off, v, 0.38, color=col, label=f"{ev.capitalize()} evaluator")
    for yi, r in zip(y, F):
        ax.annotate(f"{r['linear_pct']:.2f}%", (max(r["linear_pct"], 1e-3), yi - 0.2), fontsize=6, color=INK,
                    xytext=(2, 0), textcoords="offset points", va="center")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("random-policy feasibility (%; 100,000 samples; log scale; 0 drawn at 0.001)")
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("Measured feasibility density", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, "fig_feasibility_v2")


def fig_roman_sweeps():
    D = {sw: json.loads(Path(f"out_roman_sweeps_v2/{sw}/summary.json").read_text())["rows"]
         for sw in ["arrival", "burst", "quality", "nre"] if Path(f"out_roman_sweeps_v2/{sw}/summary.json").exists()}
    if not D:
        return
    vcol = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    vlab = {"A": "A: ODC TRL 3", "B": "B: ODC TRL 7", "C": "C: no ODC"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    if "arrival" in D:
        ax = axes[0]
        for v in "ABC":
            R = [r for r in D["arrival"] if r["variant"] == v]
            ax.errorbar([r["lam"] for r in R], [r["hv_tight_mean"] for r in R], yerr=[r["hv_tight_sd"] for r in R],
                        color=vcol[v], marker="o", ms=4, capsize=2, label=vlab[v])
        ax.set_xscale("log"); ax.set_xlabel("arrival rate λ (task/s)"); ax.set_ylabel("HV (tight reference)")
        ax.set_title("Arrival-rate sweep", fontsize=8, color=INK); ax.legend(fontsize=6.5)
    if "burst" in D:
        ax = axes[1]
        for j, reg in enumerate(["baseline", "burst"]):
            R = {r["variant"]: r for r in D["burst"] if r["condition"] == reg}
            ax.errorbar(np.arange(3) + (j - 0.5) * 0.25, [R[v]["hv_tight_mean"] for v in "ABC"],
                        yerr=[R[v]["hv_tight_sd"] for v in "ABC"], fmt=["o", "s"][j], ms=6, capsize=3,
                        color=["#2a78d6", "#eb6834"][j], linestyle="none", label=f"{reg} (λ={R['A']['lam']})")
        ax.set_xticks(range(3)); ax.set_xticklabels([vlab[v] for v in "ABC"], fontsize=7, rotation=15)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title("Baseline vs burst", fontsize=8, color=INK); ax.legend(fontsize=6.5, loc="lower left")
        ax.set_ylabel("HV (tight reference)")
    if "quality" in D:
        ax = axes[2]
        for v in "ABC":
            R = [r for r in D["quality"] if r["variant"] == v]
            ax.errorbar([r["q"] for r in R], [r["hv_tight_mean"] for r in R], yerr=[r["hv_tight_sd"] for r in R],
                        color=vcol[v], marker="o", ms=4, capsize=2, label=vlab[v])
        ax.set_xlabel("minimum quality threshold"); ax.set_ylabel("HV (tight reference)")
        ax.set_title("Quality-threshold sweep", fontsize=8, color=INK)
    if "nre" in D:
        ax = axes[3]; R = D["nre"]
        x = [r["nre"] / 1000 for r in R]
        ax.errorbar(x, [r["p_odc_mean"] for r in R], yerr=[r["p_odc_sd"] for r in R], color=vcol["B"], marker="o",
                    ms=4, capsize=2, label=f"{P_ODC} mean ± SD")
        ax.plot(x, [r["p_odc_median"] for r in R], color=INK, linestyle=":", marker="s", ms=3, label=f"{P_ODC} median")
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("ODC NRE (M USD)"); ax.set_ylabel(f"{P_ODC} over Pareto points")
        ax.set_ylim(0, 0.62)   # headroom so the legend clears the error bars
        ax.set_title("ODC NRE sweep (Variant B)", fontsize=8, color=INK); ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.suptitle("Roman Transient sweeps (nonlinear evaluator, MOPSO, 10 seeds per condition, mean ± SD)", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig_roman_sweeps_v2")


def main():
    for f in [fig_roman_sweeps, fig_feasibility, fig_hv_distributions, fig_pareto_fronts, fig_convergence, fig_ablation, fig_lin_vs_nl,
              fig_sensitivity, fig_bo, fig_odc, fig_roman, fig_pointcount, fig_archive, fig_selection]:
        try:
            f()
        except Exception as e:  # keep going; report which figure failed
            print(f"[{f.__name__}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
