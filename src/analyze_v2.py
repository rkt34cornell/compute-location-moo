#!/usr/bin/env python3
"""
analyze_v2.py — metrics, statistics and derived analyses (v2).

Inputs (produced by run_benchmark_v2.py and exp_feasibility_v2.py):
  out_benchmark_v2/            main suite (normalized violation, Deb handling)
  out_benchmark_v2_ids_legacy/ IDS, current data, legacy raw-violation + penalty
  out_roman_v2/                ROMAN suite
  out_archive_v2/              MOPSO archive-cap study
  out_feasibility_v2/feasibility.json
Legacy references: out_multiseed*/summary.json, out_gd_igd/gd_igd.csv,
  out_roman_multiseed/summary.json, out_roman_hv_recompute/recomputed_summary.json

Metric definitions (all on the 2-D latency-cost projection used for HV; the
optimizers' fronts are nondominated in the full 41/42-metric space):
  HV       2-D hypervolume in raw minimization units w.r.t. a fixed per-workload
           reference point (run_multiseed_benchmark.WCFG; ROMAN: (1e8 ms, 0.12)
           and, as in the manuscript figure, the tighter evaluator-specific
           points (1e5 ms | 2e6 ms, 0.06 USD)). Only feasible points count; a run
           with no feasible point scores 0.
  Normalization  per configuration, min-max over the pooled feasible points of
           all methods and seeds -> [0,1]^2.
  Reference front  2-D nondominated subset of that pooled set (all methods incl.
           epsilon2d, all seeds).
  GD       RMS form, as implemented in exp_gd_igd.gd_vs_ref:
           GD = sqrt( (1/|A|) * sum_{a in A} min_{r in R} ||a - r||_2^2 ).
           (Van Veldhuizen's original form, sqrt(sum d^2)/|A|, differs by a
           factor sqrt(|A|).) Also reported on the 2-D nondominated subset of A.
  IGD      arithmetic mean: IGD = (1/|R|) * sum_{r in R} min_{a in A} ||r - a||_2.
  Spacing  normalized spacing: population SD (ddof=0) of the L1 nearest-neighbour
           distances among the run's points in the normalized space; undefined
           (excluded, counted) for fronts with < 2 points.
  Front size  number of feasible points returned.
  Feasibility  (a) share of seeds that returned a feasible front;
           (b) search feasibility rate = feasible evaluations / evaluations.
Statistics: two-sided Wilcoxon signed-rank test on per-seed HV, paired by seed
index (scipy zero_method='wilcox'); Holm correction within each configuration
over all method pairs; effect sizes: matched-pairs rank-biserial correlation
r_rb = (W+ - W-)/(W+ + W-) and Vargha-Delaney A12 = P(HV_A > HV_B) + 0.5 P(=).
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, wilcoxon, kendalltau

import moo_core as mc
from run_benchmark_v2 import hv_ref_keys, MAIN_WORKLOADS
from run_multiseed_benchmark import hv2d, to_min_2d

OUT = Path("out_analysis_v2")
METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon", "epsilon2d"]
LEGACY_METHODS = ["random", "mopso", "nsga2", "spea2", "moead", "epsilon"]
ROMAN_TIGHT_REF = {"linear": np.array([1.0e5, 0.06]), "nonlinear": np.array([2.0e6, 0.06])}


# ── loading ───────────────────────────────────────────────────────────────────

def load_runs(d: Path):
    if not (d / "runs").exists():
        return []
    return [json.loads(p.read_text()) for p in sorted((d / "runs").glob("*.json"))]


def cfg_key(r):
    return r["workload"] if not r["variant"] else f"{r['workload']}|{r['variant']}|{r['evaluator']}"


@lru_cache(maxsize=None)
def workload_for(wl_name, variant):
    wl = mc.load_workload(Path("data.json"), wl_name)
    return mc.apply_odc_variant(wl, variant) if variant else wl


def feasible_pts(d: Path, r):
    """Feasible 2-D min-space points of a run (raw units)."""
    V = np.load(d / "fronts" / f"{r['job_id']}_V.npy")
    wl = workload_for(r["workload"], r["variant"])
    mk = [m.key for m in wl.metrics]
    f = np.array([mc.feasibility_and_violation(wl, dict(zip(mk, row)))[0] for row in V], bool)
    lk, ck, _ = hv_ref_keys(r["workload"])
    return to_min_2d(V[f], wl, mk, lk, ck) if f.any() else np.zeros((0, 2))


def nd2d(pts):
    if pts.shape[0] == 0:
        return pts
    o = np.lexsort((pts[:, 1], pts[:, 0]))
    s = pts[o]
    best = np.minimum.accumulate(np.concatenate([[np.inf], s[:-1, 1]]))
    return s[s[:, 1] < best]


# ── front-quality metrics ─────────────────────────────────────────────────────

def gd_rms(A, R):
    if A.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    d, _ = cKDTree(R).query(A, k=1)
    return float(np.sqrt(np.mean(d ** 2)))


def igd_mean(A, R):
    if A.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    d, _ = cKDTree(A).query(R, k=1)
    return float(np.mean(d))


def spacing_l1(A):
    if A.shape[0] < 2:
        return np.nan
    d, _ = cKDTree(A).query(A, k=2, p=1)
    return float(np.std(d[:, 1]))


def msd(x):
    x = np.asarray([v for v in x if v == v], float)
    if x.size == 0:
        return np.nan, np.nan, 0
    return float(x.mean()), (float(x.std(ddof=1)) if x.size > 1 else 0.0), int(x.size)


def config_metrics(d: Path, runs, extra_hv_ref=None):
    """Per configuration: per-method metric summaries + per-seed HV vectors."""
    by_cfg = {}
    for r in runs:
        by_cfg.setdefault(cfg_key(r), []).append(r)
    rows, per_seed = [], {}
    for cfg, R in by_cfg.items():
        methods = [m for m in METHODS if any(x["method"] == m for x in R)]
        pts = {m: {x["seed"]: feasible_pts(d, x) for x in R if x["method"] == m} for m in methods}
        pooled = np.vstack([p for m in methods for p in pts[m].values() if p.shape[0]] or [np.zeros((0, 2))])
        lo = pooled.min(0) if pooled.size else np.zeros(2)
        hi = pooled.max(0) if pooled.size else np.ones(2)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        ref_front = nd2d((pooled - lo) / span) if pooled.size else np.zeros((0, 2))
        per_seed[cfg] = {}
        for m in methods:
            X = sorted([x for x in R if x["method"] == m], key=lambda x: x["seed"])
            hv = [x["hv"] for x in X]
            norm = [(pts[m][x["seed"]] - lo) / span for x in X]
            gd = [gd_rms(a, ref_front) for a in norm]
            gd2 = [gd_rms(nd2d(a), ref_front) for a in norm]
            igd = [igd_mean(a, ref_front) for a in norm]
            sp = [spacing_l1(a) for a in norm]
            nf = [x["n_front"] for x in X]
            row = {"config": cfg, "method": m, "n_seeds": len(X),
                   "hv_mean": float(np.mean(hv)), "hv_sd": float(np.std(hv, ddof=1)) if len(hv) > 1 else 0.0,
                   "front_mean": float(np.mean(nf)), "front_sd": float(np.std(nf, ddof=1)) if len(nf) > 1 else 0.0,
                   "front_min": int(min(nf)), "front_max": int(max(nf)),
                   "seeds_feasible": int(sum(x["feasible_found"] for x in X)),
                   "search_feasible_rate_mean": float(np.mean([x["search_feasible_rate"] for x in X])),
                   "evals_mean": float(np.mean([x["n_evals"] + (2000 if m.startswith("epsilon") else 0) for x in X])),
                   "n_ref_front": int(ref_front.shape[0])}
            for name, vals in [("gd", gd), ("gd_nd2d", gd2), ("igd", igd), ("spacing_norm", sp)]:
                mu, s, n = msd(vals)
                row[f"{name}_mean"], row[f"{name}_sd"], row[f"{name}_n"] = mu, s, n
            if extra_hv_ref is not None:
                ev = X[0]["evaluator"]
                ref = extra_hv_ref[ev]
                hvt = [hv2d(pts[m][x["seed"]], ref) if pts[m][x["seed"]].shape[0] else 0.0 for x in X]
                row["hv_tight_mean"], row["hv_tight_sd"], _ = msd(hvt)
                per_seed[cfg].setdefault("_tight", {})[m] = np.array(hvt)
            rows.append(row)
            per_seed[cfg][m] = np.array(hv)
    return rows, per_seed


# ── statistics ────────────────────────────────────────────────────────────────

def holm(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return out


def pair_stats(a, b):
    diff = a - b
    nz = diff[diff != 0]
    if nz.size == 0:
        return dict(W=np.nan, p=1.0, r_rb=0.0, A12=0.5, n_nonzero=0)
    r = rankdata(np.abs(nz))
    wp, wm = r[nz > 0].sum(), r[nz < 0].sum()
    try:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        W, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        W, p = np.nan, np.nan
    gt = (a[:, None] > b[None, :]).mean(); eq = (a[:, None] == b[None, :]).mean()
    return dict(W=W, p=p, r_rb=float((wp - wm) / (wp + wm)), A12=float(gt + 0.5 * eq), n_nonzero=int(nz.size))


def pairwise(per_seed):
    rows = []
    for cfg, d in per_seed.items():
        methods = [m for m in METHODS if m in d]
        recs = []
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                s = pair_stats(d[a], d[b])
                recs.append(dict(config=cfg, method_a=a, method_b=b, test="Wilcoxon signed-rank (two-sided, paired by seed)",
                                 W=s["W"], p_raw=s["p"], r_rank_biserial=s["r_rb"], A12=s["A12"],
                                 n_pairs=len(d[a]), n_nonzero=s["n_nonzero"],
                                 mean_diff=float(d[a].mean() - d[b].mean())))
        adj = holm([r["p_raw"] for r in recs])
        for r, q in zip(recs, adj):
            r["p_holm"] = float(q); r["n_tests_in_family"] = len(recs)
            r["significant_holm_0.05"] = bool(q == q and q < 0.05)
        rows += recs
    return rows


# ── selection rule ────────────────────────────────────────────────────────────

def rule_choice(f_pct, t1, t2):
    return "nsga2" if f_pct < t1 else ("mopso" if f_pct < t2 else "random")


def selection_analysis(metric_rows, feas, cfgs, eval_key):
    # ROMAN rows use the tight reference (the saturated 1e8 ms point compresses all HVs to ~1.116e7)
    hv = {(r["config"], r["method"]): r.get("hv_tight_mean", r["hv_mean"]) for r in metric_rows}
    out = {"workloads": [], "policies": {}}
    for cfg in cfgs:
        oracle_m = max(LEGACY_METHODS, key=lambda m: hv.get((cfg, m), -np.inf))
        out["workloads"].append({"config": cfg, "feasibility_pct": feas[cfg], "oracle": oracle_m,
                                 "oracle_hv": hv[(cfg, oracle_m)],
                                 **{f"hv_{m}": hv.get((cfg, m)) for m in LEGACY_METHODS}})

    def regret(choice_fn):
        rg = []
        for w in out["workloads"]:
            m = choice_fn(w)
            o = w["oracle_hv"]
            rg.append(100.0 * (o - w[f"hv_{m}"]) / o if o > 0 else 0.0)
        return {"mean_regret_pct": float(np.mean(rg)), "median_regret_pct": float(np.median(rg)),
                "worst_regret_pct": float(np.max(rg)),
                "per_workload_pct": dict(zip([w["config"] for w in out["workloads"]], map(float, rg))),
                "choices": [choice_fn(w) for w in out["workloads"]]}

    out["policies"]["rule_nominal(t1=1%,t2=10%)"] = regret(lambda w: rule_choice(w["feasibility_pct"], 1.0, 10.0))
    out["policies"]["always_mopso"] = regret(lambda w: "mopso")
    out["policies"]["always_random"] = regret(lambda w: "random")
    out["policies"]["always_nsga2"] = regret(lambda w: "nsga2")
    out["policies"]["oracle"] = regret(lambda w: w["oracle"])
    grid = []
    for t1 in [0.2, 0.5, 1.0, 2.0, 2.8]:
        for t2 in [5.0, 10.0, 15.0, 19.6]:
            r = regret(lambda w: rule_choice(w["feasibility_pct"], t1, t2))
            grid.append({"t1_pct": t1, "t2_pct": t2, **{k: r[k] for k in ["mean_regret_pct", "median_regret_pct", "worst_regret_pct"]}})
    out["rule_threshold_grid"] = grid
    out["rule_best_over_grid"] = min(grid, key=lambda g: (g["mean_regret_pct"], g["worst_regret_pct"]))
    out["rule_worst_over_grid"] = max(grid, key=lambda g: (g["mean_regret_pct"], g["worst_regret_pct"]))
    out["feasibility_source"] = eval_key
    return out


# ── point-count-controlled HV ─────────────────────────────────────────────────

def greedy_best_k(pts, ref, k):
    P = nd2d(pts)
    if P.shape[0] <= k:
        return hv2d(P, ref)
    chosen = []
    remaining = list(range(P.shape[0]))
    for _ in range(k):
        best, bi = -1.0, None
        for i in remaining:
            h = hv2d(P[chosen + [i]], ref)
            if h > best:
                best, bi = h, i
        chosen.append(bi); remaining.remove(bi)
    return hv2d(P[chosen], ref)


def pointcount_analysis(d: Path, runs, ks=(5, 10, 25, 50), reps=100):
    by_cfg = {}
    for r in runs:
        by_cfg.setdefault(cfg_key(r), []).append(r)
    rows = []
    rng = np.random.default_rng(2026)
    for cfg, R in by_cfg.items():
        ref = hv_ref_keys(R[0]["workload"])[2]
        pts = {(x["method"], x["seed"]): feasible_pts(d, x) for x in R}
        sizes = {m: np.median([pts[(m, x["seed"])].shape[0] for x in R if x["method"] == m])
                 for m in METHODS if any(x["method"] == m for x in R)}
        nontriv = [s for m, s in sizes.items() if s >= 2]
        kmin = int(min(nontriv)) if nontriv else 2
        for k in sorted(set([kmin, *ks])):
            for m in sizes:
                rand_hv, best_hv, n_short = [], [], 0
                for x in R:
                    if x["method"] != m:
                        continue
                    A = pts[(m, x["seed"])]
                    if A.shape[0] == 0:
                        rand_hv.append(0.0); best_hv.append(0.0); continue
                    if A.shape[0] <= k:
                        n_short += 1
                        h = hv2d(A, ref); rand_hv.append(h); best_hv.append(h); continue
                    hs = [hv2d(A[rng.choice(A.shape[0], k, replace=False)], ref) for _ in range(reps)]
                    rand_hv.append(float(np.mean(hs)))
                    best_hv.append(greedy_best_k(A, ref, k))
                rows.append({"config": cfg, "k": k, "k_is_min_nontrivial": k == kmin, "method": m,
                             "median_front": float(sizes[m]),
                             "random_subsample_hv_mean": float(np.mean(rand_hv)),
                             "random_subsample_hv_sd": float(np.std(rand_hv, ddof=1)),
                             "greedy_best_k_hv_mean": float(np.mean(best_hv)),
                             "greedy_best_k_hv_sd": float(np.std(best_hv, ddof=1)),
                             "seeds_with_front_le_k": n_short, "n_seeds": len(rand_hv), "reps": reps})
    return rows


def rankings_from(rows, key, cfg_field="config", group=("k",)):
    out = {}
    for r in rows:
        g = (r[cfg_field],) + tuple(r[x] for x in group)
        out.setdefault(g, []).append((r[key], r["method"]))
    return {g: [m for _, m in sorted(v, key=lambda t: -t[0])] for g, v in out.items()}


# ── legacy references ─────────────────────────────────────────────────────────

def legacy_main():
    src = {"IDS": "out_multiseed_IDS", "SUNCATCHER": "out_multiseed_SUNCATCHER", "JWST": "out_multiseed_JWST",
           "RST": "out_multiseed_RST", "LEO_OOS": "out_multiseed", "DEEP_SPACE_CRUISE": "out_multiseed"}
    gd = {}
    p = Path("out_gd_igd/gd_igd.csv")
    if p.exists():
        for r in csv.DictReader(open(p)):
            gd[(r["label"], r["method"])] = r
    rows = []
    for wl, d in src.items():
        S = json.loads((Path(d) / "summary.json").read_text())
        for m, s in S["summary"][wl].items():
            raw = S["raw"][wl][m]
            sp = [x["spacing"] for x in raw]
            g = gd.get((wl, m), {})
            rows.append({"config": wl, "method": m, "hv_mean": s["hv_mean"], "hv_sd": s["hv_std"],
                         "front_mean": s["pareto_size_mean"], "front_sd": s["pareto_size_std"],
                         "spacing_raw_mean": float(np.mean(sp)),
                         "gd_mean": float(g["gd_mean"]) if g else np.nan, "igd_mean": float(g["igd_mean"]) if g else np.nan,
                         "source": d, "note": "stale (pre-HIL IDS data)" if wl == "IDS" else
                         ("GD/IGD row stale (pre-correction fronts)" if wl == "RST" else "")})
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    OUT.mkdir(exist_ok=True)
    summary = {}
    feas = {}
    fj = Path("out_feasibility_v2/feasibility.json")
    if fj.exists():
        for r in json.loads(fj.read_text()):
            if r["variant"]:
                for ev in ["linear", "nonlinear"]:
                    feas[f"{r['workload']}|{r['variant']}|{ev}"] = r[f"{ev}_pct"]
            else:
                feas[r["workload"]] = r["linear_pct"]

    # main suite
    d_main = Path("out_benchmark_v2")
    runs = load_runs(d_main)
    main_rows, main_seed = config_metrics(d_main, runs)
    write_csv(OUT / "main_metrics.csv", main_rows)
    pw = pairwise(main_seed)
    write_csv(OUT / "main_pairwise_tests.csv", pw)
    summary["main_n_runs"] = len(runs)

    # IDS legacy-handling control on current data
    d_leg = Path("out_benchmark_v2_ids_legacy")
    leg_runs = load_runs(d_leg)
    if leg_runs:
        leg_rows, leg_seed = config_metrics(d_leg, leg_runs)
        write_csv(OUT / "ids_legacy_current_metrics.csv", leg_rows)
        write_csv(OUT / "ids_legacy_current_pairwise.csv", pairwise(leg_seed))

    # ROMAN
    d_rom = Path("out_roman_v2")
    rom_runs = load_runs(d_rom)
    if rom_runs:
        rom_rows, rom_seed = config_metrics(d_rom, rom_runs, extra_hv_ref=ROMAN_TIGHT_REF)
        write_csv(OUT / "roman_metrics.csv", rom_rows)
        write_csv(OUT / "roman_pairwise_tests.csv", pairwise({k: {m: v for m, v in d.items() if m != "_tight"}
                                                              for k, d in rom_seed.items()}))
        write_csv(OUT / "roman_pairwise_tests_tightref.csv", pairwise({k: d["_tight"] for k, d in rom_seed.items()}))
        voo = []
        for wl in ["ROMAN_TRANSIENT", "ROMAN_TRANSIENT_FOLLOWUP"]:
            for ev in ["linear", "nonlinear"]:
                for m in METHODS:
                    g = {v: next((r for r in rom_rows if r["config"] == f"{wl}|{v}|{ev}" and r["method"] == m), None)
                         for v in "ABC"}
                    if not all(g.values()):
                        continue
                    for refname, key in [("saturated(1e8,0.12)", "hv_mean"), ("tight", "hv_tight_mean")]:
                        hb, hc = g["B"][key], g["C"][key]
                        voo.append({"workload": wl, "evaluator": ev, "method": m, "hv_ref": refname,
                                    "hv_A": g["A"][key], "hv_B": hb, "hv_C": hc, "delta_BC": hb - hc,
                                    "delta_BC_pct": 100 * (hb - hc) / hc if hc else np.nan})
        write_csv(OUT / "roman_value_of_odc.csv", voo)
        leg = Path("out_roman_hv_recompute/recomputed_summary.json")
        if leg.exists():
            L = json.loads(leg.read_text())["summary"]
            cmp_rows = []
            for cfg in sorted({r["config"] for r in rom_rows}):
                new = {r["method"]: r["hv_tight_mean"] for r in rom_rows if r["config"] == cfg}
                old = {m: v["hv_mean"] for m, v in L.get(cfg, {}).items()}
                lo = ">".join(sorted(old, key=lambda m: -old[m]))
                no = ">".join(sorted([m for m in new if m in LEGACY_METHODS], key=lambda m: -new[m]))
                for m in LEGACY_METHODS:
                    cmp_rows.append({"config": cfg, "method": m, "hv_tight_legacy": old.get(m), "hv_tight_v2": new.get(m),
                                     "legacy_order": lo, "v2_order": no, "order_changed": lo != no,
                                     "best_changed": lo.split(">")[0] != no.split(">")[0]})
            write_csv(OUT / "roman_legacy_vs_v2.csv", cmp_rows)

    # selection rule
    if feas:
        main_cfgs = [w for w in MAIN_WORKLOADS if w in feas]
        summary["selection_main"] = selection_analysis(main_rows, feas, main_cfgs, "linear random-policy feasibility")
        if rom_runs:
            rc = sorted({r["config"] for r in rom_rows if r["config"] in feas})
            summary["selection_roman"] = selection_analysis(rom_rows, feas, rc, "evaluator-matched random-policy feasibility")
            summary["selection_all"] = selection_analysis(main_rows + rom_rows, feas, main_cfgs + rc, "both")

    # point-count-controlled HV (main suite)
    pc = pointcount_analysis(d_main, runs)
    write_csv(OUT / "pointcount_hv_main.csv", pc)
    full_rank = rankings_from([dict(r, k="full") for r in main_rows if r["method"] in LEGACY_METHODS],
                              "hv_mean")
    pc_l = [r for r in pc if r["method"] in LEGACY_METHODS]
    rk_rand = rankings_from(pc_l, "random_subsample_hv_mean")
    rk_best = rankings_from(pc_l, "greedy_best_k_hv_mean")
    rk = []
    for (cfg, k), order in rk_rand.items():
        fo = full_rank[(cfg, "full")]
        pos = lambda o: [o.index(m) for m in LEGACY_METHODS]
        rk.append({"config": cfg, "k": k, "full_front_order": ">".join(fo),
                   "random_subsample_order": ">".join(order), "greedy_best_k_order": ">".join(rk_best[(cfg, k)]),
                   "kendall_tau_full_vs_random": float(kendalltau(pos(fo), pos(order)).statistic),
                   "kendall_tau_full_vs_greedy": float(kendalltau(pos(fo), pos(rk_best[(cfg, k)])).statistic),
                   "top_full": fo[0], "top_random_subsample": order[0], "top_greedy": rk_best[(cfg, k)][0]})
    write_csv(OUT / "pointcount_rankings_main.csv", rk)

    # archive study
    d_arc = Path("out_archive_v2")
    arc = load_runs(d_arc)
    if arc:
        rows = []
        for wl in sorted({r["workload"] for r in arc}):
            others = {r["method"]: r["hv_mean"] for r in main_rows if r["config"] == wl and r["method"] != "mopso"}
            best_other = max(others, key=others.get)
            for cap in sorted({r["archive_max"] for r in arc}):
                X = [r for r in arc if r["workload"] == wl and r["archive_max"] == cap]
                h = np.array([r["hv"] for r in X])
                rows.append({"workload": wl, "archive_max": cap, "n_seeds": len(X), "hv_mean": float(h.mean()),
                             "hv_sd": float(h.std(ddof=1)), "front_mean": float(np.mean([r["n_front"] for r in X])),
                             "best_other_method": best_other, "best_other_hv_mean": others[best_other],
                             "mopso_minus_best_other": float(h.mean() - others[best_other]),
                             "mopso_rank_among_methods": 1 + sum(v > h.mean() for v in others.values())})
        write_csv(OUT / "archive_cap.csv", rows)

    write_csv(OUT / "legacy_main_metrics.csv", legacy_main())
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print("analysis v2 written to", OUT)


if __name__ == "__main__":
    main()
