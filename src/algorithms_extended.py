#!/usr/bin/env python3
"""
algorithms_extended.py

Additional MOO algorithms for the compute-location benchmark:
  - NSGA-II   (Deb et al. 2002) — the canonical population-based MOO algorithm
  - SPEA2     (Zitzler et al. 2001) — strength Pareto evolutionary algorithm
  - MOEA/D    (Zhang & Li 2007) — decomposition-based; Tchebycheff scalarization
  - Epsilon-constraint — systematic single-objective sweeps over constraint levels

All operate over p ∈ Δ^{J-1} (simplex) using the moo_core evaluator.
Output: (archive_P, archive_V) matching the mopso signature so run_extended.py
can treat them uniformly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import moo_core as _mc
from moo_core import (
    Direction,
    Workload,
    aggregate_metrics_linear,
    feasibility_and_violation,
    pareto_front_indices,
    project_to_simplex,
    to_minimization,
)


# ─────────────────────────── shared helpers ────────────────────────────────────

def _eval(workload: Workload, metric_keys: Sequence[str], p: np.ndarray):
    pm = aggregate_metrics_linear(workload, p)
    raw = np.array([pm[k] for k in metric_keys], dtype=float)
    feasible, viol, _ = feasibility_and_violation(workload, pm)
    return raw, feasible, float(viol)


def _deb_mode() -> bool:
    return _mc.CONSTRAINT_HANDLING == "deb"


class _BestInfeasible:
    """Least-violation candidate seen during a run (documented no-feasible fallback).

    Every algorithm applies the same rule: if no feasible solution is retained,
    return the single candidate with the smallest violation among all evaluated
    candidates. Runners re-check feasibility and score such a result as
    'no feasible front' (HV = 0, front size 0).
    """
    def __init__(self):
        self.p = None; self.raw = None; self.viol = np.inf

    def update(self, p, raw, feas, viol):
        if not feas and viol < self.viol:
            self.p = np.array(p, dtype=float).copy()
            self.raw = np.array(raw, dtype=float).copy()
            self.viol = float(viol)

    def result(self):
        return self.p[None, :], self.raw[None, :]


def _pareto_dom_matrix(costs: np.ndarray) -> np.ndarray:
    """D[p, q] = True iff costs[p] Pareto-dominates costs[q] (minimization)."""
    le = np.all(costs[:, None, :] <= costs[None, :, :], axis=2)
    lt = np.any(costs[:, None, :] < costs[None, :, :], axis=2)
    return le & lt


def _deb_dom_matrix(costs: np.ndarray, feas: np.ndarray, viol: np.ndarray) -> np.ndarray:
    """Deb constrained domination: feasible beats infeasible; infeasible pairs by
    violation; feasible pairs by Pareto dominance on `costs` (unpenalized)."""
    feas = np.asarray(feas, dtype=bool); viol = np.asarray(viol, dtype=float)
    D = _pareto_dom_matrix(costs) & feas[:, None] & feas[None, :]
    D |= feas[:, None] & ~feas[None, :]
    D |= (~feas[:, None]) & (~feas[None, :]) & (viol[:, None] < viol[None, :])
    return D


def _sign_vector(workload: Workload, metric_keys: Sequence[str]) -> np.ndarray:
    key_dir = {m.key: m.direction for m in workload.metrics}
    return np.array(
        [1.0 if key_dir[k] == Direction.MINIMIZE else -1.0 for k in metric_keys],
        dtype=float,
    )


def _deb_dominates(a_raw, a_feas, a_viol, b_raw, b_feas, b_viol, sign) -> bool:
    """Return True if a dominates b under Deb's constraint handling."""
    if a_feas and not b_feas:
        return True
    if not a_feas and b_feas:
        return False
    if not a_feas and not b_feas:
        return a_viol < b_viol
    # both feasible — Pareto dominance in minimization space
    a_min = sign * a_raw
    b_min = sign * b_raw
    return bool(np.all(a_min <= b_min) and np.any(a_min < b_min))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NSGA-II
# ══════════════════════════════════════════════════════════════════════════════

def _fast_nondominated_sort_min(costs: np.ndarray,
                                dom: Optional[np.ndarray] = None) -> List[List[int]]:
    """Classic fast non-dominated sort. costs: (N,d) minimization.

    The pairwise dominance test is vectorized; the front bookkeeping (and hence
    the order of indices within each front) is identical to the original
    double loop. Pass `dom` (D[p,q] = p dominates q) to sort under a
    non-Pareto relation such as Deb constrained domination.
    """
    N = costs.shape[0]
    D = _pareto_dom_matrix(costs) if dom is None else dom
    S: List[List[int]] = [np.nonzero(D[p])[0].tolist() for p in range(N)]  # dominated set
    n: List[int] = D.sum(axis=0).astype(int).tolist()                      # domination count
    fronts: List[List[int]] = [[p for p in range(N) if n[p] == 0]]

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance_nsga(costs: np.ndarray, front: List[int]) -> np.ndarray:
    """Crowding distance for a single front."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    m = costs.shape[1]
    dist = np.zeros(n)
    sub = costs[front]
    for j in range(m):
        order = np.argsort(sub[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fj = sub[order, j]
        denom = max(fj[-1] - fj[0], 1e-12)
        # same per-element arithmetic as the original k-loop
        dist[order[1:-1]] += (fj[2:] - fj[:-2]) / denom
    return dist


def _sbx_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
                    eta: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover then project to simplex."""
    n = p1.shape[0]
    c1, c2 = p1.copy(), p2.copy()
    for i in range(n):
        if rng.random() > 0.5:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return project_to_simplex(c1), project_to_simplex(c2)


def _polynomial_mutation(p: np.ndarray, rng: np.random.Generator,
                          eta_m: float = 20.0, pm_rate: float | None = None) -> np.ndarray:
    n = p.shape[0]
    rate = pm_rate if pm_rate is not None else 1.0 / n
    q = p.copy()
    for i in range(n):
        if rng.random() < rate:
            u = rng.random()
            if u < 0.5:
                delta = (2.0 * u) ** (1.0 / (eta_m + 1)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1))
            q[i] = q[i] + delta * 0.1
    return project_to_simplex(q)


def nsga2(
    *,
    workload: Workload,
    metric_keys: Sequence[str],
    pop_size: int = 100,
    generations: int = 200,
    seed: int = 0,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NSGA-II on the compute-location simplex.
    Returns (P_pareto, V_pareto) — positions and raw metric values of
    the final nondominated front.
    """
    rng = np.random.default_rng(seed)
    J = len(workload.tiers)
    d = len(metric_keys)
    sign = _sign_vector(workload, metric_keys)

    # ── initialise population ──────────────────────────────────────────────
    pop_P = rng.dirichlet(np.ones(J), size=pop_size)
    pop_raw = np.zeros((pop_size, d), dtype=float)
    pop_feas = np.zeros(pop_size, dtype=bool)
    pop_viol = np.zeros(pop_size, dtype=float)

    deb = _deb_mode()
    best_inf = _BestInfeasible()
    for i in range(pop_size):
        pop_raw[i], pop_feas[i], pop_viol[i] = _eval(workload, metric_keys, pop_P[i])
        best_inf.update(pop_P[i], pop_raw[i], pop_feas[i], pop_viol[i])

    def make_costs(raw, feas, viol):
        """Costs in minimisation space (legacy: plus infeasible penalty)."""
        costs = sign * raw  # (N, d)
        if deb:
            return costs  # constraints enter through the Deb dominance matrix
        # Infeasible individuals get pushed to worst + violation magnitude
        for i in range(raw.shape[0]):
            if not feas[i]:
                costs[i] += viol[i] * 10.0
        return costs

    def sort_fronts(costs, feas, viol):
        return _fast_nondominated_sort_min(
            costs, _deb_dom_matrix(costs, feas, viol) if deb else None)

    for gen in range(generations):
        costs = make_costs(pop_raw, pop_feas, pop_viol)
        fronts = sort_fronts(costs, pop_feas, pop_viol)

        # Assign rank and crowding
        rank = np.zeros(pop_size, dtype=int)
        crowding = np.zeros(pop_size, dtype=float)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
            cd = _crowding_distance_nsga(costs, front)
            for k, idx in enumerate(front):
                crowding[idx] = cd[k]

        # ── generate offspring ────────────────────────────────────────────
        offspring_P = []
        while len(offspring_P) < pop_size:
            # binary tournament
            def tournament(a, b):
                if rank[a] < rank[b]:
                    return a
                if rank[b] < rank[a]:
                    return b
                return a if crowding[a] >= crowding[b] else b

            i1 = tournament(int(rng.integers(pop_size)), int(rng.integers(pop_size)))
            i2 = tournament(int(rng.integers(pop_size)), int(rng.integers(pop_size)))
            c1, c2 = _sbx_crossover(pop_P[i1], pop_P[i2], rng, eta=eta_c)
            c1 = _polynomial_mutation(c1, rng, eta_m=eta_m)
            c2 = _polynomial_mutation(c2, rng, eta_m=eta_m)
            offspring_P.extend([c1, c2])

        offspring_P = np.array(offspring_P[:pop_size])
        off_raw = np.zeros((pop_size, d), dtype=float)
        off_feas = np.zeros(pop_size, dtype=bool)
        off_viol = np.zeros(pop_size, dtype=float)
        for i in range(pop_size):
            off_raw[i], off_feas[i], off_viol[i] = _eval(workload, metric_keys, offspring_P[i])
            best_inf.update(offspring_P[i], off_raw[i], off_feas[i], off_viol[i])

        # ── combine + select ──────────────────────────────────────────────
        comb_P    = np.vstack([pop_P, offspring_P])
        comb_raw  = np.vstack([pop_raw, off_raw])
        comb_feas = np.concatenate([pop_feas, off_feas])
        comb_viol = np.concatenate([pop_viol, off_viol])

        comb_costs = make_costs(comb_raw, comb_feas, comb_viol)
        fronts_c = sort_fronts(comb_costs, comb_feas, comb_viol)

        rank_c = np.zeros(2 * pop_size, dtype=int)
        crowd_c = np.zeros(2 * pop_size, dtype=float)
        for r, front in enumerate(fronts_c):
            for idx in front:
                rank_c[idx] = r
            cd = _crowding_distance_nsga(comb_costs, front)
            for k, idx in enumerate(front):
                crowd_c[idx] = cd[k]

        # Select best pop_size individuals
        order = np.lexsort((-crowd_c, rank_c))
        sel = order[:pop_size]
        pop_P    = comb_P[sel]
        pop_raw  = comb_raw[sel]
        pop_feas = comb_feas[sel]
        pop_viol = comb_viol[sel]

    # ── extract final nondominated feasible front ──────────────────────────
    feas_idx = np.where(pop_feas)[0]
    if feas_idx.size == 0:
        # documented fallback: least violation among all evaluated candidates
        return best_inf.result()

    V_feas = pop_raw[feas_idx]
    par_idx = pareto_front_indices(V_feas, workload.metrics, list(metric_keys))
    return pop_P[feas_idx[par_idx]], V_feas[par_idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SPEA2
# ══════════════════════════════════════════════════════════════════════════════

def spea2(
    *,
    workload: Workload,
    metric_keys: Sequence[str],
    pop_size: int = 100,
    archive_size: int = 100,
    generations: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SPEA2: Strength Pareto Evolutionary Algorithm 2.
    Returns (P_pareto, V_pareto).
    """
    rng = np.random.default_rng(seed)
    J = len(workload.tiers)
    d = len(metric_keys)
    sign = _sign_vector(workload, metric_keys)

    deb = _deb_mode()
    best_inf = _BestInfeasible()

    def eval_pop(P):
        raw = np.zeros((P.shape[0], d))
        feas = np.zeros(P.shape[0], dtype=bool)
        viol = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            raw[i], feas[i], viol[i] = _eval(workload, metric_keys, P[i])
            best_inf.update(P[i], raw[i], feas[i], viol[i])
        return raw, feas, viol

    def min_costs(raw, feas, viol):
        c = sign * raw
        if deb:
            return c  # constraints enter through the Deb dominance matrix
        for i in range(raw.shape[0]):
            if not feas[i]:
                c[i] += viol[i] * 10.0
        return c

    pop_P = rng.dirichlet(np.ones(J), size=pop_size)
    pop_raw, pop_feas, pop_viol = eval_pop(pop_P)
    arch_P = np.zeros((0, J))
    arch_raw = np.zeros((0, d))
    arch_feas = np.zeros(0, dtype=bool)
    arch_viol = np.zeros(0)

    for gen in range(generations):
        # Combine pop + archive
        comb_P    = np.vstack([pop_P, arch_P]) if arch_P.shape[0] > 0 else pop_P.copy()
        comb_raw  = np.vstack([pop_raw, arch_raw]) if arch_raw.shape[0] > 0 else pop_raw.copy()
        comb_feas = np.concatenate([pop_feas, arch_feas]) if arch_feas.shape[0] > 0 else pop_feas.copy()
        comb_viol = np.concatenate([pop_viol, arch_viol]) if arch_viol.shape[0] > 0 else pop_viol.copy()
        N = comb_P.shape[0]
        costs = min_costs(comb_raw, comb_feas, comb_viol)

        # Strength = number of solutions dominated; raw fitness = summed
        # strength of dominators (integer-valued, so the vectorized sums are exact)
        D = _deb_dom_matrix(costs, comb_feas, comb_viol) if deb else _pareto_dom_matrix(costs)
        strength = D.sum(axis=1).astype(float)
        dominated_by = (D * strength[:, None]).sum(axis=0)

        # k-NN density estimate (clamp k so it never exceeds N-1)
        k = max(1, min(int(np.sqrt(N)), N - 1))
        density = np.zeros(N)
        diff = costs[:, None, :] - costs[None, :, :]
        dists = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        for i in range(N):
            sorted_d = np.sort(dists[i])
            density[i] = 1.0 / (sorted_d[k] + 2.0)

        fitness = dominated_by + density

        # Archive: nondominated individuals (fitness < 1)
        nondom = np.where(fitness < 1.0)[0]
        if nondom.size <= archive_size:
            new_arch_idx = list(nondom)
            # Fill with best dominated if archive underfull
            if len(new_arch_idx) < archive_size:
                dominated = np.where(fitness >= 1.0)[0]
                order = dominated[np.argsort(fitness[dominated])]
                fill = order[:archive_size - len(new_arch_idx)]
                new_arch_idx.extend(fill.tolist())
        else:
            # Truncate: iteratively remove most crowded
            idx = list(nondom)
            while len(idx) > archive_size:
                sub_dists = dists[np.ix_(idx, idx)]
                # minimum distance for each, then remove smallest
                min_d = np.sort(sub_dists, axis=1)[:, 1]  # skip self (0)
                remove = idx[int(np.argmin(min_d))]
                idx.remove(remove)
            new_arch_idx = idx

        arch_P    = comb_P[new_arch_idx]
        arch_raw  = comb_raw[new_arch_idx]
        arch_feas = comb_feas[new_arch_idx]
        arch_viol = comb_viol[new_arch_idx]

        # Mating: binary tournament on archive fitness
        arch_fit = fitness[new_arch_idx]
        new_pop = []
        while len(new_pop) < pop_size:
            i1 = int(rng.integers(arch_P.shape[0]))
            i2 = int(rng.integers(arch_P.shape[0]))
            parent1 = arch_P[i1 if arch_fit[i1] <= arch_fit[i2] else i2]
            i3 = int(rng.integers(arch_P.shape[0]))
            i4 = int(rng.integers(arch_P.shape[0]))
            parent2 = arch_P[i3 if arch_fit[i3] <= arch_fit[i4] else i4]
            c1, c2 = _sbx_crossover(parent1, parent2, rng)
            c1 = _polynomial_mutation(c1, rng)
            new_pop.append(c1)
        pop_P = np.array(new_pop[:pop_size])
        pop_raw, pop_feas, pop_viol = eval_pop(pop_P)

    # Return feasible Pareto front from archive
    feas_idx = np.where(arch_feas)[0]
    if feas_idx.size == 0:
        # documented fallback: least violation among all evaluated candidates
        return best_inf.result()
    V_feas = arch_raw[feas_idx]
    par_idx = pareto_front_indices(V_feas, workload.metrics, list(metric_keys))
    return arch_P[feas_idx[par_idx]], V_feas[par_idx]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MOEA/D  (Tchebycheff decomposition)
# ══════════════════════════════════════════════════════════════════════════════

def _uniform_weight_vectors(d: int, H: int, max_vecs: int = 120) -> np.ndarray:
    """Generate weight vectors on the d-dim simplex; cap at max_vecs via random sampling.
    NOTE: H is accepted for interface compatibility but currently ignored — a proper
    simplex lattice design is deferred (materializing C(H+d-1, d-1) tuples OOM'd for
    the full-metric dimensionality this code passes in)."""
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(d), size=max_vecs)
    return W.astype(float)


def moead(
    *,
    workload: Workload,
    metric_keys: Sequence[str],
    H: int = 12,           # weight vector divisions (pop_size ≈ C(H+d-1, d-1))
    T: int = 5,            # neighbourhood size
    generations: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MOEA/D with Tchebycheff scalarization.
    Returns (P_pareto, V_pareto).
    """
    rng = np.random.default_rng(seed)
    J = len(workload.tiers)
    d = len(metric_keys)
    sign = _sign_vector(workload, metric_keys)

    W = _uniform_weight_vectors(d, H)
    N = W.shape[0]

    # Normalise weights
    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    # Replace any zero weight with small epsilon
    W = np.where(W < 1e-6, 1e-6, W)
    W = W / W.sum(axis=1, keepdims=True)

    # Neighbourhood
    dists = np.array([[np.linalg.norm(W[i] - W[j]) for j in range(N)] for i in range(N)])
    B = np.argsort(dists, axis=1)[:, :T]

    # Init population
    pop_P = rng.dirichlet(np.ones(J), size=N)
    pop_raw = np.zeros((N, d))
    pop_feas = np.zeros(N, dtype=bool)
    pop_viol = np.zeros(N)
    deb = _deb_mode()
    best_inf = _BestInfeasible()
    for i in range(N):
        pop_raw[i], pop_feas[i], pop_viol[i] = _eval(workload, metric_keys, pop_P[i])
        best_inf.update(pop_P[i], pop_raw[i], pop_feas[i], pop_viol[i])

    # Ideal point (minimisation)
    z_ideal = (sign * pop_raw).min(axis=0)

    def tchebycheff(obj_min, w, z):
        return float(np.max(w * np.abs(obj_min - z)))

    def scalar_with_penalty(i):
        obj = sign * pop_raw[i]
        base = tchebycheff(obj, W[i], z_ideal)
        if not pop_feas[i]:
            base += pop_viol[i] * 10.0
        return base

    for gen in range(generations):
        for i in range(N):
            # Pick 2 parents from neighbourhood
            nb = B[i]
            p1, p2 = rng.choice(nb, 2, replace=False)
            c, _ = _sbx_crossover(pop_P[p1], pop_P[p2], rng)
            c = _polynomial_mutation(c, rng)

            c_raw, c_feas, c_viol = _eval(workload, metric_keys, c)
            best_inf.update(c, c_raw, c_feas, c_viol)
            c_obj = sign * c_raw

            # Update ideal
            z_ideal = np.minimum(z_ideal, c_obj)

            # Update neighbours (limit replacements to preserve diversity)
            nr = 2  # max replacements per offspring
            replacements = 0
            for j in nb:
                if replacements >= nr:
                    break
                old_val = tchebycheff(sign * pop_raw[j], W[j], z_ideal)
                new_val = tchebycheff(c_obj, W[j], z_ideal)
                if deb:
                    # Deb rules: feasible beats infeasible; infeasible pairs by
                    # violation; feasible pairs by Tchebycheff value
                    if c_feas != pop_feas[j]:
                        better = bool(c_feas)
                    elif not c_feas:
                        better = c_viol <= pop_viol[j]
                    else:
                        better = new_val <= old_val
                else:
                    if not pop_feas[j]:
                        old_val += pop_viol[j] * 10.0
                    if not c_feas:
                        new_val += c_viol * 10.0
                    better = new_val <= old_val
                if better:
                    pop_P[j] = c
                    pop_raw[j] = c_raw
                    pop_feas[j] = c_feas
                    pop_viol[j] = c_viol
                    replacements += 1

    feas_idx = np.where(pop_feas)[0]
    if feas_idx.size == 0:
        # documented fallback: least violation among all evaluated candidates
        return best_inf.result()
    V_feas = pop_raw[feas_idx]
    par_idx = pareto_front_indices(V_feas, workload.metrics, list(metric_keys))
    return pop_P[feas_idx[par_idx]], V_feas[par_idx]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Epsilon-constraint sweep
# ══════════════════════════════════════════════════════════════════════════════

# Diagnostics of the most recent epsilon_constraint call (per process).
EPSILON_LAST_RUN: Dict[str, object] = {}

def epsilon_constraint(
    *,
    workload: Workload,
    metric_keys: Sequence[str],
    primary_key: Optional[str] = None,
    secondary_keys: Optional[Sequence[str]] = None,
    n_levels: int = 30,
    samples_per_level: int = 500,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Epsilon-constraint: fix one objective as primary, sweep epsilon levels
    on the secondary objectives, return Pareto-filtered results.

    secondary_keys=None (legacy) constrains every non-primary metric at the
    same level index simultaneously. With 40+ metrics, few or no samples meet
    all levels, so the run usually ends in the fallback (see EPSILON_LAST_RUN).
    Passing e.g. secondary_keys=[cost_key] gives the classical bi-objective
    epsilon-constraint sweep.
    """
    EPSILON_LAST_RUN.clear()
    rng = np.random.default_rng(seed)
    J = len(workload.tiers)
    d = len(metric_keys)

    if primary_key is None:
        primary_key = metric_keys[0]
    pri_idx = list(metric_keys).index(primary_key)

    key_dir = {m.key: m.direction for m in workload.metrics}

    # Sample full simplex first to get ranges
    P_all = rng.dirichlet(np.ones(J), size=2000)
    V_all = np.zeros((2000, d))
    for i in range(2000):
        pm = aggregate_metrics_linear(workload, P_all[i])
        V_all[i] = [pm[k] for k in metric_keys]

    # Epsilon levels along each secondary objective
    if secondary_keys is None:
        secondary_idx = [i for i in range(d) if i != pri_idx]
    else:
        secondary_idx = [list(metric_keys).index(k) for k in secondary_keys]
    eps_levels: List[np.ndarray] = []
    for i in secondary_idx:
        lo, hi = V_all[:, i].min(), V_all[:, i].max()
        eps_levels.append(np.linspace(lo, hi, n_levels))

    results_P: List[np.ndarray] = []
    results_V: List[np.ndarray] = []
    best_inf = _BestInfeasible()
    best_feas, best_feas_prim = None, np.inf

    for level_idx in range(n_levels):
        # Set epsilon constraints for secondary objectives
        eps_vals = {secondary_idx[k]: float(eps_levels[k][level_idx]) for k in range(len(secondary_idx))}

        # Sample and filter
        P_sample = rng.dirichlet(np.ones(J), size=samples_per_level)
        best_p, best_v, best_prim = None, None, np.inf

        for i in range(samples_per_level):
            pm = aggregate_metrics_linear(workload, P_sample[i])
            feas, viol, _ = feasibility_and_violation(workload, pm)
            raw = np.array([pm[k] for k in metric_keys])
            best_inf.update(P_sample[i], raw, feas, viol)
            if not feas:
                continue
            prim_any = -raw[pri_idx] if key_dir.get(primary_key, "min") == "max" else raw[pri_idx]
            if prim_any < best_feas_prim:
                best_feas_prim = prim_any
                best_feas = (P_sample[i].copy(), raw.copy())

            # Check epsilon constraints on secondary objectives
            ok = True
            for j, eps_v in eps_vals.items():
                if key_dir.get(metric_keys[j], "min") == "min":
                    if raw[j] > eps_v:
                        ok = False; break
                else:
                    if raw[j] < eps_v:
                        ok = False; break
            if not ok:
                continue

            # Primary objective
            prim_val = raw[pri_idx]
            if key_dir.get(primary_key, "min") == "max":
                prim_val = -prim_val

            if prim_val < best_prim:
                best_prim = prim_val
                best_p = P_sample[i].copy()
                best_v = raw.copy()

        if best_p is not None:
            results_P.append(best_p)
            results_V.append(best_v)

    EPSILON_LAST_RUN.update(levels_satisfied=len(results_P), n_levels=n_levels,
                            fallback=not results_P,
                            any_feasible=best_feas is not None)
    if not results_P:
        # No sample met every epsilon level. Documented fallback: the
        # best-primary FEASIBLE sample evaluated, if any; otherwise the least-
        # violation candidate among all evaluated. (Legacy returned the least-
        # violation point of 200 extra draws, i.e. an infeasible point even
        # when feasible samples had been seen.)
        if best_feas is not None:
            return best_feas[0][None, :], best_feas[1][None, :]
        return best_inf.result()

    P_res = np.array(results_P)
    V_res = np.array(results_V)
    par_idx = pareto_front_indices(V_res, workload.metrics, list(metric_keys))
    return P_res[par_idx], V_res[par_idx]
