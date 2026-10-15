#!/usr/bin/env python3
"""optimizers_methods.py

Research-grade implementations of:
  - Weighted-sum PSO on the simplex
  - Projected (sub)gradient descent on the simplex
  - Multi-objective PSO (MOPSO) with external nondominated archive

These optimizers are suitable for research comparison and benchmarking.
All use Deb's constraint handling and operate on the simplex Δ^{J-1} via
project_to_simplex(). The pipeline: workload → optimizer → candidate policies p 
→ metrics M(p) → Pareto filter.

Note on problem structure:
With the current evaluator M_i(p) = sum_j p_j M_i(j), both objectives and 
constraints are piecewise linear in p, so many scalarizations will converge 
to simplex vertices. This is expected behavior; richer simulators with 
nonlinear evaluators will produce interior solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from moo_core import (
    Direction,
    Metric,
    Workload,
    aggregate_metrics_linear,
    feasibility_and_violation,
    objective_matrix,
    pareto_front_indices,
    project_to_simplex,
    to_minimization,
)


# ----------------------------
# Scalarization utilities
# ----------------------------


@dataclass(frozen=True)
class Scalarizer:
    metric_keys: List[str]
    weights: np.ndarray  # shape (d,)
    penalty: float = 1e3
    normalize: bool = True


def _metric_key_to_dir(workload: Workload) -> Dict[str, Direction]:
    return {m.key: m.direction for m in workload.metrics}


def build_scalar_objective(
    workload: Workload,
    scalarizer: Scalarizer,
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], Tuple[Dict[str, float], bool, float]]]:
    """Return (objective(p), eval(p)->(metrics, feasible, violation)). Lower is better."""
    metric_keys = scalarizer.metric_keys
    weights = np.asarray(scalarizer.weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != len(metric_keys):
        raise ValueError("weights must be 1D with len == len(metric_keys)")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    if float(weights.sum()) <= 0:
        raise ValueError("weights must sum to > 0")
    weights = weights / float(weights.sum())

    # Precompute tier matrix for objective metrics.
    A = objective_matrix(workload, metric_keys)  # (J, d)
    key_dir = _metric_key_to_dir(workload)

    # Convert to minimization space for normalization.
    # Replace NaN with 0.0 for the dot-product (NaN means tier lacks this metric;
    # aggregate_metrics_linear handles renormalization, but the matrix path needs
    # finite values for p @ A_min).
    A_min = np.nan_to_num(A, nan=0.0)
    for i, k in enumerate(metric_keys):
        if key_dir[k] == Direction.MAXIMIZE:
            A_min[:, i] = -A_min[:, i]

    lo = A_min.min(axis=0)
    hi = A_min.max(axis=0)
    span = np.where((hi - lo) > 1e-12, (hi - lo), 1.0)

    def _eval(p: np.ndarray) -> Tuple[Dict[str, float], bool, float]:
        pm = aggregate_metrics_linear(workload, p)
        feasible, viol, _ = feasibility_and_violation(workload, pm)
        return pm, feasible, float(viol)

    def objective(p: np.ndarray) -> float:
        # objective metrics in minimization form
        x = p @ A_min  # (d,)
        if scalarizer.normalize:
            x = (x - lo) / span
        base = float(np.dot(weights, x))

        # constraints penalty (normalized by per-constraint span if possible)
        pm = aggregate_metrics_linear(workload, p)
        feasible, viol, per = feasibility_and_violation(workload, pm)
        if feasible:
            return base
        # Scale each constraint by its tier span if metric exists
        pen = 0.0
        for k, v in per.items():
            if v <= 0:
                continue
            # If constraint key is in objective metrics, we can reuse its span.
            if k in metric_keys:
                idx = metric_keys.index(k)
                denom = float(span[idx])
            else:
                # fallback: use raw tier range, tolerating missing keys
                vals = np.array([workload.tier_values[t].get(k, np.nan) for t in workload.tiers], dtype=float)
                valid = vals[~np.isnan(vals)]
                denom = float(max(valid.max() - valid.min(), 1.0)) if valid.size >= 2 else 1.0
            pen += float(v) / denom
        return base + scalarizer.penalty * pen

    return objective, _eval


# ----------------------------
# Projected (sub)gradient descent
# ----------------------------


def projected_gradient_descent(
    *,
    objective: Callable[[np.ndarray], float],
    p0: np.ndarray,
    steps: int = 300,
    lr: float = 0.2,
    fd_eps: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """Simple projected GD using finite-difference gradients + simplex projection."""
    p = project_to_simplex(np.asarray(p0, dtype=float))
    best_p = p.copy()
    best_f = float(objective(p))

    J = p.shape[0]
    for _ in range(steps):
        f0 = float(objective(p))
        g = np.zeros(J, dtype=float)
        for j in range(J):
            e = np.zeros(J, dtype=float)
            e[j] = 1.0
            p2 = project_to_simplex(p + fd_eps * e)
            g[j] = (float(objective(p2)) - f0) / fd_eps

        # Basic step
        p = project_to_simplex(p - lr * g)
        f = float(objective(p))
        if f < best_f:
            best_f = f
            best_p = p.copy()

        # Optional tiny noise helps escape kinks in piecewise-linear penalty
        if rng is not None and rng.random() < 0.05:
            p = project_to_simplex(p + 0.02 * rng.normal(size=J))

    return best_p, best_f


# ----------------------------
# Weighted-sum PSO on simplex
# ----------------------------


def pso_simplex(
    *,
    objective: Callable[[np.ndarray], float],
    J: int,
    particles: int = 60,
    iters: int = 250,
    w: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(J), size=particles)  # positions
    V = 0.05 * rng.normal(size=(particles, J))

    pbest = X.copy()
    pbest_f = np.array([objective(X[i]) for i in range(particles)], dtype=float)
    gbest_idx = int(np.argmin(pbest_f))
    gbest = pbest[gbest_idx].copy()
    gbest_f = float(pbest_f[gbest_idx])

    for _ in range(iters):
        r1 = rng.random((particles, J))
        r2 = rng.random((particles, J))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = X + V
        # project each particle to simplex
        for i in range(particles):
            X[i] = project_to_simplex(X[i])

        f = np.array([objective(X[i]) for i in range(particles)], dtype=float)
        improve = f < pbest_f
        pbest[improve] = X[improve]
        pbest_f[improve] = f[improve]

        idx = int(np.argmin(pbest_f))
        if float(pbest_f[idx]) < gbest_f:
            gbest_f = float(pbest_f[idx])
            gbest = pbest[idx].copy()

    return gbest, gbest_f


# ----------------------------
# MOPSO (multi-objective PSO)
# ----------------------------


def _crowding_distance(V: np.ndarray) -> np.ndarray:
    """Crowding distance in minimization objective space."""
    n, m = V.shape
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([np.inf])
    dist = np.zeros(n, dtype=float)
    for j in range(m):
        idx = np.argsort(V[:, j])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        vj = V[idx, j]
        denom = float(vj[-1] - vj[0]) if float(vj[-1] - vj[0]) > 1e-12 else 1.0
        # same per-element arithmetic as the original k-loop
        dist[idx[1:-1]] += (vj[2:] - vj[:-2]) / denom
    return dist


def mopso(
    *,
    workload: Workload,
    metric_keys: Sequence[str],
    particles: int = 80,
    iters: int = 250,
    archive_max: int = 200,
    w: float = 0.6,
    c1: float = 1.6,
    c2: float = 1.6,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (archive_P, archive_V) where V are raw metric values (not minimized)."""
    rng = np.random.default_rng(seed)
    J = len(workload.tiers)
    d = len(metric_keys)

    # Tier matrix in metric_keys order
    metrics_all = workload.metrics

    # Direction signs for minimization conversion
    key_dir = {m.key: m.direction for m in metrics_all}
    sign = np.array([1.0 if key_dir[k] == Direction.MINIMIZE else -1.0 for k in metric_keys], dtype=float)

    # Particle init
    X = rng.dirichlet(np.ones(J), size=particles)
    Vv = 0.05 * rng.normal(size=(particles, J))

    def eval_particle(p: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        pm = aggregate_metrics_linear(workload, p)
        raw = np.array([pm[k] for k in metric_keys], dtype=float)
        feasible, viol, _ = feasibility_and_violation(workload, pm)
        return raw, feasible, float(viol)

    # Personal best based on Deb's constraint handling (feasible dominates infeasible; else lower violation)
    pbest = X.copy()
    pbest_raw = np.zeros((particles, d), dtype=float)
    pbest_feas = np.zeros(particles, dtype=bool)
    pbest_viol = np.zeros(particles, dtype=float)
    for i in range(particles):
        r, feas, viol = eval_particle(X[i])
        pbest_raw[i] = r
        pbest_feas[i] = feas
        pbest_viol[i] = viol

    # External archive (store only feasible nondominated when feasible exists)
    arch_P: List[np.ndarray] = []
    arch_raw: List[np.ndarray] = []
    arch_min: List[np.ndarray] = []  # minimization-space objectives (sign * raw)
    best_infeas: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # (p, raw, viol)

    def _dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))

    def archive_update(p: np.ndarray, raw: np.ndarray, feas: bool, viol: float) -> None:
        nonlocal best_infeas
        if not feas:
            if not arch_P:
                if best_infeas is None or viol < best_infeas[2]:
                    best_infeas = (p.copy(), raw.copy(), float(viol))
            return

        # Feasible point: maintain feasible nondominated archive
        rmin = sign * raw

        # If archive empty but we have a stored infeasible best, drop it now.
        best_infeas = None

        if arch_min:
            A = np.stack(arch_min, axis=0)  # (n, d)
            dominated_by_arch = np.any(np.all(A <= rmin, axis=1) & np.any(A < rmin, axis=1))
            if dominated_by_arch:
                return
            dominated_in_arch = np.all(rmin <= A, axis=1) & np.any(rmin < A, axis=1)
            if np.any(dominated_in_arch):
                keep = np.where(~dominated_in_arch)[0].tolist()
                arch_P[:] = [arch_P[i] for i in keep]
                arch_raw[:] = [arch_raw[i] for i in keep]
                arch_min[:] = [arch_min[i] for i in keep]

        # Add new point
        arch_P.append(p.copy())
        arch_raw.append(raw.copy())
        arch_min.append(rmin.copy())

        # Prune if archive too large (crowding in minimization space)
        if len(arch_P) > archive_max:
            Vmin = np.stack(arch_min, axis=0)
            cd = _crowding_distance(Vmin)
            sel = np.argsort(-cd)[:archive_max]
            arch_P[:] = [arch_P[i] for i in sel]
            arch_raw[:] = [arch_raw[i] for i in sel]
            arch_min[:] = [arch_min[i] for i in sel]

    # seed archive
    for i in range(particles):
        raw, feas, viol = eval_particle(X[i])
        archive_update(X[i], raw, feas, viol)

    def choose_leader() -> np.ndarray:
        if arch_P:
            P_arr = np.array(arch_P)
            Vmin = np.stack(arch_min, axis=0)
        else:
            # No feasible points yet: leader is the current best infeasible (lowest violation)
            assert best_infeas is not None
            return best_infeas[0]

        if P_arr.shape[0] == 1:
            return P_arr[0]

        cd = _crowding_distance(Vmin)
        # Prefer sparse regions (large crowding).
        # Handle inf/NaN robustly: if crowding is all inf (common for small archives), sample uniformly.
        if not np.any(np.isfinite(cd)):
            idx = int(rng.integers(0, P_arr.shape[0]))
            return P_arr[idx]

        finite_max = float(np.max(cd[np.isfinite(cd)]))
        cd2 = np.where(np.isfinite(cd), cd, finite_max)
        # Softmax over cd2
        z = cd2 - float(np.max(cd2))
        wts = np.exp(z)
        s = float(np.sum(wts))
        if not np.isfinite(s) or s <= 0:
            idx = int(rng.integers(0, P_arr.shape[0]))
            return P_arr[idx]
        wts = wts / s
        idx = int(rng.choice(np.arange(P_arr.shape[0]), p=wts))
        return P_arr[idx]

    # main loop
    for _ in range(iters):
        for i in range(particles):
            leader = choose_leader()
            r1 = rng.random(J)
            r2 = rng.random(J)
            Vv[i] = w * Vv[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (leader - X[i])
            X[i] = project_to_simplex(X[i] + Vv[i])

            raw, feas, viol = eval_particle(X[i])
            # update pbest with Deb's constraint handling, then Pareto tie-break
            if feas and not pbest_feas[i]:
                take = True
            elif feas and pbest_feas[i]:
                # compare dominance in minimization space
                a = raw[None, :]
                b = pbest_raw[i][None, :]
                a_min = to_minimization(a, metrics_all, list(metric_keys))[0]
                b_min = to_minimization(b, metrics_all, list(metric_keys))[0]
                take = bool(np.all(a_min <= b_min) and np.any(a_min < b_min))
            elif (not feas) and (not pbest_feas[i]):
                take = viol < pbest_viol[i]
            else:
                take = False

            if take:
                pbest[i] = X[i].copy()
                pbest_raw[i] = raw
                pbest_feas[i] = feas
                pbest_viol[i] = viol

            archive_update(X[i], raw, feas, viol)

    if arch_P:
        return np.array(arch_P), np.array(arch_raw)
    assert best_infeas is not None
    return best_infeas[0][None, :], best_infeas[1][None, :]
