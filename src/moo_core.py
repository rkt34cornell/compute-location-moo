#!/usr/bin/env python3
"""moo_core.py

Core evaluation and optimization utilities for the compute-location MOO framework.

Two-stage structure
-------------------
Stage 1 (FC-internal):
    Select the best intra-spacecraft processor configuration from candidate
    processor elements (OBC, FPGA, MCU, GPU, SEC_PROC). FC-internal variables
    (proc_type, bus_type, mem_type, reconfigurable) determine which FC profile
    enters Stage 2. These are stored with the prefix 'FC_' in data.json
    and are scored only for FC tiers.

Stage 2 (inter-tier MOO):
    Optimize a policy p over the simplex Delta^{J-1} covering all tiers
    (FC_best, ODC, GSE, TDC). All 38 inter-tier variables enter this stage.
    FC-internal variables (FC_*) are included in scoring only for the FC tier;
    their values are 0 for non-FC tiers by convention.

Missing data handling
---------------------
If a tier is missing a variable value, that variable is simply excluded from
the weighted utility sum for that tier (the denominator adjusts automatically).
No penalty is applied. This allows the framework to produce valid decisions
even with incomplete tier profiles, and avoids artificially penalizing
well-characterized tiers relative to poorly-characterized ones.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class Direction(str, Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"


@dataclass(frozen=True)
class Metric:
    key: str
    direction: Direction
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class Threshold:
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def satisfied_by(self, value: float) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def violation(self, value: float) -> float:
        v = 0.0
        if self.min_value is not None and value < self.min_value:
            v += float(self.min_value - value)
        if self.max_value is not None and value > self.max_value:
            v += float(value - self.max_value)
        return v


@dataclass(frozen=True)
class Workload:
    name: str
    tiers: List[str]
    metrics: List[Metric]
    tier_values: Dict[str, Dict[str, float]]
    thresholds: Dict[str, Threshold]
    mission_life_years: Optional[float] = None


# ── Data loading ───────────────────────────────────────────────────────────────

def load_workload(data_path: Path, workload_name: str) -> Workload:
    raw = json.loads(data_path.read_text())
    if "workloads" not in raw or workload_name not in raw["workloads"]:
        raise KeyError(
            f"Unknown workload {workload_name!r}. "
            f"Available: {list(raw.get('workloads', {}).keys())}"
        )
    w = raw["workloads"][workload_name]
    tiers: List[str] = list(w["tiers"])
    metrics_meta = w["metrics"]

    metrics: List[Metric] = [
        Metric(
            key=key,
            direction=Direction(str(meta["direction"])),
            unit=meta.get("unit"),
            description=meta.get("description"),
        )
        for key, meta in metrics_meta.items()
    ]

    # Missing values are absent from the dict — no substitution, no penalty.
    tier_values: Dict[str, Dict[str, float]] = {}
    tv = w["tier_values"]
    for t in tiers:
        tier_values[t] = {}
        for mk in metrics_meta.keys():
            raw_val = tv[t].get(mk)
            if raw_val is not None:
                tier_values[t][mk] = float(raw_val)

    thresholds: Dict[str, Threshold] = {
        k: Threshold(min_value=v.get("min"), max_value=v.get("max"))
        for k, v in w["constraints"]["thresholds"].items()
    }

    return Workload(
        name=workload_name,
        tiers=tiers,
        metrics=metrics,
        tier_values=tier_values,
        thresholds=thresholds,
        mission_life_years=w.get("mission_life_years"),
    )


# ── Stage 1: FC-internal processor selection ──────────────────────────────────

FC_PROCESSOR_PROFILES = {
    "OBC":      {"FC_proc_type":0.50,"FC_bus_type":0.30,"FC_reconfigurable":0.0,"FC_mem_type":0.50,
                 "throughput_gflops":0.3,"TRL":9.0,"SDC rate (per hr)":0.008,"Rad dose margin (krad)":50.0},
    "FPGA":     {"FC_proc_type":0.80,"FC_bus_type":0.80,"FC_reconfigurable":1.0,"FC_mem_type":1.00,
                 "throughput_gflops":10.0,"TRL":7.0,"SDC rate (per hr)":0.015,"Rad dose margin (krad)":30.0},
    "MCU":      {"FC_proc_type":0.40,"FC_bus_type":0.30,"FC_reconfigurable":0.0,"FC_mem_type":0.50,
                 "throughput_gflops":0.05,"TRL":9.0,"SDC rate (per hr)":0.003,"Rad dose margin (krad)":60.0},
    "GPU":      {"FC_proc_type":0.70,"FC_bus_type":1.00,"FC_reconfigurable":0.0,"FC_mem_type":0.30,
                 "throughput_gflops":100.0,"TRL":5.0,"SDC rate (per hr)":0.12,"Rad dose margin (krad)":-10.0},
    "SEC_PROC": {"FC_proc_type":0.60,"FC_bus_type":0.50,"FC_reconfigurable":0.0,"FC_mem_type":0.80,
                 "throughput_gflops":0.08,"TRL":7.0,"SDC rate (per hr)":0.004,"Rad dose margin (krad)":45.0},
}

FC_STAGE1_WEIGHTS  = {"FC_proc_type":0.20,"FC_bus_type":0.10,"FC_reconfigurable":0.10,
                       "FC_mem_type":0.15,"throughput_gflops":0.15,"TRL":0.15,
                       "SDC rate (per hr)":0.10,"Rad dose margin (krad)":0.05}
FC_STAGE1_INVERT   = {"SDC rate (per hr)"}
FC_STAGE1_BOUNDS   = {"FC_proc_type":(0.0,1.0),"FC_bus_type":(0.0,1.0),"FC_reconfigurable":(0.0,1.0),
                       "FC_mem_type":(0.0,1.0),"throughput_gflops":(0.05,100.0),"TRL":(1.0,9.0),
                       "SDC rate (per hr)":(0.001,0.15),"Rad dose margin (krad)":(-30.0,80.0)}


def stage1_select_fc_processor(
    mission_criticality: str = "mission_critical",
) -> Tuple[str, float, Dict[str, float]]:
    """
    Stage 1: score FC processor candidates; return (best_name, score, profile).

    Hard gates: TRL floor (7/5/3 by criticality), radiation margin >= 0.
    Missing variables are excluded from the sum (no penalty).
    """
    trl_floor = {"mission_critical":7, "mission_enhancing":5, "experimental":3}[mission_criticality]
    scores: Dict[str, float] = {}

    for proc_name, profile in FC_PROCESSOR_PROFILES.items():
        if profile["TRL"] < trl_floor or profile["Rad dose margin (krad)"] < 0:
            scores[proc_name] = -1.0
            continue
        total_w = weighted_sum = 0.0
        for var, w in FC_STAGE1_WEIGHTS.items():
            val = profile.get(var)
            if val is None:
                continue   # missing — skip, no penalty
            lo, hi = FC_STAGE1_BOUNDS[var]
            norm = max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9)))
            if var in FC_STAGE1_INVERT:
                norm = 1.0 - norm
            weighted_sum += w * norm
            total_w += w
        scores[proc_name] = weighted_sum / total_w if total_w > 0 else 0.0

    feasible = {k: v for k, v in scores.items() if v >= 0}
    best_name = max(feasible, key=lambda k: feasible[k]) if feasible else "OBC"
    return best_name, scores[best_name], FC_PROCESSOR_PROFILES[best_name]


# ── Stage 2: metric aggregation and feasibility ───────────────────────────────

def aggregate_metrics_linear(workload: Workload, p: np.ndarray) -> Dict[str, float]:
    """
    M_i(p) = sum_j p_j * M_i(j)  over tiers j that have metric i defined.

    Tiers with missing values for metric i are skipped; the weight is
    renormalized over present tiers. No imputation or penalty is applied.
    """
    if p.ndim != 1 or p.shape[0] != len(workload.tiers):
        raise ValueError("p must be 1D with length = number of tiers")
    out: Dict[str, float] = {}
    for m in workload.metrics:
        val = weight_sum = 0.0
        for j, t in enumerate(workload.tiers):
            tv = workload.tier_values[t].get(m.key)
            if tv is not None:
                val += p[j] * tv
                weight_sum += p[j]
        out[m.key] = val / weight_sum if weight_sum > 1e-12 else 0.0
    return out


# ── Constraint-violation mode and constraint-handling mode ────────────────────
#
# VIOLATION_MODE
#   "raw"        (legacy) V = sum_k of raw-unit threshold excess. Mixes ms, hours,
#                USD, krad, TRL levels and 0-1 scores in one sum.
#   "normalized" V = sum_k of excess_k / s_k, which is dimensionless; s_k is
#                defined in violation_scale() (threshold magnitude; mission
#                lifetime dose for the zero-threshold radiation margin).
#                Feasibility (V <= 0) is identical in both modes; only the
#                magnitude used to rank infeasible candidates changes.
#
# CONSTRAINT_HANDLING (read by algorithms_extended.py)
#   "legacy_penalty"  NSGA-II / SPEA2 add 10*V to every minimization-space
#                     objective of infeasible individuals; MOEA/D adds 10*V to
#                     the Tchebycheff value.
#   "deb"             Deb (2000) constrained domination: feasible beats
#                     infeasible; two infeasible solutions are ranked by V; two
#                     feasible ones by Pareto dominance (Tchebycheff for MOEA/D).
#
# Both default to the legacy behaviour. Set them with set_constraint_modes() or
# with the MOO_VIOLATION_MODE / MOO_CONSTRAINT_HANDLING environment variables.
VIOLATION_MODES = ("raw", "normalized")
CONSTRAINT_HANDLING_MODES = ("legacy_penalty", "deb")
VIOLATION_MODE = os.environ.get("MOO_VIOLATION_MODE", "raw")
CONSTRAINT_HANDLING = os.environ.get("MOO_CONSTRAINT_HANDLING", "legacy_penalty")
if VIOLATION_MODE not in VIOLATION_MODES:
    raise ValueError(f"MOO_VIOLATION_MODE must be one of {VIOLATION_MODES}")
if CONSTRAINT_HANDLING not in CONSTRAINT_HANDLING_MODES:
    raise ValueError(f"MOO_CONSTRAINT_HANDLING must be one of {CONSTRAINT_HANDLING_MODES}")


def set_constraint_modes(violation: Optional[str] = None,
                         handling: Optional[str] = None) -> None:
    global VIOLATION_MODE, CONSTRAINT_HANDLING
    if violation is not None:
        if violation not in VIOLATION_MODES:
            raise ValueError(f"violation mode must be one of {VIOLATION_MODES}")
        VIOLATION_MODE = violation
    if handling is not None:
        if handling not in CONSTRAINT_HANDLING_MODES:
            raise ValueError(f"handling mode must be one of {CONSTRAINT_HANDLING_MODES}")
        CONSTRAINT_HANDLING = handling


# Scale for the zero-threshold radiation constraint (Rad dose margin >= 0 krad):
# the mission lifetime total ionizing dose, RAD_DOSE_KRAD_PER_YEAR x mission life.
# 10 krad/yr is the dose-accumulation assumption already used by
# exp_mission_duration_sweep.py and the README (5 yr -> 50 krad, 10 yr -> 100 krad).
RAD_DOSE_KRAD_PER_YEAR = 10.0
RAD_MARGIN_KEY = "Rad dose margin (krad)"


def violation_scale(workload: Workload, key: str, thr: Threshold) -> float:
    """Dimensionless scale s_k for the normalized violation of constraint `key`.

    1. |threshold| when the threshold is nonzero (latency, cost, TRL, and the
       [0,1]-bounded success / quality / autonomy / key-management scores, whose
       shortfall is thereby expressed as a fraction of the required level);
    2. radiation margin >= 0: the mission lifetime dose
       RAD_DOSE_KRAD_PER_YEAR * mission_life_years;
    3. otherwise the range of metric k across the workload's destinations (1.0 if
       that range is zero). No current constraint uses rule 3.
    """
    bounds = [b for b in (thr.min_value, thr.max_value) if b is not None]
    mag = max((abs(float(b)) for b in bounds), default=0.0)
    if mag > 0.0:
        return mag
    if key == RAD_MARGIN_KEY and workload.mission_life_years:
        return RAD_DOSE_KRAD_PER_YEAR * float(workload.mission_life_years)
    vals = [workload.tier_values[t][key] for t in workload.tiers
            if key in workload.tier_values[t]]
    span = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
    return float(span) if span > 0.0 else 1.0


def violation_scales(workload: Workload) -> Dict[str, float]:
    return {k: violation_scale(workload, k, thr) for k, thr in workload.thresholds.items()}


def feasibility_and_violation(
    workload: Workload, pm: Mapping[str, float]
) -> Tuple[bool, float, Dict[str, float]]:
    normalized = VIOLATION_MODE == "normalized"
    total = 0.0
    per: Dict[str, float] = {}
    for k, thr in workload.thresholds.items():
        if k not in pm:
            continue
        v = thr.violation(float(pm[k]))
        if normalized and v > 0.0:
            v = v / violation_scale(workload, k, thr)
        per[k] = v
        total += v
    return (total <= 0.0), total, per


def is_feasible(workload: Workload, pm: Mapping[str, float]) -> bool:
    ok, _, _ = feasibility_and_violation(workload, pm)
    return ok


# ── Pareto utilities ──────────────────────────────────────────────────────────

def objective_matrix(workload: Workload, metric_keys: Sequence[str]) -> np.ndarray:
    """Tier-metric matrix A[j,i]. Missing values are np.nan (not 0.0)."""
    J = len(workload.tiers); d = len(metric_keys)
    A = np.full((J, d), np.nan, dtype=float)
    for j, t in enumerate(workload.tiers):
        for i, k in enumerate(metric_keys):
            v = workload.tier_values[t].get(k)
            if v is not None:
                A[j, i] = float(v)
    return A


def to_minimization(V: np.ndarray, metrics: Sequence[Metric], metric_keys: Sequence[str]) -> np.ndarray:
    key_to_metric = {m.key: m for m in metrics}
    Vm = V.copy()
    for i, k in enumerate(metric_keys):
        if key_to_metric[k].direction == Direction.MAXIMIZE:
            Vm[:, i] = -Vm[:, i]
    return Vm


def epsilon_compress(V: np.ndarray, metrics: Sequence[Metric], metric_keys: Sequence[str], eps: float) -> np.ndarray:
    if eps <= 0:
        return np.arange(V.shape[0], dtype=int)
    Vm = to_minimization(V, metrics, metric_keys)
    lo = Vm.min(axis=0); hi = Vm.max(axis=0)
    span = np.where((hi - lo) > 1e-12, (hi - lo), 1.0)
    Vn = (Vm - lo) / span
    boxes: Dict = {}; best: Dict = {}
    for i in range(Vn.shape[0]):
        key = tuple(np.floor(Vn[i] / eps).astype(int).tolist())
        score = float(Vn[i].sum())
        if key not in boxes or score < best[key]:
            boxes[key] = i; best[key] = score
    return np.array(list(boxes.values()), dtype=int)


def pareto_front_indices(V: np.ndarray, metrics: Sequence[Metric], metric_keys: Sequence[str]) -> np.ndarray:
    costs = to_minimization(V, metrics, metric_keys)
    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0
    while next_point_index < costs.shape[0]:
        nondominated_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_mask[next_point_index] = True
        costs = costs[nondominated_mask]
        is_efficient = is_efficient[nondominated_mask]
        next_point_index = int(np.sum(nondominated_mask[:next_point_index]) + 1)
    return is_efficient


# ── Simplex projection ────────────────────────────────────────────────────────

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1: raise ValueError("v must be 1D")
    n = v.shape[0]
    if n == 1: return np.array([1.0], dtype=float)
    u = np.sort(v)[::-1]; cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    theta = (cssv[rho[-1]] - 1.0) / float(rho[-1] + 1) if rho.size else 0.0
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    return w / s if s > 0 else np.ones(n, dtype=float) / float(n)


# ── Nonlinear evaluator (Phase 2) ─────────────────────────────────────────────

@dataclass
class NonlinearParams:
    """
    Parameters governing the seven nonlinear physical effects in the Stage 2
    evaluator. All have physically motivated defaults; adjust per mission.

    Effect 1 — M/M/1 queuing on latency:
        Each tier j operates as an M/M/1 queue with arrival rate p_j * lambda
        and service rate mu_j = throughput_gflops_j / compute_gflops_per_task.
        Queue wait W_j = rho_j / (mu_j * (1 - rho_j)) is hyperbolic in p_j
        and diverges as utilisation rho_j -> queue_max_utilization.

    Effect 2 — Contact window saturation on latency:
        When routing fraction p_j exceeds tier j's contact duty cycle c_j,
        excess tasks must queue for the next contact window. Latency scales as
        L_j^base * (p_j / c_j)^sat_exp for p_j > c_j. Strong for JWST/RST
        ground station tiers with 8–17% duty cycles.

    Effect 3 — Fixed link setup cost on energy:
        Activating any tier (p_j > setup_threshold) incurs a fixed energy
        overhead for link establishment and encryption handshake, independent
        of p_j. This creates non-convexity: splitting across three tiers at
        33% each costs 3x the fixed overhead of routing 100% to one tier.

    Effect 4 — Security protocol energy overhead:
        Energy is multiplied by a protocol-dependent factor per CCSDS SDLS:
        level 1 (none) = 1.00, 2 (MAC) = 1.02, 3 (AES-GCM) = 1.05,
        4 (TLS 1.3) = 1.08, 5 (PQC hybrid) = 1.15.

    Effect 5 — Thermal margin degradation under load:
        Thermal margin degrades superlinearly with power load:
        T_j^eff(p_j) = T_j^base - thermal_coeff * (p_j * P_j^peak)^thermal_exp.
        Exponent > 1 because radiative cooling capacity is fixed while
        heat generation scales with compute utilisation.

    Effect 6 — Link availability degradation under congestion:
        When p_j > c_j (over-routing beyond contact capacity), link
        availability drops: A_j^eff = A_j^base * (c_j / p_j)^link_exp.

    Effect 7 — SDC compound error rate:
        Silent data corruption compounds when multiple tiers are active
        simultaneously. Active tiers (p_j > sdc_threshold) multiply their
        independent SDC rates by sdc_compound_factor per additional tier,
        reflecting increased error surface from multi-tier coordination.

    Effect 8 — Eclipse-driven service-rate gating on M/M/1 utilization:
        For orbital destinations whose peak power exceeds practical
        battery-buffering capacity, compute is unavailable during the
        eclipse fraction of the orbit. Effective M/M/1 service rate is
        mu_j * eta_sun_j, where eta_sun_j in (0, 1] is the sunlit duty
        cycle of destination j. Tiers without an eclipse_duty_cycle entry
        default to eta_sun = 1 (continuous power: battery-buffered,
        L2-located, or terrestrial). For SUNCATCHER's LEO_TPU_CLUSTER at
        250 kW peak, eta_sun = 0.63 corresponds to a typical sun-synchronous
        LEO with ~35 min eclipse per ~95 min orbit.
    """
    # Effect 1: M/M/1 queuing
    lambda_tasks_per_s: float = 100.0       # task arrival rate (tasks/s)
    compute_gflops_per_task: float = 0.002  # GFLOPS required per task
    queue_max_utilization: float = 0.92     # utilisation cap (rho clipped here)

    # Effect 2: Contact window saturation
    contact_sat_exponent: float = 2.0       # latency blowup exponent
    contact_sat_scale: float = 1.0          # multiplier on saturation penalty

    # Effect 3: Fixed link setup cost
    link_setup_threshold: float = 0.02      # p_j > this activates setup cost
    link_setup_energy_fraction: float = 0.12  # fixed = 12% of marginal energy

    # Effect 4: Security overhead (by protocol level 1–5)
    sec_overhead: Dict[int, float] = field(default_factory=lambda: {
        1: 1.00, 2: 1.02, 3: 1.05, 4: 1.08, 5: 1.15
    })

    # Effect 5: Thermal degradation
    thermal_power_exponent: float = 1.35    # superlinear heat load
    thermal_coupling_coeff: float = 3e-5   # delta-C per W^exponent

    # Effect 6: Link congestion
    link_congestion_exponent: float = 0.5   # availability falloff exponent

    # Effect 7: SDC compounding
    sdc_compound_factor: float = 1.6        # per additional active tier
    sdc_active_threshold: float = 0.02      # p_j > this = "active"

    # Effect 8: Eclipse-driven service-rate gating.
    # Per-destination sunlit duty cycle in (0, 1]. Tiers whose power class
    # makes battery buffering of the eclipse impractical (e.g. LEO_TPU at
    # 250 kW peak) have eta_sun < 1; battery-buffered tiers and L2/terrestrial
    # tiers default to eta_sun = 1. Effective M/M/1 service rate becomes
    # mu_j * eta_sun, so the saturation threshold contracts proportionally.
    eclipse_duty_cycle: Dict[str, float] = field(default_factory=dict)

    # Optional burst multiplier for ROMAN_TRANSIENT-style bursty arrivals.
    # Not consumed by aggregate_metrics_nonlinear directly; experiments set
    # lambda_tasks_per_s := lambda_burst when modeling supernova-rich weeks.
    lambda_burst: float = 0.0


# Per-workload defaults: lambda and compute intensity vary by workload type.
# IDS:        high-frequency anomaly detection, lightweight ML inference
# SUNCATCHER: batch training jobs, dominated by contact saturation
# JWST:       ~1 observation/hour, large calibration pipelines
# RST:        ~6 exposures/hour, faster pipeline than JWST
WORKLOAD_NONLINEAR_PARAMS: Dict[str, NonlinearParams] = {
    "IDS": NonlinearParams(
        lambda_tasks_per_s=500.0,
        compute_gflops_per_task=0.001,
        # FC at 0.4 GFLOPS -> service rate 400/s -> rho at p_FC=0.5 is 0.625
        # -> queue wait ~4ms, materially affects 90ms base latency
    ),
    "SUNCATCHER": NonlinearParams(
        lambda_tasks_per_s=1e-4,
        compute_gflops_per_task=5e6,
        # Contact saturation dominant: LEO_TPU contact=0.7
        # Queuing secondary: GROUND_TPU at 1000 GFLOPS handles load well.
        # Eclipse cycling on LEO_TPU (250 kW peak, battery storage impractical):
        # sun-synchronous LEO sunlit fraction ~0.63 (~35 min eclipse per
        # 95 min orbit). HYBRID_SPLIT has a partial orbital component
        # (eta_sun ~0.80, partly battery-buffered). Ground destinations
        # default to 1.0.
        eclipse_duty_cycle={
            "LEO_TPU_CLUSTER": 0.63,
            "HYBRID_SPLIT":    0.80,
        },
    ),
    "JWST": NonlinearParams(
        lambda_tasks_per_s=1.0 / 3600.0,  # ~1 obs/hr
        compute_gflops_per_task=800.0,     # stage 1+2 calibration
        # GSE contact=0.42 (STScI nominal, ~10 hr/day across 2 contacts).
        # FC at 0.2 GFLOPS -> service rate ~0.00025/s -> saturated immediately.
    ),
    "RST": NonlinearParams(
        lambda_tasks_per_s=6.0 / 3600.0,  # ~6 exposures/hr
        compute_gflops_per_task=300.0,     # faster pipeline than JWST
        # GSE contact=0.45 (NEN+DSN+ESA+JAXA aggregate, ~10-12 hr/day).
        # FC at 2 GFLOPS -> service rate 6.7e-3/s vs arrival 1.67e-3/s -> ok.
    ),
    # ROMAN_TRANSIENT: real-time transient detection layered on continuous
    # survey. ~100k transients over 2-year HLTDS => ~140/day baseline => ~0.01/s.
    # Aggregate compute intensity 100 GFLOPs/task (filter 0.5 + ranking 5 +
    # classification 500 averaged across the pipeline mix). Bursts during
    # SN-rich weeks modeled as lambda_burst=0.1 (10x baseline).
    "ROMAN_TRANSIENT": NonlinearParams(
        lambda_tasks_per_s=0.01,
        compute_gflops_per_task=100.0,
        lambda_burst=0.1,
    ),
    "ROMAN_TRANSIENT_FOLLOWUP": NonlinearParams(
        lambda_tasks_per_s=0.01,
        compute_gflops_per_task=100.0,
        lambda_burst=0.1,
    ),
    # LEO_OOS: real-time on-orbit servicing autonomy. ~10 Hz pose update during
    # prox-ops => lambda ~10/s. Per-task compute intensity ~20 GFLOPs (pose
    # estimation + planning). LEO host with eclipse cycling on ODC if co-orbital.
    "LEO_OOS": NonlinearParams(
        lambda_tasks_per_s=10.0,
        compute_gflops_per_task=20.0,
        eclipse_duty_cycle={"ODC": 0.63},
    ),
    # DEEP_SPACE_CRUISE: telemetry triage. ~1 task per minute => lambda ~0.017/s.
    # Per-task compute ~50 GFLOPs (fault detection + science triage).
    # Interplanetary cruise, no eclipse cycling (always sun-illuminated).
    "DEEP_SPACE_CRUISE": NonlinearParams(
        lambda_tasks_per_s=1.0 / 60.0,
        compute_gflops_per_task=50.0,
    ),
}

# Per-stage GFLOP cost bookkeeping for ROMAN_TRANSIENT (documentation; the
# nonlinear aggregator uses the workload aggregate above).
ROMAN_STAGE_GFLOPS = {
    "filter":          0.5,    # on FC only
    "ranking":         5.0,    # FC/ODC/GSE/TDC at different quality
    "classification":  500.0,  # highest quality on TDC
}


# ── ROMAN ODC variant handling ────────────────────────────────────────────────

ODC_VARIANTS = {
    "A": "TRL=3 baseline (concept; TRL-gated for mission-critical)",
    "B": "TRL=7 mature (override JSON value to admit ODC)",
    "C": "ODC-ablation (drop ODC tier entirely; 3-tier simplex)",
}


def apply_odc_variant(workload: "Workload", variant: str) -> "Workload":
    """
    Return a new Workload with ODC tier modified per `variant`.

    Variants:
      A — no change (whatever data.json says; baseline TRL=3)
      B — override ODC's TRL to 7 (mature companion concept)
      C — drop ODC tier entirely (3-tier optimization)

    Existing workloads without an "ODC" tier are returned unchanged.
    """
    if variant not in ODC_VARIANTS:
        raise ValueError(f"Unknown ODC variant {variant!r}; must be one of {list(ODC_VARIANTS)}")
    if "ODC" not in workload.tiers:
        return workload

    new_tiers = list(workload.tiers)
    new_tier_values = {t: dict(workload.tier_values[t]) for t in workload.tiers}

    if variant == "A":
        pass  # no-op
    elif variant == "B":
        new_tier_values["ODC"]["TRL"] = 7.0
    elif variant == "C":
        new_tiers = [t for t in workload.tiers if t != "ODC"]
        new_tier_values = {t: new_tier_values[t] for t in new_tiers}

    return Workload(
        name=workload.name + f"_variant{variant}",
        tiers=new_tiers,
        metrics=workload.metrics,
        tier_values=new_tier_values,
        thresholds=workload.thresholds,
        mission_life_years=workload.mission_life_years,
    )


def _find_key(metric_keys: List[str], candidates: List[str]) -> Optional[str]:
    """Return first candidate present in metric_keys, else None."""
    s = set(metric_keys)
    for c in candidates:
        if c in s:
            return c
    return None


def aggregate_metrics_nonlinear(
    workload: Workload,
    p: np.ndarray,
    params: Optional[NonlinearParams] = None,
) -> Dict[str, float]:
    """
    Nonlinear metric aggregation for Stage 2.

    Starts from the linear aggregate M_i(p) = sum_j p_j M_i(j), then
    overrides six metric groups with physically motivated nonlinear models.
    Each effect is independent and additive/multiplicative on top of the
    linear base — the linear evaluator is recoverable by setting all
    nonlinear parameters to zero.

    Parameters
    ----------
    workload : Workload
    p        : policy vector in Delta^{J-1}
    params   : NonlinearParams (uses WORKLOAD_NONLINEAR_PARAMS defaults if None)

    Returns
    -------
    Dict mapping every metric key to its nonlinear aggregate value.
    """
    if params is None:
        params = WORKLOAD_NONLINEAR_PARAMS.get(workload.name, NonlinearParams())

    if p.ndim != 1 or p.shape[0] != len(workload.tiers):
        raise ValueError("p must be 1D with length = number of tiers")

    J = len(workload.tiers)
    mk = [m.key for m in workload.metrics]

    # ── Start from linear baseline ─────────────────────────────────────────
    out = aggregate_metrics_linear(workload, p)

    # ── Identify metric keys present in this workload ──────────────────────
    latency_key  = _find_key(mk, ["Latency p99 (ms)", "Latency (hrs)"])
    energy_key   = _find_key(mk, ["Energy per task (J)", "Energy per unit (J)", "Energy per obs (kJ)"])
    thermal_key  = _find_key(mk, ["Thermal margin (C)", "Thermal margin (\u25e6C)"])
    link_key     = _find_key(mk, ["Link availability"])
    contact_key  = _find_key(mk, ["Contact duty cycle"])
    throughput_key = _find_key(mk, ["Throughput (GFLOPS)"])
    peak_power_key = _find_key(mk, ["Peak power (W)"])
    sec_key      = _find_key(mk, ["Security protocol level"])
    sdc_key      = _find_key(mk, ["SDC rate (per hr)"])

    # ── Per-tier computations ──────────────────────────────────────────────
    # Pre-fetch tier values (None where missing)
    def tv(tier, key):
        if key is None: return None
        return workload.tier_values[tier].get(key)

    # ── Effect 1+2: Nonlinear latency ────────────────────────────────────
    if latency_key is not None and contact_key is not None and throughput_key is not None:
        lat_total = 0.0
        weight_sum = 0.0
        for j, t in enumerate(workload.tiers):
            L_base = tv(t, latency_key)
            c_j    = tv(t, contact_key)
            mu_raw = tv(t, throughput_key)   # GFLOPS
            if L_base is None: continue

            L_eff = float(L_base)

            # Effect 2: contact window saturation
            if c_j is not None and c_j > 1e-9 and p[j] > c_j:
                sat_factor = params.contact_sat_scale * (p[j] / c_j) ** params.contact_sat_exponent
                L_eff *= sat_factor

            # Effect 1+8: M/M/1 queuing with eclipse-driven service-rate gating.
            # Effective service rate mu_eff = mu_j * eta_sun_j, contracting the
            # saturation threshold for orbital destinations whose peak power
            # exceeds practical battery-buffering capacity.
            if mu_raw is not None and mu_raw > 1e-9 and params.compute_gflops_per_task > 0:
                eta_sun = params.eclipse_duty_cycle.get(t, 1.0)
                mu_j = (mu_raw / params.compute_gflops_per_task) * eta_sun
                lam_j = p[j] * params.lambda_tasks_per_s
                rho_j = lam_j / max(mu_j, 1e-12)
                rho_j = min(rho_j, params.queue_max_utilization)
                if rho_j > 0:
                    # Queue wait in same units as L_base
                    # W_j (seconds) = rho_j / (mu_j * (1 - rho_j))
                    W_j_s = rho_j / (mu_j * max(1.0 - rho_j, 1e-6))
                    # Convert to latency units: ms -> multiply by 1000; hrs -> /3600
                    if "ms" in latency_key:
                        L_eff += W_j_s * 1000.0
                    else:
                        L_eff += W_j_s / 3600.0

            lat_total  += p[j] * L_eff
            weight_sum += p[j]

        if weight_sum > 1e-12:
            out[latency_key] = lat_total / weight_sum

    # ── Effects 3+4: Nonlinear energy ────────────────────────────────────
    # NOTE (deferred Bug 6): this aggregation is a weighted SUM over per-tier
    # contributions, not a weighted mean. For the current dataset every tier
    # has an energy value, so sum(p_j) = 1 and the sum equals the mean — no
    # observable bug. If a partial-energy workload is ever added, the
    # marginal term will be biased low. A correct fix must keep Effect 3's
    # per-active-tier fixed cost separate from the renormalized marginal
    # (naive "divide by weight_sum" breaks the non-convex setup-cost model).
    if energy_key is not None:
        energy_total = 0.0
        for j, t in enumerate(workload.tiers):
            E_base = tv(t, energy_key)
            if E_base is None: continue

            # Effect 4: security protocol energy overhead
            sec_lvl = tv(t, sec_key)
            sec_factor = 1.0
            if sec_lvl is not None:
                sec_factor = params.sec_overhead.get(int(round(sec_lvl)), 1.0)

            # Marginal energy (scales with p_j)
            energy_total += p[j] * float(E_base) * sec_factor

            # Effect 3: fixed link setup overhead (non-convex: per active tier)
            if p[j] > params.link_setup_threshold:
                energy_total += params.link_setup_energy_fraction * float(E_base)

        out[energy_key] = energy_total

    # ── Effect 5: Thermal margin degradation ─────────────────────────────
    if thermal_key is not None and peak_power_key is not None:
        thermal_total = 0.0
        weight_sum = 0.0
        for j, t in enumerate(workload.tiers):
            T_base = tv(t, thermal_key)
            P_peak = tv(t, peak_power_key)
            if T_base is None: continue

            T_eff = float(T_base)
            if P_peak is not None and P_peak > 0:
                load = p[j] * float(P_peak)
                T_eff -= params.thermal_coupling_coeff * (load ** params.thermal_power_exponent)

            thermal_total  += p[j] * T_eff
            weight_sum += p[j]

        if weight_sum > 1e-12:
            out[thermal_key] = thermal_total / weight_sum

    # ── Effect 6: Link availability under congestion ──────────────────────
    if link_key is not None and contact_key is not None:
        link_total = 0.0
        weight_sum = 0.0
        for j, t in enumerate(workload.tiers):
            A_base = tv(t, link_key)
            c_j    = tv(t, contact_key)
            if A_base is None: continue

            A_eff = float(A_base)
            if c_j is not None and c_j > 1e-9 and p[j] > c_j:
                A_eff *= (c_j / p[j]) ** params.link_congestion_exponent

            link_total  += p[j] * A_eff
            weight_sum += p[j]

        if weight_sum > 1e-12:
            out[link_key] = link_total / weight_sum

    # ── Effect 7: SDC compound error rate ────────────────────────────────
    if sdc_key is not None:
        active_tiers = sum(1 for j in range(J) if p[j] > params.sdc_active_threshold)
        if active_tiers > 1:
            compound = params.sdc_compound_factor ** (active_tiers - 1)
            out[sdc_key] = out[sdc_key] * compound

    return out