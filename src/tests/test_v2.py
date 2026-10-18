#!/usr/bin/env python3
"""Unit tests for the nonlinear evaluator and the v2 constraint modes.

Run from src/:  python -m unittest tests/test_v2.py -v
"""
from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import moo_core as mc  # noqa: E402
import algorithms_extended as ae  # noqa: E402

DATA = SRC / "data.json"
ALL_WL = ["IDS", "SUNCATCHER", "JWST", "RST", "ROMAN_TRANSIENT",
          "ROMAN_TRANSIENT_FOLLOWUP", "LEO_OOS", "DEEP_SPACE_CRUISE"]


def load(name):
    return mc.load_workload(DATA, name)


def with_contact_one(wl):
    tv = {t: dict(wl.tier_values[t]) for t in wl.tiers}
    for t in wl.tiers:
        if "Contact duty cycle" in tv[t]:
            tv[t]["Contact duty cycle"] = 1.0
    return mc.Workload(wl.name, list(wl.tiers), wl.metrics, tv, wl.thresholds, wl.mission_life_years)


def params_for(name, **kw):
    return replace(mc.WORKLOAD_NONLINEAR_PARAMS.get(name, mc.NonlinearParams()), **kw)


def lat_key(wl):
    return mc._find_key([m.key for m in wl.metrics], ["Latency p99 (ms)", "Latency (hrs)"])


class TestNonlinearEvaluator(unittest.TestCase):
    rng = np.random.default_rng(0)

    def policies(self, J, n=200):
        return np.vstack([np.eye(J), self.rng.dirichlet(np.ones(J), n)])

    def test_lambda_to_zero_gives_negligible_queueing(self):
        """lambda -> 0: latency converges to the no-queue value. The wait is
        ~lambda*p/mu^2, so lambda must be small relative to mu^2 (SUNCATCHER has
        mu ~ 6e-5 /s); 1e-30 makes the term < 1e-12 of latency for every workload."""
        for name in ALL_WL:
            wl = load(name); lk = lat_key(wl)
            p_small = params_for(name, lambda_tasks_per_s=1e-30)
            p_noq = params_for(name, compute_gflops_per_task=0.0)  # skips Effect 1 branch
            for p in self.policies(len(wl.tiers)):
                a = mc.aggregate_metrics_nonlinear(wl, p, p_small)[lk]
                b = mc.aggregate_metrics_nonlinear(wl, p, p_noq)[lk]
                self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)), msg=f"{name} p={p}")

    def test_contact_duty_one_removes_saturation(self):
        """c_j = 1 for all tiers: no contact saturation (Effect 2) and no link
        congestion (Effect 6); with lambda -> 0 latency and link availability
        equal the linear aggregate."""
        for name in ALL_WL:
            wl = with_contact_one(load(name)); lk = lat_key(wl)
            pr = params_for(name, lambda_tasks_per_s=1e-30)
            for p in self.policies(len(wl.tiers)):
                nl = mc.aggregate_metrics_nonlinear(wl, p, pr)
                li = mc.aggregate_metrics_linear(wl, p)
                self.assertAlmostEqual(nl[lk], li[lk], delta=1e-9 * max(1.0, abs(li[lk])), msg=name)
                if "Link availability" in nl:
                    self.assertAlmostEqual(nl["Link availability"], li["Link availability"], places=12)

    def test_single_active_tier_has_unit_sdc_multiplier(self):
        """Exactly one tier above the activity threshold: SDC equals the linear
        aggregate (multiplier 1.6^0 = 1); two active tiers: multiplier 1.6."""
        for name in ALL_WL:
            wl = load(name); pr = params_for(name); J = len(wl.tiers)
            k = "SDC rate (per hr)"
            for j in range(J):
                p = np.full(J, 0.01 / max(J - 1, 1)); p[j] = 1.0 - p.sum() + p[j]
                p = p / p.sum()  # one tier active, others at <= 0.02
                self.assertLessEqual(sum(p > pr.sdc_active_threshold), 1)
                self.assertAlmostEqual(mc.aggregate_metrics_nonlinear(wl, p, pr)[k],
                                       mc.aggregate_metrics_linear(wl, p)[k], places=15)
            p2 = np.zeros(J); p2[0] = p2[1] = 0.5
            self.assertAlmostEqual(mc.aggregate_metrics_nonlinear(wl, p2, pr)[k],
                                   1.6 * mc.aggregate_metrics_linear(wl, p2)[k], places=15)

    def test_vertex_routing_reproduces_tier_metrics(self):
        """p = e_j reproduces tier j's metric values: exactly for the linear
        evaluator; for the nonlinear evaluator on every metric that no effect
        overrides, and on latency once queueing and contact saturation are
        switched off."""
        for name in ALL_WL:
            base = load(name); wl = with_contact_one(base); lk = lat_key(wl)
            pr = params_for(name, lambda_tasks_per_s=1e-30)
            overridden = {lk, "Energy per task (J)", "Energy per unit (J)", "Energy per obs (kJ)",
                          "Thermal margin (C)", "Link availability"}
            for j, t in enumerate(wl.tiers):
                p = np.eye(len(wl.tiers))[j]
                li = mc.aggregate_metrics_linear(base, p)
                nl = mc.aggregate_metrics_nonlinear(wl, p, pr)
                for k, v in base.tier_values[t].items():
                    self.assertEqual(li[k], v, msg=f"{name}/{t}/{k} linear")
                    if k not in overridden:
                        self.assertAlmostEqual(nl[k], wl.tier_values[t][k], places=12,
                                               msg=f"{name}/{t}/{k} nonlinear")
                self.assertAlmostEqual(nl[lk], base.tier_values[t][lk],
                                       delta=1e-9 * max(1.0, abs(base.tier_values[t][lk])))

    def test_queue_wait_matches_mm1_formula(self):
        """Effect 1 adds the exact rho/(mu(1-rho)) at a stable vertex (no utilization cap)."""
        wl = with_contact_one(load("IDS")); pr = replace(params_for("IDS"), lambda_tasks_per_s=380.0)
        p = np.array([1.0, 0, 0, 0])
        mu = 0.4 / pr.compute_gflops_per_task
        rho = pr.lambda_tasks_per_s / mu          # 0.95, above the former 0.92 cap
        expect = wl.tier_values["FC"]["Latency p99 (ms)"] + 1000.0 * rho / (mu * (1 - rho))
        self.assertAlmostEqual(mc.aggregate_metrics_nonlinear(wl, p, pr)["Latency p99 (ms)"], expect, places=9)

    def test_overloaded_destination_is_infeasible(self):
        """rho_j >= 1 marks the policy overloaded, also when re-checked from metric rows."""
        mc.set_constraint_modes("normalized", "deb")
        try:
            wl = load("IDS"); pr = params_for("IDS")       # FC: mu = 400/s, lambda = 500/s
            mk = [m.key for m in wl.metrics]
            pm = mc.aggregate_metrics_nonlinear(wl, np.array([1.0, 0, 0, 0]), pr)
            f, v, _ = mc.feasibility_and_violation(wl, pm)
            self.assertFalse(f)
            self.assertAlmostEqual(v, mc.OVERLOAD_PENALTY + 0.25, places=6)
            self.assertFalse(mc.feasibility_and_violation(wl, {k: pm[k] for k in mk})[0])
            stable = mc.aggregate_metrics_nonlinear(wl, np.array([0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]), pr)
            self.assertNotIn(mc.QUEUE_OVERLOAD_KEY, stable)
            self.assertLess(mc.feasibility_and_violation(wl, stable)[1], v)
        finally:
            mc.set_constraint_modes("raw", "legacy_penalty")


class TestConstraintModes(unittest.TestCase):
    def tearDown(self):
        mc.set_constraint_modes("raw", "legacy_penalty")

    def test_feasibility_identical_across_violation_modes(self):
        rng = np.random.default_rng(1)
        for name in ALL_WL:
            wl = load(name)
            for p in rng.dirichlet(np.ones(len(wl.tiers)), 300):
                pm = mc.aggregate_metrics_nonlinear(wl, p)
                mc.set_constraint_modes("raw"); f_raw, v_raw, per_raw = mc.feasibility_and_violation(wl, pm)
                mc.set_constraint_modes("normalized"); f_n, v_n, per_n = mc.feasibility_and_violation(wl, pm)
                self.assertEqual(f_raw, f_n)
                scales = mc.violation_scales(wl)
                expect = sum(per_raw[k] / scales[k] for k in per_raw if k != mc.QUEUE_OVERLOAD_KEY)
                expect += per_raw.get(mc.QUEUE_OVERLOAD_KEY, 0.0)   # overload term is already dimensionless
                self.assertAlmostEqual(v_n, expect, places=9)

    def test_violation_scale_rules(self):
        wl = load("IDS")
        s = mc.violation_scales(wl)
        self.assertEqual(s["Latency p99 (ms)"], 250.0)
        self.assertEqual(s["TRL"], 6.0)
        # zero threshold -> mission lifetime dose (IDS: 10 krad/yr x 5 yr)
        self.assertEqual(s["Rad dose margin (krad)"], 50.0)
        self.assertEqual(mc.violation_scales(load("SUNCATCHER"))["Rad dose margin (krad)"], 100.0)
        # derived workloads keep the mission life
        self.assertEqual(mc.violation_scales(mc.apply_odc_variant(load("ROMAN_TRANSIENT"), "C"))["Rad dose margin (krad)"], 50.0)

    def test_normalized_violation_upper_and_lower(self):
        mc.set_constraint_modes("normalized")
        wl = load("IDS")
        base = {k: (t.min_value if t.min_value is not None else t.max_value) for k, t in wl.thresholds.items()}
        ok, v, _ = mc.feasibility_and_violation(wl, base)              # exactly on every bound
        self.assertTrue(ok); self.assertEqual(v, 0.0)
        up = dict(base, **{"Latency p99 (ms)": 300.0})                  # upper bound 250 exceeded by 50
        ok, v, per = mc.feasibility_and_violation(wl, up)
        self.assertFalse(ok); self.assertAlmostEqual(per["Latency p99 (ms)"], 50 / 250)
        lo = dict(base, **{"Success probability": 0.82})               # lower bound 0.92 missed by 0.10
        ok, v, per = mc.feasibility_and_violation(wl, lo)
        self.assertAlmostEqual(per["Success probability"], 0.10 / 0.92); self.assertAlmostEqual(v, 0.10 / 0.92)
        rad = dict(base, **{"Rad dose margin (krad)": -15.0})           # zero threshold, scale 50 krad
        self.assertAlmostEqual(mc.feasibility_and_violation(wl, rad)[1], 15 / 50)
        both = dict(up, **{"Success probability": 0.82})                # contributions add
        self.assertAlmostEqual(mc.feasibility_and_violation(wl, both)[1], 50 / 250 + 0.10 / 0.92)
        over = dict(base, **{"Latency p99 (ms)": 100.0, "Success probability": 0.99})  # slack never counts
        self.assertEqual(mc.feasibility_and_violation(wl, over)[1], 0.0)

    def test_deb_moead_replacement_rule(self):
        """MOEA/D Deb rule: feasible child replaces infeasible parent; infeasible pair by V."""
        mc.set_constraint_modes("normalized", "deb")
        wl = load("LEO_OOS"); mk = [m.key for m in wl.metrics]
        P, V = ae.moead(workload=wl, metric_keys=mk, H=8, T=5, generations=3, seed=1)
        self.assertTrue(all(mc.feasibility_and_violation(wl, dict(zip(mk, r)))[0] for r in V) or len(V) == 1)

    def test_deb_dominance_matrix(self):
        costs = np.array([[1.0, 1.0], [0.0, 0.0], [5.0, 5.0], [6.0, 6.0]])
        feas = np.array([True, False, False, True])
        viol = np.array([0.0, 0.3, 0.1, 0.0])
        D = ae._deb_dom_matrix(costs, feas, viol)
        self.assertTrue(D[0, 1] and D[0, 2])      # feasible beats infeasible
        self.assertTrue(D[3, 1] and D[3, 2])
        self.assertTrue(D[2, 1] and not D[1, 2])  # lower violation wins
        self.assertTrue(D[0, 3] and not D[3, 0])  # Pareto among feasible
        self.assertFalse(D.diagonal().any())

    def test_vectorized_sort_matches_reference(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            C = np.round(rng.normal(size=(int(rng.integers(3, 30)), 3)), 1)
            ref = []
            N = C.shape[0]; S = [[] for _ in range(N)]; n = [0] * N; f0 = []
            for p in range(N):
                for q in range(N):
                    if p == q: continue
                    if np.all(C[p] <= C[q]) and np.any(C[p] < C[q]): S[p].append(q)
                    elif np.all(C[q] <= C[p]) and np.any(C[q] < C[p]): n[p] += 1
                if n[p] == 0: f0.append(p)
            fr = [f0]; i = 0
            while fr[i]:
                nxt = []
                for p in fr[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0: nxt.append(q)
                i += 1; fr.append(nxt)
            ref = [f for f in fr if f]
            self.assertEqual(ae._fast_nondominated_sort_min(C), ref)


class TestStage1CompleteData(unittest.TestCase):

    def test_complete_profiles_reproduce_reported_ranking(self):
        best, u, _ = mc.stage1_select_fc_processor("mission_critical")
        self.assertEqual(best, "FPGA")
        self.assertAlmostEqual(u, 0.7353, places=4)

    def test_missing_non_gate_attribute_is_not_scored(self):
        prof = copy.deepcopy(mc.FC_PROCESSOR_PROFILES)
        del prof["FPGA"]["throughput_gflops"]
        res = mc.stage1_score_candidates("mission_critical", prof)
        self.assertEqual(res["FPGA"]["status"], "incomplete")
        self.assertIsNone(res["FPGA"]["utility"])
        self.assertEqual(res["FPGA"]["missing"], ["throughput_gflops"])

    def test_incomplete_leader_defers_final_selection(self):
        prof = copy.deepcopy(mc.FC_PROCESSOR_PROFILES)
        del prof["FPGA"]["throughput_gflops"]
        r = mc.stage1_rank("mission_critical", prof)
        self.assertEqual(r["best"], "SEC_PROC")
        self.assertFalse(r["final"])
        self.assertEqual(r["pending"], ["FPGA"])
        with self.assertRaises(ValueError):
            mc.stage1_select_fc_processor("mission_critical", prof)

    def test_gate_failed_candidate_does_not_block_selection(self):
        prof = copy.deepcopy(mc.FC_PROCESSOR_PROFILES)
        del prof["GPU"]["throughput_gflops"]
        best, _, _ = mc.stage1_select_fc_processor("mission_critical", prof)
        self.assertEqual(best, "FPGA")

    def test_missing_gate_attribute_does_not_pass_gate(self):
        for attr in mc.FC_STAGE1_GATE_VARS:
            prof = copy.deepcopy(mc.FC_PROCESSOR_PROFILES)
            del prof["OBC"][attr]
            res = mc.stage1_score_candidates("experimental", prof)
            self.assertEqual(res["OBC"]["status"], "gate_unconfirmed")
            self.assertIsNone(res["OBC"]["utility"])


if __name__ == "__main__":
    unittest.main()
