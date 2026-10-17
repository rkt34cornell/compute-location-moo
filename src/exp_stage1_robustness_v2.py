"""
Stage 1 missing-attribute robustness under the complete-data rule.

Test A: remove one of the six non-gate attributes from one candidate (30 scenarios).
Test B: remove TRL or radiation dose margin from one candidate (10 scenarios).

For each scenario, checks that the missing attribute is detected, that the
affected candidate receives no utility, that a candidate missing a gate input
does not pass qualification, that every other candidate keeps its complete-data
utility, that the remaining candidates keep their complete-data order, and
that no final selection is issued while a candidate that could still qualify
is incomplete. The 40 scenarios are each evaluated at all three criticality
levels (120 evaluations).

Outputs: out_stage1_robustness_v2/{scenarios.csv, summary.json}
"""
import copy
import csv
import json
import os

import moo_core as mc

LEVELS = ("mission_critical", "mission_enhancing", "experimental")
NON_GATE = [v for v in mc.FC_STAGE1_WEIGHTS if v not in mc.FC_STAGE1_GATE_VARS]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_stage1_robustness_v2")


def ranked_order(res):
    order = list(mc.FC_PROCESSOR_PROFILES)
    rk = [k for k, r in res.items() if r["status"] == "ranked"]
    return sorted(rk, key=lambda k: (-res[k]["utility"], order.index(k)))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    base_profiles = mc.FC_PROCESSOR_PROFILES
    rows = []
    summary = {}

    for level in LEVELS:
        base = mc.stage1_score_candidates(level, base_profiles)
        base_rank = ranked_order(base)
        base_best, base_u, _ = mc.stage1_select_fc_processor(level, base_profiles)
        summary[level] = {
            "baseline_status": {k: r["status"] for k, r in base.items()},
            "baseline_utilities": {k: (round(r["utility"], 4) if r["utility"] is not None else None)
                                   for k, r in base.items()},
            "baseline_ranking": " > ".join(base_rank),
            "baseline_best": base_best,
            "baseline_best_utility": round(base_u, 4),
        }

        for test, attrs in (("A", NON_GATE), ("B", list(mc.FC_STAGE1_GATE_VARS))):
            for proc in base_profiles:
                for attr in attrs:
                    prof = copy.deepcopy(base_profiles)
                    del prof[proc][attr]
                    rank = mc.stage1_rank(level, prof)
                    res = rank["candidates"]
                    try:
                        mc.stage1_select_fc_processor(level, prof)
                        final_issued = True
                    except ValueError:
                        final_issued = False
                    r = res[proc]
                    can_still_qualify = r["status"] in ("incomplete", "gate_unconfirmed")
                    others = [k for k in base_profiles if k != proc]
                    others_unchanged = all(
                        res[k]["status"] == base[k]["status"] and res[k]["utility"] == base[k]["utility"]
                        for k in others)
                    rank_preserved = ranked_order(res) == [k for k in base_rank if k != proc]
                    rows.append({
                        "criticality": level,
                        "test": test,
                        "masked_processor": proc,
                        "masked_attribute": attr,
                        "baseline_status": base[proc]["status"],
                        "status": r["status"],
                        "missing_detected": r["missing"] == [attr],
                        "affected_scored": r["utility"] is not None,
                        "passed_gate_without_input": test == "B" and r["status"] in ("ranked", "incomplete"),
                        "others_unchanged": others_unchanged,
                        "remaining_order_preserved": rank_preserved,
                        "can_still_qualify": can_still_qualify,
                        "final_selection_issued": final_issued,
                        "final_issued_while_pending": final_issued and can_still_qualify,
                        "provisional_best": rank["best"],
                        "provisional_best_utility": round(rank["utility"], 4) if rank["utility"] is not None else "",
                    })

        lv = [x for x in rows if x["criticality"] == level]
        b = [x for x in lv if x["test"] == "B"]
        summary[level].update({
            "scenarios": len(lv),
            "test_A": sum(x["test"] == "A" for x in lv),
            "test_B": len(b),
            "missing_detected": sum(x["missing_detected"] for x in lv),
            "affected_candidate_scored": sum(x["affected_scored"] for x in lv),
            "gate_passed_without_input": sum(x["passed_gate_without_input"] for x in b),
            "other_candidates_changed": sum(not x["others_unchanged"] for x in lv),
            "remaining_order_changed": sum(not x["remaining_order_preserved"] for x in lv),
            "status_counts": {s: sum(x["status"] == s for x in lv)
                              for s in ("incomplete", "gate_unconfirmed", "gate_failed", "ranked")},
            "scenarios_on_gate_failed_candidates": sum(x["baseline_status"] == "gate_failed" for x in lv),
            "final_selection_withheld": sum(not x["final_selection_issued"] for x in lv),
            "final_issued_while_pending": sum(x["final_issued_while_pending"] for x in lv),
            "provisional_best_when_top_masked": sorted({x["provisional_best"] for x in lv
                                                        if x["masked_processor"] == base_best}),
        })
    summary["unique_scenarios"] = len(rows) // len(LEVELS)
    summary["evaluations"] = len(rows)

    with open(os.path.join(OUTDIR, "scenarios.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
