# Compute-Location Multi-Objective Optimization

Code, inputs, and results for *Multi-Objective Optimization of Computation Offloading in Space
Systems* (Thummala, Winton, Mittal, Falco).

The framework has two stages. Stage 1 pre-screens and ranks candidate flight-computer processors
under technology-readiness and radiation gates. Stage 2 optimizes the routing policy that splits a
workload across compute destinations (flight computer, orbital data center, ground-station edge,
terrestrial data center) for a fixed, mission-specific flight-computer configuration. In the reported
experiments the Stage-1 ranking is a separate pre-screening result and is not propagated into Stage 2.

## Layout

| Path | Contents |
|---|---|
| `src/moo_core.py` | Workload model, linear and nonlinear evaluators, constraint handling, Stage-1 utility |
| `src/optimizers_methods.py`, `src/algorithms_extended.py` | MOPSO, NSGA-II, SPEA2, MOEA/D, epsilon-constraint, Random, q-EHVI |
| `src/data.json` | Per-workload, per-destination engineering metrics and thresholds |
| `src/run_benchmark_v2.py` | Main benchmark (E2), Roman benchmark (E9), MOPSO archive study (E4) |
| `src/exp_*_v2.py` | Remaining experiments (see table below) |
| `src/analyze_v2.py` | Statistics: quality indicators, Wilcoxon/Holm tests, cardinality control, solver regret |
| `src/make_tables_v2.py`, `src/make_quality_summary_v2.py` | Regenerate every table in `tables_v2/` |
| `src/make_figures_v2.py`, `src/make_typology_figure_v2.py` | Regenerate the figures in `figures/` |
| `src/out_*_v2/` | Raw experiment outputs (fronts, per-seed metrics, summaries) |
| `src/out_audit/`, `src/out_duration_sweep/` | Metric-role audit (E8) and mission-duration sweep (E14) |
| `tables_v2/` | Every reported table as `.tex`, `.csv`, and `.json`, including tables not printed in the paper |
| `figures/` | The figures used in the paper |

## Experiments and outputs

| Experiment | Script | Output |
|---|---|---|
| E1 Random-policy feasibility | `exp_feasibility_v2.py` | `src/out_feasibility_v2/` |
| E2 Main benchmark | `run_benchmark_v2.py --suite main` | `src/out_benchmark_v2/` |
| E3 Front cardinality control | `analyze_v2.py` | `src/out_analysis_v2/pointcount_*` |
| E4 MOPSO archive size | `run_benchmark_v2.py --suite archive` | `src/out_archive_v2/` |
| E5 Solver choice (regret) | `analyze_v2.py` | `tables_v2/selection_*` |
| E6 Convergence | `exp_convergence_v2.py` | `src/out_convergence_v2/` |
| E7 q-EHVI baseline | `exp_bo_baseline.py --outdir out_bo_v2` | `src/out_bo_v2/` |
| E8 Metric typology and ablation | `exp_variable_audit.py`, `exp_variable_ablation_v2.py` | `src/out_audit/`, `src/out_var_ablation_v2/` |
| E9 Roman value of the ODC | `run_benchmark_v2.py --suite roman` | `src/out_roman_v2/` |
| E10 ODC specification sweep | `exp_odc_sweep_v2.py` | `src/out_odc_sweep_v2/` |
| E11 Roman arrival, burst, quality, NRE sweeps | `exp_roman_sweeps_v2.py` | `src/out_roman_sweeps_v2/` |
| E12 Evaluator consistency, inner loop | `exp_lin_vs_nl_v2.py`, `exp_inner_loop_v2.py` | `src/out_lin_vs_nl_v2/`, `src/out_inner_loop_v2/` |
| E13 Coefficient sensitivity, IDS arrival sweep | `exp_nl_sensitivity_v2.py`, `exp_arrival_sweep_v2.py` | `src/out_nl_sensitivity_v2/`, `src/out_arrival_sweep_v2/` |
| E14 Mission duration and radiation hardness | `exp_mission_duration_sweep.py` | `src/out_duration_sweep/` |

The hardware-in-the-loop measurements (E15) were taken with a separate NASA cFS application and are
not part of this repository.

Full pairwise statistical tests are in `src/out_analysis_v2/main_pairwise_tests.csv`,
`src/out_analysis_v2/roman_pairwise_tests*.csv`, and `tables_v2/pairwise_*.csv`. Experiment
configurations are in `tables_v2/hyperparameters.*`, constraint-violation scales in
`tables_v2/violation_scales.*`, and per-workload convergence results in `tables_v2/convergence.*`.

## Reproducing

Python 3.10 with the packages in `requirements.txt`. Run scripts from `src/`. The `run_queue_v2*.sh`
scripts list the exact command sequence used for the reported runs. They hard-code the interpreter
path of the original machine, so edit `PY=` before using them.

```bash
cd src
python run_benchmark_v2.py --suite main --outdir out_benchmark_v2
python analyze_v2.py
python make_tables_v2.py && python make_quality_summary_v2.py
python make_figures_v2.py
python -m unittest tests/test_v2.py
```

Each (workload, evaluator, algorithm, seed) run is deterministic given its seed. Seeds are listed in
the paper's seed-protocol section.

## License

MIT (see `LICENSE`).
