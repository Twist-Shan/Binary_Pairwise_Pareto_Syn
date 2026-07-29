# Binary Pairwise Pareto Set Identification: Synthetic Experiments

This repository contains the reproducible synthetic experiments for
fixed-confidence Pareto-set identification from binary pairwise comparisons.
The primary implementation is Vector-Borda Exponential Gap Elimination
(`VB-EGE`). The repository also contains the three fixed-confidence baselines
used in the main report and an isolated two-method study of a heuristic
Pareto extension of BT-GLR Track-and-Stop.

The canonical main report is
[`report/synthetic_fixed_confidence_report.pdf`](report/synthetic_fixed_confidence_report.pdf).
The isolated BT-GLR comparison is documented in
[`icml_pareto_track_stop/report/formal_certificate_v1_report.pdf`](icml_pareto_track_stop/report/formal_certificate_v1_report.pdf).

## Scientific status

- The main synthetic suite compares `VB-EGE-practical` with three
  fixed-confidence or fixed-confidence-style baselines under the same strict
  Pareto convention.
- All reported stopping-time means use independent latent instances as the
  unit of analysis. When one latent instance has repeated observation
  replicates, standard errors are clustered by latent instance.
- `UniformPairwiseBT-MLE-Cert` is an empirical plug-in certificate with a
  stabilized BT MLE. It is not presented as a proved delta-correct procedure.
- `Pareto BT-GLR Track-and-Stop` is a heuristic multi-objective extension of a
  scalar top-k method. Its large-front local-Fisher screening statistic is not
  a proved global Pareto-profile lower bound. The isolated report records this
  limitation explicitly and must not be read as a theorem-level comparison.
- Raw experiment rows and regenerated figures are intentionally not committed.
  Configurations, source code, tests, and the final PDFs are versioned.

## Problem definition

There are `K` arms and `d` objectives. Arm `i` has latent utility vector
`theta[i]`. A comparison between arms `i` and `j` on objective `r` returns

```text
Y ~ Bernoulli(sigmoid(theta[i, r] - theta[j, r])).
```

The repository uses all-coordinate strict dominance:

```text
j dominates i  <=>  theta[j, r] > theta[i, r] for every r.
```

Equality in any coordinate prevents domination. The target is therefore

```text
P(theta) = {i : no j strictly dominates i in every coordinate}.
```

For each objective, the Vector-Borda embedding is

```text
b[i, r] = mean_{j != i} sigmoid(theta[i, r] - theta[j, r]).
```

The embedding preserves coordinate-wise order, so `P(b) = P(theta)` under the
strict convention.

## Main algorithms

| Method | Sampling unit | Estimator and stopping rule | Status |
|---|---|---|---|
| `VB-EGE-practical` | Focal arm, random opponent, objective | Vector-Borda empirical means with phased accept/reject certificates | Main method |
| `UniformFocalBorda-FC` | Uniform focal arm-objective cells | Uniform Borda estimates with the same empirical Pareto-gap certificate | Baseline |
| `UniformPairwiseBT-MLE-Cert` | Uniform pair-objective cells | Coordinate-wise stabilized BT MLE and plug-in Pareto certificate | Heuristic FC baseline |
| `UniformPairwiseBT-BordaPlugIn-FC` | Uniform pair-objective cells | Smoothed pairwise probabilities plugged into Borda means | Baseline |

`UniformPairwiseBT-MLE-FC` is retained as a configuration alias and is
normalized to `UniformPairwiseBT-MLE-Cert` in summaries.

The practical VB-EGE configuration used throughout the headline experiments is

```text
sample_const = 2.0
threshold_const = 4.0
log_const = 4.0
```

These practical constants differ from the larger proof constants. Their
sensitivity and the practical-versus-theory sanity check are separate suites;
the same practical choice is then fixed for the other empirical sections.

## Experiment suites

| Config | Purpose | Principal grid | Repetitions |
|---|---|---|---:|
| [`fixed_confidence_scaling.yaml`](configs/fixed_confidence_scaling.yaml) | Scaling in arms, objectives, and latent separation | `K={16,32,64,128}`, `d={2,4,10}`, `Delta={0.5,0.75,1,1.25,1.5}` at `delta=0.05` | 500/cell |
| [`confidence_scaling_quantile.yaml`](configs/confidence_scaling_quantile.yaml) | Confidence scaling and tail quantiles | Symmetric: `delta=0.2` to `0.001`; Arena-10: `0.2` to `0.005` | 500 or 300/cell |
| [`fixed_confidence_benchmarks.yaml`](configs/fixed_confidence_benchmarks.yaml) | Eight structural benchmarks | Convex-2D/3D, Arena-4 small/medium, Arena-10, Witness-4/10, Two-group-10 | 500/setting |
| [`constants_calibration.yaml`](configs/constants_calibration.yaml) | Practical-constant sensitivity | Symmetric, Arena-10, Witness-10 | 300-500/cell |
| [`pareto_size_ablation.yaml`](configs/pareto_size_ablation.yaml) | Dependence on true Pareto-set size | Arena-4 and Arena-10, `|P|={4,8,16,32}` | 300/cell |
| [`theory_constants_sanity.yaml`](configs/theory_constants_sanity.yaml) | Practical versus theory-style constants | Symmetric and Arena-4 | 100/cell |
| [`smoke.yaml`](configs/smoke.yaml) | Fast implementation check | Small symmetric and arena instances | 20/setting |

Convex-3D and Arena-10 use a fixed latent-instance bank with ten observation
replicates per instance. The confidence sweep pairs all `delta` values within
the same latent instance. Raw rows store instance IDs, hashes, seeds, and
observation-replicate IDs so paired and clustered analyses are reproducible.

## Repository layout

```text
configs/                    Main experiment protocols
vb_ege/                     Core methods, baselines, generators, runners, plots
icml_pareto_track_stop/     Isolated Pareto BT-GLR Track-and-Stop extension
pilot_focal_bt_mle/         Constrained BT-MLE utilities used by the extension
scripts/                    Report builder and maintenance/diagnostic scripts
tests/                      Unit and end-to-end smoke tests
results/                    Local generated raw rows, summaries, and figures
report/                     Canonical main PDF; generated TeX is ignored
```

The historical `pilot_focal_bt_mle` study is not part of the published
comparison. Its box-constrained BT profile optimizer is used by the isolated
BT-GLR extension, so that supporting utility remains versioned while the old
pilot drivers and outputs stay local and ignored.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m pip install -e .
```

The equivalent non-editable dependency installation is:

```bash
python -m pip install -r requirements.txt
```

## Fast validation

Run the full test suite:

```bash
python -m pytest -q
```

Run and summarize the small smoke suite:

```bash
python -m vb_ege.run_sweep --config configs/smoke.yaml --out results/raw/smoke.csv --seed 0
python -m vb_ege.summarize --raw results/raw/smoke.csv --out results/summary/smoke_summary.csv --figdir results/figures/smoke
```

## Reproducing the main experiments

Each suite uses the same two-stage interface: run raw repetitions, then
aggregate tables and figures.

```bash
python -m vb_ege.run_sweep --config configs/fixed_confidence_scaling.yaml --out results/raw/fixed_confidence_scaling.csv --seed 2026
python -m vb_ege.summarize --raw results/raw/fixed_confidence_scaling.csv --out results/summary/fixed_confidence_scaling_summary.csv --figdir results/figures/fixed_confidence_scaling
```

```bash
python -m vb_ege.run_sweep --config configs/fixed_confidence_benchmarks.yaml --out results/raw/fixed_confidence_benchmarks.csv --seed 31415
python -m vb_ege.summarize --raw results/raw/fixed_confidence_benchmarks.csv --out results/summary/fixed_confidence_benchmarks_summary.csv --figdir results/figures/fixed_confidence_benchmarks
```

```bash
python -m vb_ege.run_sweep --config configs/confidence_scaling_quantile.yaml --out results/raw/confidence_scaling_quantile.csv --seed 314159
python -m vb_ege.summarize --raw results/raw/confidence_scaling_quantile.csv --out results/summary/confidence_scaling_quantile_summary.csv --figdir results/figures/confidence_scaling_quantile
```

Use the same commands with the remaining YAML files for constant sensitivity,
Pareto-size ablation, and theory-constant sanity checks. YAML `base_seed`
values are the authoritative protocol seeds; an explicit CLI seed overrides
the config only when the runner documents that behavior.

After all required summaries and figures exist, generate the main TeX source:

```bash
python scripts/build_report.py
```

Then compile `report/synthetic_fixed_confidence_report.tex` with a standard
LaTeX installation. Generated TeX, auxiliary files, raw rows, summaries, and
figures are ignored; the canonical PDF is committed.

## Isolated Pareto BT-GLR experiment

The isolated protocol reuses the main instance generators, seeds, repetition
counts, and strict Pareto convention while comparing only VB-EGE and the
Pareto BT-GLR heuristic.

```bash
python -m icml_pareto_track_stop.run_formal --out icml_pareto_track_stop/results/raw/formal_two_way_certificate.csv --jobs 16 --prepare-jobs 16
python -m icml_pareto_track_stop.summarize_formal --raw icml_pareto_track_stop/results/raw/formal_two_way_certificate.csv --summary-dir icml_pareto_track_stop/results/summary_certificate --figure-dir icml_pareto_track_stop/results/figures_certificate
python -m icml_pareto_track_stop.report.build_certificate_v1_formal_report
```

The runner writes resumable local checkpoints and an instance-bank cache.
Those files are ignored and can be deleted after the final CSV is validated.
See [`icml_pareto_track_stop/README.md`](icml_pareto_track_stop/README.md) for
the certificate construction, computational paths, and limitations.

## Output conventions

Raw rows contain, when applicable:

- true and recommended Pareto sets;
- set error, Hamming distance, false positives, and false negatives;
- stopping time, accepted/rejected/unresolved counts, and stopping status;
- latent and Borda classification gaps and `tau / (d H_B)`;
- pair-cell counts and coverage;
- BT-MLE convergence, ridge fallback, centered-theta RMSE, and sign accuracy;
- latent-instance and observation-replicate identifiers.

Summaries use mean stopping time plus or minus one standard error. Curves use
dark mean lines with light standard-error ribbons; grouped bars use a darker
mean-plus-or-minus-SE overlay. Confidence-scaling summaries additionally keep
upper quantiles as tail diagnostics. Error rates are reported with Wilson
intervals.

## Version-control policy

The repository intentionally excludes large or reproducible artifacts:

- raw CSV/JSONL experiment rows;
- checkpoint and cached job files;
- generated summary tables and figures;
- Python, pytest, LaTeX, editor, and OS caches;
- temporary PDF renderings.

Only source code, protocol configs, tests, documentation, `.gitkeep` directory
markers, and the canonical report PDFs should appear in Git status after a
clean run.
