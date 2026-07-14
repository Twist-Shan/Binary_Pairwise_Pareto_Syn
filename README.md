# Synthetic Experiments for Vector-Borda Pareto Set Identification

This repo implements fixed-confidence synthetic experiments for strict Pareto set identification under a coordinate-wise Bradley-Terry comparison model. The retained outputs correspond exactly to the experiments used in the synthetic report.

## Problem

There are `K` arms and `d` objectives. Arm `i` has latent vector `theta[i]`.
A coordinate-specific comparison between arms `i` and `j` in coordinate `r` returns

```text
Bernoulli(sigmoid(theta[i, r] - theta[j, r])).
```

The target is the strict Pareto set. Arm `i` is removed only when another arm is strictly larger in every coordinate. Equal coordinates therefore prevent domination.

The Borda embedding is

```text
b[i, r] = mean_{j != i} sigmoid(theta[i, r] - theta[j, r]).
```

It preserves coordinate-wise order, so the strict Pareto set of `b` is the same as the strict Pareto set of `theta`.

## Algorithms and Benchmarks

The main fixed-confidence algorithm is `VB-EGE-practical`, implemented in `vb_ege.algorithms`. It samples focal-arm Borda observations and accepts or rejects arms by empirical active-set Pareto gaps. Practical constants are exposed in YAML configs; larger theory-style constants can be supplied through `VBEGEConfig`.

The main comparison methods are:

- `UniformFocalBorda-FC`: uniformly samples arm-coordinate focal Borda cells in rounds and stops when the empirical Pareto gap clears the confidence radius.
- `UniformPairwiseBT-MLE-Cert`: uniformly samples pair-coordinate Bradley-Terry comparisons in rounds, fits a separate BT MLE for each coordinate, fixes the gauge by centering, and stops with an empirical plug-in certificate. This is a fixed-confidence-style stopping heuristic, not a proved delta-correct MLE confidence procedure.
- `UniformPairwiseBT-BordaPlugIn-FC`: uniformly samples pair-coordinate comparisons in rounds, estimates pairwise probabilities with add-1/2 smoothing, plugs them into Borda means, and stops by the same empirical-gap certificate.

Revised randomized headline benchmarks use a hierarchical design: a fixed bank of latent `theta` instances is reused across observation replications and algorithms. Every revised raw row stores `instance_id`, `theta_hash`, `instance_index`, and `observation_replicate`. The Arena-10 benchmark and constant-sensitivity sweep share the versioned `arena10_medium_v2` bank. The confidence sweep retains its own paired-delta bank so every delta value within a replication uses the same instance.

The package still contains fixed-budget helper implementations for development checks, but they are not part of the retained main results. `VB-EGE-capped` is a heuristic comparator, not the fixed-confidence theorem.

`UniformPairwiseBT-MLE-Cert` includes ridge stabilization. If the comparison graph is disconnected, optimization fails, or the estimate exceeds `max_abs_theta`, it reruns with `fallback_ridge_lambda` and records fallback metadata.

## Layout

```text
vb_ege/        core package, algorithms, baselines, runners, and plotting
configs/       smoke checks and the six retained fixed-confidence experiment suites
tests/         pytest coverage for core math, gaps, instances, algorithms, and BT-MLE
results/raw/   final CSV rows for the retained experiment suites
results/summary/ aggregated mean/SE, error, paired-ratio, and slope tables
results/figures/ figures used by the final report
report/        report generator, TeX source, figures, and final PDF
scripts/       convenience and diagnostic utilities
```

## Setup

From this folder:

```bash
python -m pip install -e .
```

or install the listed requirements:

```bash
python -m pip install -r requirements.txt
```

## Quick Checks

```bash
pytest -q
python -m vb_ege.run_sweep --config configs/smoke.yaml --out results/raw/smoke.csv --seed 0
python -m vb_ege.summarize --raw results/raw/smoke.csv --out results/summary/smoke_summary.csv --figdir results/figures/smoke
```

## Main Runs

Fixed-confidence scaling:

```bash
python -m vb_ege.run_sweep --config configs/fixed_confidence_scaling.yaml --out results/raw/fixed_confidence_scaling.csv --seed 2026
python -m vb_ege.summarize --raw results/raw/fixed_confidence_scaling.csv --out results/summary/fixed_confidence_scaling_summary.csv --figdir results/figures/fixed_confidence_scaling
```

Fixed-confidence benchmark comparisons:

```bash
python -m vb_ege.run_sweep --config configs/fixed_confidence_benchmarks.yaml --out results/raw/fixed_confidence_benchmarks.csv --seed 31415
python -m vb_ege.summarize --raw results/raw/fixed_confidence_benchmarks.csv --out results/summary/fixed_confidence_benchmarks_summary.csv --figdir results/figures/fixed_confidence_benchmarks
```

Confidence-scaling quantiles:

```bash
python -m vb_ege.run_sweep --config configs/confidence_scaling_quantile.yaml --out results/raw/confidence_scaling_quantile.csv --seed 20260708
python -m vb_ege.summarize --raw results/raw/confidence_scaling_quantile.csv --out results/summary/confidence_scaling_quantile_summary.csv --figdir results/figures/confidence_scaling_quantile
```

Constant sensitivity:

```bash
python -m vb_ege.run_sweep --config configs/constants_calibration.yaml --out results/raw/constants_calibration.csv --seed 20260708
python -m vb_ege.summarize --raw results/raw/constants_calibration.csv --out results/summary/constants_calibration_summary.csv --figdir results/figures/constants_calibration
```

Pareto-size ablation:

```bash
python -m vb_ege.run_sweep --config configs/pareto_size_ablation.yaml --out results/raw/pareto_size_ablation.csv --seed 20260708
python -m vb_ege.summarize --raw results/raw/pareto_size_ablation.csv --out results/summary/pareto_size_ablation_summary.csv --figdir results/figures/pareto_size_ablation
```

Theory/practical constants sanity check:

```bash
python -m vb_ege.run_sweep --config configs/theory_constants_sanity.yaml --out results/raw/theory_constants_sanity.csv --seed 20260708
python -m vb_ege.summarize --raw results/raw/theory_constants_sanity.csv --out results/summary/theory_constants_sanity_summary.csv --figdir results/figures/theory_constants_sanity
```

Build the report after the summaries and figures are available:

```bash
python report/build_report.py
```

## Outputs

Raw rows include the true and recommended Pareto sets, error, Hamming distance, stopping time, Borda and latent gaps, `tau / (d * H_B)`, pair-cell count and coverage, MLE convergence metadata, ridge fallback metadata, centered theta RMSE, and pairwise sign accuracy.

Summaries report error rates with Wilson intervals, mean stopping time and its standard error, mean Hamming distance, false positives and false negatives, MLE diagnostics, and pair-cell coverage. Mean uncertainty uses latent-instance cluster standard errors when an instance has repeated observations and ordinary replication standard errors otherwise. Summaries also write `<summary>_paired.csv`, containing paired baseline/VB mean stopping-time ratios with cluster-aware standard errors. All report stopping-time curves and grouped bars use dark means with light mean-plus-or-minus-one-standard-error regions; constant-sensitivity heatmaps print mean plus-or-minus standard error in each cell. Confidence-scaling tables retain q95 as a secondary tail diagnostic. Figures used by the final report are kept as PDF and PNG.
