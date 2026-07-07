# Synthetic Experiments for Vector-Borda Pareto Set Identification

This repo implements fixed-confidence synthetic experiments for strict Pareto set identification under a coordinate-wise Bradley-Terry comparison model. Fixed-budget variants are kept only as ablations and diagnostics.

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

The main fixed-confidence benchmark methods are:

- `UniformFocalBorda-FC`: uniformly samples arm-coordinate focal Borda cells in rounds and stops when the empirical Pareto gap clears the confidence radius.
- `UniformPairwiseBT-MLE-FC`: uniformly samples pair-coordinate Bradley-Terry comparisons in rounds, fits a separate BT MLE for each coordinate, fixes the gauge by centering, and stops when the empirical latent gap clears the confidence radius.
- `UniformPairwiseBT-BordaPlugIn-FC`: uniformly samples pair-coordinate comparisons in rounds, estimates pairwise probabilities with add-1/2 smoothing, plugs them into Borda means, and stops by the same empirical-gap certificate.

The repo also keeps fixed-budget methods (`UniformFocalBorda`, `UniformPairwiseBT-MLE`, `UniformPairwiseBT-BordaPlugIn`) and `VB-EGE-capped` for budget-curve ablations. `VB-EGE-capped` is a heuristic comparator, not the fixed-confidence theorem.

`UniformPairwiseBT-MLE-FC` includes ridge stabilization. If the comparison graph is disconnected, optimization fails, or the estimate exceeds `max_abs_theta`, it reruns with `fallback_ridge_lambda` and records fallback metadata.

## Layout

```text
vb_ege/        core package, algorithms, baselines, runners, plotting
configs/       smoke, fixed-confidence, fixed-budget ablation, arena-like, and MLE ablation configs
tests/         pytest coverage for core math, gaps, instances, algorithms, and BT-MLE
results/       raw, summary, and figure outputs
scripts/       convenience shell entrypoints
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
python -m vb_ege.run_sweep --config configs/smoke.yaml --out results/raw/smoke.parquet --seed 0
python -m vb_ege.summarize --raw results/raw/smoke.parquet --out results/summary/smoke_summary.csv --figdir results/figures/smoke
```

## Main Runs

Fixed-confidence scaling:

```bash
python -m vb_ege.run_sweep --config configs/fixed_confidence_scaling.yaml --out results/raw/fixed_confidence_scaling.parquet --seed 2026
python -m vb_ege.summarize --raw results/raw/fixed_confidence_scaling.parquet --out results/summary/fixed_confidence_scaling_summary.csv --figdir results/figures/fixed_confidence_scaling
```

Fixed-confidence benchmark comparisons:

```bash
python -m vb_ege.run_sweep --config configs/fixed_confidence_benchmarks.yaml --out results/raw/fixed_confidence_benchmarks.parquet --seed 31415
python -m vb_ege.summarize --raw results/raw/fixed_confidence_benchmarks.parquet --out results/summary/fixed_confidence_benchmarks_summary.csv --figdir results/figures/fixed_confidence_benchmarks
```

MLE ablation:

```bash
python -m vb_ege.run_sweep --config configs/mle_ablation.yaml --out results/raw/mle_ablation.parquet --seed 777
python -m vb_ege.summarize --raw results/raw/mle_ablation.parquet --out results/summary/mle_ablation_summary.csv --figdir results/figures/mle_ablation
```

Arena-like headline group:

```bash
python -m vb_ege.run_sweep --config configs/arena_like.yaml --out results/raw/arena_like.parquet --seed 4242
python -m vb_ege.summarize --raw results/raw/arena_like.parquet --out results/summary/arena_like_summary.csv --figdir results/figures/arena_like
```

## Outputs

Raw rows include the true and recommended Pareto sets, error, Hamming distance, stopping time, Borda and latent gaps, `tau / (d * H_B)`, pair-cell count and coverage, MLE convergence metadata, ridge fallback metadata, centered theta RMSE, and pairwise sign accuracy.

Summaries report error rates with Wilson intervals, stopping-time quantiles, mean Hamming distance, false positives and false negatives, MLE diagnostics, and pair-cell coverage. For fixed-confidence configs, `tau_by_algorithm_<experiment>.pdf` is the main comparison plot. Plots are written as both PDF and PNG.
