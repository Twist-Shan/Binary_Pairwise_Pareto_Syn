#!/usr/bin/env bash
set -euo pipefail

python -m vb_ege.run_sweep \
  --config configs/fixed_confidence_scaling.yaml \
  --out results/raw/fixed_confidence_scaling.parquet \
  --seed 2026
python -m vb_ege.summarize \
  --raw results/raw/fixed_confidence_scaling.parquet \
  --out results/summary/fixed_confidence_scaling_summary.csv \
  --figdir results/figures/fixed_confidence_scaling

python -m vb_ege.run_sweep \
  --config configs/fixed_confidence_benchmarks.yaml \
  --out results/raw/fixed_confidence_benchmarks.parquet \
  --seed 31415
python -m vb_ege.summarize \
  --raw results/raw/fixed_confidence_benchmarks.parquet \
  --out results/summary/fixed_confidence_benchmarks_summary.csv \
  --figdir results/figures/fixed_confidence_benchmarks
