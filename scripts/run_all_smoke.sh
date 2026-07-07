#!/usr/bin/env bash
set -euo pipefail

pytest -q
python -m vb_ege.run_sweep \
  --config configs/smoke.yaml \
  --out results/raw/smoke.parquet \
  --seed 0
python -m vb_ege.summarize \
  --raw results/raw/smoke.parquet \
  --out results/summary/smoke_summary.csv \
  --figdir results/figures/smoke
