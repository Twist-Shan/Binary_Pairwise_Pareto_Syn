# Retained Synthetic Results

This directory contains only the experiment outputs used by the final synthetic report.

## Raw CSVs

- `fixed_confidence_scaling.csv`: fixed-confidence scaling in `K`, `d`, and `Delta`, with `delta=0.05` fixed.
- `confidence_scaling_quantile.csv`: the separate confidence sweep over `delta`.
- `constants_calibration.csv`: practical-constant sensitivity experiments.
- `fixed_confidence_benchmarks.csv`: Convex, Witness, Arena, and Two-group benchmarks.
- `pareto_size_ablation.csv`: Arena-4 and Arena-10 Pareto-size sweeps.
- `theory_constants_sanity.csv`: practical versus theory-style constants.

The CSVs are the final merged versions. JSONL checkpoints, smoke runs, repair shards, correlated-arena runs, and under-budget runs were removed after validation because they are not part of the final report.

## Summaries

`summary/` contains the corresponding aggregate tables. The main stopping-time statistic is mean plus or minus one standard error. Paired files report baseline/VB-EGE mean ratios with cluster-aware standard errors when repeated observations share a latent instance.

## Figures

`figures/` retains only figures referenced by the report, in both PDF and PNG formats. Normalized duplicates, old median/IQR figures, and figures for removed experiments are intentionally omitted.

The final compiled report is `../report/synthetic_fixed_confidence_report.pdf`.
