# Generated Results

This directory is the local output root for the main synthetic experiments.
Large generated files are intentionally excluded from Git; only this guide is
versioned.

## Expected structure

```text
results/
  raw/       Run-level CSV files
  summary/   Aggregate, paired, slope, and diagnostic tables
  figures/   PDF and PNG plots generated from the summaries
```

The canonical completed suites are:

- `fixed_confidence_scaling`;
- `confidence_scaling_quantile`;
- `fixed_confidence_benchmarks`;
- `constants_calibration`;
- `pareto_size_ablation`;
- `theory_constants_sanity`.

Raw JSONL checkpoints, extension shards, repair files, and instance-bank caches
are intermediate artifacts. They may be removed once the final CSV has been
validated and are never required for reading the committed report PDF.

Summaries report mean stopping time plus or minus one standard error. Repeated
observations from a shared latent instance use latent-instance clustered
standard errors; independent-instance suites use ordinary replication standard
errors. Paired files report baseline/VB-EGE stopping-time ratios under the same
clustering rule.

The final main report is
[`../report/synthetic_fixed_confidence_report.pdf`](../report/synthetic_fixed_confidence_report.pdf).
Reproduction commands and protocol details are in the repository
[`README`](../README.md).
