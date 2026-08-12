# Isolated VB-EGE vs. Pareto BT-GLR Track-and-Stop

This package contains the isolated two-algorithm experiment comparing
the paper's main VB-EGE implementation (historical raw label
`VB-EGE-practical`) with a heuristic Pareto extension of BT-GLR
Track-and-Stop. It reuses the main synthetic generators, strict Pareto
convention, latent-instance seeds, and repetition counts.

The canonical report is
[`report/formal_certificate_v1_report.pdf`](report/formal_certificate_v1_report.pdf).

## Scope

The completed formal protocol has 18,200 runs per algorithm:

| Section | Runs/method |
|---|---:|
| Fixed-confidence scaling | 6,000 |
| Confidence-scaling quantiles | 5,800 |
| Eight fixed-confidence benchmarks | 4,000 |
| Pareto-size ablation | 2,400 |
| **Total** | **18,200** |

The combined run-level table therefore has 36,400 rows. It is generated
locally and excluded from Git.

## Methods

### VB-EGE

The experiment calls the unchanged `VB-EGE-practical` implementation with

```text
sample_const = 2.0
threshold_const = 4.0
log_const = 4.0
```

These are the common Algorithm 1 / Theorem 4.1 constants in the current paper.

### Pareto BT-GLR Track-and-Stop

For every objective, the method fits a box-constrained Bradley-Terry MLE. It
then constructs frontier-changing alternatives of two types:

1. **Drop:** another estimated Pareto arm dominates a current Pareto arm in
   every objective.
2. **Add:** a current non-Pareto arm `q` receives one witness objective for
   each current Pareto arm `p`, with `theta[q, r] >= theta[p, r]` on that
   witness objective.

For an estimated Pareto set of size `p`, one add-arm candidate has `d ** p`
witness assignments. Each fixed assignment is evaluated by a constrained BT
profile likelihood.

The current runner has two computational paths:

- **Exact certificate enumeration:** activated only when `K <= 8` and
  `d ** p <= 4096`. All witness assignments are profiled.
- **Screened joint certificate:** all candidate branches receive a local
  Fisher quadratic screening cost; only the selected feasible joint
  certificate is evaluated under the original constrained BT likelihood.

The `K <= 8` condition is an engineering cutoff, not a mathematical boundary.
Large `K` with a small Pareto front can still be cheap enough to enumerate.
Conversely, `d ** p` is prohibitive for large fronts. A workload-based exact
gate or a certified terminal audit is a planned methodological improvement.

The screened path can change the selected least-favourable alternative,
sampling allocation, and stopping time relative to complete enumeration. Its
local-Fisher screening value is not proved to be a global lower bound on the
full Pareto alternative profile. Consequently:

- the extension is reported as a fixed-confidence-style heuristic;
- the scalar top-k theorem is not claimed to prove Pareto delta-correctness;
- every output records `final_profile_mode` and
  `stopping_statistic_is_certified`;
- empirical error, stopping rate, capped runs, and MLE convergence are
  reported explicitly.

All current formal cells have `K >= 16`, so the committed formal report uses
the screened-joint-certificate path. This limitation is part of the report's
interpretation, not hidden as an implementation detail.

## Package layout

```text
certificate_track_stop.py       Certificate/profile implementation
scalable_track_stop.py          Fisher geometry, tracking, and shared helpers
formal_protocol.py              Main-suite instance and repetition protocol
formal_config.yaml              Formal algorithm constants
run_formal.py                   Parallel resumable runner
summarize_formal.py             Mean/SE tables, diagnostics, and figures
validate_profile_approximation.py  Exact-versus-quadratic diagnostic
report/                         Report builders and canonical PDF
results/                        Ignored local outputs with .gitkeep markers
```

The constrained coordinate-wise BT optimizer currently lives in
`pilot_focal_bt_mle/matched_fixed_budget.py` and is shared by both the exact
and screened profile implementations.

## Reproduce

From the repository root:

```powershell
python -m icml_pareto_track_stop.run_formal `
  --out icml_pareto_track_stop\results\raw\formal_two_way_certificate.csv `
  --jobs 16 `
  --prepare-jobs 16
```

Resume an interrupted run with `--resume`. The runner writes a JSONL
checkpoint next to the requested CSV and caches the expanded formal job bank at
`results/formal_certificate_jobs.pkl`. Both are ignored and may be deleted
after the final CSV has been validated.

Generate summaries and figures:

```powershell
python -m icml_pareto_track_stop.summarize_formal `
  --raw icml_pareto_track_stop\results\raw\formal_two_way_certificate.csv `
  --summary-dir icml_pareto_track_stop\results\summary_certificate `
  --figure-dir icml_pareto_track_stop\results\figures_certificate
```

Generate the report TeX source:

```powershell
python -m icml_pareto_track_stop.report.build_certificate_v1_formal_report
```

Compile `report/formal_certificate_v1_report.tex` with LaTeX. The generated
TeX, raw data, summaries, figures, checkpoints, and caches are ignored; the
canonical PDF is versioned.

Run the targeted tests from the repository root:

```bash
python -m pytest -q tests/test_certificate_track_stop.py tests/test_scalable_track_stop.py
```
