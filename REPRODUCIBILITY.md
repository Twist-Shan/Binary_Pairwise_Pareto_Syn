# Synthetic reproducibility protocol

## Environment

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv .venv --python 3.12
uv sync --extra dev
.venv\Scripts\python.exe -m pytest
```

The dependency ranges live in `pyproject.toml`; `uv.lock` is the release
resolution. The full test command must pass from a clean checkout without
untracked local Python modules.

## Canonical suites

| Stem | Config | Raw rows expected |
|---|---|---:|
| `fixed_confidence_scaling` | `configs/fixed_confidence_scaling.yaml` | 24,000 |
| `fixed_confidence_benchmarks` | `configs/fixed_confidence_benchmarks.yaml` | 16,000 |
| `confidence_scaling_quantile` | `configs/confidence_scaling_quantile.yaml` | 23,200 |
| `constants_calibration` | `configs/constants_calibration.yaml` | 19,800 |
| `pareto_size_ablation` | `configs/pareto_size_ablation.yaml` | 9,600 |
| `theory_constants_sanity` | `configs/theory_constants_sanity.yaml` | 400 |

The last filename is retained for compatibility. In the current paper,
`(2,4,4)` are the theorem constants; `(8,16,4)` is simply a more conservative
constant sensitivity profile.

## Run one suite

```powershell
.venv\Scripts\python.exe -m vb_ege.run_sweep `
  --config configs\fixed_confidence_scaling.yaml `
  --out results\raw\fixed_confidence_scaling.csv `
  --seed 2026 `
  --jobs 8

.venv\Scripts\python.exe -m vb_ege.summarize `
  --raw results\raw\fixed_confidence_scaling.csv `
  --out results\summary\fixed_confidence_scaling_summary.csv `
  --figdir results\figures\fixed_confidence_scaling

.venv\Scripts\python.exe -m scripts.verify_results `
  --raw results\raw\fixed_confidence_scaling.csv `
  --summary results\summary\fixed_confidence_scaling_summary.csv `
  --manifest results\raw\fixed_confidence_scaling.csv.manifest.json
```

The verifier checks the raw SHA-256 and regenerates summary, paired, and slope
tables in temporary storage. Figures are not rewritten during verification.

To audit the implementation-only change from the historical approximate-tie
rule to Algorithm 1's exact argmax without writing any artifact:

```powershell
.venv\Scripts\python.exe -m scripts.audit_exact_argmax_regression `
  --config configs\fixed_confidence_scaling.yaml `
  --config configs\fixed_confidence_benchmarks.yaml `
  --config configs\confidence_scaling_quantile.yaml `
  --replicate-index 0 --replicate-index 1
```

The command replays each selected configured cell with the same observation
seed under both rules and exits nonzero if the recommendation, stopping time,
phase count, or accepted/rejected state changes. It does not claim a proof for
unreplayed seeds; full figure reuse additionally requires the raw-to-summary
verification above.

## Run all main suites

Repeat the three commands above with the config, stem, and base seed below:

| Stem | Seed |
|---|---:|
| `fixed_confidence_scaling` | 2026 |
| `fixed_confidence_benchmarks` | 31415 |
| `confidence_scaling_quantile` | 314159 |
| `constants_calibration` | 271828 |
| `pareto_size_ablation` | 424242 |
| `theory_constants_sanity` | 20260708 |

The YAML `base_seed` is authoritative. A CLI `--seed` is shown explicitly so
the command log is self-contained.

## Raw row contract

Every row stores, as applicable:

- experiment/cell identity and algorithm;
- base-derived observation seed and latent-instance seed;
- latent parameter JSON and SHA-256 of the generated `theta` matrix;
- true and recommended strict Pareto sets;
- stopping state, integer query count, phases, accepted/rejected counts;
- set error, false positives/negatives, and Hamming distance;
- latent and Vector-Borda gaps/hardness;
- BT fit convergence and pair-cell coverage diagnostics;
- latent-instance and observation-replicate IDs for clustered and paired
  summaries.

## Summary conventions

- Main curves/tables use mean stopping time plus or minus one standard error.
- When complete `instance_id` values exist, standard errors cluster by latent
  instance; otherwise the replication is the analysis unit.
- Baseline/VB comparisons use paired per-replication ratios, with the same
  latent-instance clustering when available.
- A fixed bootstrap seed is used for paired ratio intervals.
- Unfinished/capped Track-and-Stop rows remain in raw output and are reported,
  not silently discarded.

## Build the committed report

After all canonical summaries and figures exist:

```powershell
.venv\Scripts\python.exe scripts\build_report.py
```

Compile `report/synthetic_fixed_confidence_report.tex` with LaTeX. Existing
figure basenames and `report/synthetic_fixed_confidence_report.pdf` are kept
stable. Rebuild figures only if verification shows the underlying summary has
changed.

## Provisional Pareto BT-GLR extension

```powershell
.venv\Scripts\python.exe -m icml_pareto_track_stop.run_formal `
  --out icml_pareto_track_stop\results\raw\formal_two_way_certificate.csv `
  --jobs 16 --prepare-jobs 16

.venv\Scripts\python.exe -m icml_pareto_track_stop.summarize_formal `
  --raw icml_pareto_track_stop\results\raw\formal_two_way_certificate.csv `
  --summary-dir icml_pareto_track_stop\results\summary_certificate `
  --figure-dir icml_pareto_track_stop\results\figures_certificate
```

The completed protocol has 36,400 rows total. The extension is deliberately
labeled non-certified and should not be described as inheriting the scalar
Track-and-Stop theorem.

## Public artifact release

The Git repository contains code, configs, tests, and canonical PDFs. Publish
large raw/summary/figure directories separately with:

1. release tag and Git commit;
2. the `*.manifest.json` files generated by the runners;
3. an archive-level SHA-256;
4. Python version and `uv.lock`;
5. a note that figures are derived from the included summaries.
