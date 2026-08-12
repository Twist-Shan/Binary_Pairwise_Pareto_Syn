# Beyond Scalar Leaderboards - Synthetic Experiments

Reproducible synthetic experiments for **Beyond Scalar Leaderboards: Adaptive
Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas**.

The main package implements Vector-Borda Exponential Gap Elimination (VB-EGE)
and the three matched baselines used in the paper. It also contains the paper's
clearly labeled, non-certified Pareto BT-GLR Track-and-Stop extension.

## What is reproduced

| Paper study | Config | Cells | Repetitions |
|---|---|---:|---:|
| Scaling in `K`, `d`, and gap | `configs/fixed_confidence_scaling.yaml` | 12 | 500/method/cell |
| Eight heterogeneous benchmarks | `configs/fixed_confidence_benchmarks.yaml` | 8 | 500/method/cell |
| Confidence scaling and tails | `configs/confidence_scaling_quantile.yaml` | 14 | 500 symmetric or 300 Arena/method/cell |
| Constant sensitivity | `configs/constants_calibration.yaml` | 54 summary cells | 300-500 |
| Pareto-size sensitivity | `configs/pareto_size_ablation.yaml` | 8 | 300/method/cell |
| Conservative-constant sensitivity | `configs/theory_constants_sanity.yaml` | 2 | 100/method/cell |

The headline VB-EGE method uses the paper's Algorithm 1 / Theorem 4.1 constants
`(C_samp,C_thr,C_log)=(2,4,4)`; baselines use their method-specific certificates.
The historical algorithm label `VB-EGE-practical` is retained in raw files and
figure code so published file names and result schemas remain stable; it
denotes the paper's main VB-EGE configuration, not a different algorithm.

## Repository map

```text
vb_ege/                     VB-EGE, baselines, generators, metrics, runners
configs/                    Exact paper protocols and seed schedules
scripts/                    Main-suite runner, verification, report builder
results/                    Local raw/summary/figure artifacts (Git-ignored)
report/                     Committed synthetic report PDF
icml_pareto_track_stop/     Isolated provisional Track-and-Stop extension
pilot_focal_bt_mle/         Shared constrained BT numerical utilities
tests/                      Algorithm, protocol, metric, and smoke tests
.github/workflows/          Locked clean-checkout test workflow
REPRODUCIBILITY.md          Full commands, expected rows, and verification
```

Existing figure and report names are intentionally retained. Table
verification and the stratified exact-argmax replay described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) determine whether a figure rebuild is
needed; this refactor never rewrites a figure merely because code moved.

## Five-minute validation

PowerShell:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv .venv --python 3.12
uv sync --extra dev
.venv\Scripts\python.exe -m pytest

.venv\Scripts\python.exe -m vb_ege.run_sweep `
  --config configs\smoke.yaml `
  --out results\raw\smoke.csv `
  --seed 0
.venv\Scripts\python.exe -m vb_ege.summarize `
  --raw results\raw\smoke.csv `
  --out results\summary\smoke_summary.csv `
  --figdir results\figures\smoke
```

Each completed sweep writes a sidecar `*.manifest.json` containing the config
hash, raw hash, base seed, source-tree hash, Python version, and row count.

## Reproduce the three headline suites

```powershell
.venv\Scripts\python.exe -m vb_ege.run_sweep --config configs\fixed_confidence_scaling.yaml --out results\raw\fixed_confidence_scaling.csv --seed 2026 --jobs 8
.venv\Scripts\python.exe -m vb_ege.summarize --raw results\raw\fixed_confidence_scaling.csv --out results\summary\fixed_confidence_scaling_summary.csv --figdir results\figures\fixed_confidence_scaling

.venv\Scripts\python.exe -m vb_ege.run_sweep --config configs\fixed_confidence_benchmarks.yaml --out results\raw\fixed_confidence_benchmarks.csv --seed 31415 --jobs 8
.venv\Scripts\python.exe -m vb_ege.summarize --raw results\raw\fixed_confidence_benchmarks.csv --out results\summary\fixed_confidence_benchmarks_summary.csv --figdir results\figures\fixed_confidence_benchmarks

.venv\Scripts\python.exe -m vb_ege.run_sweep --config configs\confidence_scaling_quantile.yaml --out results\raw\confidence_scaling_quantile.csv --seed 314159 --jobs 8
.venv\Scripts\python.exe -m vb_ege.summarize --raw results\raw\confidence_scaling_quantile.csv --out results\summary\confidence_scaling_quantile_summary.csv --figdir results\figures\confidence_scaling_quantile
```

Then verify raw hashes and rebuild the tables in a temporary directory:

```powershell
.venv\Scripts\python.exe -m scripts.verify_results `
  --raw results\raw\fixed_confidence_scaling.csv `
  --summary results\summary\fixed_confidence_scaling_summary.csv `
  --manifest results\raw\fixed_confidence_scaling.csv.manifest.json
```

Add `--skip-figures` to `vb_ege.summarize` when only deterministic CSV tables
need to be refreshed.

Use the same pattern for the benchmark and confidence suites. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for all six suites, expected row
counts, report construction, and reference-result publication.

## Algorithm contract

For phase `m`, VB-EGE uses

```text
epsilon_m = 2^(-m)
L_m       = log(4 K d m^2 / delta)
n_m       = ceil(2 L_m / epsilon_m^2)
r_m       = sqrt(L_m / (2 n_m))
```

Active focal arm-objective cells are cumulatively sampled to `n_m`. Opponents
are always drawn uniformly from the original `K-1` roster, including arms
already removed from the active set. After every removal, empirical Pareto
membership and identification gaps are recomputed on the remaining active
set. Removal is allowed only when the maximum gap is strictly larger than
`4 r_m`; non-Pareto-first tie-breaking is applied only among exact empirical
gap maximizers.

The default implementation uses a Binomial batch with the exact Borda mean.
This has the same law as individually drawing a fresh uniform opponent and
binary outcome for every query.

## Methods and interpretation

- `VB-EGE-practical`: paper's main VB-EGE configuration; historical label
  retained for artifact compatibility.
- `UniformFocalBorda-FC`: uniform focal Borda sampling with the matched
  frontier certificate.
- `UniformPairwiseBT-MLE-Cert`: stabilized BT-MLE plug-in heuristic; not a
  proved delta-correct confidence region.
- `UniformPairwiseBT-BordaPlugIn-FC`: uniform pair-objective sampling with
  smoothed Borda reconstruction.
- `Pareto BT-GLR Track-and-Stop`: provisional screened multi-objective
  extension. Its screening statistic is not a proved global profile lower
  bound; every raw row records `stopping_statistic_is_certified=false`.

Raw data and regenerated plots are intentionally not committed because they
are large. For a public release, upload them as a versioned GitHub Release or
Zenodo archive together with the generated manifests; do not publish a PDF
without its raw/summary hash inventory.
