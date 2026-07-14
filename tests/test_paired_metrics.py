import pandas as pd

from vb_ege.metrics import paired_tau_ratios, summarize_runs
from vb_ege.plotting import _mean_se_bounds


def test_paired_tau_ratios_use_per_replication_ratios_and_canonical_mle_name():
    rows = []
    for rep, vb_tau, mle_tau in [(0, 10.0, 20.0), (1, 100.0, 50.0), (2, 20.0, 40.0)]:
        rows.extend(
            [
                {
                    "experiment_id": "e",
                    "algorithm": "VB-EGE-practical",
                    "replicate_id": f"r{rep}",
                    "tau": vb_tau,
                    "K": 4,
                    "d": 2,
                    "delta": 0.05,
                },
                {
                    "experiment_id": "e",
                    "algorithm": "UniformPairwiseBT-MLE-FC",
                    "replicate_id": f"r{rep}",
                    "tau": mle_tau,
                    "K": 4,
                    "d": 2,
                    "delta": 0.05,
                },
            ]
        )

    result = paired_tau_ratios(pd.DataFrame(rows), n_boot=100)

    assert len(result) == 1
    assert result.iloc[0]["algorithm"] == "UniformPairwiseBT-MLE-Cert"
    assert result.iloc[0]["median_ratio"] == 2.0
    assert result.iloc[0]["mean_ratio"] == 1.5
    assert result.iloc[0]["se_ratio"] > 0.0


def test_mean_tau_ci_uses_latent_instances_as_clusters():
    rows = []
    for instance_id, taus in [("a", [10.0, 12.0]), ("b", [20.0, 22.0])]:
        for tau in taus:
            rows.append(
                {
                    "experiment_id": "hierarchical",
                    "algorithm": "VB-EGE-practical",
                    "instance_id": instance_id,
                    "tau": tau,
                    "error": False,
                    "K": 4,
                    "d": 2,
                    "delta": 0.05,
                }
            )

    result = summarize_runs(pd.DataFrame(rows)).iloc[0]

    assert result["mean_tau"] == 16.0
    assert result["mean_tau_ci_unit"] == "instance_id"
    assert result["mean_tau_ci_n"] == 2
    assert result["se_tau"] == 5.0


def test_mean_plot_bounds_use_one_standard_error_not_the_retained_ci_columns():
    summary = pd.DataFrame(
        {
            "mean_tau": [10.0, 20.0],
            "se_tau": [2.0, 0.0],
            "mean_tau_ci_lower": [1.0, 1.0],
            "mean_tau_ci_upper": [19.0, 39.0],
        }
    )

    lower, upper = _mean_se_bounds(summary)

    assert lower.tolist() == [8.0, 20.0]
    assert upper.tolist() == [12.0, 20.0]
