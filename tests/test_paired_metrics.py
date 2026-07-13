import pandas as pd

from vb_ege.metrics import paired_tau_ratios


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
