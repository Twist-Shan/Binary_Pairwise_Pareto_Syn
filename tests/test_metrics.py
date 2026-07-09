from vb_ege.compat import import_pandas_quietly
from vb_ege.metrics import summarize_runs


pd = import_pandas_quietly()


def test_summarize_runs_deduplicates_repeated_seed_within_cell():
    rows = [
        {
            "experiment_id": "exp",
            "algorithm": "alg",
            "K": 4,
            "d": 2,
            "delta": 0.05,
            "param_rho": 0.6,
            "run_id": 0,
            "seed": 123,
            "error": False,
            "tau": 10,
        },
        {
            "experiment_id": "exp",
            "algorithm": "alg",
            "K": 4,
            "d": 2,
            "delta": 0.05,
            "param_rho": 0.6,
            "run_id": 1,
            "seed": 123,
            "error": False,
            "tau": 20,
        },
        {
            "experiment_id": "exp",
            "algorithm": "alg",
            "K": 4,
            "d": 2,
            "delta": 0.05,
            "param_rho": 0.6,
            "run_id": 2,
            "seed": 456,
            "error": True,
            "tau": 30,
        },
    ]
    summary = summarize_runs(pd.DataFrame(rows))

    assert len(summary) == 1
    rec = summary.iloc[0]
    assert rec["n_reps"] == 2
    assert rec["error_rate"] == 0.5
    assert rec["mean_tau"] == 25.0
