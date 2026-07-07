import numpy as np

from vb_ege.baselines import allocate_pair_coordinate_budget, run_uniform_pairwise_bt_mle
from vb_ege.bt_mle import BTMLEConfig, fit_bt_mle, fit_bt_mle_coordinate
from vb_ege.instances import symmetric_hard
from vb_ege.metrics import pairwise_sign_accuracy
from vb_ege.core import strict_pareto_set
from vb_ege.baselines import simulate_pairwise_counts


def test_bt_mle_recovers_order_high_budget():
    rng = np.random.default_rng(0)
    theta = 2.0 * rng.normal(size=(8, 3))
    counts = allocate_pair_coordinate_budget(8, 3, 200000, rng)
    W, Npair = simulate_pairwise_counts(theta, counts, rng)
    fit = fit_bt_mle(W, Npair, BTMLEConfig())
    assert pairwise_sign_accuracy(fit["theta_hat"], theta) > 0.98


def test_bt_mle_pareto_recovery_symmetric():
    theta, _ = symmetric_hard(K=8, d=4, Delta=2.0, seed=0, permute=False)
    out = run_uniform_pairwise_bt_mle(theta, budget=50000, rng=np.random.default_rng(1))
    assert out["recommended"] == strict_pareto_set(theta)


def test_bt_mle_gauge_invariance():
    rng = np.random.default_rng(2)
    theta = rng.normal(size=(7, 3))
    shifts = np.array([10.0, -2.5, 4.0])
    counts = allocate_pair_coordinate_budget(7, 3, 120000, rng)
    W1, N1 = simulate_pairwise_counts(theta, counts, np.random.default_rng(10))
    W2, N2 = simulate_pairwise_counts(theta + shifts, counts, np.random.default_rng(10))
    fit1 = fit_bt_mle(W1, N1, BTMLEConfig())
    fit2 = fit_bt_mle(W2, N2, BTMLEConfig())
    assert np.allclose(fit1["theta_hat"], fit2["theta_hat"], atol=1e-10)


def test_bt_mle_fallback_on_separation():
    K = 5
    W = np.zeros((K, K), dtype=int)
    N = np.zeros((K, K), dtype=int)
    for j in range(1, K):
        W[0, j] = 20
        W[j, 0] = 0
        N[0, j] = 20
        N[j, 0] = 20
    cfg = BTMLEConfig(ridge_lambda=0.0, max_abs_theta=2.0, fallback_ridge_lambda=1e-3)
    out = fit_bt_mle_coordinate(W, N, cfg)
    assert out["ridge_fallback_used"]
    assert np.all(np.isfinite(out["theta_hat"]))


def test_pair_budget_balanced_random():
    rng = np.random.default_rng(3)
    K, d = 6, 4
    C = d * K * (K - 1) // 2
    counts = allocate_pair_coordinate_budget(K, d, C + 7, rng)
    upper = counts[:, np.triu_indices(K, 1)[0], np.triu_indices(K, 1)[1]]
    assert np.all(upper >= 1)
    assert counts.sum() == C + 7
