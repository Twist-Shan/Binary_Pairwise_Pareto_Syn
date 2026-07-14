from vb_ege.core import strict_pareto_set
from vb_ege.instances import (
    arena_tradeoff_frontier,
    correlated_arena_like,
    convex_frontier_2d,
    convex_frontier_3d,
    highdim_two_group,
    unique_witness_d,
)


def test_arena_target_pareto_size():
    theta, _ = arena_tradeoff_frontier(
        K=16, d=4, s=4, margin_low=0.08, margin_high=0.20, seed=0
    )
    assert len(strict_pareto_set(theta)) == 4


def test_unique_witness_target_pareto_size():
    theta, _ = unique_witness_d(
        K=15, d=3, s=5, q_per_p=2, margin_low=0.03, margin_high=0.12, seed=1
    )
    assert len(strict_pareto_set(theta)) == 5


def test_highdim_two_group_has_no_low_pareto_assumption_break():
    theta, meta = highdim_two_group(K_low=10, K_high=4, d=5, seed=2)
    assert len(strict_pareto_set(theta)) == meta["expected_pareto_size"]


def test_convex_frontier_size():
    theta, _ = convex_frontier_2d(K=20, s=6, seed=3)
    assert len(strict_pareto_set(theta)) == 6


def test_convex_frontier_3d_size():
    theta, meta = convex_frontier_3d(K=24, s=8, seed=4)
    assert theta.shape == (24, 3)
    assert len(strict_pareto_set(theta)) == 8
    assert meta["expected_pareto_size"] == 8


def test_correlated_arena_target_pareto_size_and_metadata():
    for rho in (0.0, 0.6, 0.9):
        theta, meta = correlated_arena_like(
            K=32,
            d=4,
            s=8,
            rho=rho,
            margin_low=0.08,
            margin_high=0.25,
            seed=int(100 * rho) + 4,
        )
        assert len(strict_pareto_set(theta)) == 8
        assert meta["expected_pareto_size"] == 8
        assert "achieved_objective_correlation_mean" in meta["params"]
