import numpy as np

from pilot_focal_bt_mle.matched_fixed_budget import (
    TranscriptStats,
    _project_box_zero_sum,
    _project_box_zero_sum_orders,
    add_focal_random_opponent_queries,
    fit_box_constrained_bt,
    focal_borda_recommendation,
)
from vb_ege.core import strict_pareto_set
from vb_ege.instances import symmetric_hard


def test_matched_estimators_recover_easy_instance():
    theta, _ = symmetric_hard(K=6, d=2, Delta=1.5, seed=3)
    stats = TranscriptStats.zeros(K=6, d=2)
    add_focal_random_opponent_queries(
        theta,
        stats,
        num_queries=50_000,
        rng=np.random.default_rng(4),
    )
    truth = strict_pareto_set(theta)
    assert focal_borda_recommendation(stats) == truth
    mle = fit_box_constrained_bt(stats.pair_wins, box_bound=2.0)
    assert mle["converged_all"]
    assert mle["recommended"] == truth


def test_transcript_accounting_is_exact():
    theta, _ = symmetric_hard(K=5, d=3, Delta=0.5, seed=7)
    stats = TranscriptStats.zeros(K=5, d=3)
    add_focal_random_opponent_queries(
        theta,
        stats,
        num_queries=1234,
        rng=np.random.default_rng(8),
    )
    assert stats.num_queries == 1234
    assert int(stats.focal_counts.sum()) == 1234
    assert int(stats.pair_wins.sum()) == 1234


def test_projection_enforces_box_gauge_and_tie():
    value = np.array([4.0, -3.0, 2.0, -1.0])
    projected = _project_box_zero_sum(value, box_bound=1.0, tied_pair=(0, 2))
    assert np.isclose(projected.sum(), 0.0)
    assert np.max(np.abs(projected)) <= 1.0 + 1e-10
    assert np.isclose(projected[0], projected[2])


def test_projection_enforces_star_order_constraints():
    value = np.array([3.0, -2.0, 1.0, 0.0])
    projected = _project_box_zero_sum_orders(
        value,
        box_bound=1.5,
        order_constraints=((1, 0), (1, 2)),
    )
    assert np.isclose(projected.sum(), 0.0)
    assert np.max(np.abs(projected)) <= 1.5 + 1e-10
    assert projected[1] >= projected[0] - 1e-10
    assert projected[1] >= projected[2] - 1e-10
