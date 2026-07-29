import numpy as np

from pilot_focal_bt_mle.audit_conventions import usual_pareto_set
from pilot_focal_bt_mle.matched_fixed_budget import (
    TranscriptStats,
    _project_box_zero_sum,
    _project_box_zero_sum_orders,
    add_focal_random_opponent_queries,
    fit_box_constrained_bt,
    focal_borda_recommendation,
)
from pilot_focal_bt_mle.pareto_track_and_stop import (
    ParetoTrackStopConfig,
    _add_assignment_plan,
    _estimated_dominators,
    run_pareto_bt_glr_track_and_stop,
)
from pilot_focal_bt_mle.sequential_fc import (
    FocalBTMLEFCConfig,
    run_focal_bt_mle_fc,
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


def test_isolated_pilot_uses_all_coordinate_strict_dominance():
    theta = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    assert strict_pareto_set(theta) == (0, 1)
    assert usual_pareto_set(theta) == (0,)
    assert _estimated_dominators(theta, arm=1) == []
    assert _estimated_dominators(theta, arm=2) == [0, 1]


def test_add_assignment_plan_reserves_exact_label_for_all_arms():
    theta_hat = np.array(
        [
            [1.0, 1.0],
            [0.0, 2.0],
            [-2.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    exact = _add_assignment_plan(theta_hat, arm=3, max_assignments=8)
    assert exact == {
        "mode": "all-arm-exact",
        "roster": (0, 1, 2),
        "num_assignments": 8,
        "exact": True,
    }

    relaxed = _add_assignment_plan(theta_hat, arm=3, max_assignments=4)
    assert relaxed == {
        "mode": "current-dominator-relaxation",
        "roster": (0, 1),
        "num_assignments": 4,
        "exact": False,
    }

    single = _add_assignment_plan(theta_hat, arm=3, max_assignments=1)
    assert single["mode"] == "single-dominator-bound"
    assert single["exact"] is False


def test_profile_likelihood_fc_stops_on_easy_instance():
    theta, _ = symmetric_hard(K=4, d=2, Delta=2.0, seed=1)
    theta = theta + 100.0
    result = run_focal_bt_mle_fc(
        theta,
        FocalBTMLEFCConfig(
            delta=0.05,
            box_bound=3.0,
            max_queries=200_000,
        ),
        np.random.default_rng(2),
    )
    assert result["stopped"]
    assert result["error"] is False
    assert result["recommended"] == strict_pareto_set(theta)


def test_pareto_glr_track_and_stop_stops_on_easy_instance():
    theta = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-0.5, -0.5],
        ]
    )
    result = run_pareto_bt_glr_track_and_stop(
        theta,
        ParetoTrackStopConfig(
            delta=0.05,
            box_bound=2.0,
            max_queries=100_000,
            growth_factor=1.8,
            max_add_assignments=64,
            optimizer_tol=1e-6,
            optimizer_max_iter=500,
        ),
        np.random.default_rng(7),
    )
    assert result["stopped"]
    assert result["error"] is False
    assert result["recommended"] == strict_pareto_set(theta)
