from __future__ import annotations

import numpy as np

from icml_pareto_track_stop.scalable_track_stop import (
    ScalableParetoTrackStopConfig,
    _fit_bt,
    _integer_allocation,
    _profile_geometry,
    _quadratic_order_profile,
    pareto_quadratic_profile_glr,
    run_scalable_pareto_track_and_stop,
)
from pilot_focal_bt_mle.matched_fixed_budget import (
    fit_box_constrained_bt_coordinate,
)
from vb_ege.core import stable_sigmoid
from vb_ege.instances import make_instance


def _expected_pair_wins(theta: np.ndarray, count: int) -> np.ndarray:
    K, d = theta.shape
    wins = np.zeros((d, K, K), dtype=float)
    for r in range(d):
        for i in range(K):
            for j in range(i + 1, K):
                probability = float(stable_sigmoid(theta[i, r] - theta[j, r]))
                wins[r, i, j] = count * probability
                wins[r, j, i] = count * (1.0 - probability)
    return wins


def test_integer_allocation_preserves_large_total():
    allocation = _integer_allocation(np.array([0.1, 0.2, 0.7]), 10**12 + 17)
    assert allocation.dtype == np.int64
    assert int(allocation.sum()) == 10**12 + 17
    assert np.all(allocation >= 0)


def test_local_quadratic_profile_matches_exact_profile_near_mle():
    theta = np.array(
        [
            [0.4],
            [0.1],
            [-0.1],
            [-0.4],
        ]
    )
    pair_wins = _expected_pair_wins(theta, count=5000)
    config = ScalableParetoTrackStopConfig(
        box_bound=2.0,
        optimizer_tol=1e-10,
        optimizer_max_iter=5000,
    )
    fit = _fit_bt(pair_wins, config, initial=None)
    _, _, resistances = _profile_geometry(pair_wins, fit["theta_hat"])
    approximation = _quadratic_order_profile(
        fit["theta_hat"][:, 0],
        resistances[0],
        better=1,
        worse=0,
    )
    constrained = fit_box_constrained_bt_coordinate(
        pair_wins[0],
        box_bound=2.0,
        initial=fit["theta_hat"][:, 0],
        max_iter=5000,
        tol=1e-10,
        order_constraints=((1, 0),),
    )
    exact = constrained["nll"] - fit["coordinate_results"][0]["nll"]
    assert approximation > 0.0
    assert exact > 0.0
    assert abs(approximation / exact - 1.0) < 0.15


def test_pareto_profile_statistic_has_a_frontier_witness():
    theta, _ = make_instance(
        "arena_tradeoff_frontier",
        {
            "K": 8,
            "d": 3,
            "s": 3,
            "margin_low": 0.15,
            "margin_high": 0.25,
            "alpha": 0.7,
        },
        seed=7,
    )
    pair_wins = _expected_pair_wins(theta, count=1000)
    config = ScalableParetoTrackStopConfig(box_bound=2.0)
    fit = _fit_bt(pair_wins, config, initial=None)
    _, pseudoinverses, resistances = _profile_geometry(
        pair_wins, fit["theta_hat"]
    )
    result = pareto_quadratic_profile_glr(
        fit["theta_hat"],
        pseudoinverses,
        resistances,
    )
    assert result["statistic"] > 0.0
    assert result["witness"]["kind"] in {
        "drop",
        "add-single-dominator-lower-bound",
    }


def test_smoke_run_stops_and_recovers_strict_pareto_set():
    theta, _ = make_instance(
        "symmetric_hard",
        {"K": 8, "d": 3, "Delta": 0.8},
        seed=11,
    )
    result = run_scalable_pareto_track_and_stop(
        theta,
        ScalableParetoTrackStopConfig(
            delta=0.05,
            box_bound=2.0,
            max_queries=10**10,
            optimizer_tol=1e-8,
            optimizer_max_iter=2000,
        ),
        np.random.default_rng(19),
    )
    assert result["stopped"]
    assert result["error"] is False
    assert result["statistic_type"] == "local-quadratic-profile"
    assert result["glr_is_conservative_lower_bound"] is False
