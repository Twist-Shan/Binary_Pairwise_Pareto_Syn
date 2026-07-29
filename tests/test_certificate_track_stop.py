from __future__ import annotations

from itertools import product

import numpy as np

from icml_pareto_track_stop.certificate_track_stop import (
    ParetoCertificateTrackStopConfig,
    _linear_minimum,
    pareto_certificate_profile,
    run_pareto_certificate_track_and_stop,
)
from icml_pareto_track_stop.scalable_track_stop import _fit_bt
from vb_ege.core import stable_sigmoid, strict_pareto_set
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


def test_star_linear_oracle_matches_exhaustive_small_polytope():
    grid = np.linspace(-1.0, 1.0, 9)
    feasible = np.array(
        [
            point
            for point in product(grid, repeat=3)
            if abs(sum(point)) <= 1e-12 and point[0] >= point[1]
        ]
    )
    gradients = (
        np.array([0.7, -1.2, 0.4]),
        np.array([-0.3, 1.1, -0.8]),
        np.array([1.0, 1.0, -2.0]),
    )
    for gradient in gradients:
        expected = float(np.min(feasible @ gradient))
        actual = _linear_minimum(gradient, 1.0, ((0, 1),))
        assert abs(actual - expected) <= 1e-10


def test_exact_add_enumeration_brackets_profile_and_returns_feasible_witness():
    theta, _ = make_instance(
        "arena_tradeoff_frontier",
        {
            "K": 6,
            "d": 2,
            "s": 2,
            "margin_low": 0.18,
            "margin_high": 0.28,
        },
        seed=31,
    )
    pair_wins = _expected_pair_wins(theta, count=1000)
    config = ParetoCertificateTrackStopConfig(
        box_bound=2.0,
        optimizer_tol=1e-10,
        optimizer_max_iter=5000,
        max_exact_add_assignments=100,
    )
    fit = _fit_bt(pair_wins, config, initial=None)
    profile = pareto_certificate_profile(
        pair_wins,
        fit["theta_hat"],
        fit["coordinate_results"],
        config,
    )

    assert profile["assignment_search_exact"]
    assert profile["lower"] <= profile["estimate"] + 1e-8
    assert profile["estimate"] <= profile["upper"] + 1e-8
    assert profile["num_constrained_fits"] > 0
    assert profile["all_profile_fits_converged"]
    assert profile["estimated_pareto"] == strict_pareto_set(fit["theta_hat"])


def test_certificate_pool_keeps_screening_lower_separate_from_feasible_upper():
    theta, _ = make_instance(
        "arena_tradeoff_frontier",
        {
            "K": 7,
            "d": 3,
            "s": 3,
            "margin_low": 0.18,
            "margin_high": 0.28,
        },
        seed=32,
    )
    pair_wins = _expected_pair_wins(theta, count=800)
    exact_config = ParetoCertificateTrackStopConfig(
        box_bound=2.0,
        max_exact_add_assignments=1000,
        optimizer_tol=1e-9,
        optimizer_max_iter=4000,
    )
    fit = _fit_bt(pair_wins, exact_config, initial=None)
    exact = pareto_certificate_profile(
        pair_wins,
        fit["theta_hat"],
        fit["coordinate_results"],
        exact_config,
    )
    pooled_config = ParetoCertificateTrackStopConfig(
        box_bound=2.0,
        max_exact_add_assignments=1,
        optimizer_tol=1e-9,
        optimizer_max_iter=4000,
    )
    pooled = pareto_certificate_profile(
        pair_wins,
        fit["theta_hat"],
        fit["coordinate_results"],
        pooled_config,
    )

    assert exact["assignment_search_exact"]
    assert not pooled["assignment_search_exact"]
    assert pooled["lower"] <= exact["upper"] + 1e-7
    assert pooled["lower"] <= pooled["upper"] + 1e-7
    assert pooled["alternative_theta"] is not None


def test_certificate_smoke_run_stops_and_recovers_strict_pareto_set():
    theta, _ = make_instance(
        "symmetric_hard",
        {"K": 4, "d": 2, "Delta": 1.2},
        seed=33,
    )
    result = run_pareto_certificate_track_and_stop(
        theta,
        ParetoCertificateTrackStopConfig(
            delta=0.10,
            box_bound=2.0,
            growth_factor=2.0,
            max_queries=10**9,
            optimizer_tol=1e-8,
            optimizer_max_iter=3000,
            max_exact_add_assignments=100,
            max_phases=40,
        ),
        np.random.default_rng(34),
    )

    assert result["stopped"]
    assert result["error"] is False
    assert result["final_assignment_search_exact"]
    assert result["glr_is_conservative_lower_bound"]
    assert result["statistic_type"] == "pareto-certificate-profile-lower-bound"


def test_screened_large_front_path_stops_with_a_joint_certificate():
    theta, _ = make_instance(
        "symmetric_hard",
        {"K": 6, "d": 3, "Delta": 1.0},
        seed=35,
    )
    result = run_pareto_certificate_track_and_stop(
        theta,
        ParetoCertificateTrackStopConfig(
            delta=0.10,
            box_bound=2.0,
            burnin_per_cell=3,
            growth_factor=3.0,
            max_queries=10**9,
            max_phases=24,
            optimizer_tol=1e-7,
            optimizer_max_iter=1500,
            exact_profile_max_arms=4,
        ),
        np.random.default_rng(36),
    )

    assert result["stopped"]
    assert result["error"] is False
    assert result["final_profile_mode"] == "screened-joint-certificate"
    assert not result["stopping_statistic_is_certified"]
    assert "screened-joint-certificate" in result["final_alternative_kind"]
    assert result["statistic_type"] == "pareto-screened-joint-certificate"
