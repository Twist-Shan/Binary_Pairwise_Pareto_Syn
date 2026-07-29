"""Scalable heuristic Pareto extension of BT-GLR Track-and-Stop.

The ICML 2026 algorithm of Goldberger and Rudi targets scalar top-k
identification.  This module keeps its constrained BT MLE, likelihood-ratio
threshold, tracking, and forced-exploration ingredients, while replacing the
scalar boundary-pair alternative by a Pareto-frontier-changing alternative.

Exact add-to-frontier profiles are combinatorial.  The stopping statistic here
uses a local quadratic profile approximation based on the observed Fisher
information and graph effective resistances.  This is a fixed-confidence-style
heuristic, not a proved delta-correct extension of the scalar top-k theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pilot_focal_bt_mle.matched_fixed_budget import (
    fit_box_constrained_bt_coordinate,
)
from vb_ege.core import center_columns, stable_sigmoid, strict_pareto_set
from vb_ege.metrics import hamming_set_distance, set_error


@dataclass(frozen=True)
class ScalableParetoTrackStopConfig:
    delta: float = 0.05
    box_bound: float = 2.0
    mixture_lambda: float = 0.1
    sigma2: float = 0.25
    burnin_per_cell: int = 2
    growth_factor: float = 1.8
    max_queries: int = 10**18
    mirror_step: float = 0.8
    forced_exploration_power: float = 1.0 / 3.0
    optimizer_tol: float = 1e-8
    optimizer_max_iter: int = 2000
    max_phases: int = 96


def _cell_arrays(K: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_one, j_one = np.triu_indices(K, k=1)
    m = len(i_one)
    r = np.repeat(np.arange(d, dtype=np.int64), m)
    i = np.tile(i_one.astype(np.int64), d)
    j = np.tile(j_one.astype(np.int64), d)
    return r, i, j


def _integer_allocation(weights: np.ndarray, total: int) -> np.ndarray:
    """Round a probability vector to nonnegative integers summing to total."""

    if total < 0:
        raise ValueError("total must be nonnegative")
    if total == 0:
        return np.zeros_like(weights, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    raw = weights * float(total)
    allocation = np.floor(raw).astype(np.int64)
    remainder = int(total - sum(int(value) for value in allocation))
    fractions = raw - allocation
    order = np.argsort(fractions)
    if remainder > 0:
        quotient, extra = divmod(remainder, len(allocation))
        if quotient:
            allocation += quotient
        if extra:
            allocation[order[-extra:]] += 1
    elif remainder < 0:
        need = -remainder
        for index in order:
            if need == 0:
                break
            removable = min(int(allocation[index]), need)
            allocation[index] -= removable
            need -= removable
        if need:
            raise RuntimeError("failed to repair integer allocation")
    if sum(int(value) for value in allocation) != total:
        raise RuntimeError("integer allocation does not sum to total")
    return allocation


def _sample_additions(
    theta: np.ndarray,
    pair_wins: np.ndarray,
    cells: tuple[np.ndarray, np.ndarray, np.ndarray],
    additions: np.ndarray,
    rng: np.random.Generator,
) -> int:
    r, i, j = cells
    additions = np.asarray(additions, dtype=np.int64)
    active = additions > 0
    if not np.any(active):
        return 0
    rr, ii, jj = r[active], i[active], j[active]
    counts = additions[active]
    probabilities = stable_sigmoid(theta[ii, rr] - theta[jj, rr])
    wins_i = rng.binomial(counts, probabilities).astype(np.int64)
    pair_wins[rr, ii, jj] += wins_i
    pair_wins[rr, jj, ii] += counts - wins_i
    return int(sum(int(value) for value in counts))


def _fit_bt(
    pair_wins: np.ndarray,
    config: ScalableParetoTrackStopConfig,
    initial: np.ndarray | None,
) -> dict:
    d, K, _ = pair_wins.shape
    theta_hat = np.zeros((K, d), dtype=float)
    coordinate_results = []
    for r in range(d):
        result = fit_box_constrained_bt_coordinate(
            pair_wins[r],
            box_bound=config.box_bound,
            initial=None if initial is None else initial[:, r],
            max_iter=config.optimizer_max_iter,
            tol=config.optimizer_tol,
        )
        theta_hat[:, r] = result["theta_hat"]
        coordinate_results.append(result)
    return {
        "theta_hat": theta_hat,
        "coordinate_results": coordinate_results,
        "converged_all": all(result["converged"] for result in coordinate_results),
    }


def _profile_geometry(
    pair_wins: np.ndarray,
    theta_hat: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return local Fisher Laplacians, pseudoinverses, and resistances."""

    d, K, _ = pair_wins.shape
    laplacians: list[np.ndarray] = []
    pseudoinverses: list[np.ndarray] = []
    resistances: list[np.ndarray] = []
    for r in range(d):
        counts = pair_wins[r] + pair_wins[r].T
        differences = theta_hat[:, r, None] - theta_hat[None, :, r]
        probabilities = stable_sigmoid(differences)
        fisher_edges = counts * probabilities * (1.0 - probabilities)
        laplacian = np.diag(fisher_edges.sum(axis=1)) - fisher_edges
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        tolerance = max(1.0, float(eigenvalues[-1])) * K * np.finfo(float).eps
        inverse = np.zeros_like(eigenvalues)
        positive = eigenvalues > tolerance
        inverse[positive] = 1.0 / eigenvalues[positive]
        pseudoinverse = (eigenvectors * inverse) @ eigenvectors.T
        diagonal = np.diag(pseudoinverse)
        resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * pseudoinverse
        resistance = np.maximum(resistance, 0.0)
        laplacians.append(laplacian)
        pseudoinverses.append(pseudoinverse)
        resistances.append(resistance)
    return laplacians, pseudoinverses, resistances


def _quadratic_order_profile(
    theta_r: np.ndarray,
    resistance: np.ndarray,
    better: int,
    worse: int,
) -> float:
    """Approximate the profile loss for theta[better] >= theta[worse]."""

    violation = float(theta_r[worse] - theta_r[better])
    if violation <= 0.0:
        return 0.0
    effective_resistance = float(resistance[better, worse])
    if not np.isfinite(effective_resistance) or effective_resistance <= 0.0:
        return 0.0
    return violation * violation / (2.0 * effective_resistance)


def pareto_quadratic_profile_glr(
    theta_hat: np.ndarray,
    pseudoinverses: list[np.ndarray],
    resistances: list[np.ndarray],
) -> dict:
    """Compute the heuristic quadratic frontier-changing profile statistic."""

    K, d = theta_hat.shape
    pareto = set(strict_pareto_set(theta_hat))
    best_value = np.inf
    best_witness: dict | None = None

    # Remove p by allowing j to weakly dominate it in every coordinate.
    for p in sorted(pareto):
        for j in range(K):
            if j == p:
                continue
            coordinate_values = [
                _quadratic_order_profile(
                    theta_hat[:, r],
                    resistances[r],
                    better=j,
                    worse=p,
                )
                for r in range(d)
            ]
            value = float(sum(coordinate_values))
            if value < best_value:
                best_value = value
                best_witness = {
                    "kind": "drop",
                    "arm": p,
                    "competitor": j,
                    "coordinates": tuple(range(d)),
                    "coordinate_values": tuple(coordinate_values),
                }

    # Add q.  Any valid add-q alternative must break every current strict
    # dominator.  The maximum over dominators of their cheapest break is safe.
    for q in sorted(set(range(K)) - pareto):
        dominators = [
            j
            for j in range(K)
            if j != q and np.all(theta_hat[j] > theta_hat[q])
        ]
        dominator_bounds = []
        for j in dominators:
            coordinate_values = [
                _quadratic_order_profile(
                    theta_hat[:, r],
                    resistances[r],
                    better=q,
                    worse=j,
                )
                for r in range(d)
            ]
            coordinate = int(np.argmin(coordinate_values))
            dominator_bounds.append(
                (float(coordinate_values[coordinate]), j, coordinate)
            )
        if dominator_bounds:
            value, j, coordinate = max(dominator_bounds, key=lambda item: item[0])
            if value < best_value:
                best_value = value
                best_witness = {
                    "kind": "add-single-dominator-lower-bound",
                    "arm": q,
                    "competitor": j,
                    "coordinates": (coordinate,),
                    "coordinate_values": (value,),
                    "num_current_dominators": len(dominators),
                }

    if best_witness is None:
        return {
            "statistic": 0.0,
            "witness": None,
            "estimated_pareto": tuple(sorted(pareto)),
        }

    alternative = theta_hat.copy()
    arm = int(best_witness["arm"])
    competitor = int(best_witness["competitor"])
    if best_witness["kind"] == "drop":
        better, worse = competitor, arm
    else:
        better, worse = arm, competitor
    for r in best_witness["coordinates"]:
        gap = float(alternative[worse, r] - alternative[better, r])
        resistance = float(resistances[r][better, worse])
        if gap <= 0.0 or resistance <= 0.0:
            continue
        contrast = np.zeros(K, dtype=float)
        contrast[better] = 1.0
        contrast[worse] = -1.0
        alternative[:, r] += (
            gap * (pseudoinverses[r] @ contrast) / resistance
        )
    best_witness["alternative_theta"] = alternative
    return {
        "statistic": float(best_value),
        "witness": best_witness,
        "estimated_pareto": tuple(sorted(pareto)),
    }


def global_mixture_threshold(
    theta_hat: np.ndarray,
    laplacians: list[np.ndarray],
    config: ScalableParetoTrackStopConfig,
) -> float:
    logdet = 0.0
    scale = config.sigma2 / config.mixture_lambda
    for laplacian in laplacians:
        eigenvalues = np.maximum(np.linalg.eigvalsh(laplacian), 0.0)
        logdet += float(np.log1p(scale * eigenvalues).sum())
    return float(
        np.log(1.0 / config.delta)
        + 0.5 * config.mixture_lambda * np.sum(theta_hat**2)
        + 0.5 * logdet
    )


def _cell_kl(
    theta_hat: np.ndarray,
    alternative: np.ndarray,
    cells: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    r, i, j = cells
    p = np.asarray(stable_sigmoid(theta_hat[i, r] - theta_hat[j, r]))
    q = np.asarray(stable_sigmoid(alternative[i, r] - alternative[j, r]))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    q = np.clip(q, 1e-12, 1.0 - 1e-12)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def run_scalable_pareto_track_and_stop(
    theta: np.ndarray,
    config: ScalableParetoTrackStopConfig,
    rng: np.random.Generator,
) -> dict:
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    centered = center_columns(theta)
    if np.max(np.abs(centered)) >= config.box_bound:
        raise ValueError("centered theta must lie in the interior of the box")
    if not (0.0 < config.delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if config.growth_factor <= 1.0:
        raise ValueError("growth_factor must exceed one")

    cells = _cell_arrays(K, d)
    C = len(cells[0])
    pair_wins = np.zeros((d, K, K), dtype=np.int64)
    burnin = np.full(C, config.burnin_per_cell, dtype=np.int64)
    tau = _sample_additions(theta, pair_wins, cells, burnin, rng)

    log_weights = np.full(C, -np.log(C), dtype=float)
    weights = np.full(C, 1.0 / C, dtype=float)
    theta_initial = None
    history = []
    stopped = False
    final_fit = None
    final_glr = None
    final_threshold = np.inf

    for phase in range(1, config.max_phases + 1):
        final_fit = _fit_bt(pair_wins, config, theta_initial)
        theta_initial = final_fit["theta_hat"]
        profile_laplacians, pseudoinverses, resistances = _profile_geometry(
            pair_wins, final_fit["theta_hat"]
        )
        final_glr = pareto_quadratic_profile_glr(
            final_fit["theta_hat"],
            pseudoinverses,
            resistances,
        )
        count_laplacians = []
        for r in range(d):
            counts = pair_wins[r] + pair_wins[r].T
            count_laplacians.append(np.diag(counts.sum(axis=1)) - counts)
        final_threshold = global_mixture_threshold(
            final_fit["theta_hat"], count_laplacians, config
        )
        stopped = bool(
            final_fit["converged_all"]
            and final_glr["statistic"] >= final_threshold
        )
        witness = final_glr["witness"]
        history.append(
            {
                "phase": phase,
                "tau": int(tau),
                "glr_lower_bound": float(final_glr["statistic"]),
                "statistic_type": "local-quadratic-profile",
                "threshold": float(final_threshold),
                "stopped": stopped,
                "mle_converged_all": bool(final_fit["converged_all"]),
                "witness_kind": None if witness is None else witness["kind"],
            }
        )
        if stopped or tau >= config.max_queries:
            break

        if witness is not None:
            information = _cell_kl(
                final_fit["theta_hat"],
                witness["alternative_theta"],
                cells,
            )
            scale = max(float(np.max(information)), 1e-15)
            step = config.mirror_step / np.sqrt(phase)
            log_weights += step * information / scale
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= weights.sum()

        target_tau = min(
            config.max_queries,
            max(tau + C, int(np.ceil(tau * config.growth_factor))),
        )
        batch = int(target_tau - tau)
        rho = max(float(target_tau), 1.0) ** (
            -config.forced_exploration_power
        )
        mixed = (1.0 - rho) * weights + rho / C
        additions = _integer_allocation(mixed, batch)
        tau += _sample_additions(theta, pair_wins, cells, additions, rng)

    if final_fit is None or final_glr is None:
        raise RuntimeError("Track-and-Stop did not execute a stopping check")
    recommended = strict_pareto_set(final_fit["theta_hat"])
    truth = strict_pareto_set(theta)
    witness = final_glr["witness"]
    return {
        "algorithm": "Pareto BT-GLR Track-and-Stop",
        "recommended": recommended,
        "tau": int(tau),
        "stopped": bool(stopped),
        "error": bool(set_error(recommended, truth)) if stopped else None,
        "hamming": int(hamming_set_distance(recommended, truth)) if stopped else None,
        "num_phases": len(history),
        "num_accepted": len(recommended) if stopped else 0,
        "num_rejected": K - len(recommended) if stopped else 0,
        "num_unresolved": 0 if stopped else K,
        "mle_converged_all": bool(final_fit["converged_all"]),
        "final_glr_lower_bound": float(final_glr["statistic"]),
        "final_glr_threshold": float(final_threshold),
        "final_alternative_kind": None if witness is None else witness["kind"],
        "glr_is_conservative_lower_bound": False,
        "statistic_type": "local-quadratic-profile",
        "history": history,
    }
