"""Matched-transcript focal Borda and box-constrained BT-MLE estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vb_ege.core import stable_sigmoid, strict_pareto_set


@dataclass
class TranscriptStats:
    focal_counts: np.ndarray
    focal_wins: np.ndarray
    pair_wins: np.ndarray
    num_queries: int = 0

    @classmethod
    def zeros(cls, K: int, d: int) -> "TranscriptStats":
        return cls(
            focal_counts=np.zeros((K, d), dtype=np.int64),
            focal_wins=np.zeros((K, d), dtype=np.int64),
            pair_wins=np.zeros((d, K, K), dtype=np.int64),
        )


def add_focal_random_opponent_queries(
    theta: np.ndarray,
    stats: TranscriptStats,
    num_queries: int,
    rng: np.random.Generator,
) -> None:
    """Append i.i.d. focal/random-opponent observations to sufficient statistics."""

    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    num_queries = int(num_queries)
    if num_queries < 0:
        raise ValueError("num_queries must be nonnegative")
    if num_queries == 0:
        return

    focal = rng.integers(0, K, size=num_queries)
    objective = rng.integers(0, d, size=num_queries)
    opponent = rng.integers(0, K - 1, size=num_queries)
    opponent = np.where(opponent >= focal, opponent + 1, opponent)
    probability = stable_sigmoid(theta[focal, objective] - theta[opponent, objective])
    outcome = rng.binomial(1, probability).astype(np.int64)

    np.add.at(stats.focal_counts, (focal, objective), 1)
    np.add.at(stats.focal_wins, (focal, objective), outcome)

    winner = np.where(outcome == 1, focal, opponent)
    loser = np.where(outcome == 1, opponent, focal)
    np.add.at(stats.pair_wins, (objective, winner, loser), 1)
    stats.num_queries += num_queries


def focal_borda_estimate(stats: TranscriptStats) -> np.ndarray:
    return np.divide(
        stats.focal_wins,
        stats.focal_counts,
        out=np.full(stats.focal_counts.shape, 0.5, dtype=float),
        where=stats.focal_counts > 0,
    )


def focal_borda_recommendation(stats: TranscriptStats) -> tuple[int, ...]:
    return strict_pareto_set(focal_borda_estimate(stats))


def _coordinate_nll_and_grad(theta: np.ndarray, pair_wins: np.ndarray) -> tuple[float, np.ndarray]:
    K = len(theta)
    upper_i, upper_j = np.triu_indices(K, k=1)
    wij = pair_wins[upper_i, upper_j].astype(float)
    wji = pair_wins[upper_j, upper_i].astype(float)
    difference = theta[upper_i] - theta[upper_j]
    value = np.sum(
        wij * np.logaddexp(0.0, -difference)
        + wji * np.logaddexp(0.0, difference)
    )
    residual = (wij + wji) * stable_sigmoid(difference) - wij
    gradient = np.zeros(K, dtype=float)
    np.add.at(gradient, upper_i, residual)
    np.add.at(gradient, upper_j, -residual)
    return float(value), gradient


def _project_box_zero_sum(
    value: np.ndarray,
    box_bound: float,
    tied_pair: tuple[int, int] | None = None,
) -> np.ndarray:
    """Project onto the box/gauge set, optionally imposing theta_i=theta_j."""

    value = np.asarray(value, dtype=float)
    base = value.copy()
    if tied_pair is not None:
        i, j = tied_pair
        tied_value = 0.5 * (base[i] + base[j])
        base[i] = tied_value
        base[j] = tied_value
    centered = base - float(base.mean())
    if float(np.max(np.abs(centered))) <= box_bound:
        return centered
    lower = float(np.min(base) - box_bound)
    upper = float(np.max(base) + box_bound)
    for _ in range(80):
        shift = 0.5 * (lower + upper)
        projected = np.clip(base - shift, -box_bound, box_bound)
        if float(projected.sum()) > 0.0:
            lower = shift
        else:
            upper = shift
    projected = np.clip(base - 0.5 * (lower + upper), -box_bound, box_bound)
    if tied_pair is not None:
        i, j = tied_pair
        projected[i] = projected[j] = 0.5 * (projected[i] + projected[j])
    return projected


def _project_box_zero_sum_orders(
    value: np.ndarray,
    box_bound: float,
    order_constraints: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Project onto box, zero-sum, and a single star-shaped order cone."""

    better_arms = {better for better, _ in order_constraints}
    if len(better_arms) != 1:
        raise ValueError("order constraints must form a single star")
    better = next(iter(better_arms))
    worse_arms = tuple(worse for _, worse in order_constraints)
    ordered = _project_order_star(value, better, worse_arms)
    # The order cone is translation invariant, and clipping is monotone, so
    # the box/zero-sum projection preserves every star inequality.
    return _project_box_zero_sum(ordered, box_bound)


def _project_order_star(
    value: np.ndarray,
    better: int,
    worse_arms: tuple[int, ...],
) -> np.ndarray:
    """Project onto x_better >= x_worse for every arm in a star."""

    projected = np.asarray(value, dtype=float).copy()
    ordered = sorted(worse_arms, key=lambda arm: projected[arm], reverse=True)
    active = []
    boundary = float(projected[better])
    for worse in ordered:
        if projected[worse] <= boundary:
            break
        active.append(worse)
        boundary = float(
            (value[better] + sum(value[arm] for arm in active))
            / (1 + len(active))
        )
    if active:
        projected[better] = boundary
        projected[active] = boundary
    return projected


def _gradient_lipschitz_bound(pair_wins: np.ndarray) -> float:
    counts = pair_wins + pair_wins.T
    weights = 0.25 * counts
    laplacian = np.diag(weights.sum(axis=1)) - weights
    largest = float(np.linalg.eigvalsh(laplacian)[-1])
    return max(largest, 1.0)


def fit_box_constrained_bt_coordinate(
    pair_wins: np.ndarray,
    box_bound: float,
    initial: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-8,
    tied_pair: tuple[int, int] | None = None,
    order_constraints: tuple[tuple[int, int], ...] = (),
) -> dict:
    """Fit an unpenalized BT MLE under sum(theta)=0 and ||theta||_inf <= B."""

    pair_wins = np.asarray(pair_wins, dtype=float)
    K = pair_wins.shape[0]
    if pair_wins.shape != (K, K):
        raise ValueError("pair_wins must be square")
    if box_bound <= 0:
        raise ValueError("box_bound must be positive")
    if tied_pair is not None and order_constraints:
        raise ValueError("tied_pair and order_constraints cannot be combined")

    def project(value):
        if order_constraints:
            return _project_box_zero_sum_orders(
                value,
                box_bound,
                order_constraints=order_constraints,
            )
        return _project_box_zero_sum(value, box_bound, tied_pair=tied_pair)

    if initial is None:
        x0 = np.zeros(K, dtype=float)
    else:
        x0 = np.asarray(initial, dtype=float).copy()
        if x0.shape != (K,):
            raise ValueError("initial has the wrong shape")
    x = project(x0)

    accelerated = x.copy()
    momentum = 1.0
    step = 1.0 / _gradient_lipschitz_bound(pair_wins)
    converged = False
    objective, _ = _coordinate_nll_and_grad(x, pair_wins)

    for iteration in range(1, int(max_iter) + 1):
        _, gradient = _coordinate_nll_and_grad(accelerated, pair_wins)
        candidate = project(accelerated - step * gradient)
        candidate_objective, _ = _coordinate_nll_and_grad(candidate, pair_wins)

        # Monotone restart keeps FISTA stable near an active box constraint.
        if candidate_objective > objective + 1e-12:
            accelerated = x
            momentum = 1.0
            _, gradient = _coordinate_nll_and_grad(accelerated, pair_wins)
            candidate = project(accelerated - step * gradient)
            candidate_objective, _ = _coordinate_nll_and_grad(candidate, pair_wins)

        change = float(np.linalg.norm(candidate - x, ord=np.inf))
        scale = 1.0 + float(np.linalg.norm(x, ord=np.inf))
        previous = x
        x = candidate
        objective = candidate_objective
        if change <= tol * scale:
            converged = True
            break

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = x + ((momentum - 1.0) / next_momentum) * (x - previous)
        accelerated = project(accelerated)
        momentum = next_momentum
    else:
        iteration = int(max_iter)

    theta_hat = project(x)
    feasible = (
        abs(float(theta_hat.sum())) <= 1e-6
        and float(np.max(np.abs(theta_hat))) <= box_bound + 1e-6
    )
    if tied_pair is not None:
        i, j = tied_pair
        feasible = feasible and abs(float(theta_hat[i] - theta_hat[j])) <= 1e-6
    if order_constraints:
        feasible = feasible and all(
            theta_hat[better] + 1e-6 >= theta_hat[worse]
            for better, worse in order_constraints
        )
    return {
        "theta_hat": theta_hat,
        "converged": bool(converged and feasible),
        "message": "projected-gradient tolerance reached" if converged else "max_iter reached",
        "niter": int(iteration),
        "nll": float(objective),
    }


def fit_box_constrained_bt(
    pair_wins: np.ndarray,
    box_bound: float,
    initial: np.ndarray | None = None,
) -> dict:
    pair_wins = np.asarray(pair_wins, dtype=float)
    d, K, K_again = pair_wins.shape
    if K != K_again:
        raise ValueError("pair_wins must have shape (d, K, K)")
    if initial is not None and np.asarray(initial).shape != (K, d):
        raise ValueError("initial must have shape (K, d)")

    theta_hat = np.zeros((K, d), dtype=float)
    coordinate_results = []
    for r in range(d):
        init_r = None if initial is None else np.asarray(initial)[:, r]
        result = fit_box_constrained_bt_coordinate(
            pair_wins[r],
            box_bound=box_bound,
            initial=init_r,
        )
        theta_hat[:, r] = result["theta_hat"]
        coordinate_results.append(result)
    return {
        "theta_hat": theta_hat,
        "recommended": strict_pareto_set(theta_hat),
        "converged_all": all(result["converged"] for result in coordinate_results),
        "coordinate_results": coordinate_results,
    }
