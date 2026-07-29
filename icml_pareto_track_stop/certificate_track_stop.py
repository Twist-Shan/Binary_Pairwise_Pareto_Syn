"""Certificate-based Pareto extension of BT-GLR Track-and-Stop.

The scalar ICML Track-and-Stop construction is retained at the BT likelihood,
tracking, and mixture-threshold levels.  Its scalar top-k alternative is
replaced by the exact Pareto-frontier alternative decomposition:

* remove a current Pareto arm by letting another current Pareto arm dominate it;
* add a non-Pareto arm by assigning, for every current Pareto arm, one
  coordinate on which the candidate is at least as good.

For small fronts all add certificates are enumerated.  For larger fronts the
implementation combines a valid screening lower bound with a pool of feasible
certificates.  The resulting method is an explicit, reproducible heuristic
extension; the scalar paper's delta-correctness theorem does not automatically
extend to this multi-objective stopping rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from pilot_focal_bt_mle.matched_fixed_budget import (
    _coordinate_nll_and_grad,
    fit_box_constrained_bt_coordinate,
)
from vb_ege.core import center_columns, stable_sigmoid, strict_pareto_set
from vb_ege.metrics import hamming_set_distance, set_error

from .scalable_track_stop import (
    _cell_arrays,
    _cell_kl,
    _fit_bt,
    _integer_allocation,
    _profile_geometry,
    _quadratic_order_profile,
    _sample_additions,
    global_mixture_threshold,
)


@dataclass(frozen=True)
class ParetoCertificateTrackStopConfig:
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
    max_exact_add_assignments: int = 4096
    max_certificate_pool: int = 64
    certificate_local_search_passes: int = 2
    profile_interval_atol: float = 1e-7
    profile_interval_rtol: float = 1e-5
    exact_profile_max_arms: int = 8
    screened_certificate_pool: int = 8


def _linear_minimum(
    gradient: np.ndarray,
    box_bound: float,
    constraints: tuple[tuple[int, int], ...],
) -> float:
    """Minimize a linear form over the box/gauge/order polytope."""

    gradient = np.asarray(gradient, dtype=float)
    K = len(gradient)

    def bounded_sum_minimum(
        coefficients: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target: float,
    ) -> float:
        if target < float(lower.sum()) - 1e-9:
            return np.inf
        if target > float(upper.sum()) + 1e-9:
            return np.inf
        value = lower.copy()
        remaining = max(0.0, target - float(lower.sum()))
        order = np.argsort(coefficients, kind="stable")
        for index in order:
            addition = min(float(upper[index] - lower[index]), remaining)
            value[index] += addition
            remaining -= addition
            if remaining <= 1e-10:
                break
        if remaining > 1e-7:
            return np.inf
        return float(np.dot(coefficients, value))

    if not constraints:
        lower = np.full(K, -box_bound, dtype=float)
        upper = np.full(K, box_bound, dtype=float)
        return bounded_sum_minimum(gradient, lower, upper, 0.0)

    better_arms = {better for better, _ in constraints}
    if len(better_arms) != 1:
        raise ValueError("order constraints must form a single star")
    better = next(iter(better_arms))
    worse = {worse for _, worse in constraints}
    remaining_arms = [arm for arm in range(K) if arm != better]
    remaining_gradient = gradient[remaining_arms]
    sorted_arms = [
        remaining_arms[index]
        for index in np.argsort(remaining_gradient, kind="stable")
    ]

    # Conditional on x_better=t, the remaining coordinates solve a bounded
    # continuous-knapsack problem. Its value is piecewise linear in t. The
    # breakpoints occur when the required mass equals a sorted prefix's total
    # capacity, so evaluating those breakpoints and the box endpoints is exact.
    candidates = {-box_bound, box_bound}
    num_worse = 0
    constant_capacity = 0.0
    for arm in sorted_arms:
        if arm in worse:
            num_worse += 1
            constant_capacity += box_bound
        else:
            constant_capacity += 2.0 * box_bound
        breakpoint = (
            (K - 1) * box_bound - constant_capacity
        ) / (num_worse + 1)
        if -box_bound - 1e-12 <= breakpoint <= box_bound + 1e-12:
            candidates.add(float(np.clip(breakpoint, -box_bound, box_bound)))

    best = np.inf
    for t in candidates:
        lower = np.full(K - 1, -box_bound, dtype=float)
        upper = np.array(
            [t if arm in worse else box_bound for arm in remaining_arms],
            dtype=float,
        )
        conditional = bounded_sum_minimum(
            remaining_gradient,
            lower,
            upper,
            -t,
        )
        if np.isfinite(conditional):
            best = min(best, float(gradient[better] * t + conditional))
    if not np.isfinite(best):
        raise RuntimeError("profile linear-minimization polytope is infeasible")
    return float(best)


def _objective_bounds(
    pair_wins_r: np.ndarray,
    theta_r: np.ndarray,
    box_bound: float,
    constraints: tuple[tuple[int, int], ...],
) -> tuple[float, float, float]:
    """Return a convex objective's lower estimate, value, and FW gap."""

    value, gradient = _coordinate_nll_and_grad(theta_r, pair_wins_r)
    linear_minimum = _linear_minimum(gradient, box_bound, constraints)
    fw_gap = max(0.0, float(np.dot(gradient, theta_r)) - linear_minimum)
    return max(0.0, value - fw_gap), float(value), float(fw_gap)


def _canonical_constraints(
    constraints: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(set((int(better), int(worse)) for better, worse in constraints)))


def _profile_coordinate(
    pair_wins_r: np.ndarray,
    theta_hat_r: np.ndarray,
    unconstrained_result: dict,
    unconstrained_lower: float,
    constraints: tuple[tuple[int, int], ...],
    config: ParetoCertificateTrackStopConfig,
) -> dict:
    """Bracket one coordinate-wise constrained profile likelihood."""

    constraints = _canonical_constraints(constraints)
    already_feasible = all(
        theta_hat_r[better] >= theta_hat_r[worse]
        for better, worse in constraints
    )
    if already_feasible:
        constrained_theta = theta_hat_r.copy()
        constrained_value = float(unconstrained_result["nll"])
        constrained_lower = unconstrained_lower
        converged = bool(unconstrained_result["converged"])
        constrained_gap = constrained_value - constrained_lower
    else:
        result = fit_box_constrained_bt_coordinate(
            pair_wins_r,
            box_bound=config.box_bound,
            initial=theta_hat_r,
            max_iter=config.optimizer_max_iter,
            tol=config.optimizer_tol,
            order_constraints=constraints,
        )
        constrained_theta = np.asarray(result["theta_hat"], dtype=float)
        constrained_lower, constrained_value, constrained_gap = _objective_bounds(
            pair_wins_r,
            constrained_theta,
            config.box_bound,
            constraints,
        )
        converged = bool(result["converged"])

    unconstrained_upper = float(unconstrained_result["nll"])
    lower = max(0.0, constrained_lower - unconstrained_upper)
    estimate = max(0.0, constrained_value - unconstrained_upper)
    upper = max(estimate, constrained_value - unconstrained_lower)
    return {
        "lower": float(lower),
        "estimate": float(estimate),
        "upper": float(upper),
        "theta": constrained_theta,
        "converged": converged,
        "fw_gap": float(constrained_gap),
    }


def _branch_from_grouped_constraints(
    grouped: list[list[tuple[int, int]]],
    coordinate_profile,
    theta_hat: np.ndarray,
) -> dict:
    candidate = theta_hat.copy()
    lower = 0.0
    estimate = 0.0
    upper = 0.0
    converged = True
    max_fw_gap = 0.0
    for r, constraints in enumerate(grouped):
        result = coordinate_profile(r, _canonical_constraints(constraints))
        lower += result["lower"]
        estimate += result["estimate"]
        upper += result["upper"]
        candidate[:, r] = result["theta"]
        converged = converged and result["converged"]
        max_fw_gap = max(max_fw_gap, result["fw_gap"])
    return {
        "lower": float(lower),
        "estimate": float(estimate),
        "upper": float(upper),
        "alternative_theta": candidate,
        "converged": bool(converged),
        "max_coordinate_fw_gap": float(max_fw_gap),
    }


def _assignment_branch(
    q: int,
    pareto: tuple[int, ...],
    assignment: tuple[int, ...],
    d: int,
    coordinate_profile,
    theta_hat: np.ndarray,
) -> dict:
    grouped: list[list[tuple[int, int]]] = [[] for _ in range(d)]
    for p, r in zip(pareto, assignment):
        grouped[int(r)].append((q, p))
    return _branch_from_grouped_constraints(
        grouped,
        coordinate_profile,
        theta_hat,
    )


def _seed_assignments(
    q: int,
    pareto: tuple[int, ...],
    d: int,
    coordinate_profile,
    pool: set[tuple[int, ...]],
) -> set[tuple[int, ...]]:
    assignments = set(pool)
    assignments.update(tuple([r] * len(pareto)) for r in range(d))
    cheapest = []
    for p in pareto:
        values = [
            coordinate_profile(r, ((q, p),))["estimate"]
            for r in range(d)
        ]
        cheapest.append(int(np.argmin(values)))
    assignments.add(tuple(cheapest))
    return assignments


def _search_certificate_assignments(
    q: int,
    pareto: tuple[int, ...],
    d: int,
    coordinate_profile,
    theta_hat: np.ndarray,
    pool: set[tuple[int, ...]],
    passes: int,
) -> tuple[list[tuple[tuple[int, ...], dict]], set[tuple[int, ...]]]:
    assignments = _seed_assignments(q, pareto, d, coordinate_profile, pool)
    evaluated: dict[tuple[int, ...], dict] = {}

    def evaluate(assignment: tuple[int, ...]) -> dict:
        if assignment not in evaluated:
            evaluated[assignment] = _assignment_branch(
                q,
                pareto,
                assignment,
                d,
                coordinate_profile,
                theta_hat,
            )
        return evaluated[assignment]

    for assignment in list(assignments):
        evaluate(assignment)
    current = min(evaluated, key=lambda item: evaluated[item]["estimate"])
    for _ in range(max(0, passes)):
        neighborhood = {current}
        for index in range(len(pareto)):
            for r in range(d):
                proposal = list(current)
                proposal[index] = r
                neighborhood.add(tuple(proposal))
        for assignment in neighborhood:
            evaluate(assignment)
        next_assignment = min(
            neighborhood,
            key=lambda item: evaluated[item]["estimate"],
        )
        assignments.update(neighborhood)
        if next_assignment == current:
            break
        current = next_assignment
    ranked = sorted(evaluated.items(), key=lambda item: item[1]["estimate"])
    return ranked, assignments


def pareto_certificate_profile(
    pair_wins: np.ndarray,
    theta_hat: np.ndarray,
    coordinate_results: list[dict],
    config: ParetoCertificateTrackStopConfig,
    certificate_pool: dict[
        tuple[tuple[int, ...], int], set[tuple[int, ...]]
    ] | None = None,
) -> dict:
    """Bracket the BT likelihood distance to a changed Pareto frontier."""

    K, d = theta_hat.shape
    pareto = tuple(sorted(strict_pareto_set(theta_hat)))
    nonpareto = tuple(sorted(set(range(K)) - set(pareto)))
    certificate_pool = {} if certificate_pool is None else certificate_pool

    unconstrained_lowers = []
    unconstrained_fw_gaps = []
    for r in range(d):
        lower, _, gap = _objective_bounds(
            pair_wins[r],
            theta_hat[:, r],
            config.box_bound,
            (),
        )
        unconstrained_lowers.append(lower)
        unconstrained_fw_gaps.append(gap)

    cache: dict[tuple[int, tuple[tuple[int, int], ...]], dict] = {}

    def coordinate_profile(
        r: int,
        constraints: tuple[tuple[int, int], ...],
    ) -> dict:
        canonical = _canonical_constraints(constraints)
        key = (int(r), canonical)
        if key not in cache:
            cache[key] = _profile_coordinate(
                pair_wins[r],
                theta_hat[:, r],
                coordinate_results[r],
                unconstrained_lowers[r],
                canonical,
                config,
            )
        return cache[key]

    branches: list[dict] = []

    # By Pareto-front geometry it is sufficient to consider domination by
    # another arm on the current frontier.
    for p in pareto:
        for p_prime in pareto:
            if p_prime == p:
                continue
            grouped = [[(p_prime, p)] for _ in range(d)]
            branch = _branch_from_grouped_constraints(
                grouped,
                coordinate_profile,
                theta_hat,
            )
            branch.update(
                {
                    "kind": f"drop:{p}:via-pareto:{p_prime}",
                    "assignment_search_exact": True,
                }
            )
            branches.append(branch)

    add_summaries = []
    for q in nonpareto:
        single_profiles = {
            (p, r): coordinate_profile(r, ((q, p),))
            for p in pareto
            for r in range(d)
        }
        screening_lower = max(
            min(single_profiles[(p, r)]["lower"] for r in range(d))
            for p in pareto
        )
        assignment_count = d ** len(pareto)
        exact_search = assignment_count <= config.max_exact_add_assignments
        key = (pareto, q)
        if exact_search:
            assignment_items = [
                (
                    assignment,
                    _assignment_branch(
                        q,
                        pareto,
                        assignment,
                        d,
                        coordinate_profile,
                        theta_hat,
                    ),
                )
                for assignment in product(range(d), repeat=len(pareto))
            ]
            discovered = {item[0] for item in assignment_items}
        else:
            assignment_items, discovered = _search_certificate_assignments(
                q,
                pareto,
                d,
                coordinate_profile,
                theta_hat,
                certificate_pool.get(key, set()),
                config.certificate_local_search_passes,
            )

        ranked = sorted(
            assignment_items,
            key=lambda item: item[1]["estimate"],
        )
        best_assignment, best_feasible = ranked[0]
        retained = {
            assignment for assignment, _ in ranked[: config.max_certificate_pool]
        }
        retained.update(sorted(discovered)[: config.max_certificate_pool])
        certificate_pool[key] = set(
            sorted(retained)[: config.max_certificate_pool]
        )

        if exact_search:
            branch_lower = min(item[1]["lower"] for item in assignment_items)
        else:
            branch_lower = float(screening_lower)
        branch = dict(best_feasible)
        branch.update(
            {
                "lower": float(branch_lower),
                "kind": (
                    f"add:{q}:exact-witness-enumeration"
                    if exact_search
                    else f"add:{q}:certificate-pool"
                ),
                "assignment": tuple(int(r) for r in best_assignment),
                "assignment_search_exact": bool(exact_search),
            }
        )
        branches.append(branch)
        add_summaries.append(
            {
                "arm": int(q),
                "screening_lower": float(screening_lower),
                "assignment_count": int(assignment_count),
                "assignments_evaluated": int(len(assignment_items)),
                "assignment_search_exact": bool(exact_search),
                "best_assignment": tuple(int(r) for r in best_assignment),
            }
        )

    if not branches:
        raise RuntimeError("no Pareto-frontier-changing branch was generated")

    lower = min(branch["lower"] for branch in branches)
    estimated_best = min(branches, key=lambda branch: branch["estimate"])
    upper = min(branch["upper"] for branch in branches)
    interval = max(0.0, upper - lower)
    tolerance = config.profile_interval_atol + config.profile_interval_rtol * max(
        1.0,
        abs(upper),
    )
    assignment_search_exact = all(
        branch["assignment_search_exact"] for branch in branches
    )
    profile_interval_closed = bool(assignment_search_exact and interval <= tolerance)
    return {
        "lower": float(lower),
        "estimate": float(estimated_best["estimate"]),
        "upper": float(upper),
        "interval_width": float(interval),
        "profile_interval_closed": profile_interval_closed,
        "assignment_search_exact": bool(assignment_search_exact),
        "alternative_theta": estimated_best["alternative_theta"],
        "alternative_kind": estimated_best["kind"],
        "estimated_pareto": pareto,
        "num_constrained_fits": int(len(cache)),
        "all_profile_fits_converged": bool(
            all(item["converged"] for item in cache.values())
        ),
        "max_coordinate_fw_gap": float(
            max(
                unconstrained_fw_gaps
                + [item["fw_gap"] for item in cache.values()]
            )
        ),
        "add_summaries": add_summaries,
        "profile_mode": "exact-certificate-enumeration",
        "stopping_statistic_is_certified": True,
    }


def pareto_screened_certificate_profile(
    pair_wins: np.ndarray,
    theta_hat: np.ndarray,
    coordinate_results: list[dict],
    config: ParetoCertificateTrackStopConfig,
    certificate_pool: dict[
        tuple[tuple[int, ...], int], set[tuple[int, ...]]
    ] | None = None,
) -> dict:
    """Scalable large-front oracle with global quadratic screening.

    The screening statistic is evaluated over every remove branch and every
    add arm, while the original BT profile is solved only for the selected
    joint certificate.  This is a computational heuristic: the local Fisher
    screening value is not a proved lower bound on the exact BT profile.
    """

    K, d = theta_hat.shape
    pareto = tuple(sorted(strict_pareto_set(theta_hat)))
    nonpareto = tuple(sorted(set(range(K)) - set(pareto)))
    certificate_pool = {} if certificate_pool is None else certificate_pool
    _, _, resistances = _profile_geometry(pair_wins, theta_hat)

    candidates: list[dict] = []
    for p in pareto:
        for p_prime in pareto:
            if p == p_prime:
                continue
            coordinate_values = tuple(
                _quadratic_order_profile(
                    theta_hat[:, r],
                    resistances[r],
                    better=p_prime,
                    worse=p,
                )
                for r in range(d)
            )
            candidates.append(
                {
                    "screening": float(sum(coordinate_values)),
                    "kind": f"drop:{p}:via-pareto:{p_prime}",
                    "grouped": [[(p_prime, p)] for _ in range(d)],
                    "assignment": None,
                }
            )

    add_summaries = []
    for q in nonpareto:
        single = np.empty((len(pareto), d), dtype=float)
        for p_index, p in enumerate(pareto):
            for r in range(d):
                single[p_index, r] = _quadratic_order_profile(
                    theta_hat[:, r],
                    resistances[r],
                    better=q,
                    worse=p,
                )
        screening = float(np.max(np.min(single, axis=1)))
        key = (pareto, q)
        assignments = set(certificate_pool.get(key, set()))
        assignments.add(tuple(int(r) for r in np.argmin(single, axis=1)))
        assignments.update(tuple([r] * len(pareto)) for r in range(d))
        ranked = sorted(
            assignments,
            key=lambda assignment: float(
                sum(
                    single[p_index, int(r)]
                    for p_index, r in enumerate(assignment)
                )
            ),
        )
        retained = ranked[: max(1, config.screened_certificate_pool)]
        certificate_pool[key] = set(retained)
        assignment = retained[0]
        grouped: list[list[tuple[int, int]]] = [[] for _ in range(d)]
        for p, r in zip(pareto, assignment):
            grouped[int(r)].append((q, p))
        candidates.append(
            {
                "screening": screening,
                "kind": f"add:{q}:screened-joint-certificate",
                "grouped": grouped,
                "assignment": assignment,
            }
        )
        add_summaries.append(
            {
                "arm": int(q),
                "screening_lower": screening,
                "assignment_count": int(d ** len(pareto)),
                "assignments_evaluated": int(len(retained)),
                "assignment_search_exact": False,
                "best_assignment": tuple(int(r) for r in assignment),
            }
        )

    if not candidates:
        raise RuntimeError("no Pareto-frontier-changing branch was generated")
    selected = min(candidates, key=lambda item: item["screening"])

    unconstrained_lowers = []
    unconstrained_fw_gaps = []
    for r in range(d):
        lower, _, gap = _objective_bounds(
            pair_wins[r],
            theta_hat[:, r],
            config.box_bound,
            (),
        )
        unconstrained_lowers.append(lower)
        unconstrained_fw_gaps.append(gap)

    cache: dict[tuple[int, tuple[tuple[int, int], ...]], dict] = {}

    def coordinate_profile(
        r: int,
        constraints: tuple[tuple[int, int], ...],
    ) -> dict:
        canonical = _canonical_constraints(constraints)
        key = (int(r), canonical)
        if key not in cache:
            cache[key] = _profile_coordinate(
                pair_wins[r],
                theta_hat[:, r],
                coordinate_results[r],
                unconstrained_lowers[r],
                canonical,
                config,
            )
        return cache[key]

    active = _branch_from_grouped_constraints(
        selected["grouped"],
        coordinate_profile,
        theta_hat,
    )
    stopping_statistic = min(float(selected["screening"]), active["lower"])
    return {
        "lower": float(stopping_statistic),
        "estimate": float(active["estimate"]),
        "upper": float(active["upper"]),
        "interval_width": float(max(0.0, active["upper"] - active["lower"])),
        "profile_interval_closed": False,
        "assignment_search_exact": False,
        "alternative_theta": active["alternative_theta"],
        "alternative_kind": selected["kind"],
        "estimated_pareto": pareto,
        "num_constrained_fits": int(len(cache)),
        "all_profile_fits_converged": bool(
            all(item["converged"] for item in cache.values())
        ),
        "max_coordinate_fw_gap": float(
            max(
                unconstrained_fw_gaps
                + [item["fw_gap"] for item in cache.values()]
            )
        ),
        "add_summaries": add_summaries,
        "profile_mode": "screened-joint-certificate",
        "stopping_statistic_is_certified": False,
        "screening_statistic": float(selected["screening"]),
        "active_certificate_lower": float(active["lower"]),
    }


def _count_laplacians(pair_wins: np.ndarray) -> list[np.ndarray]:
    d = pair_wins.shape[0]
    result = []
    for r in range(d):
        counts = pair_wins[r] + pair_wins[r].T
        result.append(np.diag(counts.sum(axis=1)) - counts)
    return result


def _tracked_batch_allocation(
    cumulative_target: np.ndarray,
    current_counts: np.ndarray,
    mixed_weights: np.ndarray,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative_target = cumulative_target + batch * mixed_weights
    deficits = np.maximum(cumulative_target - current_counts, 0.0)
    if not np.any(deficits > 0.0):
        additions = _integer_allocation(mixed_weights, batch)
    else:
        additions = _integer_allocation(deficits, batch)
    return additions, cumulative_target


def run_pareto_certificate_track_and_stop(
    theta: np.ndarray,
    config: ParetoCertificateTrackStopConfig,
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
    if config.max_exact_add_assignments < 1:
        raise ValueError("max_exact_add_assignments must be positive")

    cells = _cell_arrays(K, d)
    C = len(cells[0])
    pair_wins = np.zeros((d, K, K), dtype=np.int64)
    burnin = np.full(C, config.burnin_per_cell, dtype=np.int64)
    tau = _sample_additions(theta, pair_wins, cells, burnin, rng)

    log_weights = np.full(C, -np.log(C), dtype=float)
    weights = np.full(C, 1.0 / C, dtype=float)
    cumulative_target = burnin.astype(float)
    theta_initial = None
    history = []
    stopped = False
    final_fit = None
    final_profile = None
    final_threshold = np.inf
    certificate_pool: dict[
        tuple[tuple[int, ...], int], set[tuple[int, ...]]
    ] = {}

    for phase in range(1, config.max_phases + 1):
        final_fit = _fit_bt(pair_wins, config, theta_initial)
        theta_initial = final_fit["theta_hat"]
        estimated_pareto_size = len(
            strict_pareto_set(final_fit["theta_hat"])
        )
        exact_profile = bool(
            K <= config.exact_profile_max_arms
            and d ** estimated_pareto_size
            <= config.max_exact_add_assignments
        )
        profile_function = (
            pareto_certificate_profile
            if exact_profile
            else pareto_screened_certificate_profile
        )
        final_profile = profile_function(
            pair_wins,
            final_fit["theta_hat"],
            final_fit["coordinate_results"],
            config,
            certificate_pool=certificate_pool,
        )
        final_threshold = global_mixture_threshold(
            final_fit["theta_hat"],
            _count_laplacians(pair_wins),
            config,
        )
        stopped = bool(
            final_fit["converged_all"]
            and final_profile["all_profile_fits_converged"]
            and final_profile["lower"] >= final_threshold
        )
        history.append(
            {
                "phase": int(phase),
                "tau": int(tau),
                "glr_lower_bound": float(final_profile["lower"]),
                "glr_estimate": float(final_profile["estimate"]),
                "glr_upper_bound": float(final_profile["upper"]),
                "profile_interval_width": float(final_profile["interval_width"]),
                "profile_interval_closed": bool(
                    final_profile["profile_interval_closed"]
                ),
                "assignment_search_exact": bool(
                    final_profile["assignment_search_exact"]
                ),
                "threshold": float(final_threshold),
                "stopped": bool(stopped),
                "mle_converged_all": bool(final_fit["converged_all"]),
                "profile_fits_converged_all": bool(
                    final_profile["all_profile_fits_converged"]
                ),
                "alternative_kind": final_profile["alternative_kind"],
                "profile_mode": final_profile["profile_mode"],
                "stopping_statistic_is_certified": bool(
                    final_profile["stopping_statistic_is_certified"]
                ),
                "num_constrained_fits": int(
                    final_profile["num_constrained_fits"]
                ),
            }
        )
        if stopped or tau >= config.max_queries:
            break

        information = _cell_kl(
            final_fit["theta_hat"],
            final_profile["alternative_theta"],
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
        r, i, j = cells
        current_counts = pair_wins[r, i, j] + pair_wins[r, j, i]
        additions, cumulative_target = _tracked_batch_allocation(
            cumulative_target,
            current_counts,
            mixed,
            batch,
        )
        tau += _sample_additions(theta, pair_wins, cells, additions, rng)

    if final_fit is None or final_profile is None:
        raise RuntimeError("Track-and-Stop did not execute a stopping check")
    recommended = strict_pareto_set(final_fit["theta_hat"])
    truth = strict_pareto_set(theta)
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
        "profile_fits_converged_all": bool(
            final_profile["all_profile_fits_converged"]
        ),
        "final_glr_lower_bound": float(final_profile["lower"]),
        "final_glr_estimate": float(final_profile["estimate"]),
        "final_glr_upper_bound": float(final_profile["upper"]),
        "final_glr_threshold": float(final_threshold),
        "final_profile_interval_width": float(
            final_profile["interval_width"]
        ),
        "final_profile_interval_closed": bool(
            final_profile["profile_interval_closed"]
        ),
        "final_assignment_search_exact": bool(
            final_profile["assignment_search_exact"]
        ),
        "final_alternative_kind": final_profile["alternative_kind"],
        "final_profile_mode": final_profile["profile_mode"],
        "stopping_statistic_is_certified": bool(
            final_profile["stopping_statistic_is_certified"]
        ),
        "glr_is_conservative_lower_bound": bool(
            final_profile["stopping_statistic_is_certified"]
        ),
        "statistic_type": (
            "pareto-certificate-profile-lower-bound"
            if final_profile["stopping_statistic_is_certified"]
            else "pareto-screened-joint-certificate"
        ),
        "history": history,
    }
