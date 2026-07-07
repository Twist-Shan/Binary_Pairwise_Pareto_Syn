"""VB-EGE fixed-confidence algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import borda, strict_pareto_set, stable_sigmoid
from .gaps import empirical_pareto_and_gaps


@dataclass
class VBEGEConfig:
    delta: float
    sample_const: float = 8.0
    threshold_const: float = 16.0
    log_const: float = 4.0
    max_phases: int = 60
    max_queries: int | None = None
    batch_sampling: bool = True
    explicit_focal_sampling: bool = False
    seed: int | None = None
    tie_break_nonpareto_first: bool = True
    return_history: bool = True


@dataclass
class VBEGEResult:
    recommended: tuple[int, ...]
    accepted: tuple[int, ...]
    rejected: tuple[int, ...]
    active_final: tuple[int, ...]
    stopped: bool
    tau: int
    num_phases: int
    N: np.ndarray
    S: np.ndarray
    b_hat: np.ndarray
    history: list[dict]
    error: bool | None
    true_pareto: tuple[int, ...] | None


def choose_removal_arm(
    active: list[int] | tuple[int, ...],
    gap_by_arm: dict[int, float],
    empirical_pareto: tuple[int, ...],
    tie_break_nonpareto_first: bool = True,
) -> int:
    """Choose the empirical-gap maximizer with deterministic tie-breaking."""

    emp_pareto = set(empirical_pareto)
    max_gap = max(gap_by_arm[i] for i in active)
    candidates = [i for i in active if np.isclose(gap_by_arm[i], max_gap)]
    if tie_break_nonpareto_first:
        nonpareto = [i for i in candidates if i not in emp_pareto]
        if nonpareto:
            return int(min(nonpareto))
    return int(min(candidates))


def _as_rng(rng=None, seed=None):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _current_b_hat(S: np.ndarray, N: np.ndarray) -> np.ndarray:
    return np.divide(S, N, out=np.full_like(S, 0.5, dtype=float), where=N > 0)


def _sample_explicit_focal(
    theta: np.ndarray,
    i: int,
    r: int,
    add: int,
    rng: np.random.Generator,
) -> int:
    K = theta.shape[0]
    opponents = rng.integers(0, K - 1, size=add)
    opponents = np.where(opponents >= i, opponents + 1, opponents)
    probs = stable_sigmoid(theta[i, r] - theta[opponents, r])
    return int(rng.binomial(1, probs).sum())


def run_vb_ege(theta, config: VBEGEConfig, rng=None) -> VBEGEResult:
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    rng = _as_rng(rng, config.seed)
    b = borda(theta)
    true_pareto = strict_pareto_set(theta)
    active = list(range(K))
    accepted: set[int] = set()
    rejected: set[int] = set()
    N = np.zeros((K, d), dtype=int)
    S = np.zeros((K, d), dtype=float)
    history: list[dict] = []
    tau = 0
    stopped = False
    last_phase = 0

    for m in range(1, config.max_phases + 1):
        last_phase = m
        if len(active) <= 1:
            accepted.update(active)
            active.clear()
            stopped = True
            break

        eps_m = 2.0 ** (-m)
        log_term = float(np.log(config.log_const * K * d * m * m / config.delta))
        n_m = int(np.ceil(config.sample_const / (eps_m**2) * log_term))
        r_m = float(np.sqrt(log_term / (2.0 * n_m)))

        hit_budget = False
        for i in list(active):
            for r in range(d):
                add = n_m - int(N[i, r])
                if add <= 0:
                    continue
                if config.max_queries is not None:
                    remaining = config.max_queries - tau
                    if remaining <= 0:
                        hit_budget = True
                        break
                    add = min(add, remaining)
                if config.explicit_focal_sampling:
                    wins = _sample_explicit_focal(theta, i, r, add, rng)
                else:
                    wins = int(rng.binomial(add, b[i, r]))
                S[i, r] += wins
                N[i, r] += add
                tau += add
                if config.max_queries is not None and tau >= config.max_queries:
                    hit_budget = True
                    break
            if hit_budget:
                break

        b_hat = _current_b_hat(S, N)
        if hit_budget:
            recommended = tuple(sorted(accepted | set(active)))
            return VBEGEResult(
                recommended=recommended,
                accepted=tuple(sorted(accepted)),
                rejected=tuple(sorted(rejected)),
                active_final=tuple(active),
                stopped=False,
                tau=tau,
                num_phases=m,
                N=N,
                S=S,
                b_hat=b_hat,
                history=history,
                error=None,
                true_pareto=true_pareto,
            )

        removed_this_phase: list[dict] = []
        while active:
            emp = empirical_pareto_and_gaps(b_hat, active)
            if not emp["gap_by_arm"]:
                break
            max_gap = max(emp["gap_by_arm"].values())
            if not (max_gap > config.threshold_const * r_m):
                break
            arm = choose_removal_arm(
                active,
                emp["gap_by_arm"],
                emp["pareto_set"],
                config.tie_break_nonpareto_first,
            )
            active.remove(arm)
            if arm in set(emp["pareto_set"]):
                accepted.add(arm)
                decision = "accepted"
            else:
                rejected.add(arm)
                decision = "rejected"
            removed_this_phase.append(
                {"arm": arm, "decision": decision, "gap_hat": float(max_gap)}
            )

        if config.return_history:
            history.append(
                {
                    "phase": m,
                    "eps": eps_m,
                    "n_m": n_m,
                    "r_m": r_m,
                    "tau": tau,
                    "active": tuple(active),
                    "accepted": tuple(sorted(accepted)),
                    "rejected": tuple(sorted(rejected)),
                    "removed": removed_this_phase,
                }
            )

        if len(active) <= 1:
            accepted.update(active)
            active.clear()
            stopped = True
            break
        if len(accepted) + len(rejected) == K:
            stopped = True
            break

    b_hat = _current_b_hat(S, N)
    if stopped:
        recommended = tuple(sorted(accepted))
    else:
        emp = empirical_pareto_and_gaps(b_hat, active)
        recommended = tuple(sorted(accepted | set(emp["pareto_set"])))
    error = set(recommended) != set(true_pareto)
    return VBEGEResult(
        recommended=recommended,
        accepted=tuple(sorted(accepted)),
        rejected=tuple(sorted(rejected)),
        active_final=tuple(active),
        stopped=stopped,
        tau=tau,
        num_phases=last_phase,
        N=N,
        S=S,
        b_hat=b_hat,
        history=history,
        error=error,
        true_pareto=true_pareto,
    )
