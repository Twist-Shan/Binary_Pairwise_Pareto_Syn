"""Gap utilities for strict Pareto identification."""

from __future__ import annotations

import numpy as np

from .core import strict_pareto_set


def pairwise_m_matrix(x: np.ndarray) -> np.ndarray:
    """m[i, j] = min_r (x[j, r] - x[i, r]) with diagonal ignored."""

    x = np.asarray(x, dtype=float)
    diffs = x[None, :, :] - x[:, None, :]
    m = diffs.min(axis=2)
    np.fill_diagonal(m, -np.inf)
    return m


def pairwise_M_matrix(x: np.ndarray) -> np.ndarray:
    """M[i, j] = max_r (x[i, r] - x[j, r])."""

    x = np.asarray(x, dtype=float)
    diffs = x[:, None, :] - x[None, :, :]
    M = diffs.max(axis=2)
    np.fill_diagonal(M, 0.0)
    return M


def compute_gaps(x: np.ndarray) -> dict:
    """Compute strict-Pareto raw and identification gaps."""

    x = np.asarray(x, dtype=float)
    K = x.shape[0]
    pareto_set = strict_pareto_set(x)
    pareto = set(pareto_set)
    m = pairwise_m_matrix(x)
    M = pairwise_M_matrix(x)
    delta_star = m.max(axis=1)
    delta = np.empty(K, dtype=float)

    if K == 1:
        delta[:] = np.inf
    else:
        for i in range(K):
            if i not in pareto:
                delta[i] = delta_star[i]
                continue
            vals = []
            for j in range(K):
                if i == j:
                    continue
                term = min(M[i, j], max(M[j, i], 0.0) + max(delta_star[j], 0.0))
                vals.append(term)
            delta[i] = min(vals) if vals else np.inf

    delta_min = float(np.min(delta)) if len(delta) else np.inf
    if np.any(delta <= 0.0) or not np.all(np.isfinite(delta)):
        H = np.inf if np.any(delta <= 0.0) else float(np.nansum(1.0 / delta**2))
    else:
        H = float(np.sum(1.0 / delta**2))
    return {
        "pareto_set": pareto_set,
        "delta_star": delta_star,
        "delta": delta,
        "delta_min": delta_min,
        "H": H,
        "m": m,
        "M": M,
    }


def empirical_pareto_and_gaps(x_hat: np.ndarray, active: list[int] | tuple[int, ...]) -> dict:
    """Compute empirical active-set Pareto membership and gaps.

    Returned arm indices in ``pareto_set`` are global arm indices. Gap arrays
    are aligned to the ``active`` ordering and also exposed as a global dict.
    """

    active_tuple = tuple(int(i) for i in active)
    if not active_tuple:
        return {
            "active": active_tuple,
            "pareto_set": tuple(),
            "pareto_set_local": tuple(),
            "delta": np.array([], dtype=float),
            "delta_star": np.array([], dtype=float),
            "gap_by_arm": {},
            "details": None,
        }
    x_active = np.asarray(x_hat, dtype=float)[list(active_tuple)]
    details = compute_gaps(x_active)
    pareto_local = details["pareto_set"]
    pareto_global = tuple(active_tuple[i] for i in pareto_local)
    gap_by_arm = {
        active_tuple[pos]: float(details["delta"][pos]) for pos in range(len(active_tuple))
    }
    return {
        "active": active_tuple,
        "pareto_set": pareto_global,
        "pareto_set_local": pareto_local,
        "delta": details["delta"],
        "delta_star": details["delta_star"],
        "gap_by_arm": gap_by_arm,
        "details": details,
    }
