"""Core utilities for coordinate-wise Bradley-Terry Pareto experiments."""

from __future__ import annotations

import numpy as np


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""

    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.ndim(x) == 0:
        return float(out)
    return out


def bt_prob(theta: np.ndarray, i: int, j: int, r: int) -> float:
    """Bradley-Terry probability that arm i beats arm j on coordinate r."""

    theta = np.asarray(theta, dtype=float)
    return float(stable_sigmoid(theta[i, r] - theta[j, r]))


def borda(theta: np.ndarray) -> np.ndarray:
    """Coordinate-wise Borda embedding under the BT model."""

    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2:
        raise ValueError("theta must have shape (K, d)")
    K, _ = theta.shape
    if K < 2:
        raise ValueError("borda requires at least two arms")
    diffs = theta[:, None, :] - theta[None, :, :]
    probs = stable_sigmoid(diffs)
    idx = np.arange(K)
    probs[idx, idx, :] = 0.0
    return probs.sum(axis=1) / (K - 1)


def dominates_strict(x_j: np.ndarray, x_i: np.ndarray, atol: float = 0.0) -> bool:
    """Return True iff x_j strictly dominates x_i coordinate-wise."""

    return bool(np.all(np.asarray(x_j) > np.asarray(x_i) + atol))


def strict_pareto_set(x: np.ndarray, atol: float = 0.0) -> tuple[int, ...]:
    """Strict Pareto set using zero-based arm indices.

    Arm i is removed only if some arm j is larger than i by more than atol
    in every coordinate.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (K, d)")
    K = x.shape[0]
    pareto: list[int] = []
    for i in range(K):
        dominated = any(
            j != i and dominates_strict(x[j], x[i], atol=atol) for j in range(K)
        )
        if not dominated:
            pareto.append(i)
    return tuple(pareto)


def dynamic_range_B(theta: np.ndarray) -> float:
    """Maximum coordinate-wise latent BT range."""

    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2:
        raise ValueError("theta must have shape (K, d)")
    return float(np.max(np.max(theta, axis=0) - np.min(theta, axis=0)))


def kappa_B(B: float) -> float:
    """Logistic curvature condition proxy over a dynamic range B."""

    deriv = stable_sigmoid(B) * stable_sigmoid(-B)
    return float(1.0 / max(float(deriv), np.finfo(float).tiny))


def center_columns(theta: np.ndarray) -> np.ndarray:
    """Center every coordinate, matching the BT identifiability gauge."""

    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2:
        raise ValueError("theta must have shape (K, d)")
    return theta - theta.mean(axis=0, keepdims=True)
