"""Synthetic instance generators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .core import dominates_strict, strict_pareto_set


def _meta(name: str, theta: np.ndarray, seed: int | None, params: dict, permutation):
    return {
        "name": name,
        "K": int(theta.shape[0]),
        "d": int(theta.shape[1]),
        "expected_pareto_size": None,
        "params": dict(params),
        "seed": seed,
        "permutation": None if permutation is None else [int(i) for i in permutation],
    }


def _permute(theta: np.ndarray, rng: np.random.Generator, permute: bool):
    if not permute:
        return theta, None
    perm = rng.permutation(theta.shape[0])
    return theta[perm], perm


def _pareto_anchors(
    s: int,
    d: int,
    rng: np.random.Generator,
    alpha: float = 0.7,
    max_attempts: int = 10000,
) -> np.ndarray:
    for _ in range(max_attempts):
        z = rng.dirichlet(alpha * np.ones(d), size=s)
        anchors = 0.15 + 0.75 * z / z.max(axis=1, keepdims=True)
        anchors *= rng.uniform(0.94, 1.06, size=(s, 1))
        anchors += rng.uniform(-0.005, 0.005, size=anchors.shape)
        if len(strict_pareto_set(anchors)) == s:
            return anchors
    raise RuntimeError("could not sample mutually non-dominated Pareto anchors")


def symmetric_hard(
    K: int,
    d: int,
    Delta: float,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    theta = np.full((K, d), -float(Delta), dtype=float)
    theta[0, :] = 0.0
    theta, perm = _permute(theta, rng, permute)
    meta = _meta(
        "symmetric_hard",
        theta,
        seed,
        {"K": K, "d": d, "Delta": Delta, "permute": permute},
        perm,
    )
    meta["expected_pareto_size"] = 1
    return theta, meta


def arena_tradeoff_frontier(
    K: int,
    d: int,
    s: int,
    margin_low: float,
    margin_high: float,
    alpha: float = 0.7,
    scale: float = 1.0,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    if not 1 <= s <= K:
        raise ValueError("s must satisfy 1 <= s <= K")
    for _ in range(1000):
        theta_p = scale * _pareto_anchors(s, d, rng, alpha=alpha)
        dominated = []
        for _arm in range(K - s):
            witness = rng.integers(0, s)
            margin = rng.uniform(margin_low, margin_high, size=d)
            x = theta_p[witness] - margin
            if not dominates_strict(theta_p[witness], x):
                raise AssertionError("generated dominated arm is not dominated")
            dominated.append(x)
        theta = np.vstack([theta_p, np.asarray(dominated)])
        if len(strict_pareto_set(theta)) == s:
            theta, perm = _permute(theta, rng, permute)
            meta = _meta(
                "arena_tradeoff_frontier",
                theta,
                seed,
                {
                    "K": K,
                    "d": d,
                    "s": s,
                    "margin_low": margin_low,
                    "margin_high": margin_high,
                    "alpha": alpha,
                    "scale": scale,
                    "permute": permute,
                },
                perm,
            )
            meta["expected_pareto_size"] = s
            assert len(strict_pareto_set(theta)) == s
            return theta, meta
    raise RuntimeError("could not generate arena_tradeoff_frontier with target Pareto size")


def unique_witness_d(
    K: int,
    d: int,
    s: int,
    q_per_p: int,
    margin_low: float,
    margin_high: float,
    alpha: float = 0.7,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    expected_K = s * (q_per_p + 1)
    if K != expected_K:
        raise ValueError(f"K must equal s * (q_per_p + 1) = {expected_K}")

    for _outer in range(300):
        theta_p = _pareto_anchors(s, d, rng, alpha=alpha)
        dominated = []
        ok = True
        for p in range(s):
            made = 0
            attempts = 0
            while made < q_per_p and attempts < 10000:
                attempts += 1
                margin = rng.uniform(margin_low, margin_high, size=d)
                x = theta_p[p] - margin
                witnesses = [
                    q for q in range(s) if dominates_strict(theta_p[q], x)
                ]
                if witnesses == [p]:
                    dominated.append(x)
                    made += 1
            if made < q_per_p:
                ok = False
                break
        if not ok:
            continue
        theta = np.vstack([theta_p, np.asarray(dominated)])
        if len(strict_pareto_set(theta)) == s:
            theta, perm = _permute(theta, rng, permute)
            meta = _meta(
                "unique_witness_d",
                theta,
                seed,
                {
                    "K": K,
                    "d": d,
                    "s": s,
                    "q_per_p": q_per_p,
                    "margin_low": margin_low,
                    "margin_high": margin_high,
                    "alpha": alpha,
                    "permute": permute,
                },
                perm,
            )
            meta["expected_pareto_size"] = s
            return theta, meta
    raise RuntimeError("could not generate unique_witness_d")


def highdim_two_group(
    K_low: int,
    K_high: int,
    d: int,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    low = rng.uniform(0.20, 0.45, size=(K_low, d))
    high = rng.uniform(0.55, 0.75, size=(K_high, d))
    theta = np.vstack([low, high])
    theta, perm = _permute(theta, rng, permute)
    meta = _meta(
        "highdim_two_group",
        theta,
        seed,
        {"K_low": K_low, "K_high": K_high, "d": d, "permute": permute},
        perm,
    )
    meta["expected_pareto_size"] = len(strict_pareto_set(theta))
    return theta, meta


def convex_frontier_2d(
    K: int = 60,
    s: int = 15,
    d: int = 2,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    if d != 2:
        raise ValueError("convex_frontier_2d requires d=2")
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, s)
    theta_p = np.column_stack([0.20 + 0.65 * u, 0.85 - 0.65 * u])
    dominated = []
    for _ in range(K - s):
        witness = rng.integers(0, s)
        margin = rng.uniform(0.03, 0.18, size=d)
        dominated.append(theta_p[witness] - margin)
    theta = np.vstack([theta_p, np.asarray(dominated)])
    theta, perm = _permute(theta, rng, permute)
    meta = _meta(
        "convex_frontier_2d",
        theta,
        seed,
        {"K": K, "d": d, "s": s, "permute": permute},
        perm,
    )
    meta["expected_pareto_size"] = s
    assert len(strict_pareto_set(theta)) == s
    return theta, meta


def circle_2d(
    K: int = 200,
    s: int = 20,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.05, np.pi / 2.0 - 0.05, s)
    theta_p = np.column_stack([np.cos(angles), np.sin(angles)])
    theta_p = 0.2 + 0.7 * theta_p
    dominated = []
    for _ in range(K - s):
        witness = rng.integers(0, s)
        dominated.append(theta_p[witness] - rng.uniform(0.04, 0.20, size=2))
    theta = np.vstack([theta_p, np.asarray(dominated)])
    theta, perm = _permute(theta, rng, permute)
    meta = _meta(
        "circle_2d",
        theta,
        seed,
        {"K": K, "d": 2, "s": s, "permute": permute},
        perm,
    )
    meta["expected_pareto_size"] = s
    return theta, meta


def boundary_equality_sanity(seed: int | None = None) -> tuple[np.ndarray, dict]:
    theta = np.array(
        [
            [0.50, 0.50],
            [0.50, 0.40],
            [0.40, 0.50],
            [0.30, 0.30],
        ],
        dtype=float,
    )
    meta = _meta("boundary_equality_sanity", theta, seed, {}, None)
    meta["expected_pareto_size"] = 3
    return theta, meta


GENERATORS: dict[str, Callable[..., tuple[np.ndarray, dict]]] = {
    "symmetric_hard": symmetric_hard,
    "arena_tradeoff_frontier": arena_tradeoff_frontier,
    "unique_witness_d": unique_witness_d,
    "highdim_two_group": highdim_two_group,
    "convex_frontier_2d": convex_frontier_2d,
    "circle_2d": circle_2d,
    "boundary_equality_sanity": boundary_equality_sanity,
}


def make_instance(
    generator: str,
    params: dict | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    if generator not in GENERATORS:
        raise KeyError(f"unknown generator: {generator}")
    kwargs = dict(params or {})
    kwargs["seed"] = seed
    return GENERATORS[generator](**kwargs)
