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
    if d < 2 and s > 1:
        raise ValueError("multiple Pareto anchors require d >= 2")
    for _ in range(max_attempts):
        z = rng.dirichlet(alpha * np.ones(d), size=s)
        # A constant row sum makes strict dominance among distinct anchors impossible.
        anchors = 0.15 + 0.75 * z
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


def _rescale_columns(x: np.ndarray, low: float = 0.20, high: float = 0.90) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mins = x.min(axis=0, keepdims=True)
    maxs = x.max(axis=0, keepdims=True)
    span = maxs - mins
    scaled = np.full_like(x, (low + high) / 2.0)
    mask = span > 1e-12
    scaled[:, mask.ravel()] = low + (high - low) * (
        (x[:, mask.ravel()] - mins[:, mask.ravel()]) / span[:, mask.ravel()]
    )
    return scaled


def _objective_correlation(theta: np.ndarray) -> tuple[float, list[float]]:
    if theta.shape[1] < 2:
        return float("nan"), []
    corr = np.corrcoef(theta, rowvar=False)
    offdiag = corr[np.triu_indices(theta.shape[1], k=1)]
    offdiag = offdiag[np.isfinite(offdiag)]
    return (float(offdiag.mean()) if offdiag.size else float("nan"), [float(x) for x in offdiag])


def correlated_arena_like(
    K: int,
    d: int,
    s: int,
    rho: float,
    margin_low: float,
    margin_high: float,
    latent_rank: int = 3,
    alpha: float = 0.7,
    scale: float = 1.0,
    seed: int | None = None,
    permute: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    if not 1 <= s <= K:
        raise ValueError("s must satisfy 1 <= s <= K")
    if not 0.0 <= rho <= 0.98:
        raise ValueError("rho must be in [0, 0.98]")
    if latent_rank < 1:
        raise ValueError("latent_rank must be positive")

    for _outer in range(2000):
        common = rng.normal(size=(latent_rank, 1))
        noise = rng.normal(size=(latent_rank, d))
        A = np.sqrt(rho) * common @ np.ones((1, d)) + np.sqrt(1.0 - rho) * noise
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A = A / np.maximum(norms, 1e-12)

        z = rng.dirichlet(alpha * np.ones(latent_rank), size=s)
        raw = z @ A
        theta_p = scale * _rescale_columns(raw, low=0.20, high=0.90)
        theta_p *= rng.uniform(0.96, 1.04, size=(s, 1))
        theta_p += rng.uniform(-0.003, 0.003, size=theta_p.shape)
        if len(strict_pareto_set(theta_p)) != s:
            continue

        dominated = []
        for _arm in range(K - s):
            witness = rng.integers(0, s)
            margin_common = rng.uniform(margin_low, margin_high)
            margin_noise = rng.uniform(margin_low, margin_high, size=d)
            margin = rho * margin_common + (1.0 - rho) * margin_noise
            x = theta_p[witness] - margin
            if not dominates_strict(theta_p[witness], x):
                raise AssertionError("generated dominated arm is not dominated")
            dominated.append(x)
        theta = np.vstack([theta_p, np.asarray(dominated)])
        if len(strict_pareto_set(theta)) != s:
            continue
        theta, perm = _permute(theta, rng, permute)
        if len(strict_pareto_set(theta)) != s:
            continue
        corr_mean, corr_offdiag = _objective_correlation(theta)
        meta = _meta(
            "correlated_arena_like",
            theta,
            seed,
            {
                "K": K,
                "d": d,
                "s": s,
                "rho": rho,
                "latent_rank": latent_rank,
                "margin_low": margin_low,
                "margin_high": margin_high,
                "alpha": alpha,
                "scale": scale,
                "permute": permute,
                "achieved_objective_correlation_mean": corr_mean,
                "achieved_objective_correlation_offdiag": corr_offdiag,
            },
            perm,
        )
        meta["expected_pareto_size"] = s
        return theta, meta
    raise RuntimeError("could not generate correlated_arena_like with target Pareto size")


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
    "correlated_arena_like": correlated_arena_like,
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
