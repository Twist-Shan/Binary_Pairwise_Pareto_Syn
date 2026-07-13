"""Uniform-sampling baselines and capped VB-EGE wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .algorithms import VBEGEConfig, run_vb_ege
from .bt_mle import BTMLEConfig, fit_bt_mle
from .core import borda, strict_pareto_set
from .gaps import compute_gaps, empirical_pareto_and_gaps
from .metrics import (
    centered_theta_rmse,
    hamming_set_distance,
    pairwise_sign_accuracy,
    set_error,
)


def _as_rng(rng=None):
    return rng if rng is not None else np.random.default_rng()


def _balanced_counts(num_cells: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(num_cells, budget // num_cells, dtype=int)
    rem = budget % num_cells
    if rem:
        extra = rng.choice(num_cells, size=rem, replace=False)
        counts[extra] += 1
    return counts


@dataclass
class FixedConfidenceBaselineConfig:
    delta: float
    sample_const: float = 2.0
    threshold_const: float = 4.0
    log_const: float = 4.0
    max_phases: int = 40
    max_queries: int | None = None
    allocation_scheme: str = "balanced_random"
    mle_radius_scale: float = 1.0


def _fc_config_from(config=None) -> FixedConfidenceBaselineConfig:
    if isinstance(config, FixedConfidenceBaselineConfig):
        return config
    cfg = dict(config or {})
    if "max_rounds" in cfg and "max_phases" not in cfg:
        cfg["max_phases"] = cfg["max_rounds"]
    allowed = set(FixedConfidenceBaselineConfig.__dataclass_fields__)
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    if "delta" not in cfg:
        cfg["delta"] = 0.05
    return FixedConfidenceBaselineConfig(**cfg)


def _phase_schedule(
    phase: int,
    num_cells: int,
    config: FixedConfidenceBaselineConfig,
) -> tuple[int, float, float]:
    eps_m = 2.0 ** (-phase)
    log_term = float(np.log(config.log_const * num_cells * phase * phase / config.delta))
    n_m = int(np.ceil(config.sample_const / (eps_m**2) * log_term))
    radius = float(np.sqrt(log_term / (2.0 * n_m)))
    return n_m, radius, log_term


def _gap_certificate(x_hat: np.ndarray, radius: float, threshold_const: float) -> tuple[bool, dict]:
    gaps = compute_gaps(x_hat)
    delta_min = float(gaps["delta_min"])
    stopped = bool(np.isfinite(delta_min) and delta_min > threshold_const * radius)
    return stopped, gaps


def _base_result(
    algorithm: str,
    theta: np.ndarray,
    recommended,
    tau: int,
    stopped: bool,
    num_phases: int,
    pareto_size_hat: int,
) -> dict:
    true_pareto = strict_pareto_set(theta)
    return {
        "algorithm": algorithm,
        "recommended": tuple(recommended),
        "tau": int(tau),
        "error": set_error(recommended, true_pareto),
        "hamming": hamming_set_distance(recommended, true_pareto),
        "stopped": bool(stopped),
        "num_phases": int(num_phases),
        "num_accepted": np.nan,
        "num_rejected": np.nan,
        "pareto_size_hat": int(pareto_size_hat),
    }


def run_uniform_focal_borda(theta, budget: int, rng) -> dict:
    rng = _as_rng(rng)
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    b = borda(theta)
    counts = _balanced_counts(K * d, int(budget), rng).reshape(K, d)
    S = rng.binomial(counts, b)
    b_hat = np.divide(S, counts, out=np.full((K, d), 0.5, dtype=float), where=counts > 0)
    recommended = strict_pareto_set(b_hat)
    true_pareto = strict_pareto_set(theta)
    return {
        "algorithm": "UniformFocalBorda",
        "recommended": recommended,
        "tau": int(counts.sum()),
        "error": set_error(recommended, true_pareto),
        "hamming": hamming_set_distance(recommended, true_pareto),
        "stopped": True,
        "num_phases": np.nan,
        "num_accepted": np.nan,
        "num_rejected": np.nan,
        "pareto_size_hat": len(recommended),
        "b_hat": b_hat,
        "N": counts,
        "S": S,
    }


def run_uniform_focal_borda_fc(theta, config=None, rng=None) -> dict:
    rng = _as_rng(rng)
    config = _fc_config_from(config)
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    b = borda(theta)
    N = np.zeros((K, d), dtype=int)
    S = np.zeros((K, d), dtype=float)
    history: list[dict] = []
    tau = 0
    stopped = False
    recommended = tuple(range(K))
    b_hat = np.full((K, d), 0.5, dtype=float)
    radius = np.nan

    for phase in range(1, config.max_phases + 1):
        n_m, radius, _log_term = _phase_schedule(phase, K * d, config)
        hit_budget = False
        for i in range(K):
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
                S[i, r] += int(rng.binomial(add, b[i, r]))
                N[i, r] += add
                tau += add
                if config.max_queries is not None and tau >= config.max_queries:
                    hit_budget = True
                    break
            if hit_budget:
                break
        b_hat = np.divide(S, N, out=np.full((K, d), 0.5, dtype=float), where=N > 0)
        recommended = strict_pareto_set(b_hat)
        stopped, gaps = _gap_certificate(b_hat, radius, config.threshold_const)
        history.append(
            {
                "phase": phase,
                "n_m": n_m,
                "radius": radius,
                "tau": tau,
                "delta_min_hat": float(gaps["delta_min"]),
                "stopped": stopped,
            }
        )
        if stopped or hit_budget:
            break

    result = _base_result(
        "UniformFocalBorda-FC",
        theta,
        recommended,
        tau,
        stopped,
        len(history),
        len(recommended),
    )
    result.update(
        {
            "b_hat": b_hat,
            "N": N,
            "S": S,
            "history": history,
            "fc_radius": radius,
        }
    )
    return result


def _pair_cells(K: int, d: int) -> list[tuple[int, int, int]]:
    return [(r, i, j) for r in range(d) for i in range(K) for j in range(i + 1, K)]


def allocate_pair_coordinate_budget(
    K: int,
    d: int,
    budget: int,
    rng,
    scheme: str = "balanced_random",
) -> np.ndarray:
    rng = _as_rng(rng)
    budget = int(budget)
    counts = np.zeros((d, K, K), dtype=int)
    cells = _pair_cells(K, d)
    C = len(cells)
    if scheme == "balanced_random":
        flat_counts = _balanced_counts(C, budget, rng)
    elif scheme == "with_replacement":
        flat_counts = np.zeros(C, dtype=int)
        if budget:
            draws = rng.integers(0, C, size=budget)
            np.add.at(flat_counts, draws, 1)
    else:
        raise ValueError(f"unknown allocation scheme: {scheme}")
    for n, (r, i, j) in zip(flat_counts, cells):
        counts[r, i, j] = int(n)
    return counts


def simulate_pairwise_counts(theta, counts: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    rng = _as_rng(rng)
    theta = np.asarray(theta, dtype=float)
    d, K, _ = counts.shape
    W = np.zeros((d, K, K), dtype=int)
    Npair = np.zeros((d, K, K), dtype=int)
    for r in range(d):
        for i in range(K):
            for j in range(i + 1, K):
                n = int(counts[r, i, j])
                if n <= 0:
                    continue
                p = 1.0 / (1.0 + np.exp(-(theta[i, r] - theta[j, r])))
                wij = int(rng.binomial(n, p))
                wji = n - wij
                W[r, i, j] = wij
                W[r, j, i] = wji
                Npair[r, i, j] = n
                Npair[r, j, i] = n
    return W, Npair


def _add_pairwise_samples(
    theta: np.ndarray,
    W: np.ndarray,
    Npair: np.ndarray,
    target_per_cell: int,
    rng: np.random.Generator,
    max_add: int | None = None,
) -> int:
    K, d = theta.shape
    added_total = 0
    for r in range(d):
        for i in range(K):
            for j in range(i + 1, K):
                add = target_per_cell - int(Npair[r, i, j])
                if add <= 0:
                    continue
                if max_add is not None:
                    remaining = max_add - added_total
                    if remaining <= 0:
                        return added_total
                    add = min(add, remaining)
                p = 1.0 / (1.0 + np.exp(-(theta[i, r] - theta[j, r])))
                wij = int(rng.binomial(add, p))
                wji = add - wij
                W[r, i, j] += wij
                W[r, j, i] += wji
                Npair[r, i, j] += add
                Npair[r, j, i] += add
                added_total += add
    return added_total


def _pairwise_borda_hat(W: np.ndarray, Npair: np.ndarray) -> np.ndarray:
    d, K, _ = W.shape
    p_hat = np.full((d, K, K), 0.5, dtype=float)
    mask = Npair > 0
    p_hat[mask] = (W[mask] + 0.5) / (Npair[mask] + 1.0)
    for r in range(d):
        np.fill_diagonal(p_hat[r], 0.0)
    return p_hat.sum(axis=2).T / (K - 1)


def _mle_config_from(config=None) -> tuple[BTMLEConfig, str]:
    if config is None:
        return BTMLEConfig(), "balanced_random"
    if isinstance(config, BTMLEConfig):
        return config, "balanced_random"
    if isinstance(config, dict):
        cfg = dict(config)
        allocation = cfg.pop("allocation_scheme", "balanced_random")
        allowed = set(BTMLEConfig.__dataclass_fields__)
        cfg = {k: v for k, v in cfg.items() if k in allowed}
        return BTMLEConfig(**cfg), allocation
    raise TypeError("config must be None, dict, or BTMLEConfig")


def run_uniform_pairwise_bt_mle(theta, budget: int, rng, config=None) -> dict:
    rng = _as_rng(rng)
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    mle_config, allocation = _mle_config_from(config)
    counts = allocate_pair_coordinate_budget(K, d, budget, rng, scheme=allocation)
    W, Npair = simulate_pairwise_counts(theta, counts, rng)
    fit = fit_bt_mle(W, Npair, mle_config)
    theta_hat = fit["theta_hat"]
    recommended = strict_pareto_set(theta_hat)
    true_pareto = strict_pareto_set(theta)
    C = d * K * (K - 1) // 2
    return {
        "algorithm": "UniformPairwiseBT-MLE",
        "recommended": recommended,
        "tau": int(counts.sum()),
        "error": set_error(recommended, true_pareto),
        "hamming": hamming_set_distance(recommended, true_pareto),
        "stopped": True,
        "num_phases": np.nan,
        "num_accepted": np.nan,
        "num_rejected": np.nan,
        "pareto_size_hat": len(recommended),
        "theta_hat": theta_hat,
        "W": W,
        "Npair": Npair,
        "pair_cell_count": C,
        "cell_coverage": float(budget / C) if C else np.nan,
        "theta_rmse_centered": centered_theta_rmse(theta_hat, theta),
        "pairwise_sign_accuracy": pairwise_sign_accuracy(theta_hat, theta),
        **fit,
        "mle_config": asdict(mle_config),
        "mle_allocation_scheme": allocation,
    }


def run_uniform_pairwise_bt_mle_fc(theta, config=None, rng=None) -> dict:
    rng = _as_rng(rng)
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    fc_config = _fc_config_from(config)
    mle_config, allocation = _mle_config_from(config)
    if allocation != "balanced_random":
        raise ValueError("fixed-confidence pairwise MLE currently uses balanced per-cell rounds")
    W = np.zeros((d, K, K), dtype=int)
    Npair = np.zeros((d, K, K), dtype=int)
    C = d * K * (K - 1) // 2
    history: list[dict] = []
    tau = 0
    stopped = False
    fit = None
    theta_hat = np.zeros_like(theta)
    recommended = tuple(range(K))
    radius = np.nan

    for phase in range(1, fc_config.max_phases + 1):
        n_m, radius, _log_term = _phase_schedule(phase, C, fc_config)
        max_add = None if fc_config.max_queries is None else fc_config.max_queries - tau
        if max_add is not None and max_add <= 0:
            break
        added = _add_pairwise_samples(theta, W, Npair, n_m, rng, max_add=max_add)
        tau += added
        fit = fit_bt_mle(W, Npair, mle_config)
        theta_hat = fit["theta_hat"]
        recommended = strict_pareto_set(theta_hat)
        stopped, gaps = _gap_certificate(
            theta_hat, fc_config.mle_radius_scale * radius, fc_config.threshold_const
        )
        history.append(
            {
                "phase": phase,
                "n_m": n_m,
                "radius": fc_config.mle_radius_scale * radius,
                "tau": tau,
                "delta_min_hat": float(gaps["delta_min"]),
                "stopped": stopped,
                "mle_converged_all": fit["mle_converged_all"],
                "mle_ridge_fallback_any": fit["mle_ridge_fallback_any"],
            }
        )
        if stopped or (fc_config.max_queries is not None and tau >= fc_config.max_queries):
            break

    if fit is None:
        fit = fit_bt_mle(W, Npair, mle_config)
        theta_hat = fit["theta_hat"]
        recommended = strict_pareto_set(theta_hat)
    result = _base_result(
        "UniformPairwiseBT-MLE-Cert",
        theta,
        recommended,
        tau,
        stopped,
        len(history),
        len(recommended),
    )
    result.update(
        {
            "theta_hat": theta_hat,
            "W": W,
            "Npair": Npair,
            "history": history,
            "pair_cell_count": C,
            "cell_coverage": float(tau / C) if C else np.nan,
            "theta_rmse_centered": centered_theta_rmse(theta_hat, theta),
            "pairwise_sign_accuracy": pairwise_sign_accuracy(theta_hat, theta),
            "fc_radius": fc_config.mle_radius_scale * radius,
            **fit,
            "mle_config": asdict(mle_config),
            "mle_allocation_scheme": allocation,
        }
    )
    return result


def run_uniform_pairwise_bt_borda_plugin(theta, budget: int, rng, config=None) -> dict:
    rng = _as_rng(rng)
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    allocation = "balanced_random"
    if isinstance(config, dict):
        allocation = config.get("allocation_scheme", allocation)
    counts = allocate_pair_coordinate_budget(K, d, budget, rng, scheme=allocation)
    W, Npair = simulate_pairwise_counts(theta, counts, rng)
    b_hat = _pairwise_borda_hat(W, Npair)
    recommended = strict_pareto_set(b_hat)
    true_pareto = strict_pareto_set(theta)
    C = d * K * (K - 1) // 2
    return {
        "algorithm": "UniformPairwiseBT-BordaPlugIn",
        "recommended": recommended,
        "tau": int(counts.sum()),
        "error": set_error(recommended, true_pareto),
        "hamming": hamming_set_distance(recommended, true_pareto),
        "stopped": True,
        "num_phases": np.nan,
        "num_accepted": np.nan,
        "num_rejected": np.nan,
        "pareto_size_hat": len(recommended),
        "b_hat": b_hat,
        "W": W,
        "Npair": Npair,
        "pair_cell_count": C,
        "cell_coverage": float(budget / C) if C else np.nan,
        "mle_allocation_scheme": allocation,
    }


def run_uniform_pairwise_bt_borda_plugin_fc(theta, config=None, rng=None) -> dict:
    rng = _as_rng(rng)
    theta = np.asarray(theta, dtype=float)
    K, d = theta.shape
    fc_config = _fc_config_from(config)
    if fc_config.allocation_scheme != "balanced_random":
        raise ValueError("fixed-confidence pairwise Borda currently uses balanced per-cell rounds")
    W = np.zeros((d, K, K), dtype=int)
    Npair = np.zeros((d, K, K), dtype=int)
    C = d * K * (K - 1) // 2
    history: list[dict] = []
    tau = 0
    stopped = False
    b_hat = np.full((K, d), 0.5, dtype=float)
    recommended = tuple(range(K))
    radius = np.nan

    for phase in range(1, fc_config.max_phases + 1):
        n_m, radius, _log_term = _phase_schedule(phase, C, fc_config)
        max_add = None if fc_config.max_queries is None else fc_config.max_queries - tau
        if max_add is not None and max_add <= 0:
            break
        added = _add_pairwise_samples(theta, W, Npair, n_m, rng, max_add=max_add)
        tau += added
        b_hat = _pairwise_borda_hat(W, Npair)
        recommended = strict_pareto_set(b_hat)
        stopped, gaps = _gap_certificate(b_hat, radius, fc_config.threshold_const)
        history.append(
            {
                "phase": phase,
                "n_m": n_m,
                "radius": radius,
                "tau": tau,
                "delta_min_hat": float(gaps["delta_min"]),
                "stopped": stopped,
            }
        )
        if stopped or (fc_config.max_queries is not None and tau >= fc_config.max_queries):
            break

    result = _base_result(
        "UniformPairwiseBT-BordaPlugIn-FC",
        theta,
        recommended,
        tau,
        stopped,
        len(history),
        len(recommended),
    )
    result.update(
        {
            "b_hat": b_hat,
            "W": W,
            "Npair": Npair,
            "history": history,
            "pair_cell_count": C,
            "cell_coverage": float(tau / C) if C else np.nan,
            "mle_allocation_scheme": fc_config.allocation_scheme,
            "fc_radius": radius,
        }
    )
    return result


def _vbege_config_from(config, budget: int | None = None) -> VBEGEConfig:
    if isinstance(config, VBEGEConfig):
        cfg = config
    else:
        cfg = VBEGEConfig(**dict(config or {}))
    if budget is not None:
        cfg = VBEGEConfig(**{**asdict(cfg), "max_queries": int(budget)})
    return cfg


def run_vb_ege_capped(theta, budget: int, config, rng) -> dict:
    cfg = _vbege_config_from(config, budget=budget)
    result = run_vb_ege(theta, cfg, rng=rng)
    true_pareto = strict_pareto_set(theta)
    if result.stopped:
        recommended = result.recommended
    else:
        emp = empirical_pareto_and_gaps(result.b_hat, result.active_final)
        recommended = tuple(sorted(set(result.accepted) | set(emp["pareto_set"])))
    return {
        "algorithm": "VB-EGE-capped",
        "recommended": recommended,
        "tau": int(result.tau),
        "error": set_error(recommended, true_pareto),
        "hamming": hamming_set_distance(recommended, true_pareto),
        "stopped": result.stopped,
        "num_phases": result.num_phases,
        "num_accepted": len(result.accepted),
        "num_rejected": len(result.rejected),
        "pareto_size_hat": len(recommended),
        "b_hat": result.b_hat,
        "N": result.N,
        "S": result.S,
        "history": result.history,
        "accepted": result.accepted,
        "rejected": result.rejected,
        "active_final": result.active_final,
    }
