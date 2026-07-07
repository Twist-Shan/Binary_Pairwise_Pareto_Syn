"""Coordinate-wise Bradley-Terry maximum-likelihood estimation."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

try:  # SciPy may be unavailable or ABI-incompatible in lightweight envs.
    with contextlib.redirect_stderr(io.StringIO()):
        from scipy.optimize import minimize
except Exception:  # pragma: no cover - exercised only in broken SciPy installs.
    minimize = None

from .core import stable_sigmoid, strict_pareto_set


@dataclass
class BTMLEConfig:
    ridge_lambda: float = 1e-8
    fallback_ridge_lambda: float = 1e-4
    max_abs_theta: float = 20.0
    max_iter: int = 1000
    tol: float = 1e-9
    method: str = "L-BFGS-B"
    init: str = "borda_logit"
    gauge: str = "last_zero"
    check_graph_connected: bool = True


def pair_graph_connected(N_r: np.ndarray) -> bool:
    N_r = np.asarray(N_r)
    K = N_r.shape[0]
    if K <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        neighbors = np.where(N_r[i] > 0)[0]
        for j in neighbors:
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == K


def borda_logit_init(W_r: np.ndarray, N_r: np.ndarray) -> np.ndarray:
    W_r = np.asarray(W_r, dtype=float)
    N_r = np.asarray(N_r, dtype=float)
    K = W_r.shape[0]
    p = np.full((K, K), 0.5, dtype=float)
    mask = N_r > 0
    p[mask] = (W_r[mask] + 0.5) / (N_r[mask] + 1.0)
    np.fill_diagonal(p, 0.0)
    b = p.sum(axis=1) / max(K - 1, 1)
    b = np.clip(b, 1e-6, 1.0 - 1e-6)
    theta0 = np.log(b / (1.0 - b))
    return theta0 - theta0.mean()


def bt_neg_loglik_and_grad(
    z: np.ndarray,
    W_r: np.ndarray,
    N_r: np.ndarray,
    ridge_lambda: float,
) -> tuple[float, np.ndarray]:
    W_r = np.asarray(W_r, dtype=float)
    N_r = np.asarray(N_r, dtype=float)
    K = W_r.shape[0]
    theta = np.zeros(K, dtype=float)
    theta[: K - 1] = z
    grad = np.zeros(K, dtype=float)
    nll = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            wij = W_r[i, j]
            wji = W_r[j, i]
            n = wij + wji
            if n <= 0:
                continue
            x = theta[i] - theta[j]
            nll += wij * np.logaddexp(0.0, -x) + wji * np.logaddexp(0.0, x)
            diff = n * stable_sigmoid(x) - wij
            grad[i] += diff
            grad[j] -= diff
    if ridge_lambda > 0.0:
        nll += 0.5 * ridge_lambda * float(theta @ theta)
        grad += ridge_lambda * theta
    return float(nll), grad[: K - 1]


def _bt_hessian_reduced(
    z: np.ndarray,
    W_r: np.ndarray,
    N_r: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    K = W_r.shape[0]
    theta = np.zeros(K, dtype=float)
    theta[: K - 1] = z
    H = np.zeros((K - 1, K - 1), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            n = W_r[i, j] + W_r[j, i]
            if n <= 0:
                continue
            p = stable_sigmoid(theta[i] - theta[j])
            w = n * p * (1.0 - p)
            if i < K - 1:
                H[i, i] += w
            if j < K - 1:
                H[j, j] += w
            if i < K - 1 and j < K - 1:
                H[i, j] -= w
                H[j, i] -= w
    if ridge_lambda > 0.0:
        H += ridge_lambda * np.eye(K - 1)
    return H


def _newton_minimize(
    z0: np.ndarray,
    W_r: np.ndarray,
    N_r: np.ndarray,
    ridge_lambda: float,
    max_iter: int,
    tol: float,
) -> dict:
    z = z0.astype(float, copy=True)
    val, grad = bt_neg_loglik_and_grad(z, W_r, N_r, ridge_lambda)
    success = False
    message = "max_iter reached"
    for it in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(grad, ord=np.inf))
        if grad_norm <= tol:
            success = True
            message = "gradient tolerance reached"
            break
        H = _bt_hessian_reduced(z, W_r, N_r, ridge_lambda)
        H.flat[:: H.shape[0] + 1] += 1e-10
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        step_norm = float(np.linalg.norm(step))
        if step_norm > 50.0:
            step *= 50.0 / step_norm
        directional = float(grad @ step)
        if directional <= 0.0 or not np.isfinite(directional):
            step = grad
            directional = float(grad @ step)
        line = 1.0
        accepted = False
        for _ in range(30):
            z_new = z - line * step
            val_new, grad_new = bt_neg_loglik_and_grad(z_new, W_r, N_r, ridge_lambda)
            if np.isfinite(val_new) and val_new <= val - 1e-4 * line * directional:
                z, val, grad = z_new, val_new, grad_new
                accepted = True
                break
            line *= 0.5
        if not accepted:
            message = "line search failed"
            break
        if np.linalg.norm(line * step, ord=np.inf) <= tol * (1.0 + np.linalg.norm(z, ord=np.inf)):
            success = True
            message = "step tolerance reached"
            break
    else:
        it = max_iter
    return {
        "x": z,
        "success": success,
        "nit": it,
        "fun": val,
        "message": message,
    }


def _initial_z(W_r: np.ndarray, N_r: np.ndarray, config: BTMLEConfig) -> np.ndarray:
    K = W_r.shape[0]
    if config.init == "zeros":
        theta0 = np.zeros(K, dtype=float)
    elif config.init == "borda_logit":
        theta0 = borda_logit_init(W_r, N_r)
    else:
        raise ValueError(f"unknown init: {config.init}")
    theta0 = theta0 - theta0[-1]
    return theta0[: K - 1]


def _fit_once(
    W_r: np.ndarray,
    N_r: np.ndarray,
    config: BTMLEConfig,
    ridge_lambda: float,
) -> dict:
    K = W_r.shape[0]
    if K == 1:
        return {
            "theta_hat": np.zeros(1),
            "success": True,
            "niter": 0,
            "fun": 0.0,
            "message": "single arm",
        }
    z0 = _initial_z(W_r, N_r, config)

    def objective(z):
        return bt_neg_loglik_and_grad(z, W_r, N_r, ridge_lambda)

    if minimize is not None:
        res = minimize(
            objective,
            z0,
            jac=True,
            method=config.method,
            tol=config.tol,
            options={"maxiter": config.max_iter},
        )
        x = res.x
        success = bool(res.success)
        nit = int(getattr(res, "nit", 0))
        fun = float(res.fun) if np.isfinite(res.fun) else np.inf
        message = str(res.message)
    else:
        res = _newton_minimize(
            z0, W_r, N_r, ridge_lambda, max_iter=config.max_iter, tol=config.tol
        )
        x = res["x"]
        success = bool(res["success"])
        nit = int(res["nit"])
        fun = float(res["fun"]) if np.isfinite(res["fun"]) else np.inf
        message = str(res["message"])
    theta = np.zeros(K, dtype=float)
    theta[: K - 1] = x
    theta = theta - theta.mean()
    return {
        "theta_hat": theta,
        "success": success,
        "niter": nit,
        "fun": fun,
        "message": message,
    }


def fit_bt_mle_coordinate(W_r: np.ndarray, N_r: np.ndarray, config: BTMLEConfig) -> dict:
    W_r = np.asarray(W_r, dtype=float)
    N_r = np.asarray(N_r, dtype=float)
    connected = pair_graph_connected(N_r)
    primary = _fit_once(W_r, N_r, config, config.ridge_lambda)
    max_abs = float(np.max(np.abs(primary["theta_hat"])))
    needs_fallback = (
        (config.check_graph_connected and not connected)
        or (not primary["success"])
        or (not np.isfinite(max_abs))
        or (max_abs > config.max_abs_theta)
    )
    result = primary
    ridge_used = config.ridge_lambda
    if needs_fallback:
        result = _fit_once(W_r, N_r, config, config.fallback_ridge_lambda)
        ridge_used = config.fallback_ridge_lambda

    theta = result["theta_hat"]
    return {
        "theta_hat": theta,
        "converged": bool(result["success"]),
        "ridge_fallback_used": bool(needs_fallback),
        "pure_mle_failed": bool(
            config.ridge_lambda == 0.0
            and (needs_fallback or not primary["success"])
        ),
        "max_abs_theta": float(np.max(np.abs(theta))),
        "niter": int(result["niter"]),
        "fun": float(result["fun"]),
        "message": result["message"],
        "graph_connected": bool(connected),
        "ridge_lambda_used": float(ridge_used),
    }


def fit_bt_mle(W: np.ndarray, Npair: np.ndarray, config: BTMLEConfig | None = None) -> dict:
    config = config or BTMLEConfig()
    W = np.asarray(W, dtype=float)
    Npair = np.asarray(Npair, dtype=float)
    d, K, _ = W.shape
    theta_hat = np.zeros((K, d), dtype=float)
    coordinate_results = []
    for r in range(d):
        res = fit_bt_mle_coordinate(W[r], Npair[r], config)
        theta_hat[:, r] = res["theta_hat"]
        coordinate_results.append(res)
    return {
        "theta_hat": theta_hat,
        "recommended": strict_pareto_set(theta_hat),
        "coordinate_results": coordinate_results,
        "mle_converged_all": all(r["converged"] for r in coordinate_results),
        "mle_ridge_fallback_any": any(
            r["ridge_fallback_used"] for r in coordinate_results
        ),
        "mle_max_abs_theta": float(np.max(np.abs(theta_hat))),
        "mle_mean_niter": float(np.mean([r["niter"] for r in coordinate_results])),
        "pure_mle_failed_any": any(r["pure_mle_failed"] for r in coordinate_results),
    }
