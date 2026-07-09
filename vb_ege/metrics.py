"""Metrics and summary helpers."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from .core import center_columns
from .compat import import_pandas_quietly

pd = import_pandas_quietly()


def set_error(recommended, true_pareto) -> bool:
    return set(recommended) != set(true_pareto)


def hamming_set_distance(A, B) -> int:
    return len(set(A).symmetric_difference(set(B)))


def wilson_ci(num_errors: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = num_errors / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def _nanmean(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").mean())


def _nanquantile(s: pd.Series, q: float) -> float:
    values = pd.to_numeric(s, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw replicate rows into plotting-ready grouped metrics."""

    preferred = [
        "experiment_id",
        "algorithm",
        "K",
        "d",
        "budget",
        "delta",
        "sample_const",
        "threshold_const",
        "mle_ridge_lambda",
        "mle_init",
        "mle_allocation_scheme",
    ]
    # Achieved quantities are run metadata, not design parameters.  In
    # particular, correlated-arena instances have a different achieved
    # correlation for each seed; grouping by it would split a 300-rep cell
    # into many one-rep summaries.
    param_cols = sorted(
        c
        for c in df.columns
        if c.startswith("param_")
        and not c.startswith("param_achieved_objective_correlation")
    )
    meta_cols = sorted(c for c in df.columns if c.startswith("meta_"))
    group_cols = [c for c in preferred if c in df.columns] + param_cols + meta_cols
    if not group_cols:
        raise ValueError("no grouping columns found")
    if "seed" in df.columns:
        dedupe_cols = group_cols + ["seed"]
        if "run_id" in df.columns:
            df = df.sort_values("run_id")
        df = df.drop_duplicates(dedupe_cols, keep="last")

    records: list[dict] = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))
        n = len(g)
        errors = int(pd.to_numeric(g["error"], errors="coerce").fillna(False).sum())
        lo, hi = wilson_ci(errors, n)
        rec.update(
            {
                "n_reps": n,
                "error_rate": errors / n if n else np.nan,
                "wilson_lower": lo,
                "wilson_upper": hi,
                "mean_tau": _nanmean(g.get("tau", pd.Series(dtype=float))),
                "median_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.50),
                "q80_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.80),
                "q90_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.90),
                "q95_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.95),
                "q99_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.99),
                "mean_stopped": _nanmean(g.get("stopped", pd.Series(dtype=float))),
                "mean_hamming": _nanmean(g.get("hamming", pd.Series(dtype=float))),
                "mean_pareto_size_true": _nanmean(
                    g.get("pareto_size_true", pd.Series(dtype=float))
                ),
                "mean_pareto_target": _nanmean(
                    g.get("pareto_target", pd.Series(dtype=float))
                ),
                "mean_pareto_size_hat": _nanmean(
                    g.get("pareto_size_hat", pd.Series(dtype=float))
                ),
                "mean_num_accepted": _nanmean(
                    g.get("num_accepted", pd.Series(dtype=float))
                ),
                "mean_num_rejected": _nanmean(
                    g.get("num_rejected", pd.Series(dtype=float))
                ),
                "mean_final_phase": _nanmean(
                    g.get("num_phases", pd.Series(dtype=float))
                ),
                "mean_H_B": _nanmean(g.get("H_B", pd.Series(dtype=float))),
                "mean_H_theta": _nanmean(g.get("H_theta", pd.Series(dtype=float))),
                "mean_delta_min_B": _nanmean(
                    g.get("delta_min_B", pd.Series(dtype=float))
                ),
                "mean_delta_min_theta": _nanmean(
                    g.get("delta_min_theta", pd.Series(dtype=float))
                ),
                "mean_norm_tau_B": _nanmean(
                    g.get("norm_tau_B", pd.Series(dtype=float))
                ),
                "mean_norm_tau_B_logdelta": _nanmean(
                    g.get("norm_tau_B_logdelta", pd.Series(dtype=float))
                ),
                "mean_theta_rmse_centered": _nanmean(
                    g.get("theta_rmse_centered", pd.Series(dtype=float))
                ),
                "mean_pairwise_sign_accuracy": _nanmean(
                    g.get("pairwise_sign_accuracy", pd.Series(dtype=float))
                ),
                "mean_mle_converged_all": _nanmean(
                    g.get("mle_converged_all", pd.Series(dtype=float))
                ),
                "mean_mle_ridge_fallback_any": _nanmean(
                    g.get("mle_ridge_fallback_any", pd.Series(dtype=float))
                ),
                "mean_pair_cell_coverage": _nanmean(
                    g.get("pair_cell_coverage", pd.Series(dtype=float))
                ),
                "mean_achieved_objective_correlation": _nanmean(
                    g.get("achieved_objective_correlation", pd.Series(dtype=float))
                ),
                "mean_false_positive_count": _nanmean(
                    g.get("false_positive_count", pd.Series(dtype=float))
                ),
                "mean_false_negative_count": _nanmean(
                    g.get("false_negative_count", pd.Series(dtype=float))
                ),
                "median_final_phase": _nanquantile(
                    g.get("num_phases", pd.Series(dtype=float)), 0.50
                ),
                "q95_final_phase": _nanquantile(
                    g.get("num_phases", pd.Series(dtype=float)), 0.95
                ),
            }
        )
        records.append(rec)
    return pd.DataFrame.from_records(records)


def loglog_slope(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return (np.nan, np.nan)
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return (np.nan, np.nan)
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return (float(slope), float(intercept))


def centered_theta_rmse(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    theta_hat_c = center_columns(np.asarray(theta_hat, dtype=float))
    theta_true_c = center_columns(np.asarray(theta_true, dtype=float))
    return float(np.sqrt(np.mean((theta_hat_c - theta_true_c) ** 2)))


def pairwise_sign_accuracy(
    theta_hat: np.ndarray, theta_true: np.ndarray, atol: float = 1e-12
) -> float:
    theta_hat = np.asarray(theta_hat, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)
    K, d = theta_true.shape
    correct = 0
    total = 0
    for r in range(d):
        for i in range(K):
            for j in range(i + 1, K):
                true_diff = theta_true[i, r] - theta_true[j, r]
                if abs(true_diff) <= atol:
                    continue
                total += 1
                hat_diff = theta_hat[i, r] - theta_hat[j, r]
                if np.sign(hat_diff) == np.sign(true_diff):
                    correct += 1
    return float(correct / total) if total else np.nan
