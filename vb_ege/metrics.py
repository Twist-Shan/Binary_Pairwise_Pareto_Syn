"""Metrics and summary helpers."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from .core import center_columns
from .compat import import_pandas_quietly

pd = import_pandas_quietly()


ALGORITHM_NAME_ALIASES = {
    "UniformPairwiseBT-MLE-FC": "UniformPairwiseBT-MLE-Cert",
}


def canonicalize_algorithm_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "algorithm" in out:
        out["algorithm"] = out["algorithm"].replace(ALGORITHM_NAME_ALIASES)
    return out


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

    df = canonicalize_algorithm_names(df)

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
                "q25_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.25),
                "q75_tau": _nanquantile(g.get("tau", pd.Series(dtype=float)), 0.75),
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
                "mean_pair_cell_count": _nanmean(
                    g.get("pair_cell_count", pd.Series(dtype=float))
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


def paired_tau_ratios(
    df: pd.DataFrame,
    baseline: str = "VB-EGE-practical",
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Compute robust per-replication stopping-time ratios against ``baseline``."""

    work = canonicalize_algorithm_names(df)
    required = {"experiment_id", "algorithm", "tau"}
    if not required.issubset(work.columns):
        return pd.DataFrame()
    work = work.copy()
    work["tau"] = pd.to_numeric(work["tau"], errors="coerce")
    work = work[np.isfinite(work["tau"]) & (work["tau"] > 0)]
    if work.empty:
        return pd.DataFrame()

    design_cols = [
        c
        for c in ["experiment_id", "K", "d", "budget", "delta"]
        if c in work.columns
    ]
    design_cols += sorted(
        c
        for c in work.columns
        if (c.startswith("param_") and not c.startswith("param_achieved_"))
        or c.startswith("meta_")
    )

    if "replicate_id" in work and work["replicate_id"].notna().all():
        work["_pair_id"] = work["replicate_id"].astype(str)
    else:
        sort_cols = [c for c in ["run_id"] if c in work.columns]
        if sort_cols:
            work = work.sort_values(sort_cols)
        work["_pair_id"] = work.groupby(
            design_cols + ["algorithm"], dropna=False
        ).cumcount()

    records: list[dict] = []
    for key, g in work.groupby(design_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        if g.duplicated(["_pair_id", "algorithm"]).any():
            continue
        wide = g.pivot(index="_pair_id", columns="algorithm", values="tau")
        if baseline not in wide:
            continue
        for algorithm in wide.columns:
            if algorithm == baseline:
                continue
            paired = wide[[baseline, algorithm]].dropna()
            if paired.empty:
                continue
            ratios = (paired[algorithm] / paired[baseline]).to_numpy(dtype=float)
            valid = np.isfinite(ratios) & (ratios > 0)
            ratios = ratios[valid]
            if not len(ratios):
                continue
            log_ratios = np.log(ratios)
            rng = np.random.default_rng(20260708 + len(records))
            cluster_values = None
            if "instance_id" in g and g["instance_id"].notna().any():
                pair_instances = (
                    g.dropna(subset=["instance_id"])
                    .drop_duplicates("_pair_id")
                    .set_index("_pair_id")["instance_id"]
                )
                ratio_index = paired.index.to_numpy()[valid]
                instance_ids = pair_instances.reindex(ratio_index).to_numpy()
                if pd.notna(instance_ids).all():
                    cluster_values = [
                        log_ratios[instance_ids == instance_id]
                        for instance_id in pd.unique(instance_ids)
                    ]
            if cluster_values and len(cluster_values) > 1:
                boot = np.empty(n_boot, dtype=float)
                for draw_index in range(n_boot):
                    sampled = rng.integers(0, len(cluster_values), size=len(cluster_values))
                    boot[draw_index] = np.median(
                        np.concatenate([cluster_values[index] for index in sampled])
                    )
            elif len(log_ratios) == 1:
                boot = np.repeat(log_ratios[0], n_boot)
            else:
                draws = rng.integers(0, len(log_ratios), size=(n_boot, len(log_ratios)))
                boot = np.median(log_ratios[draws], axis=1)
            rec = dict(zip(design_cols, key))
            rec.update(
                {
                    "algorithm": algorithm,
                    "baseline": baseline,
                    "paired_n": int(len(ratios)),
                    "median_ratio": float(np.exp(np.median(log_ratios))),
                    "q25_ratio": float(np.exp(np.quantile(log_ratios, 0.25))),
                    "q75_ratio": float(np.exp(np.quantile(log_ratios, 0.75))),
                    "q90_ratio": float(np.exp(np.quantile(log_ratios, 0.90))),
                    "bootstrap_ci_lower": float(np.exp(np.quantile(boot, 0.025))),
                    "bootstrap_ci_upper": float(np.exp(np.quantile(boot, 0.975))),
                    "bootstrap_unit": "instance_id" if cluster_values else "replication",
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
