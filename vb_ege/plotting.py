"""Matplotlib plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .compat import import_pandas_quietly
from .core import borda, strict_pareto_set

pd = import_pandas_quietly()


def _savefig(outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    alt = outpath.with_suffix(".png" if outpath.suffix.lower() == ".pdf" else ".pdf")
    plt.savefig(alt)
    plt.close()


def plot_2d_instance(theta, outpath):
    theta = np.asarray(theta)
    if theta.shape[1] != 2:
        return
    pareto = set(strict_pareto_set(theta))
    colors = ["tab:red" if i in pareto else "tab:blue" for i in range(theta.shape[0])]
    plt.figure(figsize=(5.5, 4.0))
    plt.scatter(theta[:, 0], theta[:, 1], c=colors, s=28)
    plt.xlabel("theta[0]")
    plt.ylabel("theta[1]")
    plt.title(f"Latent instance, K={theta.shape[0]}, |P|={len(pareto)}")
    _savefig(outpath)


def plot_borda_embedding_2d(theta, outpath):
    b = borda(theta)
    if b.shape[1] != 2:
        return
    pareto = set(strict_pareto_set(b))
    colors = ["tab:red" if i in pareto else "tab:blue" for i in range(b.shape[0])]
    plt.figure(figsize=(5.5, 4.0))
    plt.scatter(b[:, 0], b[:, 1], c=colors, s=28)
    plt.xlabel("Borda[0]")
    plt.ylabel("Borda[1]")
    plt.title(f"Borda embedding, K={b.shape[0]}, |P|={len(pareto)}")
    _savefig(outpath)


def _curve_title(df: pd.DataFrame, y: str) -> str:
    row = df.iloc[0]
    n_reps = int(df["n_reps"].max()) if "n_reps" in df else len(df)
    return f"{row.get('experiment_id', '')}: K={row.get('K', '')}, d={row.get('d', '')}, reps={n_reps}, {y}"


def plot_budget_curve(summary_df, outpath, x="budget"):
    df = summary_df.dropna(subset=[x, "error_rate"]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values(x)
        floor = np.maximum(g["error_rate"].to_numpy(dtype=float), 0.5 / g["n_reps"].to_numpy(dtype=float))
        plt.plot(g[x], np.log10(floor), marker="o", label=algorithm)
    plt.xscale("log")
    plt.xlabel(x)
    plt.ylabel("log10(error rate floor)")
    plt.title(_curve_title(df, "error"))
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_budget_curve_normalized(summary_df, outpath):
    df = summary_df.copy()
    if "mean_norm_tau_B" not in df:
        return
    df = df.dropna(subset=["mean_norm_tau_B", "error_rate"])
    if df.empty:
        return
    plot_budget_curve(df.rename(columns={"mean_norm_tau_B": "budget_norm"}), outpath, x="budget_norm")


def plot_hamming_curve(summary_df, outpath):
    df = summary_df.dropna(subset=["budget", "mean_hamming"]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values("budget")
        plt.plot(g["budget"], g["mean_hamming"], marker="o", label=algorithm)
    plt.xscale("log")
    plt.xlabel("budget")
    plt.ylabel("mean symmetric-difference distance")
    plt.title(_curve_title(df, "hamming"))
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_stopping_scaling(summary_df, sweep_var, outpath):
    df = summary_df.dropna(subset=[sweep_var, "mean_tau"]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values(sweep_var)
        plt.plot(g[sweep_var], g["mean_tau"], marker="o", label=algorithm)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(sweep_var)
    plt.ylabel("mean stopping time")
    plt.title(f"Stopping scaling vs {sweep_var}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_tau_quantiles_vs_x(
    summary_df,
    x_col: str,
    outpath,
    *,
    x_transform=None,
    xlabel: str | None = None,
    normalized: bool = False,
):
    y_cols = (
        ("median_tau", "q90_tau", "q95_tau")
        if not normalized
        else ("mean_norm_tau_B", None, None)
    )
    required = [x_col, y_cols[0]]
    df = summary_df.dropna(subset=required).copy()
    if df.empty:
        return
    if x_transform is not None:
        df["_x"] = x_transform(df[x_col].to_numpy(dtype=float))
    else:
        df["_x"] = df[x_col].astype(float)
    ylabel = "median stopping time" if not normalized else "mean tau / (d H_B)"
    plt.figure(figsize=(6.2, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values("_x")
        x = g["_x"].to_numpy(dtype=float)
        y = g[y_cols[0]].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=algorithm)
        if y_cols[1] and y_cols[2] and y_cols[1] in g and y_cols[2] in g:
            lo = g[y_cols[1]].to_numpy(dtype=float)
            hi = g[y_cols[2]].to_numpy(dtype=float)
            plt.fill_between(x, lo, hi, alpha=0.12)
    if (df["_x"] > 0).all() and (df[y_cols[0]] > 0).all():
        plt.yscale("log")
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel)
    plt.title(f"{_curve_title(df, ylabel)} vs {xlabel or x_col}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_baseline_ratio_vs_x(summary_df, x_col: str, outpath, baseline="VB-EGE-practical"):
    df = summary_df.dropna(subset=[x_col, "mean_tau"]).copy()
    if df.empty or baseline not in set(df["algorithm"]):
        return
    rows = []
    for key, g in df.groupby(x_col, dropna=False):
        base = g[g["algorithm"] == baseline]
        if base.empty:
            continue
        base_tau = float(base["mean_tau"].iloc[0])
        if not np.isfinite(base_tau) or base_tau <= 0:
            continue
        for _, row in g.iterrows():
            if row["algorithm"] == baseline:
                continue
            rows.append(
                {
                    x_col: key,
                    "algorithm": row["algorithm"],
                    "ratio": float(row["mean_tau"]) / base_tau,
                }
            )
    if not rows:
        return
    r = pd.DataFrame(rows).sort_values(x_col)
    plt.figure(figsize=(6.2, 4.2))
    for algorithm, g in r.groupby("algorithm"):
        plt.plot(g[x_col], g["ratio"], marker="o", label=algorithm)
    plt.xscale("log" if (r[x_col] > 0).all() else "linear")
    plt.yscale("log")
    plt.xlabel(x_col)
    plt.ylabel(f"mean tau / {baseline} mean tau")
    plt.title(f"Baseline ratios vs {x_col}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_metric_vs_x(summary_df, x_col: str, y_col: str, outpath, ylabel: str | None = None):
    df = summary_df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.2, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values(x_col)
        plt.plot(g[x_col], g[y_col], marker="o", label=algorithm)
    plt.xlabel(x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(f"{ylabel or y_col} vs {x_col}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_calibration_frontier(summary_df, outpath, y_col="error_rate"):
    df = summary_df.dropna(subset=["mean_tau", y_col, "sample_const", "threshold_const"]).copy()
    df = df[df["algorithm"] == "VB-EGE-practical"]
    if df.empty:
        return
    plt.figure(figsize=(6.4, 4.4))
    for exp_id, g in df.groupby("experiment_id"):
        plt.scatter(g["mean_tau"], g[y_col], label=exp_id, alpha=0.8)
        for _, row in g.iterrows():
            if row["sample_const"] == 2.0 and row["threshold_const"] == 4.0:
                plt.scatter(row["mean_tau"], row[y_col], marker="*", s=180, color="black")
    plt.xscale("log")
    if (df[y_col] > 0).any():
        plt.yscale("log")
    plt.xlabel("mean stopping time")
    plt.ylabel(y_col)
    plt.title(f"Constants calibration frontier: {y_col}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_calibration_heatmap(summary_df, exp_id: str, metric: str, outpath):
    df = summary_df[
        (summary_df["experiment_id"] == exp_id)
        & (summary_df["algorithm"] == "VB-EGE-practical")
    ].dropna(subset=["sample_const", "threshold_const", metric])
    if df.empty:
        return
    pivot = df.pivot_table(
        index="threshold_const",
        columns="sample_const",
        values=metric,
        aggfunc="mean",
    ).sort_index(ascending=True)
    if pivot.empty:
        return
    plt.figure(figsize=(5.8, 4.6))
    values = pivot.to_numpy(dtype=float)
    im = plt.imshow(values, origin="lower", aspect="auto")
    plt.colorbar(im, label=metric)
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("sample_const")
    plt.ylabel("threshold_const")
    plt.title(f"{metric}: {exp_id}")
    if 2.0 in pivot.columns and 4.0 in pivot.index:
        x = list(pivot.columns).index(2.0)
        y = list(pivot.index).index(4.0)
        plt.scatter([x], [y], marker="*", s=220, color="white", edgecolor="black")
    _savefig(outpath)


def plot_allocation_vs_gap(result, true_gaps, outpath):
    N = np.asarray(result.N if hasattr(result, "N") else result["N"])
    gaps = np.asarray(true_gaps["delta"])
    alloc = N.sum(axis=1)
    plt.figure(figsize=(5.5, 4.0))
    plt.scatter(gaps, alloc, s=28)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("true gap")
    plt.ylabel("samples")
    plt.title("Allocation vs true gap")
    _savefig(outpath)


def plot_mle_theta_rmse(summary_df, outpath):
    df = summary_df.dropna(subset=["budget", "mean_theta_rmse_centered"]).copy()
    df = df[df["algorithm"].astype(str).str.contains("MLE", na=False)]
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values("budget")
        plt.plot(g["budget"], g["mean_theta_rmse_centered"], marker="o", label=algorithm)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("budget")
    plt.ylabel("centered theta RMSE")
    plt.title(_curve_title(df, "MLE RMSE"))
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_mle_sign_accuracy(summary_df, outpath):
    df = summary_df.dropna(subset=["budget", "mean_pairwise_sign_accuracy"]).copy()
    df = df[df["algorithm"].astype(str).str.contains("MLE", na=False)]
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values("budget")
        plt.plot(g["budget"], g["mean_pairwise_sign_accuracy"], marker="o", label=algorithm)
    plt.xscale("log")
    plt.xlabel("budget")
    plt.ylabel("pairwise sign accuracy")
    plt.ylim(0.0, 1.02)
    plt.title(_curve_title(df, "MLE sign accuracy"))
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_pair_cell_coverage_effect(summary_df, outpath):
    df = summary_df.dropna(subset=["mean_pair_cell_coverage", "error_rate"]).copy()
    df = df[df["algorithm"].astype(str).str.contains("Pairwise", na=False)]
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values("mean_pair_cell_coverage")
        floor = np.maximum(g["error_rate"].to_numpy(dtype=float), 0.5 / g["n_reps"].to_numpy(dtype=float))
        plt.plot(g["mean_pair_cell_coverage"], np.log10(floor), marker="o", label=algorithm)
    plt.xscale("log")
    plt.xlabel("pair-cell coverage")
    plt.ylabel("log10(error rate floor)")
    plt.title("Pair-cell coverage effect")
    plt.axvline(1.0, color="black", linestyle=":", linewidth=1.0)
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_fixed_confidence_tau(summary_df, outpath):
    df = summary_df.dropna(subset=["mean_tau"]).copy()
    if df.empty:
        return
    df = df.sort_values("mean_tau")
    labels = df["algorithm"].astype(str).tolist()
    values = df["mean_tau"].to_numpy(dtype=float)
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(labels, values, color="tab:blue", alpha=0.85)
    plt.yscale("log")
    plt.ylabel("mean stopping time")
    plt.title(_curve_title(df, "fixed-confidence stopping time"))
    plt.xticks(rotation=25, ha="right")
    _savefig(outpath)
