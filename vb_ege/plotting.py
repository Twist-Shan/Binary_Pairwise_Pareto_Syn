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
