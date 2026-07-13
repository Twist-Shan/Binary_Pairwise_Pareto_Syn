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
    return f"K={row.get('K', '')}, d={row.get('d', '')}, reps={n_reps}, {y}"


def _short_series_label(value) -> str:
    s = str(value)
    lower = s.lower()
    if "arena10" in lower:
        return "Arena-10"
    if "arena4" in lower:
        return "Arena-4"
    if "witness10" in lower:
        return "Witness-10"
    if "symmetric" in lower:
        return "Symmetric"
    return s


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


def plot_budget_ratio_curve(
    summary_df,
    outpath,
    y_col: str,
    ylabel: str,
    *,
    log10_floor: bool = False,
):
    required = ["budget", "meta_tau_ref", y_col]
    df = summary_df.dropna(subset=required).copy()
    if df.empty:
        return
    df["_budget_ratio"] = df["budget"].astype(float) / df["meta_tau_ref"].astype(float)
    if log10_floor:
        if "n_reps" not in df:
            return
        y = np.maximum(
            df[y_col].to_numpy(dtype=float),
            0.5 / df["n_reps"].to_numpy(dtype=float),
        )
        df["_y"] = np.log10(y)
    else:
        df["_y"] = df[y_col].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["_budget_ratio", "_y"])
    if df.empty:
        return
    plt.figure(figsize=(6.2, 4.2))
    for algorithm, g in df.groupby("algorithm", dropna=False):
        g = g.sort_values("_budget_ratio")
        plt.plot(g["_budget_ratio"], g["_y"], marker="o", label=str(algorithm))
    plt.xscale("log")
    plt.xlabel("budget / default VB-EGE mean tau")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs under-budget ratio")
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
    plt.title(f"{_curve_title(df, ylabel)} vs {xlabel or x_col}", fontsize=9)
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_baseline_ratio_vs_x(
    summary_df,
    x_col: str,
    outpath,
    baseline="VB-EGE-practical",
    xlabel: str | None = None,
):
    df = summary_df.dropna(subset=[x_col, "mean_tau"]).copy()
    if df.empty or baseline not in set(df["algorithm"]):
        return
    group_cols = [x_col]
    multi_setting = "experiment_id" in df and df["experiment_id"].nunique(dropna=True) > 1
    if multi_setting:
        group_cols = ["experiment_id", x_col]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_cols, key))
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
                    x_col: key_map[x_col],
                    "experiment_id": key_map.get("experiment_id", ""),
                    "algorithm": row["algorithm"],
                    "ratio": float(row["mean_tau"]) / base_tau,
                }
            )
    if not rows:
        return
    r = pd.DataFrame(rows).sort_values(x_col)
    plt.figure(figsize=(6.2, 4.2))
    plot_groups = (
        r.groupby(["algorithm", "experiment_id"], dropna=False)
        if multi_setting
        else r.groupby("algorithm", dropna=False)
    )
    for key, g in plot_groups:
        if isinstance(key, tuple):
            algorithm, exp_id = key
            label = f"{algorithm} ({_short_series_label(exp_id)})"
        else:
            label = key
        g = g.sort_values(x_col)
        plt.plot(g[x_col], g["ratio"], marker="o", label=label)
    plt.xscale("log" if (r[x_col] > 0).all() else "linear")
    plt.yscale("log")
    x_label = xlabel or x_col
    plt.xlabel(x_label)
    plt.ylabel(f"mean tau / {baseline} mean tau")
    plt.title(f"Baseline ratios vs {x_label}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_metric_vs_x(
    summary_df,
    x_col: str,
    y_col: str,
    outpath,
    ylabel: str | None = None,
    series_col: str | None = None,
    xlabel: str | None = None,
):
    df = summary_df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.2, 4.2))
    group_cols = ["algorithm"]
    if series_col and series_col in df and df[series_col].nunique(dropna=True) > 1:
        group_cols.append(series_col)
    for key, g in df.groupby(group_cols, dropna=False):
        if isinstance(key, tuple):
            algorithm = key[0]
            series = key[1] if len(key) > 1 else None
            label = (
                f"{algorithm} ({_short_series_label(series)})"
                if series is not None
                else str(algorithm)
            )
        else:
            label = str(key)
        g = g.sort_values(x_col)
        plt.plot(g[x_col], g[y_col], marker="o", label=label)
    x_label = xlabel or x_col
    plt.xlabel(x_label)
    plt.ylabel(ylabel or y_col)
    plt.title(f"{ylabel or y_col} vs {x_label}")
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_calibration_frontier(summary_df, outpath, y_col="error_rate"):
    df = summary_df.dropna(subset=["median_tau", y_col, "sample_const", "threshold_const"]).copy()
    df = df[df["algorithm"] == "VB-EGE-practical"]
    if df.empty:
        return
    plt.figure(figsize=(6.4, 4.4))
    for exp_id, g in df.groupby("experiment_id"):
        plt.scatter(g["median_tau"], g[y_col], label=exp_id, alpha=0.8)
        for _, row in g.iterrows():
            if row["sample_const"] == 2.0 and row["threshold_const"] == 4.0:
                plt.scatter(row["median_tau"], row[y_col], marker="*", s=180, color="black")
    plt.xscale("log")
    if (df[y_col] > 0).any():
        plt.yscale("log")
    plt.xlabel("median stopping time")
    plt.ylabel(y_col)
    plt.title(f"Constant sensitivity: {y_col}")
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
    df = summary_df.dropna(subset=["mean_tau"]).copy()
    if df.empty or "mean_pair_cell_coverage" not in df:
        return
    param_cols = [
        c
        for c in df.columns
        if c.startswith("param_")
        and not c.startswith("param_achieved_objective_correlation")
    ]
    setting_cols = [
        c
        for c in ["experiment_id", "K", "d", "budget", "delta"]
        if c in df.columns
    ] + param_cols

    def key_from(row):
        key = []
        for col in setting_cols:
            val = row.get(col)
            if pd.isna(val):
                val = "<NA>"
            key.append(val)
        return tuple(key)

    vb_rows = df[df["algorithm"] == "VB-EGE-practical"]
    vb_tau = {key_from(row): float(row["mean_tau"]) for _, row in vb_rows.iterrows()}
    rows = []
    pairwise = df[df["algorithm"].astype(str).str.contains("Pairwise", na=False)]
    pairwise = pairwise.dropna(subset=["mean_pair_cell_coverage"])
    for _, row in pairwise.iterrows():
        base_tau = vb_tau.get(key_from(row))
        tau = float(row["mean_tau"])
        coverage = float(row["mean_pair_cell_coverage"])
        K = float(row.get("K", np.nan))
        d = float(row.get("d", np.nan))
        pair_cells = float(row.get("mean_pair_cell_count", np.nan))
        focal_cells = K * d
        if (
            not base_tau
            or base_tau <= 0
            or tau <= 0
            or coverage <= 0
            or not np.isfinite(pair_cells)
            or pair_cells <= 0
            or not np.isfinite(focal_cells)
            or focal_cells <= 0
        ):
            continue
        rows.append(
            {
                "algorithm": row["algorithm"],
                "cell_multiplier": pair_cells / focal_cells,
                "per_cell_burden_ratio": coverage / (base_tau / focal_cells),
            }
        )
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in plot_df.groupby("algorithm"):
        plt.scatter(
            g["cell_multiplier"],
            g["per_cell_burden_ratio"],
            label=algorithm,
            alpha=0.8,
            s=28,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("pairwise / focal basic-cell count")
    plt.ylabel("pairwise / VB samples per basic cell")
    plt.title("Pairwise total-cost decomposition")
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    plt.legend(fontsize=8)
    _savefig(outpath)


def plot_fixed_confidence_tau(summary_df, outpath):
    df = summary_df.dropna(subset=["median_tau", "q25_tau", "q75_tau"]).copy()
    if df.empty:
        return
    df = df.sort_values("median_tau")
    labels = df["algorithm"].astype(str).tolist()
    values = df["median_tau"].to_numpy(dtype=float)
    lower = values - df["q25_tau"].to_numpy(dtype=float)
    upper = df["q75_tau"].to_numpy(dtype=float) - values
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(
        labels,
        values,
        yerr=np.vstack([lower, upper]),
        color="tab:blue",
        alpha=0.85,
        capsize=3,
    )
    plt.yscale("log")
    plt.ylabel("median stopping time (IQR)")
    plt.title(_curve_title(df, "fixed-confidence median stopping time"))
    plt.xticks(rotation=25, ha="right")
    _savefig(outpath)


def plot_paired_ratio_by_setting(paired_df, outpath):
    required = {
        "experiment_id",
        "algorithm",
        "median_ratio",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    }
    if paired_df.empty or not required.issubset(paired_df.columns):
        return
    df = paired_df.dropna(subset=list(required)).copy()
    if df.empty:
        return
    settings = list(dict.fromkeys(df["experiment_id"].astype(str)))
    algorithms = list(dict.fromkeys(df["algorithm"].astype(str)))
    y0 = np.arange(len(settings), dtype=float)
    width = 0.7 / max(1, len(algorithms))
    plt.figure(figsize=(8.0, max(4.2, 0.55 * len(settings) + 1.5)))
    for index, algorithm in enumerate(algorithms):
        g = df[df["algorithm"] == algorithm].set_index("experiment_id")
        xs = []
        lows = []
        highs = []
        ys = []
        for setting_index, setting in enumerate(settings):
            if setting not in g.index:
                continue
            row = g.loc[setting]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x = float(row["median_ratio"])
            xs.append(x)
            lows.append(x - float(row["bootstrap_ci_lower"]))
            highs.append(float(row["bootstrap_ci_upper"]) - x)
            ys.append(y0[setting_index] + (index - (len(algorithms) - 1) / 2) * width)
        if xs:
            plt.errorbar(
                xs,
                ys,
                xerr=np.vstack([lows, highs]),
                fmt="o",
                capsize=3,
                label=algorithm,
            )
    labels = [s.removeprefix("fc_").replace("_", " ") for s in settings]
    plt.yticks(y0, labels)
    plt.xscale("log")
    plt.axvline(1.0, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("median paired stopping-time ratio to VB-EGE (95% bootstrap CI)")
    plt.ylabel("benchmark setting")
    plt.title("Paired fixed-confidence sample cost")
    plt.legend(fontsize=8)
    _savefig(outpath)
