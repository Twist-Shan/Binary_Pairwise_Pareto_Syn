"""Matplotlib plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from .compat import import_pandas_quietly
from .core import borda, strict_pareto_set

pd = import_pandas_quietly()

ALGORITHM_ORDER = [
    "VB-EGE-practical",
    "VB-EGE-theory",
    "UniformFocalBorda-FC",
    "UniformPairwiseBT-MLE-Cert",
    "UniformPairwiseBT-BordaPlugIn-FC",
]
ALGORITHM_COLORS = {
    "UniformFocalBorda-FC": "#8EC5E8",
    "UniformPairwiseBT-BordaPlugIn-FC": "#F7C59F",
    "UniformPairwiseBT-MLE-Cert": "#95D5B2",
    "VB-EGE-practical": "#F28E8E",
    "VB-EGE-theory": "#C4B0E3",
}
ALGORITHM_LINE_COLORS = {
    "UniformFocalBorda-FC": "#3E86B8",
    "UniformPairwiseBT-BordaPlugIn-FC": "#D47732",
    "UniformPairwiseBT-MLE-Cert": "#398F68",
    "VB-EGE-practical": "#C94F5D",
    "VB-EGE-theory": "#8068AD",
}
ALGORITHM_MARKERS = {
    "UniformFocalBorda-FC": "s",
    "UniformPairwiseBT-BordaPlugIn-FC": "D",
    "UniformPairwiseBT-MLE-Cert": "^",
    "VB-EGE-practical": "o",
    "VB-EGE-theory": "P",
}
ALGORITHM_LINESTYLES = {
    "UniformFocalBorda-FC": "--",
    "UniformPairwiseBT-BordaPlugIn-FC": "-",
    "UniformPairwiseBT-MLE-Cert": "-.",
    "VB-EGE-practical": "-",
    "VB-EGE-theory": ":",
}
ALGORITHM_LABELS = {
    "UniformFocalBorda-FC": "Focal Borda",
    "UniformPairwiseBT-BordaPlugIn-FC": "Borda plug-in",
    "UniformPairwiseBT-MLE-Cert": "BT-MLE",
    "VB-EGE-practical": "VB-EGE",
    "VB-EGE-theory": "Theory-style",
}
MACARON_SERIES_COLORS = [
    "#8EC5E8",
    "#F28E8E",
    "#95D5B2",
    "#F7C59F",
    "#C4B0E3",
    "#F6D6A8",
]
DARK_SERIES_COLORS = [
    "#3E86B8",
    "#C94F5D",
    "#398F68",
    "#D47732",
    "#8068AD",
    "#B58A45",
]
MACARON_CMAP = LinearSegmentedColormap.from_list(
    "macaron",
    ["#FFF8E7", "#F6D6A8", "#F7C59F", "#F28E8E", "#C4B0E3"],
)
METRIC_LABELS = {
    "error_rate": "empirical error",
    "wilson_upper": "95% Wilson upper bound",
    "median_tau": "median stopping time",
    "mean_tau": "mean stopping time",
    "mean_stopped": "stopped fraction",
}


def _ordered_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"].dropna().astype(str))
    ordered = [name for name in ALGORITHM_ORDER if name in present]
    return ordered + sorted(present - set(ordered))


def _algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm)


def _mean_se_bounds(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = g["mean_tau"].to_numpy(dtype=float)
    if "se_tau" in g.columns:
        se = pd.to_numeric(g["se_tau"], errors="coerce").to_numpy(dtype=float)
        se = np.where(np.isfinite(se), np.maximum(se, 0.0), 0.0)
        lower = values - se
        upper = values + se
        return np.maximum(lower, np.finfo(float).tiny), np.maximum(upper, values)
    if {"mean_tau_ci_lower", "mean_tau_ci_upper"}.issubset(g.columns):
        ci_lower = g["mean_tau_ci_lower"].to_numpy(dtype=float)
        ci_upper = g["mean_tau_ci_upper"].to_numpy(dtype=float)
        se = np.maximum(ci_upper - ci_lower, 0.0) / (2.0 * 1.96)
        lower = values - se
        upper = values + se
        return np.maximum(lower, np.finfo(float).tiny), np.maximum(upper, values)
    return values, values


def _fill_color(algorithm: str, fallback: str = "#8EC5E8") -> str:
    return ALGORITHM_COLORS.get(algorithm, fallback)


def _line_color(algorithm: str, fallback: str = "#3E86B8") -> str:
    return ALGORITHM_LINE_COLORS.get(algorithm, fallback)


def _shade_mean_ribbon(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fill_color: str,
) -> None:
    plt.fill_between(
        x,
        lower,
        upper,
        color=fill_color,
        alpha=0.48,
        linewidth=0,
        zorder=1,
    )
    if np.any(np.isfinite(upper - lower) & ((upper - lower) > 0)):
        plt.plot(x, lower, color=fill_color, alpha=0.95, linewidth=0.75, zorder=2)
        plt.plot(x, upper, color=fill_color, alpha=0.95, linewidth=0.75, zorder=2)


def _add_legend_below(ncol: int = 2) -> None:
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(ncol, len(labels)),
        frameon=False,
        fontsize=7.5,
        columnspacing=1.1,
        handlelength=2.0,
    )


def _shade_bar_intervals(
    centers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    width: float,
    color: str,
) -> None:
    for center, lo, hi in zip(centers, lower, upper):
        if not (np.isfinite(center) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        lo = max(float(lo), np.finfo(float).tiny)
        hi = max(float(hi), lo)
        plt.fill_between(
            [center - 0.42 * width, center + 0.42 * width],
            [lo, lo],
            [hi, hi],
            color=color,
            alpha=0.42,
            linewidth=0,
            zorder=4,
        )


def _savefig(outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
    alt = outpath.with_suffix(".png" if outpath.suffix.lower() == ".pdf" else ".pdf")
    plt.savefig(alt, bbox_inches="tight", pad_inches=0.04)
    plt.close()


def plot_2d_instance(theta, outpath):
    theta = np.asarray(theta)
    if theta.shape[1] != 2:
        return
    pareto = set(strict_pareto_set(theta))
    colors = ["#F28E8E" if i in pareto else "#8EC5E8" for i in range(theta.shape[0])]
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
    colors = ["#F28E8E" if i in pareto else "#8EC5E8" for i in range(b.shape[0])]
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
    for algorithm in _ordered_algorithms(df):
        g = df[df["algorithm"] == algorithm]
        g = g.sort_values(x)
        floor = np.maximum(g["error_rate"].to_numpy(dtype=float), 0.5 / g["n_reps"].to_numpy(dtype=float))
        plt.plot(
            g[x],
            np.log10(floor),
            marker="o",
            color=_line_color(algorithm),
            label=_algorithm_label(algorithm),
        )
    plt.xscale("log")
    plt.xlabel(x)
    plt.ylabel("log10(error rate floor)")
    plt.title(_curve_title(df, "error"))
    _add_legend_below()
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
    for algorithm in _ordered_algorithms(df):
        g = df[df["algorithm"] == algorithm]
        g = g.sort_values("budget")
        plt.plot(
            g["budget"],
            g["mean_hamming"],
            marker="o",
            color=_line_color(algorithm),
            label=_algorithm_label(algorithm),
        )
    plt.xscale("log")
    plt.xlabel("budget")
    plt.ylabel("mean symmetric-difference distance")
    plt.title(_curve_title(df, "hamming"))
    _add_legend_below()
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
    for algorithm in _ordered_algorithms(df):
        g = df[df["algorithm"] == algorithm]
        g = g.sort_values("_budget_ratio")
        plt.plot(
            g["_budget_ratio"],
            g["_y"],
            marker="o",
            color=_line_color(algorithm),
            label=_algorithm_label(algorithm),
        )
    plt.xscale("log")
    plt.xlabel("budget / default VB-EGE mean tau")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs under-budget ratio")
    _add_legend_below()
    _savefig(outpath)


def plot_stopping_scaling(summary_df, sweep_var, outpath, xlabel: str | None = None):
    df = summary_df.dropna(subset=[sweep_var, "mean_tau"]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.6))
    for algorithm in _ordered_algorithms(df):
        g = df[df["algorithm"] == algorithm]
        g = g.sort_values(sweep_var)
        x = g[sweep_var].to_numpy(dtype=float)
        y = g["mean_tau"].to_numpy(dtype=float)
        lower, upper = _mean_se_bounds(g)
        fill_color = _fill_color(algorithm)
        line_color = _line_color(algorithm)
        _shade_mean_ribbon(x, lower, upper, fill_color)
        plt.plot(
            x,
            y,
            marker=ALGORITHM_MARKERS.get(algorithm, "o"),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            linewidth=1.9,
            color=line_color,
            markerfacecolor=("white" if algorithm == "UniformFocalBorda-FC" else fill_color),
            markeredgecolor=line_color,
            markeredgewidth=0.9,
            markersize=5.2 if algorithm == "UniformFocalBorda-FC" else 4.6,
            zorder=3,
            label=_algorithm_label(algorithm),
        )
    plt.xscale("log")
    plt.yscale("log")
    x_label = xlabel or sweep_var
    plt.xlabel(x_label)
    plt.ylabel("mean stopping time")
    plt.title(f"Stopping scaling vs {x_label}")
    _add_legend_below()
    _savefig(outpath)


def plot_stopping_scaling_bar(summary_df, sweep_var, outpath, xlabel: str | None = None):
    df = summary_df.dropna(subset=[sweep_var, "mean_tau"]).copy()
    if df.empty:
        return
    x_values = sorted(df[sweep_var].astype(float).unique())
    algorithms = _ordered_algorithms(df)
    centers = np.arange(len(x_values), dtype=float)
    width = 0.78 / max(1, len(algorithms))
    plt.figure(figsize=(7.4, 4.9))
    for index, algorithm in enumerate(algorithms):
        g = df[df["algorithm"] == algorithm].copy()
        g[sweep_var] = g[sweep_var].astype(float)
        g = g.set_index(sweep_var).reindex(x_values)
        values = g["mean_tau"].to_numpy(dtype=float)
        offset = (index - (len(algorithms) - 1) / 2.0) * width
        bar_centers = centers + offset
        color = _fill_color(algorithm)
        line_color = _line_color(algorithm)
        plt.bar(
            bar_centers,
            values,
            width=width,
            color=color,
            alpha=0.66,
            edgecolor=line_color,
            linewidth=0.75,
            label=_algorithm_label(algorithm),
        )
        lower, upper = _mean_se_bounds(g)
        _shade_bar_intervals(bar_centers, lower, upper, width, line_color)
    plt.yscale("log")
    plt.xticks(centers, [f"{value:.3g}" for value in x_values])
    x_label = xlabel or sweep_var
    plt.xlabel(x_label)
    plt.ylabel("mean stopping time")
    plt.title(f"Grouped stopping times vs {x_label}")
    _add_legend_below()
    _savefig(outpath)


def plot_benchmark_grouped_bars(
    summary_df,
    setting_ids: list[str],
    setting_labels: list[str],
    outpath,
):
    df = summary_df[summary_df["experiment_id"].isin(setting_ids)].copy()
    df = df.dropna(subset=["mean_tau", "se_tau"])
    if df.empty:
        return
    algorithms = _ordered_algorithms(df)
    centers = np.arange(len(setting_ids), dtype=float)
    width = 0.78 / max(1, len(algorithms))
    plt.figure(figsize=(9.0, 5.1))
    for index, algorithm in enumerate(algorithms):
        g = df[df["algorithm"] == algorithm].set_index("experiment_id").reindex(setting_ids)
        values = g["mean_tau"].to_numpy(dtype=float)
        offset = (index - (len(algorithms) - 1) / 2.0) * width
        bar_centers = centers + offset
        color = _fill_color(algorithm)
        line_color = _line_color(algorithm)
        plt.bar(
            bar_centers,
            values,
            width=width,
            color=color,
            alpha=0.66,
            edgecolor=line_color,
            linewidth=0.75,
            label=_algorithm_label(algorithm),
        )
        _shade_bar_intervals(
            bar_centers,
            _mean_se_bounds(g)[0],
            _mean_se_bounds(g)[1],
            width,
            line_color,
        )
    plt.yscale("log")
    plt.xticks(centers, setting_labels)
    plt.xlabel("benchmark setting")
    plt.ylabel("mean stopping time")
    _add_legend_below()
    _savefig(outpath)


def plot_tau_mean_vs_x(
    summary_df,
    x_col: str,
    outpath,
    *,
    x_transform=None,
    xlabel: str | None = None,
    normalized: bool = False,
):
    y_cols = (
        ("mean_tau", "se_tau")
        if not normalized
        else ("mean_norm_tau_B", None)
    )
    required = [x_col, y_cols[0]]
    df = summary_df.dropna(subset=required).copy()
    if df.empty:
        return
    if x_transform is not None:
        df["_x"] = x_transform(df[x_col].to_numpy(dtype=float))
    else:
        df["_x"] = df[x_col].astype(float)
    ylabel = "mean stopping time" if not normalized else "mean tau / (d H_B)"
    plt.figure(figsize=(6.2, 4.6))
    for algorithm in _ordered_algorithms(df):
        g = df[df["algorithm"] == algorithm]
        g = g.sort_values("_x")
        x = g["_x"].to_numpy(dtype=float)
        y = g[y_cols[0]].to_numpy(dtype=float)
        fill_color = _fill_color(algorithm)
        line_color = _line_color(algorithm)
        if not normalized:
            lo, hi = _mean_se_bounds(g)
            _shade_mean_ribbon(x, lo, hi, fill_color)
        plt.plot(
            x,
            y,
            marker=ALGORITHM_MARKERS.get(algorithm, "o"),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            linewidth=1.9,
            color=line_color,
            markerfacecolor=("white" if algorithm == "UniformFocalBorda-FC" else fill_color),
            markeredgecolor=line_color,
            markeredgewidth=0.9,
            markersize=5.2 if algorithm == "UniformFocalBorda-FC" else 4.6,
            zorder=3,
            label=_algorithm_label(algorithm),
        )
    if (df["_x"] > 0).all() and (df[y_cols[0]] > 0).all():
        plt.yscale("log")
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel.capitalize()} vs {xlabel or x_col}", fontsize=9)
    _add_legend_below()
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
            label = f"{_algorithm_label(algorithm)} ({_short_series_label(exp_id)})"
        else:
            algorithm = key
            label = _algorithm_label(algorithm)
        g = g.sort_values(x_col)
        plt.plot(
            g[x_col],
            g["ratio"],
            marker="o",
            color=_line_color(algorithm),
            label=label,
        )
    plt.xscale("log" if (r[x_col] > 0).all() else "linear")
    plt.yscale("log")
    x_label = xlabel or x_col
    plt.xlabel(x_label)
    plt.ylabel(f"mean tau / {baseline} mean tau")
    plt.title(f"Baseline ratios vs {x_label}")
    _add_legend_below()
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
    for color_index, (key, g) in enumerate(df.groupby(group_cols, dropna=False)):
        if isinstance(key, tuple):
            algorithm = key[0]
            series = key[1] if len(key) > 1 else None
            label = (
                f"{_algorithm_label(algorithm)} ({_short_series_label(series)})"
                if series is not None
                else str(algorithm)
            )
        else:
            algorithm = str(key)
            label = _algorithm_label(algorithm)
        g = g.sort_values(x_col)
        color = ALGORITHM_LINE_COLORS.get(
            algorithm,
            DARK_SERIES_COLORS[color_index % len(DARK_SERIES_COLORS)],
        )
        plt.plot(g[x_col], g[y_col], marker="o", color=color, label=label)
    x_label = xlabel or x_col
    plt.xlabel(x_label)
    plt.ylabel(ylabel or y_col)
    plt.title(f"{ylabel or y_col} vs {x_label}")
    _add_legend_below()
    _savefig(outpath)


def plot_calibration_frontier(summary_df, outpath, y_col="error_rate"):
    df = summary_df.dropna(subset=["mean_tau", "se_tau", y_col, "sample_const", "threshold_const"]).copy()
    df = df[df["algorithm"] == "VB-EGE-practical"]
    if df.empty:
        return
    plt.figure(figsize=(6.4, 4.4))
    for color_index, (exp_id, g) in enumerate(df.groupby("experiment_id")):
        color = MACARON_SERIES_COLORS[color_index % len(MACARON_SERIES_COLORS)]
        edge_color = DARK_SERIES_COLORS[color_index % len(DARK_SERIES_COLORS)]
        plt.scatter(
            g["mean_tau"],
            g[y_col],
            label=_short_series_label(exp_id),
            color=color,
            s=38,
            alpha=0.82,
            edgecolor=edge_color,
            linewidth=0.9,
        )
        for _, row in g.iterrows():
            if row["sample_const"] == 2.0 and row["threshold_const"] == 4.0:
                plt.scatter(row["mean_tau"], row[y_col], marker="*", s=180, color="black")
    plt.xscale("log")
    if (df[y_col] > 0).any():
        plt.yscale("log")
    plt.xlabel("mean stopping time")
    metric_label = METRIC_LABELS.get(y_col, y_col)
    plt.ylabel(metric_label)
    plt.title(f"Constant sensitivity: {metric_label}")
    _add_legend_below(ncol=3)
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
    im = plt.imshow(values, origin="lower", aspect="auto", cmap=MACARON_CMAP)
    metric_label = METRIC_LABELS.get(metric, metric)
    plt.colorbar(im, label=metric_label)
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel(r"sample constant $c_s$")
    plt.ylabel(r"threshold constant $c_\theta$")
    plt.title(_short_series_label(exp_id))
    if metric == "mean_tau" and "se_tau" in df.columns:
        se_pivot = df.pivot_table(
            index="threshold_const",
            columns="sample_const",
            values="se_tau",
            aggfunc="mean",
        ).reindex(index=pivot.index, columns=pivot.columns)
        midpoint = (np.nanmin(values) + np.nanmax(values)) / 2.0
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                mean = values[row_index, col_index]
                se = float(se_pivot.iloc[row_index, col_index])
                if not (np.isfinite(mean) and np.isfinite(se)):
                    continue
                color = "#2B2B2B" if mean <= midpoint else "white"
                plt.text(
                    col_index,
                    row_index,
                    f"{mean:.1e}\n±{se:.1e}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=color,
                )
    if 2.0 in pivot.columns and 4.0 in pivot.index:
        x = list(pivot.columns).index(2.0)
        y = list(pivot.index).index(4.0)
        plt.gca().add_patch(
            Rectangle(
                (x - 0.48, y - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#2B2B2B",
                linewidth=1.8,
            )
        )
    _savefig(outpath)


def plot_allocation_vs_gap(result, true_gaps, outpath):
    N = np.asarray(result.N if hasattr(result, "N") else result["N"])
    gaps = np.asarray(true_gaps["delta"])
    alloc = N.sum(axis=1)
    plt.figure(figsize=(5.5, 4.0))
    plt.scatter(
        gaps,
        alloc,
        s=34,
        color="#8EC5E8",
        edgecolor="#3E86B8",
        linewidth=0.9,
        alpha=0.82,
    )
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
        plt.plot(
            g["budget"],
            g["mean_theta_rmse_centered"],
            marker="o",
            color=_line_color(algorithm, "#398F68"),
            label=_algorithm_label(algorithm),
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("budget")
    plt.ylabel("centered theta RMSE")
    plt.title(_curve_title(df, "MLE RMSE"))
    _add_legend_below()
    _savefig(outpath)


def plot_mle_sign_accuracy(summary_df, outpath):
    df = summary_df.dropna(subset=["budget", "mean_pairwise_sign_accuracy"]).copy()
    df = df[df["algorithm"].astype(str).str.contains("MLE", na=False)]
    if df.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    for algorithm, g in df.groupby("algorithm"):
        g = g.sort_values("budget")
        plt.plot(
            g["budget"],
            g["mean_pairwise_sign_accuracy"],
            marker="o",
            color=_line_color(algorithm, "#398F68"),
            label=_algorithm_label(algorithm),
        )
    plt.xscale("log")
    plt.xlabel("budget")
    plt.ylabel("pairwise sign accuracy")
    plt.ylim(0.0, 1.02)
    plt.title(_curve_title(df, "MLE sign accuracy"))
    _add_legend_below()
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
            label=_algorithm_label(algorithm),
            color=_fill_color(algorithm),
            alpha=0.82,
            edgecolor=_line_color(algorithm),
            linewidth=0.9,
            s=38,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("pairwise / focal basic-cell count")
    plt.ylabel("pairwise / VB samples per basic cell")
    plt.title("Pairwise total-cost decomposition")
    plt.axhline(1.0, color="#7A7A7A", linestyle=":", linewidth=1.0)
    _add_legend_below()
    _savefig(outpath)


def plot_fixed_confidence_tau(summary_df, outpath):
    df = summary_df.dropna(subset=["mean_tau", "se_tau"]).copy()
    if df.empty:
        return
    df = df.sort_values("mean_tau")
    algorithms = df["algorithm"].astype(str).tolist()
    labels = [_algorithm_label(name) for name in algorithms]
    values = df["mean_tau"].to_numpy(dtype=float)
    lower, upper = _mean_se_bounds(df)
    colors = [_fill_color(name, MACARON_SERIES_COLORS[i % len(MACARON_SERIES_COLORS)]) for i, name in enumerate(algorithms)]
    edge_colors = [_line_color(name, DARK_SERIES_COLORS[i % len(DARK_SERIES_COLORS)]) for i, name in enumerate(algorithms)]
    centers = np.arange(len(labels), dtype=float)
    width = 0.68
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(
        centers,
        values,
        width=width,
        color=colors,
        alpha=0.66,
        edgecolor=edge_colors,
        linewidth=0.75,
    )
    for center, lo, hi, color in zip(centers, lower, upper, colors):
        _shade_bar_intervals(
            np.asarray([center]),
            np.asarray([lo]),
            np.asarray([hi]),
            width,
            color,
        )
    plt.yscale("log")
    plt.ylabel("mean stopping time")
    plt.title("Practical vs theory-style constants")
    plt.xticks(centers, labels)
    _savefig(outpath)


def plot_paired_ratio_by_setting(paired_df, outpath):
    required = {
        "experiment_id",
        "algorithm",
        "mean_ratio",
        "se_ratio",
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
        ci_lowers = []
        ci_uppers = []
        ys = []
        for setting_index, setting in enumerate(settings):
            if setting not in g.index:
                continue
            row = g.loc[setting]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x = float(row["mean_ratio"])
            xs.append(x)
            se = max(float(row["se_ratio"]), 0.0)
            ci_lowers.append(max(x - se, np.finfo(float).tiny))
            ci_uppers.append(x + se)
            ys.append(y0[setting_index] + (index - (len(algorithms) - 1) / 2) * width)
        if xs:
            fill_color = _fill_color(algorithm, MACARON_SERIES_COLORS[index % len(MACARON_SERIES_COLORS)])
            line_color = _line_color(algorithm, DARK_SERIES_COLORS[index % len(DARK_SERIES_COLORS)])
            for y, lo, hi in zip(ys, ci_lowers, ci_uppers):
                plt.fill_betweenx(
                    [y - 0.38 * width, y + 0.38 * width],
                    [lo, lo],
                    [hi, hi],
                    color=fill_color,
                    alpha=0.36,
                    linewidth=0,
                )
            plt.scatter(
                xs,
                ys,
                color=line_color,
                edgecolor="white",
                linewidth=0.45,
                s=30,
                label=_algorithm_label(algorithm),
            )
    labels = [s.removeprefix("fc_").replace("_", " ") for s in settings]
    plt.yticks(y0, labels)
    plt.xscale("log")
    plt.axvline(1.0, color="#7A7A7A", linestyle=":", linewidth=1.0)
    plt.xlabel("mean paired stopping-time ratio to VB-EGE")
    plt.ylabel("benchmark setting")
    plt.title("Paired fixed-confidence sample cost")
    _add_legend_below()
    _savefig(outpath)
