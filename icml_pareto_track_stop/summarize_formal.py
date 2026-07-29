"""Summarize and plot the full formal two-algorithm experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vb_ege.compat import import_pandas_quietly


pd = import_pandas_quietly()

ROOT = Path(__file__).resolve().parent
RAW_DEFAULT = ROOT / "results" / "raw" / "formal_two_way_certificate.csv"
SUMMARY_DIR = ROOT / "results" / "summary_certificate"
FIGURE_DIR = ROOT / "results" / "figures_certificate"

ALGORITHMS = [
    "VB-EGE-practical",
    "Pareto BT-GLR Track-and-Stop",
]
LABELS = {
    "VB-EGE-practical": "VB-EGE",
    "Pareto BT-GLR Track-and-Stop": "Pareto BT-GLR T&S",
}
COLORS = {
    "VB-EGE-practical": "#C94F5D",
    "Pareto BT-GLR Track-and-Stop": "#3E86B8",
}
FILLS = {
    "VB-EGE-practical": "#F28E8E",
    "Pareto BT-GLR Track-and-Stop": "#8EC5E8",
}
BENCHMARK_LABELS = {
    "fc_convex2d": "Convex-2D",
    "fc_convex3d": "Convex-3D",
    "fc_arena4_small": "Arena-4 small",
    "fc_arena4_medium": "Arena-4 medium",
    "fc_arena10_medium": "Arena-10 medium",
    "fc_witness4": "Witness-4",
    "fc_witness10": "Witness-10",
    "fc_twogroup10_medium": "Two-group-10",
}


def _wilson_upper(errors: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return np.nan
    p = errors / n
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    radius = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float((center + radius) / denominator)


def _add_parameter_columns(raw):
    parsed = raw["theta_params_json"].map(json.loads)
    keys = sorted({key for params in parsed for key in params})
    for key in keys:
        raw[f"param_{key}"] = parsed.map(lambda params: params.get(key, np.nan))
    raw["params_key"] = raw["theta_params_json"]
    return raw


def _as_bool(series):
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _cluster_mean_se(frame, value: str) -> tuple[float, float, int]:
    clustered = frame.groupby("instance_id", dropna=False)[value].mean()
    values = clustered.to_numpy(dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se, len(values)


def summarize(raw):
    group_columns = [
        "section",
        "experiment_id",
        "params_key",
        "K",
        "d",
        "delta",
        "algorithm",
    ]
    rows = []
    for key, frame in raw.groupby(group_columns, dropna=False, sort=False):
        values = dict(zip(group_columns, key))
        mean_tau, se_tau, n_instances = _cluster_mean_se(frame, "tau")
        stopped = _as_bool(frame["stopped"])
        stopped_frame = frame[stopped]
        errors = int(stopped_frame["error"].fillna(0).astype(int).sum())
        n_stopped = int(stopped.sum())
        tau = frame["tau"].to_numpy(dtype=float)
        values.update(
            {
                "n": len(frame),
                "n_instances": n_instances,
                "mean_tau": mean_tau,
                "se_tau": se_tau,
                "median_tau": float(np.median(tau)),
                "q25_tau": float(np.quantile(tau, 0.25)),
                "q75_tau": float(np.quantile(tau, 0.75)),
                "q95_tau": float(np.quantile(tau, 0.95)),
                "max_tau": float(np.max(tau)),
                "stop_rate": float(stopped.mean()),
                "n_stopped": n_stopped,
                "errors_stopped": errors,
                "error_rate_stopped": errors / n_stopped if n_stopped else np.nan,
                "wilson95_error_upper": _wilson_upper(errors, n_stopped),
                "mle_convergence_rate": float(
                    _as_bool(frame["mle_converged_all"]).mean()
                ),
                "mean_phases": float(frame["num_phases"].mean()),
                "mean_elapsed_seconds": float(frame["elapsed_seconds"].mean()),
            }
        )
        params = json.loads(values["params_key"])
        for param_key, param_value in params.items():
            if isinstance(param_value, (int, float, str, bool)):
                values[f"param_{param_key}"] = param_value
        rows.append(values)
    return pd.DataFrame(rows)


def paired_summary(raw):
    index = [
        "cell_key",
        "section",
        "experiment_id",
        "params_key",
        "K",
        "d",
        "delta",
        "instance_id",
    ]
    tau = raw.pivot(index=index, columns="algorithm", values="tau").reset_index()
    stopped = (
        raw.pivot(index=index, columns="algorithm", values="stopped")
        .reset_index()
    )
    paired = tau[index].copy()
    paired["vbege_tau"] = tau["VB-EGE-practical"].to_numpy(dtype=float)
    paired["track_tau"] = tau["Pareto BT-GLR Track-and-Stop"].to_numpy(dtype=float)
    paired["both_stopped"] = (
        _as_bool(stopped["VB-EGE-practical"])
        & _as_bool(stopped["Pareto BT-GLR Track-and-Stop"])
    )
    paired["track_over_vbege"] = paired["track_tau"] / paired["vbege_tau"]
    paired["log_track_over_vbege"] = np.log(paired["track_over_vbege"])

    group_columns = [
        "section",
        "experiment_id",
        "params_key",
        "K",
        "d",
        "delta",
    ]
    rows = []
    for key, frame in paired.groupby(group_columns, dropna=False, sort=False):
        values = dict(zip(group_columns, key))
        ratios_by_instance = frame.groupby("instance_id")["track_over_vbege"].mean()
        log_by_instance = frame.groupby("instance_id")["log_track_over_vbege"].mean()
        ratios = ratios_by_instance.to_numpy(dtype=float)
        logs = log_by_instance.to_numpy(dtype=float)
        values.update(
            {
                "n_pairs": len(frame),
                "n_instances": len(ratios),
                "both_stopped_rate": float(frame["both_stopped"].mean()),
                "mean_track_over_vbege": float(ratios.mean()),
                "se_track_over_vbege": (
                    float(ratios.std(ddof=1) / np.sqrt(len(ratios)))
                    if len(ratios) > 1
                    else 0.0
                ),
                "geometric_mean_track_over_vbege": float(np.exp(logs.mean())),
                "median_track_over_vbege": float(
                    np.median(frame["track_over_vbege"])
                ),
                "track_faster_rate": float(
                    (frame["track_over_vbege"] < 1.0).mean()
                ),
            }
        )
        params = json.loads(values["params_key"])
        for param_key, param_value in params.items():
            if isinstance(param_value, (int, float, str, bool)):
                values[f"param_{param_key}"] = param_value
        rows.append(values)
    return pd.DataFrame(rows)


def _line_with_se(axis, frame, x_column, algorithm):
    frame = frame.sort_values(x_column)
    x = frame[x_column].to_numpy(dtype=float)
    mean = frame["mean_tau"].to_numpy(dtype=float)
    se = frame["se_tau"].to_numpy(dtype=float)
    axis.plot(
        x,
        mean,
        color=COLORS[algorithm],
        marker="o" if algorithm == "VB-EGE-practical" else "s",
        linewidth=2.0,
        markersize=5,
        label=LABELS[algorithm],
    )
    axis.fill_between(
        x,
        np.maximum(mean - se, np.finfo(float).tiny),
        mean + se,
        color=FILLS[algorithm],
        alpha=0.55,
        linewidth=0,
    )


def plot_scaling(summary, output: Path):
    section = summary[summary["section"] == "fixed_confidence_scaling"]
    definitions = [
        ("fc_K_scaling_d4", "param_K", "number of arms $K$", "$K$ scaling"),
        ("fc_d_scaling_K64", "param_d", "number of objectives $d$", "$d$ scaling"),
        (
            "fc_gap_scaling_K64_d10",
            "param_Delta",
            "latent separation $\\Delta$",
            "gap scaling",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    for axis, (experiment, x_column, xlabel, title) in zip(axes, definitions):
        frame = section[section["experiment_id"] == experiment]
        for algorithm in ALGORITHMS:
            _line_with_se(
                axis,
                frame[frame["algorithm"] == algorithm],
                x_column,
                algorithm,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("mean stopping time")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_confidence(summary, output: Path):
    section = summary[summary["section"] == "confidence_scaling_quantiles"].copy()
    section["log_inv_delta"] = np.log(1.0 / section["delta"])
    experiments = [
        ("conf_quantile_symmetric_K64_d10", "Symmetric $K=64,d=10$"),
        ("conf_quantile_arena10_medium", "Arena-10 medium"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for axis, (experiment, title) in zip(axes, experiments):
        frame = section[section["experiment_id"] == experiment]
        for algorithm in ALGORITHMS:
            _line_with_se(
                axis,
                frame[frame["algorithm"] == algorithm],
                "log_inv_delta",
                algorithm,
            )
        axis.set_yscale("log")
        axis.set_xlabel("$\\log(1/\\delta)$")
        axis.set_title(title)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("mean stopping time")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_benchmarks(summary, output: Path):
    section = summary[summary["section"] == "fixed_confidence_benchmarks"]
    settings = list(BENCHMARK_LABELS)
    x = np.arange(len(settings), dtype=float)
    width = 0.36
    fig, axis = plt.subplots(figsize=(15.2, 5.2))
    plotted = section[section["algorithm"].isin(ALGORITHMS)]
    positive_lower = np.maximum(
        plotted["mean_tau"].to_numpy(dtype=float)
        - plotted["se_tau"].to_numpy(dtype=float),
        np.finfo(float).tiny,
    )
    bar_floor = 0.65 * float(np.min(positive_lower))
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        frame = (
            section[section["algorithm"] == algorithm]
            .set_index("experiment_id")
            .loc[settings]
        )
        positions = x + (algorithm_index - 0.5) * width
        means = frame["mean_tau"].to_numpy(dtype=float)
        ses = frame["se_tau"].to_numpy(dtype=float)
        axis.bar(
            positions,
            means - bar_floor,
            bottom=bar_floor,
            width=width,
            color=FILLS[algorithm],
            edgecolor=COLORS[algorithm],
            linewidth=1.2,
            label=LABELS[algorithm],
        )
        lower = np.maximum(means - ses, np.finfo(float).tiny)
        axis.bar(
            positions,
            2.0 * ses,
            bottom=lower,
            width=width,
            color=COLORS[algorithm],
            alpha=0.38,
            linewidth=0,
        )
        for position, mean, n, n_stopped in zip(
            positions,
            means,
            frame["n"].to_numpy(dtype=int),
            frame["n_stopped"].to_numpy(dtype=int),
            strict=True,
        ):
            n_capped = int(n - n_stopped)
            if n_capped:
                axis.annotate(
                    f"{n_capped}/{n} capped",
                    xy=(position, mean),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=COLORS[algorithm],
                )
    axis.set_yscale("log")
    axis.set_ylim(bottom=bar_floor)
    axis.set_ylabel("mean stopping time")
    axis.set_xlabel("benchmark setting")
    axis.set_xticks(x)
    axis.set_xticklabels([BENCHMARK_LABELS[item] for item in settings])
    axis.grid(axis="y", alpha=0.18)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def diagnostic_summary(raw):
    rows = []
    group_columns = ["section", "algorithm"]
    for key, frame in raw.groupby(group_columns, sort=False, dropna=False):
        stopped = _as_bool(frame["stopped"])
        stopped_frame = frame[stopped]
        errors = int(stopped_frame["error"].fillna(0).astype(int).sum())
        row = {
            "section": key[0],
            "algorithm": key[1],
            "n": int(len(frame)),
            "stop_rate": float(stopped.mean()),
            "errors_stopped": errors,
            "n_stopped": int(stopped.sum()),
            "error_rate_stopped": (
                errors / int(stopped.sum()) if int(stopped.sum()) else np.nan
            ),
            "wilson95_error_upper": _wilson_upper(errors, int(stopped.sum())),
            "mle_convergence_rate": float(
                _as_bool(frame["mle_converged_all"]).mean()
            ),
            "mean_elapsed_seconds": float(frame["elapsed_seconds"].mean()),
        }
        if key[1] == "Pareto BT-GLR Track-and-Stop":
            certified = _as_bool(frame["stopping_statistic_is_certified"])
            row.update(
                {
                    "certified_statistic_rate": float(certified.mean()),
                    "profile_modes": "; ".join(
                        f"{name}:{count}"
                        for name, count in frame["final_profile_mode"]
                        .fillna("")
                        .value_counts()
                        .items()
                    ),
                    "mean_final_statistic_over_threshold": float(
                        (
                            frame["final_glr_statistic"].astype(float)
                            / frame["final_glr_threshold"].astype(float)
                        ).mean()
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pareto_size(summary, output: Path):
    section = summary[summary["section"] == "pareto_size_ablation"]
    experiments = [
        ("pareto_size_arena4_K64", "Arena-4, $K=64$"),
        ("pareto_size_arena10_K64", "Arena-10, $K=64$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for axis, (experiment, title) in zip(axes, experiments):
        frame = section[section["experiment_id"] == experiment]
        for algorithm in ALGORITHMS:
            _line_with_se(
                axis,
                frame[frame["algorithm"] == algorithm],
                "param_s",
                algorithm,
            )
        axis.set_yscale("log")
        axis.set_xlabel("true Pareto size $|P|$")
        axis.set_title(title)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("mean stopping time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    raw = _add_parameter_columns(raw)
    summary = summarize(raw)
    paired = paired_summary(raw)
    diagnostics = diagnostic_summary(raw)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_dir / "formal_two_way_summary.csv", index=False)
    paired.to_csv(args.summary_dir / "formal_two_way_paired.csv", index=False)
    diagnostics.to_csv(
        args.summary_dir / "formal_two_way_diagnostics.csv",
        index=False,
    )

    plot_scaling(summary, args.figure_dir / "formal_scaling.pdf")
    plot_confidence(summary, args.figure_dir / "formal_confidence.pdf")
    plot_benchmarks(summary, args.figure_dir / "formal_benchmarks.pdf")
    plot_pareto_size(summary, args.figure_dir / "formal_pareto_size.pdf")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
