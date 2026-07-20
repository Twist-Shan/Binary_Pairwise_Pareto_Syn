"""Summarize raw sweep outputs and produce standard figures."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from .compat import import_pandas_quietly
from .io_utils import ensure_parent
from .metrics import loglog_slope, paired_tau_ratios, summarize_runs
from .plotting import (
    plot_baseline_ratio_vs_x,
    plot_benchmark_grouped_bars,
    plot_budget_ratio_curve,
    plot_budget_curve,
    plot_budget_curve_normalized,
    plot_calibration_frontier,
    plot_calibration_heatmap,
    plot_fixed_confidence_tau,
    plot_hamming_curve,
    plot_metric_vs_x,
    plot_mle_sign_accuracy,
    plot_mle_theta_rmse,
    plot_pair_cell_coverage_effect,
    plot_paired_ratio_by_setting,
    plot_stopping_scaling,
    plot_stopping_scaling_bar,
    plot_tau_mean_vs_x,
)

pd = import_pandas_quietly()


def _read_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p.with_suffix(".csv"), low_memory=False)
    return pd.read_csv(p, low_memory=False)


def _write_slopes(summary: pd.DataFrame, out: Path) -> None:
    rows = []
    for exp_id, g in summary.groupby("experiment_id"):
        candidates = [
            ("K", "K"),
            ("d", "d"),
            ("Delta", "param_Delta"),
            ("inv_gap2", "mean_delta_min_B"),
            ("log_delta", "delta"),
        ]
        for name, col in candidates:
            if col not in g.columns:
                continue
            x = g[col]
            if name == "inv_gap2":
                x = 1.0 / (x**2)
            if name == "log_delta":
                x = (1.0 / x).apply(lambda v: None if v <= 0 else v)
            slope, intercept = loglog_slope(x, g["mean_tau"])
            if pd.notna(slope):
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "sweep_var": name,
                        "slope": slope,
                        "intercept": intercept,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)


def _suffix_for_exp(exp_id: str) -> str:
    if "symmetric" in exp_id:
        return "symmetric"
    if "arena10" in exp_id:
        return "arena10"
    if "arena4" in exp_id:
        return "arena4"
    if "witness10" in exp_id:
        return "witness10"
    return exp_id


def _write_confidence_slopes(summary: pd.DataFrame, out: Path) -> None:
    rows = []
    for (exp_id, algorithm), g in summary.groupby(["experiment_id", "algorithm"]):
        if "delta" not in g or g["delta"].nunique(dropna=True) < 2:
            continue
        x = pd.to_numeric(g["delta"], errors="coerce").apply(
            lambda v: None if pd.isna(v) or v <= 0 else float(__import__("math").log(1.0 / v))
        )
        for metric in ["mean_tau", "q95_tau", "mean_norm_tau_B"]:
            if metric not in g:
                continue
            slope, intercept = loglog_slope(x, g[metric])
            if pd.notna(slope):
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "algorithm": algorithm,
                        "metric": metric,
                        "slope_loglog_vs_log_inv_delta": slope,
                        "intercept": intercept,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)


def _write_special_figures(summary: pd.DataFrame, figdir: Path, out: Path) -> None:
    name = figdir.name
    if name == "fixed_confidence_benchmarks":
        plot_benchmark_grouped_bars(
            summary,
            [
                "fc_convex2d",
                "fc_convex3d",
                "fc_witness4",
                "fc_witness10",
                "fc_arena4_small",
                "fc_arena4_medium",
                "fc_arena10_medium",
                "fc_twogroup10_medium",
            ],
            [
                "Convex-2D",
                "Convex-3D",
                "Witness-4",
                "Witness-10",
                "Arena-4\nsmall",
                "Arena-4\nmedium",
                "Arena-10\nmedium",
                "Two-group-10",
            ],
            figdir / "benchmark_group_all.pdf",
        )
    elif name == "confidence_scaling_quantile":
        transform = lambda x: __import__("numpy").log(1.0 / x)
        for exp_id, g in summary.groupby("experiment_id"):
            suffix = _suffix_for_exp(exp_id)
            transformed = g.assign(
                log_inv_delta=transform(g["delta"].to_numpy(dtype=float))
            )
            plot_stopping_scaling(
                transformed,
                "log_inv_delta",
                figdir / f"tau_mean_ci_vs_log_inv_delta_{suffix}.pdf",
                xlabel="log(1/delta)",
            )
            plot_stopping_scaling_bar(
                transformed,
                "log_inv_delta",
                figdir / f"tau_mean_ci_bar_vs_log_inv_delta_{suffix}.pdf",
                xlabel="log(1/delta)",
            )
        _write_confidence_slopes(summary, out.with_name(out.stem + "_slopes.csv"))
    elif name == "constants_calibration":
        plot_calibration_frontier(summary, figdir / "error_tau_frontier_all.pdf", "error_rate")
        plot_calibration_frontier(summary, figdir / "wilson_tau_frontier_all.pdf", "wilson_upper")
        for exp_id in summary["experiment_id"].dropna().unique():
            suffix = _suffix_for_exp(str(exp_id))
            plot_calibration_heatmap(
                summary,
                str(exp_id),
                "mean_tau",
                figdir / f"heatmap_mean_tau_by_constants_{suffix}.pdf",
            )
            plot_calibration_heatmap(
                summary,
                str(exp_id),
                "error_rate",
                figdir / f"heatmap_error_by_constants_{suffix}.pdf",
            )
            plot_calibration_heatmap(
                summary,
                str(exp_id),
                "mean_stopped",
                figdir / f"heatmap_stopped_fraction_by_constants_{suffix}.pdf",
            )
    elif name == "pareto_size_ablation":
        for exp_id, g in summary.groupby("experiment_id"):
            suffix = _suffix_for_exp(exp_id)
            plot_tau_mean_vs_x(
                g,
                "param_s",
                figdir / f"tau_vs_pareto_size_{suffix}.pdf",
                xlabel="true Pareto size |P|",
            )
        vb = summary[summary["algorithm"] == "VB-EGE-practical"]
        if not vb.empty:
            plot_metric_vs_x(
                vb,
                "param_s",
                "mean_num_accepted",
                figdir / "accepted_phase_profile.pdf",
                ylabel="mean accepted arms",
                xlabel="true Pareto size |P|",
            )
            plot_metric_vs_x(
                vb,
                "param_s",
                "mean_num_rejected",
                figdir / "rejected_phase_profile.pdf",
                ylabel="mean rejected arms",
                xlabel="true Pareto size |P|",
            )
    elif name == "correlated_arena":
        for exp_id, g in summary.groupby("experiment_id"):
            suffix = _suffix_for_exp(exp_id)
            x_col = "mean_achieved_objective_correlation"
            plot_tau_mean_vs_x(
                g,
                x_col,
                figdir / f"tau_vs_rho_{suffix}.pdf",
                xlabel="achieved objective correlation",
            )
            plot_tau_mean_vs_x(
                g,
                x_col,
                figdir / f"norm_tau_vs_rho_{suffix}.pdf",
                xlabel="achieved objective correlation",
                normalized=True,
            )
        plot_baseline_ratio_vs_x(
            summary,
            "mean_achieved_objective_correlation",
            figdir / "baseline_ratio_vs_rho.pdf",
            xlabel="achieved objective correlation",
        )
        plot_metric_vs_x(
            summary,
            "mean_achieved_objective_correlation",
            "mean_pair_cell_coverage",
            figdir / "pair_cell_coverage_vs_rho.pdf",
            ylabel="samples per pair-coordinate cell",
            series_col="experiment_id",
            xlabel="achieved objective correlation",
        )
        vb = summary[summary["algorithm"] == "VB-EGE-practical"]
        plot_metric_vs_x(
            vb,
            "mean_achieved_objective_correlation",
            "mean_delta_min_B",
            figdir / "delta_min_B_vs_rho.pdf",
            ylabel="mean Borda delta_min",
            series_col="experiment_id",
            xlabel="achieved objective correlation",
        )
    elif name == "under_budget_stress":
        for exp_id, g in summary.groupby("experiment_id"):
            suffix = _suffix_for_exp(exp_id)
            plot_budget_ratio_curve(
                g,
                figdir / f"error_vs_budget_ratio_{suffix}.pdf",
                "error_rate",
                "log10(error rate floor)",
                log10_floor=True,
            )
            plot_budget_ratio_curve(
                g,
                figdir / f"hamming_vs_budget_ratio_{suffix}.pdf",
                "mean_hamming",
                "mean symmetric-difference distance",
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figdir", required=True)
    args = parser.parse_args(argv)

    raw = _read_raw(args.raw)
    summary = summarize_runs(raw)
    paired = paired_tau_ratios(raw)
    out = Path(args.out)
    ensure_parent(out)
    summary.to_csv(out, index=False)
    paired_out = out.with_name(out.stem + "_paired.csv")
    if not paired.empty:
        paired.to_csv(paired_out, index=False)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    for exp_id, g in summary.groupby("experiment_id"):
        plot_budget_curve(g, figdir / f"error_vs_budget_abs_{exp_id}.pdf")
        plot_budget_curve_normalized(g, figdir / f"error_vs_budget_norm_{exp_id}.pdf")
        plot_hamming_curve(g, figdir / f"hamming_vs_budget_{exp_id}.pdf")
        plot_fixed_confidence_tau(g, figdir / f"tau_by_algorithm_{exp_id}.pdf")
        plot_mle_theta_rmse(g, figdir / f"mle_theta_rmse_{exp_id}.pdf")
        plot_mle_sign_accuracy(g, figdir / f"mle_pairwise_sign_accuracy_{exp_id}.pdf")
        for col in ["K", "d", "param_Delta", "delta", "param_s", "param_rho"]:
            if col in g.columns and g[col].nunique(dropna=True) > 1:
                xlabel = {
                    "K": "number of arms K",
                    "d": "number of objectives d",
                    "param_Delta": "latent separation Delta",
                    "delta": "target failure probability delta",
                    "param_s": "target Pareto size |P|",
                    "param_rho": "target objective correlation rho",
                }.get(col, col)
                plot_stopping_scaling(
                    g,
                    col,
                    figdir / f"tau_vs_{col}_{exp_id}.pdf",
                    xlabel=xlabel,
                )
                if figdir.name == "fixed_confidence_scaling":
                    plot_stopping_scaling_bar(
                        g,
                        col,
                        figdir / f"tau_bar_vs_{col}_{exp_id}.pdf",
                        xlabel=xlabel,
                    )
    plot_pair_cell_coverage_effect(summary, figdir / "pair_cell_coverage_effect.pdf")
    plot_paired_ratio_by_setting(paired, figdir / "paired_ratio_by_setting.pdf")
    if figdir.name != "fixed_confidence_benchmarks":
        _write_slopes(summary, out.with_name(out.stem + "_slopes.csv"))
    _write_special_figures(summary, figdir, out)
    print(f"wrote summary to {out}")


if __name__ == "__main__":
    main()
