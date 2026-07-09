"""Summarize raw sweep outputs and produce standard figures."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from .compat import import_pandas_quietly
from .io_utils import ensure_parent
from .metrics import loglog_slope, summarize_runs
from .plotting import (
    plot_baseline_ratio_vs_x,
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
    plot_stopping_scaling,
    plot_tau_quantiles_vs_x,
)

pd = import_pandas_quietly()


def _read_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p.with_suffix(".csv"))
    return pd.read_csv(p)


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
        for metric in ["median_tau", "q95_tau", "mean_norm_tau_B"]:
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
    if name == "confidence_scaling_quantile":
        transform = lambda x: __import__("numpy").log(1.0 / x)
        for exp_id, g in summary.groupby("experiment_id"):
            suffix = _suffix_for_exp(exp_id)
            plot_tau_quantiles_vs_x(
                g,
                "delta",
                figdir / f"tau_quantiles_vs_log_inv_delta_{suffix}.pdf",
                x_transform=transform,
                xlabel="log(1/delta)",
            )
            plot_tau_quantiles_vs_x(
                g,
                "delta",
                figdir / f"norm_tau_quantiles_vs_log_inv_delta_{suffix}.pdf",
                x_transform=transform,
                xlabel="log(1/delta)",
                normalized=True,
            )
            if "mean_norm_tau_B_logdelta" in g:
                plot_metric_vs_x(
                    g.assign(log_inv_delta=transform(g["delta"].to_numpy(dtype=float))),
                    "log_inv_delta",
                    "mean_norm_tau_B_logdelta",
                    figdir / f"tau_over_log_inv_delta_{suffix}.pdf",
                    ylabel="mean normalized tau / log(1/delta)",
                )
        vb = summary[summary["algorithm"] == "VB-EGE-practical"]
        if not vb.empty and "delta" in vb:
            plot_metric_vs_x(
                vb.assign(log_inv_delta=transform(vb["delta"].to_numpy(dtype=float))),
                "log_inv_delta",
                "mean_final_phase",
                figdir / "final_phase_distribution_vs_delta.pdf",
                ylabel="mean final phase",
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
            plot_tau_quantiles_vs_x(
                g,
                "param_s",
                figdir / f"tau_vs_pareto_size_{suffix}.pdf",
                xlabel="true Pareto size |P|",
            )
            plot_tau_quantiles_vs_x(
                g,
                "param_s",
                figdir / f"norm_tau_vs_pareto_size_{suffix}.pdf",
                xlabel="true Pareto size |P|",
                normalized=True,
            )
        plot_baseline_ratio_vs_x(
            summary,
            "param_s",
            figdir / "baseline_ratio_vs_pareto_size.pdf",
        )
        vb = summary[summary["algorithm"] == "VB-EGE-practical"]
        if not vb.empty:
            plot_metric_vs_x(
                vb,
                "param_s",
                "mean_num_accepted",
                figdir / "accepted_phase_profile.pdf",
                ylabel="mean accepted arms",
            )
            plot_metric_vs_x(
                vb,
                "param_s",
                "mean_num_rejected",
                figdir / "rejected_phase_profile.pdf",
                ylabel="mean rejected arms",
            )
    elif name == "correlated_arena":
        for exp_id, g in summary.groupby("experiment_id"):
            suffix = _suffix_for_exp(exp_id)
            x_col = "param_rho"
            plot_tau_quantiles_vs_x(
                g,
                x_col,
                figdir / f"tau_vs_rho_{suffix}.pdf",
                xlabel="target objective correlation rho",
            )
            plot_tau_quantiles_vs_x(
                g,
                x_col,
                figdir / f"norm_tau_vs_rho_{suffix}.pdf",
                xlabel="target objective correlation rho",
                normalized=True,
            )
        plot_baseline_ratio_vs_x(summary, "param_rho", figdir / "baseline_ratio_vs_rho.pdf")
        plot_metric_vs_x(
            summary,
            "param_rho",
            "mean_pair_cell_coverage",
            figdir / "pair_cell_coverage_vs_rho.pdf",
            ylabel="mean pair-cell coverage",
            series_col="experiment_id",
        )
        vb = summary[summary["algorithm"] == "VB-EGE-practical"]
        plot_metric_vs_x(
            vb,
            "param_rho",
            "mean_delta_min_B",
            figdir / "delta_min_B_vs_rho.pdf",
            ylabel="mean Borda delta_min",
            series_col="experiment_id",
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
    out = Path(args.out)
    ensure_parent(out)
    summary.to_csv(out, index=False)
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
                plot_stopping_scaling(g, col, figdir / f"tau_vs_{col}_{exp_id}.pdf")
    plot_pair_cell_coverage_effect(summary, figdir / "pair_cell_coverage_effect.pdf")
    _write_slopes(summary, out.with_name(out.stem + "_slopes.csv"))
    _write_special_figures(summary, figdir, out)
    print(f"wrote summary to {out}")


if __name__ == "__main__":
    main()
