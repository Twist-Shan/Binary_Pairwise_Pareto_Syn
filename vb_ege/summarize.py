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
    plot_budget_curve,
    plot_budget_curve_normalized,
    plot_fixed_confidence_tau,
    plot_hamming_curve,
    plot_mle_sign_accuracy,
    plot_mle_theta_rmse,
    plot_pair_cell_coverage_effect,
    plot_stopping_scaling,
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
        for col in ["K", "d", "param_Delta", "delta"]:
            if col in g.columns and g[col].nunique(dropna=True) > 1:
                plot_stopping_scaling(g, col, figdir / f"tau_vs_{col}_{exp_id}.pdf")
    plot_pair_cell_coverage_effect(summary, figdir / "pair_cell_coverage_effect.pdf")
    _write_slopes(summary, out.with_name(out.stem + "_slopes.csv"))
    print(f"wrote summary to {out}")


if __name__ == "__main__":
    main()
