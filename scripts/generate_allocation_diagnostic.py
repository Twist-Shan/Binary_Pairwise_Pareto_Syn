"""Generate an arm-allocation diagnostic for two heterogeneous instances."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vb_ege.algorithms import VBEGEConfig, run_vb_ege
from vb_ege.core import borda, strict_pareto_set
from vb_ege.gaps import compute_gaps
from vb_ege.instances import make_instance


SETTINGS = [
    (
        "Arena-4 medium",
        "arena_tradeoff_frontier",
        {"K": 64, "d": 4, "s": 12, "margin_low": 0.08, "margin_high": 0.25, "alpha": 0.7},
        4242,
    ),
    (
        "Two-group-10",
        "highdim_two_group",
        {"K_low": 48, "K_high": 16, "d": 10},
        4242,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    cfg = VBEGEConfig(
        delta=0.05,
        sample_const=2.0,
        threshold_const=4.0,
        log_const=4.0,
        max_phases=45,
    )
    for ax, (title, generator, params, seed) in zip(axes, SETTINGS):
        theta, _ = make_instance(generator, params, seed)
        result = run_vb_ege(theta, cfg, rng=np.random.default_rng(seed + 10_000))
        gaps = np.asarray(compute_gaps(borda(theta))["delta"], dtype=float)
        allocation = np.asarray(result.N, dtype=float).sum(axis=1)
        pareto = set(strict_pareto_set(theta))
        is_pareto = np.array([i in pareto for i in range(len(gaps))])
        valid = np.isfinite(gaps) & (gaps > 0) & np.isfinite(allocation) & (allocation > 0)
        ax.scatter(
            gaps[valid & ~is_pareto],
            allocation[valid & ~is_pareto],
            s=30,
            color="#8EC5E8",
            edgecolor="#3E86B8",
            linewidth=0.85,
            alpha=0.82,
            label="dominated arm",
        )
        ax.scatter(
            gaps[valid & is_pareto],
            allocation[valid & is_pareto],
            s=42,
            marker="^",
            color="#F28E8E",
            edgecolor="#C94F5D",
            linewidth=0.9,
            alpha=0.86,
            label="Pareto arm",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"arm Borda certificate gap $\Delta_i^B$")
        ax.set_title(title)
        ax.grid(alpha=0.2, which="both")
    for ax in axes:
        ax.set_ylabel("VB-EGE samples allocated to arm")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Adaptive allocation by Borda gap")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
