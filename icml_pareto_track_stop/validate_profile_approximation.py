"""Numerically compare the quadratic order profile to exact constrained fits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from icml_pareto_track_stop.scalable_track_stop import (
    ScalableParetoTrackStopConfig,
    _fit_bt,
    _profile_geometry,
    _quadratic_order_profile,
)
from pilot_focal_bt_mle.matched_fixed_budget import (
    fit_box_constrained_bt_coordinate,
)
from vb_ege.core import stable_sigmoid
from vb_ege.instances import make_instance


ROOT = Path(__file__).resolve().parent


def _sample_pair_wins(
    theta: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    K, d = theta.shape
    wins = np.zeros((d, K, K), dtype=float)
    for r in range(d):
        for i in range(K):
            for j in range(i + 1, K):
                probability = float(stable_sigmoid(theta[i, r] - theta[j, r]))
                wins_i = int(rng.binomial(count, probability))
                wins[r, i, j] = wins_i
                wins[r, j, i] = count - wins_i
    return wins


def run_validation(n_instances: int, constraints_per_instance: int) -> list[dict]:
    rows = []
    config = ScalableParetoTrackStopConfig(
        box_bound=2.0,
        optimizer_tol=1e-10,
        optimizer_max_iter=5000,
    )
    generators = [
        (
            "symmetric_hard",
            {"K": 8, "d": 3, "Delta": 0.8},
        ),
        (
            "arena_tradeoff_frontier",
            {
                "K": 8,
                "d": 3,
                "s": 3,
                "margin_low": 0.12,
                "margin_high": 0.25,
                "alpha": 0.7,
            },
        ),
        (
            "unique_witness_d",
            {
                "K": 8,
                "d": 3,
                "s": 2,
                "q_per_p": 3,
                "margin_low": 0.08,
                "margin_high": 0.20,
            },
        ),
        (
            "convex_frontier_3d",
            {
                "K": 8,
                "d": 3,
                "s": 3,
                "margin_low": 0.08,
                "margin_high": 0.20,
                "alpha": 1.0,
            },
        ),
    ]
    for instance_index in range(n_instances):
        generator, params = generators[instance_index % len(generators)]
        theta, _ = make_instance(generator, params, seed=7000 + instance_index)
        rng = np.random.default_rng(17000 + instance_index)
        pair_wins = _sample_pair_wins(theta, count=1000, rng=rng)
        fit = _fit_bt(pair_wins, config, initial=None)
        _, _, resistances = _profile_geometry(pair_wins, fit["theta_hat"])
        K, d = fit["theta_hat"].shape
        candidates = []
        for r in range(d):
            for better in range(K):
                for worse in range(K):
                    if (
                        better != worse
                        and fit["theta_hat"][better, r]
                        < fit["theta_hat"][worse, r]
                    ):
                        candidates.append((r, better, worse))
        rng.shuffle(candidates)
        accepted = 0
        for r, better, worse in candidates:
            approximation = _quadratic_order_profile(
                fit["theta_hat"][:, r],
                resistances[r],
                better,
                worse,
            )
            constrained = fit_box_constrained_bt_coordinate(
                pair_wins[r],
                box_bound=2.0,
                initial=fit["theta_hat"][:, r],
                max_iter=5000,
                tol=1e-10,
                order_constraints=((better, worse),),
            )
            exact = (
                constrained["nll"]
                - fit["coordinate_results"][r]["nll"]
            )
            # Near-ties can produce a numerically zero constrained loss.  They
            # carry no useful relative-error information for this diagnostic.
            if exact <= 1e-10 or approximation <= 1e-12:
                continue
            rows.append(
                {
                    "generator": generator,
                    "instance_index": instance_index,
                    "coordinate": r,
                    "better": better,
                    "worse": worse,
                    "quadratic_profile": float(approximation),
                    "exact_profile": float(exact),
                    "quadratic_over_exact": float(approximation / exact),
                    "exact_converged": bool(constrained["converged"]),
                }
            )
            accepted += 1
            if accepted == constraints_per_instance:
                break
        if accepted < constraints_per_instance:
            raise RuntimeError(
                f"instance {instance_index} supplied only {accepted} "
                "numerically informative profile constraints"
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=40)
    parser.add_argument("--constraints-per-instance", type=int, default=6)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "summary" / "profile_validation.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results" / "figures" / "profile_validation.pdf",
    )
    args = parser.parse_args()
    rows = run_validation(args.instances, args.constraints_per_instance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    exact = np.array([row["exact_profile"] for row in rows], dtype=float)
    approximation = np.array(
        [row["quadratic_profile"] for row in rows], dtype=float
    )
    ratio = approximation / exact
    correlation = float(np.corrcoef(np.log(exact), np.log(approximation))[0, 1])
    print(
        {
            "n": len(rows),
            "log_correlation": correlation,
            "median_ratio": float(np.median(ratio)),
            "q05_ratio": float(np.quantile(ratio, 0.05)),
            "q95_ratio": float(np.quantile(ratio, 0.95)),
            "all_exact_converged": all(row["exact_converged"] for row in rows),
        }
    )

    lower = min(float(exact.min()), float(approximation.min()))
    upper = max(float(exact.max()), float(approximation.max()))
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.scatter(
        exact,
        approximation,
        s=18,
        alpha=0.55,
        color="#28795D",
        edgecolors="none",
    )
    axis.plot([lower, upper], [lower, upper], color="#C9515D", linewidth=1.5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("exact constrained profile likelihood")
    axis.set_ylabel("local quadratic approximation")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
