"""Isolated smoke comparison for the certificate-based Pareto BT-GLR method."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from icml_pareto_track_stop.certificate_track_stop import (
    ParetoCertificateTrackStopConfig,
    run_pareto_certificate_track_and_stop,
)
from vb_ege.algorithms import VBEGEConfig, run_vb_ege
from vb_ege.core import strict_pareto_set
from vb_ege.instances import make_instance
from vb_ege.metrics import set_error


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "smoke_certificate"

SETTINGS = (
    (
        "symmetric-hard",
        "symmetric_hard",
        {"K": 6, "d": 3, "Delta": 1.0},
    ),
    (
        "arena-3",
        "arena_tradeoff_frontier",
        {
            "K": 6,
            "d": 3,
            "s": 2,
            "margin_low": 0.20,
            "margin_high": 0.30,
        },
    ),
    (
        "unique-witness",
        "unique_witness_d",
        {
            "K": 6,
            "d": 3,
            "s": 2,
            "q_per_p": 2,
            "margin_low": 0.18,
            "margin_high": 0.28,
        },
    ),
    (
        "convex-3d",
        "convex_frontier_3d",
        {
            "K": 6,
            "d": 3,
            "s": 2,
            "margin_low": 0.12,
            "margin_high": 0.22,
        },
    ),
)
SETTING_INDEX = {
    setting_id: index
    for index, (setting_id, _generator, _params) in enumerate(SETTINGS)
}


def _run_one(
    setting_id: str,
    generator: str,
    params: dict,
    replicate: int,
    delta: float,
) -> list[dict]:
    instance_seed = 51000 + 100 * SETTING_INDEX[setting_id] + replicate
    theta, _ = make_instance(generator, params, seed=instance_seed)
    truth = strict_pareto_set(theta)
    rows = []

    started = time.perf_counter()
    vb = run_vb_ege(
        theta,
        VBEGEConfig(
            delta=delta,
            sample_const=2.0,
            threshold_const=4.0,
            log_const=4.0,
            max_queries=10**12,
        ),
        rng=np.random.default_rng(instance_seed + 1_000_000),
    )
    rows.append(
        {
            "setting": setting_id,
            "generator": generator,
            "replicate": replicate,
            "instance_seed": instance_seed,
            "algorithm": "VB-EGE",
            "K": theta.shape[0],
            "d": theta.shape[1],
            "pareto_size": len(truth),
            "delta": delta,
            "tau": int(vb.tau),
            "stopped": bool(vb.stopped),
            "error": (
                int(set_error(vb.recommended, truth)) if vb.stopped else ""
            ),
            "mle_converged_all": "",
            "profile_fits_converged_all": "",
            "assignment_search_exact": "",
            "glr_lower": "",
            "glr_estimate": "",
            "glr_upper": "",
            "glr_threshold": "",
            "profile_interval_width": "",
            "alternative_kind": "",
            "elapsed_seconds": time.perf_counter() - started,
        }
    )

    started = time.perf_counter()
    track = run_pareto_certificate_track_and_stop(
        theta,
        ParetoCertificateTrackStopConfig(
            delta=delta,
            box_bound=2.0,
            burnin_per_cell=3,
            growth_factor=3.0,
            max_queries=10**9,
            optimizer_tol=1e-6,
            optimizer_max_iter=800,
            max_phases=18,
            max_exact_add_assignments=4096,
        ),
        rng=np.random.default_rng(instance_seed + 2_000_000),
    )
    rows.append(
        {
            "setting": setting_id,
            "generator": generator,
            "replicate": replicate,
            "instance_seed": instance_seed,
            "algorithm": "Pareto BT-GLR Track-and-Stop",
            "K": theta.shape[0],
            "d": theta.shape[1],
            "pareto_size": len(truth),
            "delta": delta,
            "tau": int(track["tau"]),
            "stopped": bool(track["stopped"]),
            "error": (
                int(track["error"]) if track["stopped"] else ""
            ),
            "mle_converged_all": track["mle_converged_all"],
            "profile_fits_converged_all": track[
                "profile_fits_converged_all"
            ],
            "assignment_search_exact": track[
                "final_assignment_search_exact"
            ],
            "glr_lower": track["final_glr_lower_bound"],
            "glr_estimate": track["final_glr_estimate"],
            "glr_upper": track["final_glr_upper_bound"],
            "glr_threshold": track["final_glr_threshold"],
            "profile_interval_width": track[
                "final_profile_interval_width"
            ],
            "alternative_kind": track["final_alternative_kind"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["setting"], row["algorithm"]) for row in rows})
    for setting, algorithm in keys:
        group = [
            row
            for row in rows
            if row["setting"] == setting and row["algorithm"] == algorithm
        ]
        stopped = [row for row in group if row["stopped"]]
        errors = [int(row["error"]) for row in stopped]
        result.append(
            {
                "setting": setting,
                "algorithm": algorithm,
                "runs": len(group),
                "stop_rate": len(stopped) / len(group),
                "error_rate_among_stopped": (
                    float(np.mean(errors)) if errors else None
                ),
                "mean_tau": float(np.mean([row["tau"] for row in group])),
                "mean_elapsed_seconds": float(
                    np.mean([row["elapsed_seconds"] for row in group])
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be positive")

    rows = []
    for setting_id, generator, params in SETTINGS:
        for replicate in range(args.reps):
            print(
                f"[smoke] {setting_id} replicate {replicate + 1}/{args.reps}",
                flush=True,
            )
            rows.extend(
                _run_one(
                    setting_id,
                    generator,
                    params,
                    replicate,
                    args.delta,
                )
            )

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "certificate_smoke.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    (args.out / "certificate_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"[smoke] wrote {csv_path}")


if __name__ == "__main__":
    main()
