"""Run the full formal two-algorithm comparison with resumable checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

# Process-level parallelism is the intended execution model. Prevent each
# worker from starting a nested BLAS thread pool and competing for the same
# cores; this changes wall-clock time only, not the numerical protocol.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from tqdm import tqdm

from icml_pareto_track_stop.formal_protocol import (
    FormalJob,
    load_or_build_formal_jobs,
    protocol_counts,
)
from icml_pareto_track_stop.certificate_track_stop import (
    ParetoCertificateTrackStopConfig,
    run_pareto_certificate_track_and_stop,
)
from vb_ege.algorithms import VBEGEConfig, run_vb_ege
from vb_ege.core import strict_pareto_set
from vb_ege.gaps import compute_gaps
from vb_ege.io_utils import dumps_json
from vb_ege.metrics import hamming_set_distance, set_error


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "formal_config.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "raw" / "formal_two_way_certificate.csv"
DEFAULT_JOB_CACHE = ROOT / "results" / "formal_certificate_jobs.pkl"


def _run_job(job: FormalJob) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(job.seed)
    if job.algorithm == "VB-EGE-practical":
        result_raw = run_vb_ege(
            job.theta,
            VBEGEConfig(**job.algorithm_config),
            rng=rng,
        )
        result = {
            "recommended": result_raw.recommended,
            "tau": result_raw.tau,
            "stopped": result_raw.stopped,
            "num_phases": result_raw.num_phases,
            "num_accepted": len(result_raw.accepted),
            "num_rejected": len(result_raw.rejected),
            "num_unresolved": len(result_raw.active_final),
            "mle_converged_all": True,
            "final_glr_lower_bound": np.nan,
            "final_glr_threshold": np.nan,
            "final_alternative_kind": "",
            "statistic_type": "",
        }
    elif job.algorithm == "Pareto BT-GLR Track-and-Stop":
        track_config = dict(job.algorithm_config)
        track_config["delta"] = job.delta
        result = run_pareto_certificate_track_and_stop(
            job.theta,
            ParetoCertificateTrackStopConfig(**track_config),
            rng=rng,
        )
    else:
        raise ValueError(f"unknown algorithm: {job.algorithm}")

    truth = strict_pareto_set(job.theta)
    recommended = tuple(result["recommended"])
    stopped = bool(result["stopped"])
    gaps = compute_gaps(job.theta)
    return {
        "run_id": job.run_id,
        "cell_key": job.cell_key,
        "section": job.section,
        "experiment_id": job.experiment_id,
        "algorithm": job.algorithm,
        "replicate_index": job.replicate_index,
        "replicate_id": job.meta.get("replicate_id"),
        "instance_id": job.meta.get("instance_id"),
        "instance_bank_id": job.meta.get("instance_bank_id"),
        "instance_index": job.meta.get("instance_index"),
        "observation_replicate": job.meta.get("observation_replicate"),
        "instance_seed": job.meta.get("instance_seed"),
        "observation_seed": job.seed,
        "theta_hash": job.meta.get("theta_hash"),
        "theta_params_json": dumps_json(job.meta["params"]),
        "K": int(job.theta.shape[0]),
        "d": int(job.theta.shape[1]),
        "delta": job.delta,
        "pareto_size_true": len(truth),
        "true_pareto_json": dumps_json(truth),
        "recommended_json": dumps_json(recommended),
        "tau": int(result["tau"]),
        "stopped": stopped,
        "error": int(set_error(recommended, truth)) if stopped else "",
        "hamming": int(hamming_set_distance(recommended, truth)) if stopped else "",
        "num_phases": int(result["num_phases"]),
        "num_accepted": int(result.get("num_accepted", 0)),
        "num_rejected": int(result.get("num_rejected", 0)),
        "num_unresolved": int(result.get("num_unresolved", 0)),
        "delta_min_theta": float(gaps["delta_min"]),
        "mle_converged_all": bool(result.get("mle_converged_all", True)),
        "final_glr_statistic": result.get("final_glr_lower_bound", ""),
        "final_glr_estimate": result.get("final_glr_estimate", ""),
        "final_glr_upper_bound": result.get("final_glr_upper_bound", ""),
        "final_glr_threshold": result.get("final_glr_threshold", ""),
        "final_profile_interval_width": result.get(
            "final_profile_interval_width", ""
        ),
        "final_profile_interval_closed": result.get(
            "final_profile_interval_closed", ""
        ),
        "final_assignment_search_exact": result.get(
            "final_assignment_search_exact", ""
        ),
        "final_profile_mode": result.get("final_profile_mode", ""),
        "stopping_statistic_is_certified": result.get(
            "stopping_statistic_is_certified", ""
        ),
        "final_alternative_kind": result.get("final_alternative_kind", ""),
        "statistic_type": result.get("statistic_type", ""),
        "elapsed_seconds": float(time.perf_counter() - started),
        "pareto_convention": "all_coordinate_strict",
        "method_version": (
            "vbege-practical-v1"
            if job.algorithm == "VB-EGE-practical"
            else "pareto-certificate-profile-v1"
        ),
    }


def _read_completed(checkpoint: Path) -> set[int]:
    if not checkpoint.exists():
        return set()
    completed = set()
    with checkpoint.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["run_id"]))
    return completed


def _append_rows(checkpoint: Path, rows: list[dict]) -> None:
    if not rows:
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _finalize(checkpoint: Path, output: Path) -> None:
    rows = []
    with checkpoint.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["run_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-cache", type=Path, default=DEFAULT_JOB_CACHE)
    parser.add_argument("--rebuild-job-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument(
        "--prepare-jobs",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--section", action="append")
    parser.add_argument("--rep-limit", type=int)
    parser.add_argument(
        "--algorithm",
        choices=("both", "vbege", "track"),
        default="both",
    )
    args = parser.parse_args()

    jobs = load_or_build_formal_jobs(
        args.config,
        cache_path=args.job_cache,
        prepare_workers=args.prepare_jobs,
        force_rebuild=args.rebuild_job_cache,
    )
    if args.section:
        selected = set(args.section)
        jobs = [job for job in jobs if job.section in selected]
    if args.rep_limit is not None:
        jobs = [job for job in jobs if job.replicate_index < args.rep_limit]
    if args.algorithm != "both":
        selected_algorithm = {
            "vbege": "VB-EGE-practical",
            "track": "Pareto BT-GLR Track-and-Stop",
        }[args.algorithm]
        jobs = [job for job in jobs if job.algorithm == selected_algorithm]

    checkpoint = args.out.with_suffix(".jsonl")
    if checkpoint.exists() and not args.resume:
        checkpoint.unlink()
    completed = _read_completed(checkpoint) if args.resume else set()
    pending = [job for job in jobs if job.run_id not in completed]
    print(f"protocol counts: {protocol_counts(jobs)}", flush=True)
    print(
        f"scheduled={len(jobs)}, completed={len(completed)}, pending={len(pending)}",
        flush=True,
    )

    buffer: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        iterator = executor.map(_run_job, pending, chunksize=1)
        for row in tqdm(
            iterator,
            total=len(jobs),
            initial=len(completed),
            desc=f"formal two-way ({args.jobs} jobs)",
        ):
            buffer.append(row)
            if len(buffer) >= args.checkpoint_every:
                _append_rows(checkpoint, buffer)
                buffer.clear()
    _append_rows(checkpoint, buffer)
    _finalize(checkpoint, args.out)
    print(f"wrote {len(jobs)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
