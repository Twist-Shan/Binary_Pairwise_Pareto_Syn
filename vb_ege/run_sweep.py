"""Expand YAML configs and run synthetic experiment sweeps."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .algorithms import VBEGEConfig, run_vb_ege
from .baselines import (
    run_uniform_focal_borda,
    run_uniform_focal_borda_fc,
    run_uniform_pairwise_bt_borda_plugin,
    run_uniform_pairwise_bt_borda_plugin_fc,
    run_uniform_pairwise_bt_mle,
    run_uniform_pairwise_bt_mle_fc,
    run_vb_ege_capped,
)
from .core import borda, dynamic_range_B, kappa_B, strict_pareto_set
from .gaps import compute_gaps
from .instances import make_instance
from .compat import import_pandas_quietly
from .io_utils import dumps_json, ensure_parent, load_yaml, stable_seed, write_dataframe
from .metrics import hamming_set_distance, set_error

pd = import_pandas_quietly()


FIXED_CONFIDENCE_ALGORITHMS = {
    "VB-EGE-practical",
    "VB-EGE-theory",
    "UniformFocalBorda-FC",
    "UniformPairwiseBT-MLE-FC",
    "UniformPairwiseBT-MLE-Cert",
    "UniformPairwiseBT-BordaPlugIn-FC",
}


def _grid_product(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _mle_grid_product(grid: dict) -> list[dict]:
    return _grid_product(grid)


def _algorithm_jobs(config: dict, exp: dict) -> list[tuple[str, dict]]:
    if "mle_grid" in config:
        return [
            (config.get("algorithm", "UniformPairwiseBT-MLE"), cfg)
            for cfg in _mle_grid_product(config["mle_grid"])
        ]
    if "constant_grid" in config:
        jobs: list[tuple[str, dict]] = []
        for cfg in _grid_product(config["constant_grid"]):
            run_cfg = {
                **cfg,
                "delta": config.get("delta", 0.05),
                "max_phases": config.get("max_phases", 60),
            }
            jobs.append(("VB-EGE-practical", run_cfg))
        for name, cfg in config.get("reference_algorithms", {}).items():
            ref_cfg = dict(cfg or {})
            ref_cfg.setdefault("delta", config.get("delta", 0.05))
            jobs.append((name, ref_cfg))
        return jobs
    algs = exp.get("algorithms", config.get("algorithms", {}))
    if isinstance(algs, list):
        return [(name, config.get("algorithms", {}).get(name, {})) for name in algs]
    return [(name, cfg or {}) for name, cfg in algs.items()]


def _expanded_experiment_cells(exp: dict) -> list[tuple[dict, float | None, str]]:
    if "grid" in exp:
        params_list = _grid_product(exp["grid"])
    else:
        params_list = [dict(exp.get("params", {}))]
    deltas = exp.get("delta_grid", [exp.get("delta")])
    pair_delta_instances = bool(exp.get("paired_delta_instances", False))
    cells = []
    for params in params_list:
        for delta in deltas:
            seed_payload = {"params": params}
            if delta is not None and not pair_delta_instances:
                seed_payload["delta"] = delta
            seed_extra = dumps_json(seed_payload)
            cells.append((params, delta, seed_extra))
    return cells


def _run_algorithm(name: str, theta, budget, cfg: dict, rng):
    if name in {"VB-EGE-practical", "VB-EGE-theory"}:
        run_cfg = VBEGEConfig(**cfg)
        res = run_vb_ege(theta, run_cfg, rng=rng)
        return {
            "algorithm": name,
            "recommended": res.recommended,
            "tau": res.tau,
            "error": res.error,
            "hamming": hamming_set_distance(res.recommended, strict_pareto_set(theta)),
            "stopped": res.stopped,
            "num_phases": res.num_phases,
            "num_accepted": len(res.accepted),
            "num_rejected": len(res.rejected),
            "pareto_size_hat": len(res.recommended),
            "N": res.N,
            "S": res.S,
            "history": res.history,
            "accepted": res.accepted,
            "rejected": res.rejected,
            "active_final": res.active_final,
        }
    if name == "VB-EGE-capped":
        return run_vb_ege_capped(theta, budget, cfg, rng)
    if name == "UniformFocalBorda":
        return run_uniform_focal_borda(theta, budget, rng)
    if name == "UniformFocalBorda-FC":
        return run_uniform_focal_borda_fc(theta, cfg, rng)
    if name == "UniformPairwiseBT-MLE":
        return run_uniform_pairwise_bt_mle(theta, budget, rng, cfg)
    if name in {"UniformPairwiseBT-MLE-FC", "UniformPairwiseBT-MLE-Cert"}:
        return run_uniform_pairwise_bt_mle_fc(theta, cfg, rng)
    if name == "UniformPairwiseBT-BordaPlugIn":
        return run_uniform_pairwise_bt_borda_plugin(theta, budget, rng, cfg)
    if name == "UniformPairwiseBT-BordaPlugIn-FC":
        return run_uniform_pairwise_bt_borda_plugin_fc(theta, cfg, rng)
    raise ValueError(f"unknown algorithm: {name}")


def _json_array_or_none(arr, max_size=4000):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size > max_size:
        return None
    return dumps_json(arr)


def _row_from_result(
    run_id: int,
    seed: int,
    algorithm: str,
    exp_id: str,
    experiment_metadata: dict,
    theta,
    meta,
    budget,
    delta,
    alg_cfg: dict,
    result: dict,
) -> dict:
    true_pareto = strict_pareto_set(theta)
    recommended = tuple(result["recommended"])
    b = borda(theta)
    gaps_B = compute_gaps(b)
    gaps_theta = compute_gaps(theta)
    B = dynamic_range_B(theta)
    K, d = theta.shape
    pair_cell_count = d * K * (K - 1) // 2
    tau = result.get("tau", np.nan)
    H_B = gaps_B["H"]
    cfg_delta = delta if delta is not None else alg_cfg.get("delta", np.nan)
    norm_tau_B = tau / (d * H_B) if np.isfinite(H_B) and H_B > 0 else np.nan
    logdelta = math.log(1.0 / cfg_delta) if cfg_delta and cfg_delta > 0 else np.nan
    false_pos = len(set(recommended) - set(true_pareto))
    false_neg = len(set(true_pareto) - set(recommended))
    mle_cfg = result.get("mle_config", {})
    row = {
        "run_id": run_id,
        "seed": seed,
        "replicate_id": meta.get("replicate_id"),
        "instance_id": meta.get("instance_id"),
        "instance_bank_id": meta.get("instance_bank_id"),
        "instance_index": meta.get("instance_index"),
        "observation_replicate": meta.get("observation_replicate"),
        "instance_seed": meta.get("instance_seed"),
        "theta_hash": meta.get("theta_hash"),
        "algorithm": algorithm,
        "experiment_id": exp_id,
        "instance_name": meta["name"],
        "K": K,
        "d": d,
        "delta": cfg_delta,
        "budget": budget,
        "sample_const": alg_cfg.get("sample_const"),
        "threshold_const": alg_cfg.get("threshold_const"),
        "log_const": alg_cfg.get("log_const"),
        "mle_ridge_lambda": mle_cfg.get("ridge_lambda", alg_cfg.get("ridge_lambda")),
        "mle_fallback_ridge_lambda": mle_cfg.get(
            "fallback_ridge_lambda", alg_cfg.get("fallback_ridge_lambda")
        ),
        "mle_init": mle_cfg.get("init", alg_cfg.get("init")),
        "mle_allocation_scheme": result.get(
            "mle_allocation_scheme", alg_cfg.get("allocation_scheme")
        ),
        "theta_params_json": dumps_json(meta["params"]),
        "true_pareto_json": dumps_json(true_pareto),
        "recommended_json": dumps_json(recommended),
        "error": bool(set_error(recommended, true_pareto)),
        "hamming": hamming_set_distance(recommended, true_pareto),
        "false_positive_count": false_pos,
        "false_negative_count": false_neg,
        "tau": tau,
        "stopped": result.get("stopped"),
        "num_phases": result.get("num_phases"),
        "num_accepted": result.get("num_accepted"),
        "num_rejected": result.get("num_rejected"),
        "pareto_size_true": len(true_pareto),
        "pareto_size_hat": len(recommended),
        "pareto_target": meta.get("expected_pareto_size"),
        "achieved_objective_correlation": meta["params"].get(
            "achieved_objective_correlation_mean"
        ),
        "H_B": gaps_B["H"],
        "H_theta": gaps_theta["H"],
        "delta_min_B": gaps_B["delta_min"],
        "delta_min_theta": gaps_theta["delta_min"],
        "B_min": B,
        "kappa_B": kappa_B(B),
        "norm_tau_B": norm_tau_B,
        "norm_tau_B_logdelta": norm_tau_B / logdelta if logdelta and logdelta > 0 else np.nan,
        "pair_cell_count": pair_cell_count,
        "pair_cell_coverage": result.get("cell_coverage"),
        "mle_converged_all": result.get("mle_converged_all"),
        "mle_ridge_fallback_any": result.get("mle_ridge_fallback_any"),
        "mle_max_abs_theta": result.get("mle_max_abs_theta"),
        "mle_mean_niter": result.get("mle_mean_niter"),
        "theta_rmse_centered": result.get("theta_rmse_centered"),
        "pairwise_sign_accuracy": result.get("pairwise_sign_accuracy"),
        "N_json_or_path": _json_array_or_none(result.get("N")),
        "history_path": None,
    }
    for key, val in meta["params"].items():
        if isinstance(val, (int, float, str, bool)):
            row[f"param_{key}"] = val
    for key, val in (experiment_metadata or {}).items():
        if isinstance(val, (int, float, str, bool)):
            row[f"meta_{key}"] = val
    return row


def _iter_experiment_instance_specs(config: dict, base_seed: int):
    for exp in config.get("experiments", []):
        if exp.get("optional", False):
            continue
        budgets = exp.get("budgets", config.get("budgets", [None]))
        for params, delta, seed_extra in _expanded_experiment_cells(exp):
            cell_exp = {**exp, "params": params}
            alg_jobs = []
            for alg_name, alg_cfg in _algorithm_jobs(config, exp):
                run_cfg = dict(alg_cfg or {})
                if delta is not None:
                    run_cfg["delta"] = delta
                alg_jobs.append((alg_name, run_cfg))
            n_reps = int(exp.get("n_reps", 1))
            observation_reps = int(exp.get("observation_reps", 1))
            if observation_reps <= 0 or n_reps % observation_reps:
                raise ValueError(
                    f"{exp['id']}: n_reps must be divisible by positive observation_reps"
                )
            n_instances = n_reps // observation_reps
            configured_instances = exp.get("n_instances")
            if configured_instances is not None and int(configured_instances) != n_instances:
                raise ValueError(
                    f"{exp['id']}: n_instances * observation_reps must equal n_reps"
                )
            bank_id = str(exp.get("instance_bank_id", exp["id"]))
            bank_seed = int(exp.get("instance_bank_seed", base_seed))
            for rep in range(n_reps):
                instance_index = rep // observation_reps
                observation_replicate = rep % observation_reps
                inst_seed = stable_seed(
                    bank_seed,
                    bank_id,
                    seed_extra,
                    instance_index,
                    "instance",
                )
                rep_exp = {
                    **cell_exp,
                    "_instance_bank_id": bank_id,
                    "_instance_bank_seed": bank_seed,
                    "_instance_index": instance_index,
                    "_observation_replicate": observation_replicate,
                }
                yield "experiment", rep_exp, rep, budgets, alg_jobs, inst_seed, seed_extra


def _iter_sweep_instance_specs(config: dict, base_seed: int):
    algorithms = config.get("algorithms", {})
    if not algorithms:
        return
    for sweep in config.get("sweeps", []):
        for params in _grid_product(sweep["grid"]):
            params_json = dumps_json(params)
            for rep in range(int(sweep.get("n_reps", 1))):
                inst_seed = stable_seed(base_seed, sweep["id"], params_json, rep, "instance")
                inst_params = {k: v for k, v in params.items() if k != "delta"}
                exp = {"id": sweep["id"], "generator": sweep["generator"], "params": inst_params}
                alg_jobs = [
                    (
                        alg_name,
                        {**(alg_cfg or {}), "delta": params.get("delta", (alg_cfg or {}).get("delta"))},
                    )
                    for alg_name, alg_cfg in algorithms.items()
                ]
                yield "sweep", exp, rep, [None], alg_jobs, inst_seed, params_json


def _prepare_instance_spec(payload):
    kind, exp, rep, budgets, alg_jobs, inst_seed, seed_extra = payload
    theta, meta = make_instance(exp["generator"], exp.get("params", {}), inst_seed)
    true_pareto = strict_pareto_set(theta)
    if strict_pareto_set(borda(theta)) != true_pareto:
        raise RuntimeError(f"Borda/Pareto mismatch in {exp['id']} rep {rep}")
    theta_bytes = np.ascontiguousarray(theta, dtype=np.float64).tobytes()
    theta_hash = hashlib.sha256(theta_bytes).hexdigest()
    bank_id = str(exp.get("_instance_bank_id", exp["id"]))
    instance_index = int(exp.get("_instance_index", rep))
    observation_replicate = int(exp.get("_observation_replicate", 0))
    meta = dict(meta)
    meta.update(
        {
            "replicate_id": f"{bank_id}:{instance_index:04d}:{observation_replicate:02d}",
            "instance_id": f"{bank_id}:{instance_index:04d}",
            "instance_bank_id": bank_id,
            "instance_index": instance_index,
            "observation_replicate": observation_replicate,
            "instance_seed": int(inst_seed),
            "theta_hash": theta_hash,
        }
    )
    return kind, exp, rep, budgets, alg_jobs, theta, meta, seed_extra


def _iter_jobs_from_prepared_instances(prepared_instances, base_seed: int):
    for kind, exp, rep, budgets, alg_jobs, theta, meta, seed_extra in prepared_instances:
        for alg_name, alg_cfg in alg_jobs:
            if kind == "experiment":
                alg_budgets = [None] if alg_name in FIXED_CONFIDENCE_ALGORITHMS else budgets
                for budget in alg_budgets:
                    if budget is None and alg_name not in FIXED_CONFIDENCE_ALGORITHMS:
                        continue
                    observation_seed = int(exp.get("observation_bank_seed", exp.get("_instance_bank_seed", base_seed)))
                    observation_bank = str(exp.get("instance_bank_id", exp.get("_instance_bank_id", exp["id"])))
                    alg_seed = stable_seed(
                        observation_seed,
                        observation_bank,
                        seed_extra,
                        rep,
                        budget,
                        alg_name,
                        alg_cfg.get("delta"),
                    )
                    yield exp, rep, budget, alg_name, alg_cfg, theta, meta, alg_seed
            else:
                alg_seed = stable_seed(base_seed, exp["id"], seed_extra, rep, alg_name)
                yield exp, rep, None, alg_name, alg_cfg, theta, meta, alg_seed


def _execute_job(payload) -> dict:
    run_id, exp, budget, alg_name, alg_cfg, theta, meta, alg_seed = payload
    rng = np.random.default_rng(alg_seed)
    result = _run_algorithm(alg_name, theta, budget, dict(alg_cfg or {}), rng)
    return _row_from_result(
        run_id,
        alg_seed,
        alg_name,
        exp["id"],
        dict(exp.get("metadata", {})),
        theta,
        meta,
        budget,
        (alg_cfg or {}).get("delta"),
        dict(alg_cfg or {}),
        result,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--experiment-id",
        action="append",
        help="Run only the named experiment id; may be supplied more than once.",
    )
    parser.add_argument(
        "--replicate-index",
        action="append",
        type=int,
        help="Run only the zero-based replication index; may be supplied more than once.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    base_seed = int(args.seed if args.seed is not None else config.get("base_seed", 0))
    instance_specs = list(_iter_experiment_instance_specs(config, base_seed))
    instance_specs.extend(list(_iter_sweep_instance_specs(config, base_seed)))
    if args.experiment_id:
        selected = set(args.experiment_id)
        instance_specs = [spec for spec in instance_specs if spec[1]["id"] in selected]
        missing = selected - {spec[1]["id"] for spec in instance_specs}
        if missing:
            raise ValueError(f"unknown or empty experiment ids: {sorted(missing)}")
    if args.replicate_index:
        selected_reps = set(args.replicate_index)
        instance_specs = [spec for spec in instance_specs if spec[2] in selected_reps]
        missing_reps = selected_reps - {spec[2] for spec in instance_specs}
        if missing_reps:
            raise ValueError(f"unknown or empty replicate indices: {sorted(missing_reps)}")
    prep_desc = f"preparing {config.get('name', Path(args.config).stem)} instances"
    if args.jobs <= 1:
        prepared_instances = [
            item for item in tqdm(
                map(_prepare_instance_spec, instance_specs),
                total=len(instance_specs),
                desc=prep_desc,
            )
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            prepared_instances = list(
                tqdm(
                    executor.map(_prepare_instance_spec, instance_specs, chunksize=1),
                    total=len(instance_specs),
                    desc=f"{prep_desc} ({args.jobs} jobs)",
                )
            )
    jobs = list(_iter_jobs_from_prepared_instances(prepared_instances, base_seed))

    out_path = Path(args.out)
    checkpoint_path = out_path.with_suffix(".jsonl")
    existing_run_ids: set[int] = set()
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                existing_run_ids.add(int(json.loads(line)["run_id"]))
        print(
            f"resuming from {checkpoint_path}; skipping {len(existing_run_ids)} completed rows"
        )

    rows: list[dict] = []

    def flush_rows() -> None:
        nonlocal rows
        if not rows:
            return
        ensure_parent(checkpoint_path)
        for attempt in range(8):
            try:
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(dumps_json(row) + "\n")
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.25 * (attempt + 1))
        last_run_id = rows[-1]["run_id"]
        print(f"checkpoint: wrote through run_id={last_run_id} to {checkpoint_path}")
        rows = []

    pending_payloads = [
        (run_id, exp, budget, alg_name, alg_cfg, theta, meta, alg_seed)
        for run_id, (exp, _rep, budget, alg_name, alg_cfg, theta, meta, alg_seed) in enumerate(jobs)
        if run_id not in existing_run_ids
    ]
    desc = f"running {config.get('name', Path(args.config).stem)}"
    progress_total = len(existing_run_ids) + len(pending_payloads)
    if args.jobs <= 1:
        iterator = map(_execute_job, pending_payloads)
        for row in tqdm(iterator, total=progress_total, initial=len(existing_run_ids), desc=desc):
            rows.append(row)
            if args.checkpoint_every > 0 and len(rows) >= args.checkpoint_every:
                flush_rows()
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_execute_job, payload) for payload in pending_payloads]
            for row in tqdm(
                (future.result() for future in concurrent.futures.as_completed(futures)),
                total=progress_total,
                initial=len(existing_run_ids),
                desc=f"{desc} ({args.jobs} jobs)",
            ):
                rows.append(row)
                if args.checkpoint_every > 0 and len(rows) >= args.checkpoint_every:
                    flush_rows()
    flush_rows()

    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            final_rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(final_rows).sort_values("run_id")
        actual_out = write_dataframe(df, out_path)
    else:
        actual_out = checkpoint_path
    print(f"wrote {len(jobs)} scheduled rows; output at {actual_out}")


if __name__ == "__main__":
    main()
