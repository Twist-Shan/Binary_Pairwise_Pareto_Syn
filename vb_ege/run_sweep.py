"""Expand YAML configs and run synthetic experiment sweeps."""

from __future__ import annotations

import argparse
import itertools
import math
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
from .io_utils import dumps_json, load_yaml, stable_seed, write_dataframe
from .metrics import hamming_set_distance, set_error

pd = import_pandas_quietly()


FIXED_CONFIDENCE_ALGORITHMS = {
    "VB-EGE-practical",
    "UniformFocalBorda-FC",
    "UniformPairwiseBT-MLE-FC",
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
    algs = exp.get("algorithms", config.get("algorithms", {}))
    if isinstance(algs, list):
        return [(name, config.get("algorithms", {}).get(name, {})) for name in algs]
    return [(name, cfg or {}) for name, cfg in algs.items()]


def _run_algorithm(name: str, theta, budget, cfg: dict, rng):
    if name == "VB-EGE-practical":
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
    if name == "UniformPairwiseBT-MLE-FC":
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
    return row


def _iter_experiment_jobs(config: dict, base_seed: int):
    for exp in config.get("experiments", []):
        budgets = exp.get("budgets", config.get("budgets", [None]))
        for rep in range(int(exp.get("n_reps", 1))):
            inst_seed = stable_seed(base_seed, exp["id"], rep, "instance")
            theta, meta = make_instance(exp["generator"], exp.get("params", {}), inst_seed)
            true_pareto = strict_pareto_set(theta)
            if strict_pareto_set(borda(theta)) != true_pareto:
                raise RuntimeError(f"Borda/Pareto mismatch in {exp['id']} rep {rep}")
            for alg_name, alg_cfg in _algorithm_jobs(config, exp):
                alg_budgets = [None] if alg_name in FIXED_CONFIDENCE_ALGORITHMS else budgets
                for budget in alg_budgets:
                    if budget is None and alg_name not in FIXED_CONFIDENCE_ALGORITHMS:
                        continue
                    alg_seed = stable_seed(base_seed, exp["id"], rep, budget, alg_name, dumps_json(alg_cfg))
                    yield exp, rep, budget, alg_name, alg_cfg, theta, meta, alg_seed


def _iter_sweep_jobs(config: dict, base_seed: int):
    algorithms = config.get("algorithms", {})
    if not algorithms:
        return
    for sweep in config.get("sweeps", []):
        for params in _grid_product(sweep["grid"]):
            for rep in range(int(sweep.get("n_reps", 1))):
                inst_seed = stable_seed(base_seed, sweep["id"], dumps_json(params), rep, "instance")
                inst_params = {k: v for k, v in params.items() if k != "delta"}
                theta, meta = make_instance(sweep["generator"], inst_params, inst_seed)
                exp = {"id": sweep["id"], "generator": sweep["generator"], "params": inst_params}
                for alg_name, alg_cfg in algorithms.items():
                    cfg = {**(alg_cfg or {}), "delta": params.get("delta", (alg_cfg or {}).get("delta"))}
                    alg_seed = stable_seed(base_seed, sweep["id"], dumps_json(params), rep, alg_name)
                    yield exp, rep, None, alg_name, cfg, theta, meta, alg_seed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    base_seed = int(args.seed if args.seed is not None else config.get("base_seed", 0))
    jobs = list(_iter_experiment_jobs(config, base_seed))
    jobs.extend(list(_iter_sweep_jobs(config, base_seed)))
    rows = []
    for run_id, (exp, _rep, budget, alg_name, alg_cfg, theta, meta, alg_seed) in enumerate(
        tqdm(jobs, desc=f"running {config.get('name', Path(args.config).stem)}")
    ):
        rng = np.random.default_rng(alg_seed)
        result = _run_algorithm(alg_name, theta, budget, dict(alg_cfg or {}), rng)
        rows.append(
            _row_from_result(
                run_id,
                alg_seed,
                alg_name,
                exp["id"],
                theta,
                meta,
                budget,
                (alg_cfg or {}).get("delta"),
                dict(alg_cfg or {}),
                result,
            )
        )
    df = pd.DataFrame(rows)
    actual_out = write_dataframe(df, args.out)
    print(f"wrote {len(df)} rows to {actual_out}")


if __name__ == "__main__":
    main()
