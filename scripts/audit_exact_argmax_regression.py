"""Compare exact Algorithm 1 argmax with the historical ``np.isclose`` rule.

This is a result-preservation audit, not a new experiment.  It replays selected
configured seeds twice and reports any change in the stopping result.  No raw,
summary, or figure artifact is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import vb_ege.algorithms as algorithms
from vb_ege.io_utils import load_yaml
from vb_ege.run_sweep import (
    _iter_experiment_instance_specs,
    _iter_jobs_from_prepared_instances,
    _iter_sweep_instance_specs,
    _prepare_instance_spec,
    _run_algorithm,
)


def _historical_close_argmax(
    active,
    gap_by_arm,
    empirical_pareto,
    tie_break_nonpareto_first=True,
):
    maximum = max(gap_by_arm[arm] for arm in active)
    candidates = [
        arm for arm in active if np.isclose(gap_by_arm[arm], maximum)
    ]
    if tie_break_nonpareto_first:
        pareto = set(empirical_pareto)
        nonpareto = [arm for arm in candidates if arm not in pareto]
        if nonpareto:
            return int(min(nonpareto))
    return int(min(candidates))


def _signature(result: dict) -> dict:
    return {
        key: result.get(key)
        for key in (
            "recommended",
            "tau",
            "stopped",
            "num_phases",
            "accepted",
            "rejected",
            "active_final",
        )
    }


def audit(config_paths: list[Path], replicate_indices: set[int]) -> dict:
    checked = 0
    differences: list[dict] = []
    exact_rule = algorithms.choose_removal_arm
    try:
        for config_path in config_paths:
            config = load_yaml(config_path)
            base_seed = int(config.get("base_seed", 0))
            specs = list(_iter_experiment_instance_specs(config, base_seed))
            specs.extend(_iter_sweep_instance_specs(config, base_seed))
            for spec in specs:
                if int(spec[2]) not in replicate_indices:
                    continue
                prepared = _prepare_instance_spec(spec)
                for job in _iter_jobs_from_prepared_instances([prepared], base_seed):
                    exp, rep, budget, name, cfg, theta, _meta, seed = job
                    if name not in {"VB-EGE-practical", "VB-EGE-theory"}:
                        continue
                    checked += 1
                    algorithms.choose_removal_arm = exact_rule
                    exact = _run_algorithm(
                        name, theta, budget, dict(cfg), np.random.default_rng(seed)
                    )
                    algorithms.choose_removal_arm = _historical_close_argmax
                    historical = _run_algorithm(
                        name, theta, budget, dict(cfg), np.random.default_rng(seed)
                    )
                    if _signature(exact) != _signature(historical):
                        differences.append(
                            {
                                "config": config_path.as_posix(),
                                "experiment_id": exp["id"],
                                "replicate": int(rep),
                                "seed": int(seed),
                                "algorithm": name,
                                "exact": _signature(exact),
                                "historical": _signature(historical),
                            }
                        )
    finally:
        algorithms.choose_removal_arm = exact_rule
    return {
        "checked_vb_ege_runs": checked,
        "replicate_indices": sorted(replicate_indices),
        "differences": differences,
        "ok": not differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--replicate-index", action="append", type=int)
    args = parser.parse_args()
    replicate_indices = set(args.replicate_index or [0])
    result = audit(args.config, replicate_indices)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
