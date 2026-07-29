"""Expand the existing formal experiment configs into paired two-way jobs."""

from __future__ import annotations

import hashlib
import concurrent.futures
import pickle
from dataclasses import dataclass
from pathlib import Path

from vb_ege.io_utils import dumps_json, load_yaml, stable_seed
from vb_ege.run_sweep import (
    _iter_experiment_instance_specs,
    _iter_jobs_from_prepared_instances,
    _iter_sweep_instance_specs,
    _prepare_instance_spec,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FormalJob:
    run_id: int
    section: str
    experiment_id: str
    replicate_index: int
    algorithm: str
    theta: object
    meta: dict
    algorithm_config: dict
    seed: int
    delta: float
    cell_key: str


def _theta_hash(theta) -> str:
    return hashlib.sha256(theta.tobytes(order="C")).hexdigest()


def protocol_fingerprint(config_path: Path) -> str:
    config_path = Path(config_path)
    suite = load_yaml(config_path)
    digest = hashlib.sha256(config_path.read_bytes())
    for source in suite["source_protocols"]:
        source_path = ROOT / str(source["config"])
        digest.update(str(source_path.relative_to(ROOT)).encode("utf-8"))
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def build_formal_jobs(
    config_path: Path,
    prepare_workers: int = 1,
) -> list[FormalJob]:
    config_path = Path(config_path)
    suite = load_yaml(config_path)
    jobs: list[FormalJob] = []
    run_id = 0

    for source in suite["source_protocols"]:
        section = str(source["section"])
        source_path = ROOT / str(source["config"])
        source_config = load_yaml(source_path)
        base_seed = int(source_config.get("base_seed", 0))
        specs = list(_iter_experiment_instance_specs(source_config, base_seed))
        specs.extend(list(_iter_sweep_instance_specs(source_config, base_seed)))

        if prepare_workers <= 1:
            prepared_specs = map(_prepare_instance_spec, specs)
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=prepare_workers
            )
            prepared_specs = executor.map(
                _prepare_instance_spec,
                specs,
                chunksize=4,
            )
        try:
            for prepared in prepared_specs:
                _, exp, rep, _, _, theta, meta, seed_extra = prepared
                vbege_jobs = [
                    item
                    for item in _iter_jobs_from_prepared_instances(
                        [prepared], base_seed
                    )
                    if item[3] == "VB-EGE-practical"
                ]
                if len(vbege_jobs) != 1:
                    raise RuntimeError(
                        f"{section}/{exp['id']}/rep={rep}: expected one VB-EGE job"
                    )
                (
                    _exp,
                    _rep,
                    _budget,
                    _algorithm,
                    vbege_config,
                    _theta,
                    _meta,
                    vbege_seed,
                ) = vbege_jobs[0]
                delta = float(vbege_config["delta"])
                params_json = dumps_json(meta["params"])
                cell_key = "|".join(
                    [
                        section,
                        exp["id"],
                        str(rep),
                        params_json,
                        f"{delta:.17g}",
                    ]
                )
                common = {
                    "section": section,
                    "experiment_id": exp["id"],
                    "replicate_index": int(rep),
                    "theta": theta,
                    "meta": meta,
                    "delta": delta,
                    "cell_key": cell_key,
                }
                jobs.append(
                    FormalJob(
                        run_id=run_id,
                        algorithm="VB-EGE-practical",
                        algorithm_config=dict(vbege_config),
                        seed=int(vbege_seed),
                        **common,
                    )
                )
                run_id += 1
                track_seed = stable_seed(
                    base_seed,
                    section,
                    exp["id"],
                    seed_extra,
                    rep,
                    "Pareto BT-GLR Track-and-Stop",
                )
                jobs.append(
                    FormalJob(
                        run_id=run_id,
                        algorithm="Pareto BT-GLR Track-and-Stop",
                        algorithm_config=dict(suite["track_stop"]),
                        seed=int(track_seed),
                        **common,
                    )
                )
                run_id += 1

                if meta.get("theta_hash") != _theta_hash(theta):
                    raise RuntimeError(
                        "formal instance hash changed during expansion"
                    )
        finally:
            if prepare_workers > 1:
                executor.shutdown()
    return jobs


def load_or_build_formal_jobs(
    config_path: Path,
    cache_path: Path,
    prepare_workers: int = 1,
    force_rebuild: bool = False,
) -> list[FormalJob]:
    fingerprint = protocol_fingerprint(config_path)
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("fingerprint") == fingerprint:
            return cached["jobs"]
    jobs = build_formal_jobs(config_path, prepare_workers=prepare_workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {"fingerprint": fingerprint, "jobs": jobs},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return jobs


def protocol_counts(jobs: list[FormalJob]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for job in jobs:
        section = counts.setdefault(job.section, {})
        section[job.algorithm] = section.get(job.algorithm, 0) + 1
    return counts
