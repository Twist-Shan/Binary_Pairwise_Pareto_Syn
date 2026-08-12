"""I/O helpers for runners."""

from __future__ import annotations

import hashlib
import contextlib
import io
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, default=json_default, sort_keys=True, separators=(",", ":"))


def stable_seed(base_seed: int, *parts) -> int:
    msg = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path) -> tuple[str, int]:
    """Hash tracked experiment source inputs by relative path and bytes."""

    directory = Path(root).resolve()
    files = sorted(
        path
        for pattern in ("*.py", "*.yaml", "*.toml")
        for path in directory.rglob(pattern)
        if path.is_file()
        and not any(part in {".venv", "results", "tmp", "__pycache__"} for part in path.parts)
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(files)


def write_run_manifest(
    output: str | Path,
    *,
    config: str | Path,
    project_root: str | Path,
    scheduled_rows: int,
    base_seed: int,
) -> Path:
    output_path = Path(output).resolve()
    config_path = Path(config).resolve()
    root = Path(project_root).resolve()
    tree_digest, source_count = source_tree_sha256(root)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    payload = {
        "schema_version": 1,
        "paper": "Beyond Scalar Leaderboards: Adaptive Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas",
        "protocol": config_path.stem,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
            "base_seed": int(base_seed),
        },
        "output": {
            "path": output_path.relative_to(root).as_posix(),
            "sha256": sha256_file(output_path),
            "scheduled_rows": int(scheduled_rows),
        },
        "implementation": {
            "source_tree_sha256": tree_digest,
            "source_file_count": source_count,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def write_dataframe(df, out: str | Path) -> Path:
    out = Path(out)
    ensure_parent(out)
    if out.suffix == ".parquet":
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                df.to_parquet(out, index=False)
        except Exception as exc:  # pragma: no cover - depends on optional engines.
            csv_out = out.with_suffix(".csv")
            df.to_csv(csv_out, index=False)
            print(f"Could not write parquet ({exc}); wrote {csv_out}")
            return csv_out
        df.to_csv(out.with_suffix(".csv"), index=False)
        return out
    else:
        df.to_csv(out, index=False)
        return out
