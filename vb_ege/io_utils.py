"""I/O helpers for runners."""

from __future__ import annotations

import hashlib
import contextlib
import io
import json
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
