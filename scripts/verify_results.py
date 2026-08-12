"""Verify raw hashes and deterministically rebuild synthetic summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from vb_ege.compat import import_pandas_quietly
from vb_ege.io_utils import sha256_file
from vb_ege.metrics import paired_tau_ratios, summarize_runs
from vb_ege.summarize import _read_raw, _write_confidence_slopes, _write_slopes


pd = import_pandas_quietly()


def _frames_equal(expected: Path, actual: Path) -> bool:
    left = pd.read_csv(expected, low_memory=False)
    right = pd.read_csv(actual, low_memory=False)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(float),
                right[column].to_numpy(float),
                equal_nan=True,
                rtol=1e-12,
                atol=1e-9,
            ):
                return False
        elif not left[column].fillna("<NA>").astype(str).equals(
            right[column].fillna("<NA>").astype(str)
        ):
            return False
    return True


def verify(raw: Path, summary: Path, manifest: Path | None) -> dict:
    raw = raw.resolve()
    summary = summary.resolve()
    failures: list[str] = []
    if manifest is not None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if sha256_file(raw) != payload["output"]["sha256"]:
            failures.append("raw SHA-256 does not match manifest")

    frame = _read_raw(str(raw))
    with tempfile.TemporaryDirectory(prefix="vbege-verify-") as directory:
        root = Path(directory)
        regenerated = root / summary.name
        summarize_runs(frame).to_csv(regenerated, index=False)
        paired = regenerated.with_name(regenerated.stem + "_paired.csv")
        paired_tau_ratios(frame).to_csv(paired, index=False)
        if summary.stem.startswith("confidence_scaling_quantile"):
            _write_confidence_slopes(
                pd.read_csv(regenerated),
                regenerated.with_name(regenerated.stem + "_slopes.csv"),
            )
        elif not summary.stem.startswith("fixed_confidence_benchmarks"):
            _write_slopes(
                pd.read_csv(regenerated),
                regenerated.with_name(regenerated.stem + "_slopes.csv"),
            )
        for generated in sorted(root.glob("*.csv")):
            expected = summary.parent / generated.name
            if not expected.exists():
                failures.append(f"missing expected summary: {expected.name}")
            elif not _frames_equal(expected, generated):
                failures.append(f"summary differs: {expected.name}")

    return {
        "raw": str(raw),
        "rows": len(frame),
        "summary": str(summary),
        "failures": failures,
        "ok": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = verify(args.raw, args.summary, args.manifest)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
