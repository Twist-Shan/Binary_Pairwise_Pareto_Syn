"""Replace selected experiment rows in a raw CSV with a clean rerun."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from vb_ege.compat import import_pandas_quietly

pd = import_pandas_quietly()


def _read(path: str):
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return pd.DataFrame([json.loads(line) for line in handle if line.strip()])
    return pd.read_csv(source, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", action="append", required=True)
    parser.add_argument("--experiment-id", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base, low_memory=False)
    replacement = pd.concat([_read(path) for path in args.replacement], ignore_index=True)
    selected = set(args.experiment_id)
    replacement = replacement[replacement["experiment_id"].isin(selected)].copy()
    identity_candidates = [
        "replicate_id",
        "algorithm",
        "budget",
        "delta",
        "sample_const",
        "threshold_const",
        "log_const",
    ]
    identity_candidates.extend(
        sorted(col for col in replacement.columns if col.startswith("param_"))
    )
    dedupe_cols = [col for col in identity_candidates if col in replacement.columns]
    if dedupe_cols:
        replacement = replacement.drop_duplicates(dedupe_cols, keep="last")
    present = set(replacement["experiment_id"].astype(str))
    if present != selected:
        raise ValueError(
            f"replacement experiments {sorted(present)} do not match requested {sorted(selected)}"
        )
    merged = pd.concat(
        [base[~base["experiment_id"].isin(selected)], replacement],
        ignore_index=True,
        sort=False,
    )
    merged["run_id"] = range(len(merged))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    merged.to_csv(temp, index=False)
    temp.replace(out)
    print(f"wrote {len(merged)} rows to {out}")


if __name__ == "__main__":
    main()
