from __future__ import annotations

import json

from vb_ege.io_utils import sha256_file, write_run_manifest


def test_run_manifest_hashes_raw_and_config(tmp_path) -> None:
    (tmp_path / "protocol.yaml").write_text("base_seed: 7\n", encoding="utf-8")
    output = tmp_path / "raw.csv"
    output.write_text("tau\n10\n", encoding="utf-8")

    manifest = write_run_manifest(
        output,
        config=tmp_path / "protocol.yaml",
        project_root=tmp_path,
        scheduled_rows=1,
        base_seed=7,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["output"]["sha256"] == sha256_file(output)
    assert payload["output"]["scheduled_rows"] == 1
    assert payload["config"]["base_seed"] == 7
