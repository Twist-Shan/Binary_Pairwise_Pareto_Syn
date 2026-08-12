from __future__ import annotations

from scripts.audit_exact_argmax_regression import audit


def test_exact_argmax_audit_smoke(tmp_path) -> None:
    config = tmp_path / "audit.yaml"
    config.write_text(
        """
base_seed: 7
algorithms:
  VB-EGE-practical:
    delta: 0.05
    sample_const: 0.02
    threshold_const: 0.1
    log_const: 4.0
    max_phases: 3
experiments:
  - id: audit_case
    generator: symmetric_hard
    params: {K: 4, d: 2, Delta: 1.0}
    n_reps: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = audit([config], {0})

    assert result["checked_vb_ege_runs"] == 1
    assert result["ok"] is True
    assert result["differences"] == []
