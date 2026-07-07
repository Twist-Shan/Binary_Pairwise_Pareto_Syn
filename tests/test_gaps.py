import numpy as np

from vb_ege.core import borda
from vb_ege.gaps import compute_gaps
from vb_ege.instances import boundary_equality_sanity, symmetric_hard


def test_symmetric_gaps_positive():
    theta, _ = symmetric_hard(K=5, d=2, Delta=1.0, seed=0, permute=False)
    out = compute_gaps(borda(theta))
    assert out["pareto_set"] == (0,)
    assert np.all(out["delta"] > 0)
    assert np.isfinite(out["H"])


def test_boundary_gap_zero_allowed():
    theta, _ = boundary_equality_sanity()
    out = compute_gaps(theta)
    assert np.any(out["delta"] <= 0.0)
    assert out["H"] == np.inf
