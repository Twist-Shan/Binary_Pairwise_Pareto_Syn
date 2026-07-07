import numpy as np

from vb_ege.core import borda, strict_pareto_set
from vb_ege.instances import boundary_equality_sanity


def test_strict_pareto_equality_boundary():
    theta, _ = boundary_equality_sanity()
    assert strict_pareto_set(theta) == (0, 1, 2)


def test_borda_order_preservation_random():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(12, 4))
    b = borda(theta)
    for r in range(theta.shape[1]):
        for i in range(theta.shape[0]):
            for j in range(theta.shape[0]):
                assert np.sign(b[i, r] - b[j, r]) == np.sign(theta[i, r] - theta[j, r])


def test_pareto_preservation_random():
    rng = np.random.default_rng(1)
    theta = rng.normal(size=(16, 3))
    assert strict_pareto_set(theta) == strict_pareto_set(borda(theta))
