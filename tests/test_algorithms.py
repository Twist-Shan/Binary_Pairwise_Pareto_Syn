import numpy as np

from vb_ege.algorithms import VBEGEConfig, choose_removal_arm, run_vb_ege
from vb_ege.baselines import (
    run_uniform_focal_borda_fc,
    run_uniform_pairwise_bt_borda_plugin_fc,
    run_uniform_pairwise_bt_mle_fc,
)
from vb_ege.instances import symmetric_hard


def test_vb_ege_symmetric_small():
    theta, _ = symmetric_hard(K=4, d=2, Delta=2.5, seed=0, permute=False)
    cfg = VBEGEConfig(
        delta=0.10,
        sample_const=1.0,
        threshold_const=2.0,
        log_const=4.0,
        max_phases=20,
        seed=0,
    )
    out = run_vb_ege(theta, cfg)
    assert out.stopped
    assert out.recommended == (0,)


def test_tie_break_nonpareto_first_unit():
    active = [0, 1]
    gap_by_arm = {0: 1.0, 1: 1.0}
    empirical_pareto = (0,)
    assert choose_removal_arm(active, gap_by_arm, empirical_pareto) == 1


def test_uniform_focal_borda_fc_symmetric_small():
    theta, _ = symmetric_hard(K=4, d=2, Delta=2.5, seed=0, permute=False)
    out = run_uniform_focal_borda_fc(
        theta,
        {
            "delta": 0.10,
            "sample_const": 1.0,
            "threshold_const": 2.0,
            "max_phases": 12,
        },
        np.random.default_rng(1),
    )
    assert out["stopped"]
    assert out["recommended"] == (0,)


def test_pairwise_fc_baselines_symmetric_small():
    theta, _ = symmetric_hard(K=4, d=2, Delta=2.5, seed=0, permute=False)
    cfg = {
        "delta": 0.10,
        "sample_const": 1.0,
        "threshold_const": 2.0,
        "max_phases": 10,
        "ridge_lambda": 1e-8,
        "fallback_ridge_lambda": 1e-4,
    }
    mle = run_uniform_pairwise_bt_mle_fc(theta, cfg, np.random.default_rng(2))
    plugin = run_uniform_pairwise_bt_borda_plugin_fc(theta, cfg, np.random.default_rng(3))
    assert mle["stopped"]
    assert plugin["stopped"]
    assert mle["recommended"] == (0,)
    assert plugin["recommended"] == (0,)
