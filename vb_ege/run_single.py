"""Run one synthetic experiment replicate and print JSON."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .algorithms import VBEGEConfig, run_vb_ege
from .baselines import (
    run_uniform_focal_borda,
    run_uniform_focal_borda_fc,
    run_uniform_pairwise_bt_borda_plugin,
    run_uniform_pairwise_bt_borda_plugin_fc,
    run_uniform_pairwise_bt_mle,
    run_uniform_pairwise_bt_mle_fc,
    run_vb_ege_capped,
)
from .instances import make_instance
from .io_utils import json_default


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--s", type=int)
    parser.add_argument("--Delta", type=float)
    parser.add_argument("--margin_low", type=float)
    parser.add_argument("--margin_high", type=float)
    parser.add_argument("--K_low", type=int)
    parser.add_argument("--K_high", type=int)
    parser.add_argument("--q_per_p", type=int)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args(argv)

    params = {
        k: v
        for k, v in vars(args).items()
        if k
        in {
            "K",
            "d",
            "s",
            "Delta",
            "margin_low",
            "margin_high",
            "K_low",
            "K_high",
            "q_per_p",
        }
        and v is not None
    }
    theta, meta = make_instance(args.instance, params, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    if args.algorithm == "VB-EGE-practical":
        result = run_vb_ege(theta, VBEGEConfig(delta=args.delta), rng=rng).__dict__
    elif args.algorithm == "VB-EGE-capped":
        if args.budget is None:
            raise ValueError("--budget is required for VB-EGE-capped")
        result = run_vb_ege_capped(theta, args.budget, {"delta": args.delta}, rng)
    elif args.algorithm == "UniformFocalBorda":
        result = run_uniform_focal_borda(theta, args.budget, rng)
    elif args.algorithm == "UniformFocalBorda-FC":
        result = run_uniform_focal_borda_fc(theta, {"delta": args.delta}, rng)
    elif args.algorithm == "UniformPairwiseBT-MLE":
        result = run_uniform_pairwise_bt_mle(theta, args.budget, rng)
    elif args.algorithm in {"UniformPairwiseBT-MLE-FC", "UniformPairwiseBT-MLE-Cert"}:
        result = run_uniform_pairwise_bt_mle_fc(theta, {"delta": args.delta}, rng)
    elif args.algorithm == "UniformPairwiseBT-BordaPlugIn":
        result = run_uniform_pairwise_bt_borda_plugin(theta, args.budget, rng)
    elif args.algorithm == "UniformPairwiseBT-BordaPlugIn-FC":
        result = run_uniform_pairwise_bt_borda_plugin_fc(theta, {"delta": args.delta}, rng)
    else:
        raise ValueError(f"unknown algorithm: {args.algorithm}")
    print(json.dumps({"meta": meta, "result": result}, default=json_default, sort_keys=True))


if __name__ == "__main__":
    main()
