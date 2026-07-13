from vb_ege.run_sweep import _expanded_experiment_cells, _iter_experiment_instance_specs


def test_paired_delta_instances_reuses_instance_seed_payload():
    exp = {
        "params": {"K": 8, "d": 2},
        "delta_grid": [0.1, 0.05],
        "paired_delta_instances": True,
    }
    cells = _expanded_experiment_cells(exp)

    assert cells[0][2] == cells[1][2]


def test_unpaired_delta_instances_includes_delta_in_seed_payload():
    exp = {
        "params": {"K": 8, "d": 2},
        "delta_grid": [0.1, 0.05],
    }
    cells = _expanded_experiment_cells(exp)

    assert cells[0][2] != cells[1][2]


def test_shared_instance_bank_ignores_section_specific_experiment_id():
    common = {
        "generator": "symmetric_hard",
        "params": {"K": 8, "d": 2, "Delta": 1.0},
        "n_reps": 2,
        "instance_bank_id": "shared-bank",
        "instance_bank_seed": 123,
    }
    config_a = {"algorithms": {"VB-EGE-practical": {}}, "experiments": [{"id": "a", **common}]}
    config_b = {"algorithms": {"VB-EGE-practical": {}}, "experiments": [{"id": "b", **common}]}

    specs_a = list(_iter_experiment_instance_specs(config_a, 1))
    specs_b = list(_iter_experiment_instance_specs(config_b, 999))

    assert [spec[5] for spec in specs_a] == [spec[5] for spec in specs_b]


def test_hierarchical_design_reuses_theta_seed_across_observation_reps():
    config = {
        "algorithms": {"VB-EGE-practical": {}},
        "experiments": [
            {
                "id": "hierarchical",
                "generator": "symmetric_hard",
                "params": {"K": 8, "d": 2, "Delta": 1.0},
                "n_reps": 6,
                "n_instances": 2,
                "observation_reps": 3,
            }
        ],
    }

    specs = list(_iter_experiment_instance_specs(config, 123))

    assert len(specs) == 6
    assert len({spec[5] for spec in specs[:3]}) == 1
    assert specs[2][5] != specs[3][5]
