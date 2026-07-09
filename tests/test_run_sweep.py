from vb_ege.run_sweep import _expanded_experiment_cells


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
