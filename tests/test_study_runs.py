"""Precision-study runs: chaos-baseline seeds, saving and reloading."""

from __future__ import annotations

import dataclasses

import numpy as np

from experiments.precision_study import study
from experiments.precision_study.study import RunConfig, evaluate, load, run, save

TINY = RunConfig(
    n_qubits=6,
    teacher_seed=1,
    init_seed=7,
    seed=2,
    precision={"phases": "bf16"},
    mc_samples=16,
    mask_refresh="step",
    steps=4,
    validation_interval=2,
    n_ops=8,
    pool_draws=64,
    eval_masks=32,
    eval_samples=64,
)


def test_the_sampling_seed_leaves_the_initial_parameters_unchanged():
    other_key = dataclasses.replace(TINY, seed=3)
    np.testing.assert_array_equal(
        study.initial_params(other_key), study.initial_params(TINY)
    )


def test_the_ulp_baseline_moves_each_parameter_by_one_float32_ulp():
    base = study.initial_params(TINY)
    nudged = study.initial_params(dataclasses.replace(TINY, theta0_ulp=True))
    np.testing.assert_array_equal(
        nudged - base, np.spacing(base.astype(np.float32)).astype(np.float64)
    )


def test_a_saved_run_reloads_and_reproduces_its_final_fp64_loss(tmp_path):
    result = run(TINY)
    save(tmp_path, TINY, result)
    study.clear_cache()
    config, arrays = load(tmp_path)
    assert config == TINY
    assert evaluate(config, arrays["final_params"]) == arrays["fp64_losses"][-1]
    assert (tmp_path / "NOTE.md").exists()
