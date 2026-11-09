"""Precision-study runs: chaos-baseline seeds, saving and reloading."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from experiments.precision_study import study
from experiments.precision_study.study import RunConfig, evaluate, load, run, save
from iqp_cond.loss import TargetMoments

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


def test_the_refreshed_arm_trains_on_exact_teacher_samples_up_to_30_qubits(tmp_path):
    config = dataclasses.replace(TINY, teacher_sample_count=256)
    assert isinstance(study.build(config).targets, np.ndarray)
    result = run(config)
    save(tmp_path, config, result)
    _, arrays = load(tmp_path)
    assert np.all(np.isfinite(arrays["fp64_losses"]))


@pytest.mark.parametrize(
    ("changes", "masks"),
    [({"refresh_source": "pool"}, "pool"), ({"mask_refresh": "cache"}, "bank")],
)
def test_moment_arms_train_on_the_pool_or_a_fixed_bank(changes, masks):
    config = dataclasses.replace(TINY, **changes)
    targets = study.build(config).targets
    assert isinstance(targets, TargetMoments)
    # The pool holds pool_draws draws, the fixed bank n_ops draws.
    expected = config.pool_draws if masks == "pool" else config.n_ops
    assert targets.bank.n_ops == expected
    assert np.all(np.isfinite(run(config).train_losses))
