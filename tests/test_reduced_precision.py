"""Reduced precision through the estimator, the loss and the trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from fp64_golden import inputs

import iqp_cond as ciqp
from iqp_cond import precision
from iqp_cond.estimator import estimate_pauli_z
from iqp_cond.loss import mmd


@pytest.fixture(scope="module")
def data() -> dict:
    return inputs()


def estimate(data, params, recipe):
    model = data["model"]
    return np.asarray(
        estimate_pauli_z(
            jnp.asarray(params),
            model.ansatz.generators,
            model.encoder,
            data["masks"],
            samples=data["samples"],
            precision=recipe,
        ).values,
    )


def loss(data, params, recipe):
    model = data["model"]
    return mmd(
        jnp.asarray(params),
        model.ansatz.generators,
        model.encoder,
        data["targets"],
        jax.random.key(3),
        sigmas=(1.0, 2.0),
        n_ops=16,
        n_samples=64,
        precision=recipe,
    ).loss


def test_gemm_inputs_already_in_the_format_change_nothing(data):
    params = data["params"].astype(ml_dtypes.bfloat16).astype(np.float64)
    recipe = precision(gemm_inputs="bf16")
    np.testing.assert_array_equal(
        estimate(data, params, recipe), estimate(data, params, "fp64")
    )
    assert float(loss(data, params, recipe)) == float(loss(data, params, "fp64"))


@pytest.mark.parametrize(
    "knob", ["gemm_inputs", "gemm_accumulate", "phases", "statistics", "ratios"]
)
def test_error_shrinks_as_the_knob_format_keeps_more_mantissa_bits(data, knob):
    reference = estimate(data, data["params"], "fp64")
    errors = [
        np.sqrt(
            np.mean(
                (estimate(data, data["params"], precision(**{knob: name})) - reference)
                ** 2
            )
        )
        for name in ("fp8_e4m3", "bf16", "fp16", "fp32")
    ]
    # 3, 7, 10 and 23 mantissa bits.
    assert errors[0] > errors[1] > errors[2] > errors[3] > 0.0


def test_split_variant_keeps_the_forward_value_of_the_uniform_variant(data):
    uniform = loss(data, data["params"], precision("bf16"))
    split = loss(data, data["params"], precision("bf16", variant="split"))
    assert float(split) == float(uniform)


def test_split_variant_differentiates_through_the_rounding_unchanged(data):
    # With bf16-representable parameters the forward pass is exact, so the
    # split gradient must equal the fp64 gradient, while the uniform variant
    # rounds the cotangent reaching the parameters to bf16.
    params = jnp.asarray(data["params"].astype(ml_dtypes.bfloat16).astype(np.float64))

    def gradient(recipe):
        return np.asarray(jax.grad(lambda p: loss(data, p, recipe))(params))

    reference = gradient("fp64")
    np.testing.assert_array_equal(
        gradient(precision(gemm_inputs="bf16", variant="split")), reference
    )
    assert not np.array_equal(gradient(precision(gemm_inputs="bf16")), reference)


def fp16_training(data, steps):
    return ciqp.train(
        data["model"],
        data["targets"],
        data["targets"],
        seed=5,
        initial_params=data["params"],
        config=ciqp.TrainingConfig(
            steps=steps,
            batch_size=8,
            validation_interval=steps,
            n_ops=16,
            mc_samples=32,
            validation_n_ops=16,
            validation_mc_samples=32,
            precision="fp16",
            initial_loss_scale=2.0**80,
        ),
    )


def test_overflowing_fp16_steps_are_skipped_and_halve_the_loss_scale(data):
    # A scale of 2^80 overflows fp16's largest value, 65504, in the first
    # rounded cotangent, so every step is skipped and the scale halves each time.
    result = fp16_training(data, steps=5)
    np.testing.assert_array_equal(result.skipped_steps, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result.loss_scales, 2.0 ** np.arange(80, 75, -1))
    np.testing.assert_array_equal(result.final_params, data["params"])


def test_fp16_training_resumes_once_the_loss_scale_fits(data):
    result = fp16_training(data, steps=90)
    skipped = result.skipped_steps
    np.testing.assert_array_equal(skipped, np.arange(1, skipped.size + 1))
    assert 0 < skipped.size < 90
    assert not np.array_equal(result.final_params, data["params"])


def test_training_records_the_parameters_at_every_checkpoint(data):
    result = ciqp.train(
        data["model"],
        data["targets"],
        data["targets"],
        seed=5,
        initial_params=data["params"],
        config=ciqp.TrainingConfig(
            steps=6,
            batch_size=8,
            validation_interval=3,
            n_ops=8,
            mc_samples=16,
            validation_n_ops=8,
            validation_mc_samples=16,
        ),
    )
    assert result.checkpoint_params.shape == (3, data["params"].size)
    np.testing.assert_array_equal(result.checkpoint_params[0], data["params"])
    np.testing.assert_array_equal(result.checkpoint_params[-1], result.final_params)
