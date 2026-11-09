"""Reduced precision through the estimator, the loss and the trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from fp64_golden import inputs

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
