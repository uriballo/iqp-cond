"""Stochastic rounding: unbiased in expectation, keyed, and off by default."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond._precision import _ELEMENT_FORMATS, _round, quantize
from iqp_cond.estimator import estimate_pauli_z
from iqp_cond.loss import mmd

KEYS = jax.random.split(jax.random.key(0), 4000)


@pytest.mark.parametrize("name", ["bf16", "fp8_e4m3", "mxfp8", "nvfp4"])
def test_stochastic_rounding_is_unbiased_where_nearest_is_not(name):
    # Below 0.85 of the largest power of two, no MX element saturates at
    # e4m3's 448, where both roundings clip.
    x = jnp.asarray(np.random.default_rng(1).uniform(0.3, 0.85, 32))
    nearest = quantize(x, name)
    mean = jnp.mean(jax.vmap(lambda key: quantize(x, name, key=key))(KEYS), axis=0)
    assert float(jnp.mean(jnp.abs(mean - x))) < 0.2 * float(
        jnp.mean(jnp.abs(nearest - x))
    )


def test_representable_values_are_left_unchanged():
    x = quantize(jnp.linspace(-3.0, 3.0, 25), "bf16")
    for key in KEYS[:20]:
        np.testing.assert_array_equal(quantize(x, "bf16", key=key), x)


def test_a_value_nearest_keeps_in_range_never_rounds_past_the_largest():
    # Scaling to the format's range can leave the amax element just above 448,
    # which nearest rounding brings back to 448; stochastic rounding must too
    # rather than round up to 480, which e4m3 turns into NaN.
    e4m3 = _ELEMENT_FORMATS["fp8_e4m3"]
    x = jnp.full(4096, 460.8)
    np.testing.assert_array_equal(_round(x, e4m3), 448.0)
    for key in KEYS[:20]:
        np.testing.assert_array_equal(_round(x, e4m3, key=key), 448.0)


def test_rounding_defaults_to_nearest():
    assert ciqp.precision(statistics="fp8_e4m3").rounding == "nearest"
    with pytest.raises(ValueError, match="rounding"):
        ciqp.precision(statistics="fp8_e4m3", rounding="up")


MODEL = ciqp.ConditionalModel(
    ciqp.Ansatz.complete(4),
    ciqp.AngleEncoder(np.zeros((1, 1, 4)), axes=("YYYY",)),
)
PARAMS = np.random.default_rng(2).normal(0.0, 0.3, MODEL.generators.shape[0])
TARGETS = np.random.default_rng(3).integers(0, 2, size=(1, 64, 4)).astype(float)
STOCHASTIC = ciqp.precision(statistics="fp8_e4m3", rounding="stochastic")


def loss(precision, key):
    return mmd(
        PARAMS,
        MODEL.generators,
        MODEL.encoder,
        TARGETS,
        key,
        sigmas=(1.0,),
        n_ops=16,
        n_samples=256,
        precision=precision,
    ).loss


def test_stochastic_results_are_reproducible_for_a_key_and_differ_from_nearest():
    key = jax.random.key(5)
    assert loss(STOCHASTIC, key) == loss(STOCHASTIC, key)
    nearest = dataclasses.replace(STOCHASTIC, rounding="nearest")
    values = {float(loss(STOCHASTIC, jax.random.key(k))) for k in range(5)}
    assert float(loss(STOCHASTIC, key)) != float(loss(nearest, key))
    assert len(values) == 5


def test_stochastic_rounding_without_a_key_is_refused():
    samples = MODEL.encoder.sample(jax.random.key(0), 32)
    with pytest.raises(ValueError, match="key"):
        estimate_pauli_z(
            PARAMS,
            MODEL.generators,
            MODEL.encoder,
            np.eye(4)[:2],
            samples=samples,
            precision=STOCHASTIC,
        )
