"""Native arithmetic against the emulation it replaces, on the CPU."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fp64_golden import inputs

from iqp_cond import precision
from iqp_cond._precision import _DTYPES, _SCALED, _cast, _scaled_operand, quantize
from iqp_cond.loss import mmd

RECIPE = dict(
    gemm_inputs="bf16",
    gemm_accumulate="fp32",
    phases="bf16",
    cosines="bf16",
    ratios="bf16",
    statistics="fp32",
    phase_gradient="bf16",
    backward_accumulate="fp32",
    variant="split",
    accumulation="hardware",
)


def awkward_values() -> jnp.ndarray:
    """Magnitudes from 1e-30 to past every format's range, with exact ties.

    Below about 1e-36 the emulation's quantum for bf16 is a float32 subnormal,
    which XLA flushes to zero on the CPU.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50_000) * np.logspace(-30, 38, 50_000)
    ties = np.array([1 + 2.0**-8, 1 + 3 * 2.0**-8, 65520.0, 65519.0, 0.0, -0.0])
    return jnp.asarray(np.concatenate([x, ties, -ties]), jnp.float32)


@pytest.mark.parametrize("scaling", ["amax", "pow2"])
@pytest.mark.parametrize("name", _DTYPES)
def test_hardware_casts_give_the_emulated_values_bit_for_bit(name, scaling):
    x = awkward_values()
    native = np.asarray(jax.jit(lambda v: _cast(v, name, scaling))(x))
    emulated = np.asarray(jax.jit(lambda v: quantize(v, name, scaling=scaling))(x))
    np.testing.assert_array_equal(native.view(np.uint32), emulated.view(np.uint32))


@pytest.mark.parametrize("name", _SCALED)
def test_block_elements_and_scales_give_the_emulated_values(name):
    # A width that is no multiple of the block size exercises the padding.
    rng = np.random.default_rng(1)
    x = jnp.asarray(
        rng.standard_normal((4, 1205)) * np.exp(3 * rng.standard_normal((4, 1205))),
        jnp.float32,
    )
    elements, scales, tensor_scale = _scaled_operand(x, name)
    size = _SCALED[name][2]
    blocks = elements.astype(jnp.float32).reshape(4, -1, size)
    values = (blocks * scales.astype(jnp.float32)[..., None]).reshape(4, -1)
    values = values[:, : x.shape[1]] / tensor_scale
    np.testing.assert_array_equal(np.asarray(values), np.asarray(quantize(x, name)))


@pytest.fixture(scope="module")
def data() -> dict:
    return inputs()


def value_and_gradient(data, recipe):
    model = data["model"]

    def loss(params):
        return mmd(
            params,
            model.ansatz.generators,
            model.encoder,
            data["targets"],
            jax.random.key(3),
            sigmas=(1.0, 2.0),
            n_ops=16,
            n_samples=64,
            precision=recipe,
        ).loss

    return jax.value_and_grad(loss)(jnp.asarray(data["params"]))


SETTINGS = {
    "recipe": RECIPE,
    "fp16 recipe": {k: "fp16" if v == "bf16" else v for k, v in RECIPE.items()},
    "fp8 forward, E5M2 backward": RECIPE
    | dict(
        gemm_inputs="fp8_e4m3",
        phases="fp8_e4m3",
        cosines="fp8_e4m3",
        ratios="fp8_e4m3",
        phase_gradient="fp8_e5m2",
    ),
    "uniform bf16, rounded sums": dict(
        gemm_inputs="bf16",
        gemm_accumulate="bf16",
        phases="bf16",
        statistics="bf16",
        ratios="bf16",
        accumulation="hardware",
    ),
    "pow2-scaled fp8 inputs": RECIPE | dict(gemm_inputs="fp8_e4m3", scaling="pow2"),
    "mxfp8 inputs": RECIPE | dict(gemm_inputs="mxfp8"),
    "nvfp4 inputs": RECIPE | dict(gemm_inputs="nvfp4"),
    "uniform with backward fields": dict(
        gemm_inputs="fp8_e4m3",
        gemm_accumulate="fp32",
        phase_gradient="fp8_e5m2",
        backward_accumulate="bf16",
        accumulation="hardware",
    ),
    "uniform mxfp8 inputs": dict(gemm_inputs="mxfp8", gemm_accumulate="fp32"),
    "fp64 accumulator": RECIPE | dict(gemm_accumulate="fp64"),
    "bf16 backward sum": RECIPE | dict(backward_accumulate="bf16"),
}


@pytest.mark.parametrize("setting", SETTINGS)
def test_a_native_step_matches_the_emulated_step(data, setting):
    knobs = SETTINGS[setting]
    emulated_loss, emulated_gradient = value_and_gradient(data, precision(**knobs))
    native_loss, native_gradient = value_and_gradient(
        data, precision(**knobs, arithmetic="native")
    )
    exact_loss, exact_gradient = value_and_gradient(data, "fp64")
    # The rounded values agree; only the order of the fp32 sums differs, far
    # below the rounding error of the formats themselves.
    rounding = float(jnp.linalg.norm(emulated_gradient - exact_gradient))
    difference = float(jnp.linalg.norm(native_gradient - emulated_gradient))
    assert difference < 1e-3 * rounding
    assert abs(float(native_loss - emulated_loss)) < 1e-3 * abs(
        float(emulated_loss - exact_loss)
    ) + 1e-6 * abs(float(exact_loss))


@pytest.mark.parametrize(
    ("knobs", "fix"),
    [
        (dict(gemm_inputs="fp8_e4m3", rounding="stochastic"), "rounding='nearest'"),
        (dict(gemm_accumulate="fp16"), "accumulation='hardware'"),
        (
            dict(backward_accumulate="bf16", accumulation="blocked"),
            "accumulation='hardware'",
        ),
    ],
)
def test_native_arithmetic_refuses_what_xla_cannot_do_and_names_the_fix(knobs, fix):
    with pytest.raises(ValueError, match=f"{fix}.*arithmetic='emulated'"):
        precision(**knobs, arithmetic="native")


@pytest.mark.parametrize(
    "name", ["fp4_e2m1", "fp6_e2m3", "mxfp6_e2m3", "mxfp4", "tf32"]
)
def test_a_format_without_a_native_product_warns_and_stays_exact(data, name):
    knobs = RECIPE | dict(gemm_inputs=name)
    with pytest.warns(UserWarning, match=f"gemm_inputs='{name}'.*exact but not faster"):
        native = value_and_gradient(data, precision(**knobs, arithmetic="native"))
    emulated = value_and_gradient(data, precision(**knobs))
    np.testing.assert_allclose(native[1], emulated[1], rtol=1e-5, atol=1e-7)


def test_formats_with_a_native_product_do_not_warn_on_the_cpu(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value_and_gradient(data, precision(**RECIPE, arithmetic="native"))


@pytest.mark.parametrize("name", ["fp4_e2m1", "fp8_e4m3", "bf16"])
def test_a_nan_stays_nan_through_a_hardware_cast(name):
    x = jnp.array([1.0, jnp.nan, -2.0], jnp.float32)
    assert jnp.isnan(_cast(x, name)[1])


def test_a_nan_flip_makes_its_block_scaled_phases_nan():
    from iqp_cond._precision import _scaled_product

    flips = jnp.ones((3, 40), jnp.float32).at[1, 5].set(jnp.nan)
    phases = _scaled_product(jnp.ones((2, 40), jnp.float32), flips, "nvfp4")
    np.testing.assert_array_equal(np.isnan(phases), [[False, True, False]] * 2)
