"""Precision configuration and the format emulation behind it."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from iqp_cond._precision import (
    PRECISIONS,
    Precision,
    accumulate,
    precision,
    quantize,
    resolve_precision,
)

ELEMENT_DTYPES = {
    "bf16": ml_dtypes.bfloat16,
    "fp16": np.float16,
    "fp8_e4m3": ml_dtypes.float8_e4m3fn,
    "fp8_e5m2": ml_dtypes.float8_e5m2,
    "fp6_e2m3": ml_dtypes.float6_e2m3fn,
    "fp6_e3m2": ml_dtypes.float6_e3m2fn,
    "fp4_e2m1": ml_dtypes.float4_e2m1fn,
}


AMAX_SCALED = {"fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1"}


@pytest.mark.parametrize("name", ELEMENT_DTYPES)
def test_element_rounding_matches_ml_dtypes_cast(name):
    dtype = ELEMENT_DTYPES[name]
    info = ml_dtypes.finfo(dtype)
    rng = np.random.default_rng(0)
    # Log-uniform magnitudes from below the smallest subnormal upwards, so
    # normals, subnormals and underflow are covered; unscaled formats also
    # overflow.  Formats below 16 bits are amax-scaled, so their tensors carry
    # the format maximum as amax: the scale is then exactly 1 and the plain
    # cast stays a valid oracle.
    low = np.log2(float(info.smallest_subnormal)) - 2
    high = np.log2(float(info.max)) + (0 if name in AMAX_SCALED else 1)
    magnitudes = 2.0 ** rng.uniform(low, high, size=20_000)
    values = magnitudes * rng.choice([-1.0, 1.0], size=magnitudes.size)
    # Exact ties between neighbouring values exercise round-half-to-even.
    half_step = float(info.smallest_subnormal) / 2
    reach = min(64, int(float(info.max) / half_step))
    grid = np.arange(-reach, reach + 1) * half_step
    values = np.concatenate([values, grid, [float(info.max), -float(info.max)]])

    with np.errstate(over="ignore"):
        expected = values.astype(dtype).astype(np.float64)
    actual = np.asarray(quantize(jnp.asarray(values), name))

    np.testing.assert_array_equal(actual, expected)


def test_amax_scaling_maps_the_largest_entry_onto_the_format_maximum():
    # amax 0.1 gives scale 6 / 0.1 = 60: [1.8, 3.6, 6] rounds in fp4 to
    # [2, 4, 6], which dequantizes to [1/30, 1/15, 0.1].  Unscaled, every entry
    # would round to 0.
    values = jnp.array([0.03, -0.06, 0.1])
    np.testing.assert_allclose(
        np.asarray(quantize(values, "fp4_e2m1")),
        [1 / 30, -1 / 15, 0.1],
        rtol=1e-15,
    )


def test_tf32_keeps_ten_mantissa_bits():
    ulp = 2.0**-10
    tf32_max = (2 - ulp) * 2.0**127
    values = jnp.array(
        [
            1 + ulp,
            1 + ulp / 2,
            1 + 3 * ulp / 2,
            -(2 + 2 * ulp),
            tf32_max,
            (2 - ulp / 4) * 2.0**127,
        ],
    )
    # 1 + ulp/2 is a tie and rounds to the even neighbour 1; 1 + 3ulp/2 is a
    # tie and rounds up to the even neighbour 1 + 2ulp.  TF32 keeps the fp32
    # exponent range: its maximum survives, and a value past it overflows.
    expected = [1 + ulp, 1.0, 1 + 2 * ulp, -(2 + 2 * ulp), tf32_max, np.inf]
    np.testing.assert_array_equal(np.asarray(quantize(values, "tf32")), expected)


def test_mx_blocks_share_a_power_of_two_scale():
    # OCP MX v1.0: each block of 32 along the last axis shares the scale
    # 2^(floor(log2 amax) - 2) for fp4 elements, and elements saturate.
    # Block 1 (amax 0.47): scale 1/16, so 0.3, 0.05, -0.47 become 4.8, 0.8,
    # -7.52 and round to 4, 1, -6.  Block 2 (amax 3): scale 1/2, so 3 and 1
    # become 6 and 2 and are exact.
    values = np.zeros(64)
    values[:3] = [0.3, 0.05, -0.47]
    values[32:34] = [3.0, 1.0]
    expected = np.zeros(64)
    expected[:3] = [0.25, 0.0625, -0.375]
    expected[32:34] = [3.0, 1.0]
    np.testing.assert_array_equal(
        np.asarray(quantize(jnp.asarray(values), "mxfp4")), expected
    )


def test_nvfp4_scales_per_tensor_and_per_block_of_16():
    # Tensor scale 6 * 448 / amax = 896.  Block 1 (amax 3): block scale
    # e4m3(3 / 6 * 896) = 448, so 3 and 1.1 become 6 and 2.2, round to 6 and 2,
    # and decode as 3 and 1.  Block 2 (amax 0.2): block scale e4m3(29.87) = 30,
    # so 0.2 and 0.07 become 5.97 and 2.09, round to 6 and 2, and decode as
    # 6 * 30 / 896 and 2 * 30 / 896.
    values = np.zeros(32)
    values[:2] = [3.0, 1.1]
    values[16:18] = [0.2, 0.07]
    expected = np.zeros(32)
    expected[:2] = [3.0, 1.0]
    expected[16:18] = [180 / 896, 60 / 896]
    np.testing.assert_allclose(
        np.asarray(quantize(jnp.asarray(values), "nvfp4")),
        expected,
        rtol=1e-15,
    )


@pytest.mark.parametrize(
    ("name", "count", "expected"),
    [
        # Element by element, fp16 would stall at 2048 (2048 + 1 is a tie that
        # rounds back to the even 2048).  Blocked partial sums of 32 keep every
        # running sum a multiple of 32, which fp16 represents up to 8192.
        ("fp16", 8192, 8192.0),
        # bf16's spacing reaches 64 at 8192, where 8192 + 32 is a tie that
        # rounds back to the even 8192, so even the blocked sum stalls there.
        ("bf16", 65536, 8192.0),
    ],
)
def test_accumulation_rounds_block_partials_and_the_running_sum(name, count, expected):
    assert float(accumulate(jnp.ones(count), name)) == expected


def test_shorthand_sets_every_knob_from_a_base_and_overrides_single_knobs():
    recipe = precision("bf16", statistics="fp32", variant="split")
    assert recipe == Precision(
        gemm_inputs="bf16",
        gemm_accumulate="bf16",
        phases="bf16",
        statistics="fp32",
        ratios="bf16",
        variant="split",
    )
    assert precision() == Precision()


@pytest.mark.parametrize(
    "build",
    [
        lambda: Precision(phases="fp12"),
        lambda: Precision(statistics="mxfp8"),
        lambda: precision("nvfp4"),
        lambda: Precision(variant="mixed"),
    ],
    ids=[
        "unknown format",
        "block format off the GEMM",
        "block base",
        "unknown variant",
    ],
)
def test_invalid_configurations_are_rejected_when_built(build):
    with pytest.raises(ValueError):
        build()


def test_block_formats_are_accepted_on_the_gemm_inputs():
    assert precision("fp32", gemm_inputs="mxfp4").gemm_inputs == "mxfp4"


@pytest.mark.parametrize(
    ("recipe", "scaled"),
    [
        (precision("fp16"), True),
        (precision("fp32", phases="fp16"), True),
        (precision("fp16", variant="split"), False),
        (precision("bf16"), False),
        (precision(), False),
    ],
)
def test_loss_scaling_follows_fp16_in_the_backward_pass(recipe, scaled):
    assert recipe.loss_scaled is scaled


def test_presets_are_uniform_recipes_named_after_their_format():
    assert set(PRECISIONS) == {"fp64", "fp32", *ELEMENT_DTYPES, "tf32"}
    assert resolve_precision("bf16") == precision("bf16")
    recipe = precision("fp32", gemm_inputs="mxfp4")
    assert resolve_precision(recipe) is recipe


@pytest.mark.parametrize("name", ["gemm_f32", "mxfp4", "fp12"])
def test_names_that_are_not_presets_are_rejected(name):
    with pytest.raises(ValueError):
        resolve_precision(name)
