"""The analytic cost model of the precision study."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from experiments.precision_study.cost import step_cost, tensor_bytes
from iqp_cond import precision


def xla_flops(function, *shapes):
    arguments = [jnp.zeros(shape, jnp.float32) for shape in shapes]
    cost = jax.jit(function).lower(*arguments).compile().cost_analysis()
    return (cost[0] if isinstance(cost, list) else cost)["flops"]


def test_gemm_counts_match_xla_for_the_contractions_alone():
    n, n_samples, n_masks = 8, 1024, 64
    m = n * (n + 1) // 2
    cost = step_cost(n, n_samples, n_masks, precision("fp32"))
    assert cost["phase_gemm"].flops == xla_flops(
        lambda a, b: a @ b.T, (n_samples, m), (n_masks, m)
    )
    assert cost["parity_gemms"].flops == xla_flops(
        lambda samples, masks, generators: (
            samples @ generators.T,
            masks @ generators.T,
        ),
        (n_samples, n),
        (n_masks, n),
        (m, n),
    )


@pytest.mark.parametrize(
    ("elements", "name", "expected"),
    [
        (1024 * 64, "fp8_e4m3", 64 * 1024),  # a 1024 x 64 phase buffer: 64 KiB
        (1024, "fp64", 8192),
        (1024, "tf32", 4096),  # stored in a 32-bit container
        (1024, "fp4_e2m1", 512),
        (1024, "mxfp4", 512 + 32),  # plus one 8-bit scale per block of 32
        (1024, "nvfp4", 512 + 64 + 4),  # e4m3 scale per 16, one fp32 tensor scale
    ],
)
def test_tensor_bytes_follow_the_format_width(elements, name, expected):
    assert tensor_bytes(elements, name) == expected
