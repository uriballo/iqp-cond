"""Named precision policies for the estimator, the loss and training.

A policy fixes four dtypes: ``compute`` for the parities, the phases and the
ratio-phase products; ``contraction`` for the final phase matrix product;
``accumulate`` for the per-sample statistics, the loss, the parameters and the
optimizer state; and ``ratios`` for the encoder's amplitude-ratio queries.
Below float64, the estimator casts the encoder's floating leaves, the samples
and the masks to ``ratios`` before the query, so the built-in encoders compute
their ratios in that precision.  Reference sampling stays float64 under every
policy, so every policy sees the same draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Precision:
    """The dtypes of one precision policy."""

    compute: jnp.dtype
    contraction: jnp.dtype
    accumulate: jnp.dtype
    ratios: jnp.dtype


PRECISIONS = {
    "fp64": Precision(jnp.float64, jnp.float64, jnp.float64, jnp.float64),
    "gemm_f32": Precision(jnp.float64, jnp.float32, jnp.float64, jnp.float64),
    "estimator_f32": Precision(jnp.float32, jnp.float32, jnp.float64, jnp.float64),
    "fp32": Precision(jnp.float32, jnp.float32, jnp.float32, jnp.float32),
}


def resolve_precision(name: str) -> Precision:
    """Return the dtypes of the named policy, or raise for an unknown name."""
    if name not in PRECISIONS:
        raise ValueError(
            f"unknown precision {name!r}; expected one of {sorted(PRECISIONS)}",
        )
    return PRECISIONS[name]
