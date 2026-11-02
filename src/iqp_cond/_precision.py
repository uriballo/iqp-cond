"""Named precision policies for the estimator, the loss and training.

A policy fixes three dtypes: ``compute`` for the parities, the phases and the
ratio-phase products; ``contraction`` for the final phase matrix product; and
``accumulate`` for the per-sample statistics, the loss, the parameters and the
optimizer state.  The encoder's sampling and ratio queries keep their own
precision, which is float64 for the built-in encoders, and are cast to
``compute`` where they enter the estimator.
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


PRECISIONS = {
    "fp64": Precision(jnp.float64, jnp.float64, jnp.float64),
    "gemm_f32": Precision(jnp.float64, jnp.float32, jnp.float64),
    "estimator_f32": Precision(jnp.float32, jnp.float32, jnp.float64),
    "fp32": Precision(jnp.float32, jnp.float32, jnp.float32),
}


def resolve_precision(name: str) -> Precision:
    """Return the dtypes of the named policy, or raise for an unknown name."""
    if name not in PRECISIONS:
        raise ValueError(
            f"unknown precision {name!r}; expected one of {sorted(PRECISIONS)}",
        )
    return PRECISIONS[name]
