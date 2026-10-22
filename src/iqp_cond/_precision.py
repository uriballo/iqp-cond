"""Validation and normalization of the estimator precision options."""

from __future__ import annotations

import jax.numpy as jnp


def resolve_estimator_dtype(
    gemm_dtype: jnp.dtype | None,
    estimator_dtype: jnp.dtype | None,
) -> jnp.dtype | None:
    """Validate precision options and normalize the estimator compute dtype."""
    if gemm_dtype is not None and estimator_dtype is not None:
        raise ValueError("gemm_dtype and estimator_dtype are mutually exclusive")
    if estimator_dtype is None:
        return None
    dtype = jnp.dtype(estimator_dtype)
    if dtype == jnp.dtype(jnp.float64):
        return None
    if dtype != jnp.dtype(jnp.float32):
        raise ValueError("estimator_dtype must be float32, float64, or None")
    return jnp.float32
