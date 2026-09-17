"""Ancilla conventions for conditional IQP models.

The model may use ``r`` extra qubits that are marginalized after measurement.
Qubits are ordered visible first, ancillas last, so a visible mask ``a`` becomes
the model mask ``(a, 0^r)``: the identity on every ancilla.  Nothing else in the
model needs to know about ancillas — the encoder supplies one angle per model
qubit and the estimator uses model-width masks.  The estimator and the loss pad
visible masks themselves, so training a model with ancillas only requires the
encoder to carry the extra columns.

Exact visible-marginal evaluation needs the full joint array and lives with the
research evaluation code, not here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["pad_visible_masks"]


def pad_visible_masks(
    masks: jnp.ndarray | np.ndarray,
    n_ancillas: int,
) -> jnp.ndarray:
    """Extend visible-qubit masks to the model register by zero padding.

    ``masks`` has shape ``(n_masks, n_visible)``; the result has shape
    ``(n_masks, n_visible + n_ancillas)``.
    """
    masks = jnp.asarray(masks)
    if n_ancillas < 0:
        raise ValueError("n_ancillas must be nonnegative")
    if masks.ndim == 1:
        masks = masks[None, :]
    if masks.ndim != 2:
        raise ValueError("masks must have shape (n_masks, n_visible)")
    if n_ancillas == 0:
        return masks
    zeros = jnp.zeros((masks.shape[0], n_ancillas), dtype=masks.dtype)
    return jnp.concatenate([masks, zeros], axis=1)
