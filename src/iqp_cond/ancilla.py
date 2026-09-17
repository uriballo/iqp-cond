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

__all__ = ["ancilla_support", "pad_visible_masks"]


def ancilla_support(
    generators: jnp.ndarray | np.ndarray,
    n_visible: int,
) -> np.ndarray:
    """Return the parameter mask whose generators touch an ancilla qubit.

    Under the visible-first ordering this is the set of parameters that have no
    data-based initialization and receive seeded noise instead; pass it to
    :func:`iqp_cond.parameters_from_data`.
    """
    rows = np.asarray(generators)
    if rows.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if not 0 <= n_visible <= rows.shape[1]:
        raise ValueError("n_visible must be between zero and the register width")
    return np.any(rows[:, n_visible:] != 0, axis=1)


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
