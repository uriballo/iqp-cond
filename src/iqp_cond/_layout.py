"""Visible-first register layout shared by the model, estimator, and loss.

The visible qubits come first and the ancillas last.  These helpers live apart
from :mod:`iqp_cond.model` so the estimator and the loss can use them without
importing the model, which itself builds on the estimator and the loss.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def ancilla_support(
    generators: jnp.ndarray | np.ndarray,
    n_visible: int,
) -> np.ndarray:
    """Return the parameter mask whose generators act on an ancilla qubit.

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
