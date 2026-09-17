"""Ancilla conventions for conditional IQP models.

The model may use ``r`` extra qubits that are marginalized after measurement.
Qubits are ordered visible first, ancillas last, so a visible mask ``a`` becomes
the model mask ``(a, 0^r)``: the identity on every ancilla.  Nothing else in the
model needs to know about ancillas — the reference supplies one angle per model
qubit and the estimator uses model-width masks.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "ancilla_only_condition_angles",
    "marginalize_ancillas",
    "pad_visible_masks",
]


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


def ancilla_only_condition_angles(
    ancilla_angles: jnp.ndarray | np.ndarray,
    n_data_qubits: int,
) -> jnp.ndarray:
    """Place all condition angles on ancillas and fix data angles to zero.

    ``ancilla_angles`` contains one angle per ancilla and has shape
    ``(n_conditions, n_ancillas)`` (or ``(n_ancillas,)`` for one condition).
    """
    if n_data_qubits < 1:
        raise ValueError("n_data_qubits must be positive")
    angles = jnp.asarray(ancilla_angles, dtype=jnp.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    if angles.ndim != 2 or angles.shape[1] < 1:
        raise ValueError(
            "ancilla_angles must have shape (n_conditions, n_ancillas)",
        )
    data_part = jnp.zeros((angles.shape[0], n_data_qubits), dtype=angles.dtype)
    return jnp.concatenate([data_part, angles], axis=1)


def marginalize_ancillas(
    joint_probabilities: np.ndarray,
    n_data_qubits: int,
    n_ancillas: int,
) -> np.ndarray:
    """Sum ancilla bits out of joint bitstring probabilities.

    Assumes little-endian bit ordering where data qubits occupy bits
    ``0..n_data_qubits-1`` and ancillas the bits above them.
    """
    joint_probabilities = np.asarray(joint_probabilities, dtype=np.float64)
    if n_ancillas == 0:
        return joint_probabilities
    if joint_probabilities.ndim == 1:
        reshaped = joint_probabilities.reshape(2**n_ancillas, 2**n_data_qubits)
        return reshaped.sum(axis=0)
    if joint_probabilities.ndim == 2:
        n_conditions = joint_probabilities.shape[0]
        reshaped = joint_probabilities.reshape(
            n_conditions,
            2**n_ancillas,
            2**n_data_qubits,
        )
        return reshaped.sum(axis=1)
    raise ValueError("joint_probabilities must be 1D or 2D")
