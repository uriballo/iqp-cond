"""Shared kernels for radial (permutation-symmetric) reference states.

A radial reference has ``|phi(z)|^2 = |phi_{|z|}|^2``, so it is fully described
by one amplitude per Hamming weight.  Both tractability operations then collapse
to lookups: sampling draws a weight from ``q(k) = C(n, k) |phi_k|^2`` and picks a
uniform bitstring of that weight, and a ratio query indexes the same table at
the flipped weight.  Each family only has to supply the shell table itself.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "binomial_row",
    "bit_weights",
    "radial_shell_ratios",
    "radial_shell_samples",
    "radial_shell_state",
]


def safe_ratio(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Divide, mapping division by zero to zero."""
    return jnp.where(denominator != 0.0, numerator / denominator, 0.0)


def bit_weights(n_qubits: int) -> np.ndarray:
    """Return the Hamming weight of every little-endian bitstring index."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    indices = np.arange(2**n_qubits, dtype=np.uint32)
    return np.bitwise_count(indices).astype(np.int64)


def binomial_row(n_qubits: int) -> jnp.ndarray:
    """Return ``(C(n, 0), ..., C(n, n))`` as float64."""
    return jnp.asarray(
        [math.comb(n_qubits, k) for k in range(n_qubits + 1)],
        dtype=jnp.float64,
    )


def radial_shell_samples(
    key: jax.Array,
    shell_amplitudes: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    """Sample bitstrings from a radial reference.

    Draws a Hamming weight ``k`` from ``q(k) = C(n, k) |phi_k|^2`` and then a
    uniformly random bitstring of that weight.

    The weight-``k`` string comes from thresholding the row's uniforms at the
    ``k``-th order statistic: the ``k`` smallest uniforms mark a uniformly
    random ``k``-subset, and ties have probability zero for continuous
    uniforms.  One ``sort`` replaces the two ``argsort`` calls (a permutation
    and its inverse) that the same draw used to need.
    """
    n_conditions, n_shell = shell_amplitudes.shape
    n_qubits = n_shell - 1
    weights = binomial_row(n_qubits)
    probabilities = weights[None, :] * jnp.abs(shell_amplitudes) ** 2
    probabilities = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    logits = jnp.log(jnp.maximum(probabilities, 1e-30))
    logits = jnp.broadcast_to(
        logits[:, None, :],
        (n_conditions, n_samples, n_shell),
    )
    weight_key, bit_key = jax.random.split(key)
    drawn = jax.random.categorical(weight_key, logits, axis=-1).astype(jnp.int32)
    flat = drawn.reshape(-1)
    uniforms = jax.random.uniform(bit_key, (flat.size, n_qubits))
    ordered = jnp.sort(uniforms, axis=-1)
    row = jnp.arange(flat.size)
    threshold = ordered[row, jnp.clip(flat - 1, 0, n_qubits - 1)]
    bits = jnp.where(
        flat[:, None] == 0,
        0.0,
        (uniforms <= threshold[:, None]).astype(jnp.float64),
    )
    return bits.reshape(n_conditions, n_samples, n_qubits)


def radial_shell_ratios(
    samples: jnp.ndarray,
    masks: jnp.ndarray,
    shell_amplitudes: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``phi_{|z xor a|} / phi_{|z|}`` for sampled strings and masks."""
    n_conditions, _n_samples, n_qubits = samples.shape
    weight = samples.sum(axis=-1)
    mask_weight = masks.sum(axis=-1)
    overlap = jnp.einsum("csn,on->cso", samples, masks)
    flipped_weight = weight[..., None] + mask_weight[None, None, :] - 2.0 * overlap
    weight_index = weight.astype(jnp.int32)
    flipped_index = jnp.clip(flipped_weight.astype(jnp.int32), 0, n_qubits)
    condition_axis = jnp.arange(n_conditions)
    denominator = shell_amplitudes[condition_axis[:, None], weight_index]
    numerator = shell_amplitudes[condition_axis[:, None, None], flipped_index]
    return safe_ratio(numerator, denominator[..., None])


def radial_shell_state(shell_amplitudes: jnp.ndarray) -> np.ndarray:
    """Expand shell amplitudes to a normalised ``(C, 2**n)`` post-Hadamard state.

    Shell tables used for sampling and ratio queries may carry an arbitrary
    positive per-condition scale, which cancels in both operations; the returned
    state is normalised so that it is a genuine amplitude vector.
    """
    weights = bit_weights(shell_amplitudes.shape[1] - 1)
    state = np.asarray(shell_amplitudes, dtype=np.complex128)[:, weights]
    norm = np.linalg.norm(state, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("reference state has zero norm")
    return state / norm
