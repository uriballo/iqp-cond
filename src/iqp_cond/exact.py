"""Exact output distributions of conditional IQP models, without a circuit simulator.

Every IQP gate is diagonal, so with ``psi = H^n |phi(c)>`` the encoder's
post-Hadamard state, the output amplitudes are ``H^n D psi``: the phase
``D(z) = exp(i sum_j theta_j chi_j(z))``, with ``chi_j(z) = (-1)^(g_j . z)``,
followed by one fast Walsh-Hadamard transform.  The cost is ``O(m 2^n)`` for
``m`` generators, whatever the circuit's depth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .encoders.base import ExactEncoder
from .model import ConditionalModel


def _phases(params: jnp.ndarray, generators: np.ndarray, n_qubits: int) -> jnp.ndarray:
    """Return ``sum_j theta_j chi_j(z)`` for every little-endian basis index."""
    index = jnp.arange(2**n_qubits, dtype=jnp.uint32)
    masks = jnp.asarray(generators @ (1 << np.arange(n_qubits)), dtype=jnp.uint32)

    def add(phase, generator):
        mask, theta = generator
        odd = jax.lax.population_count(index & mask) & 1
        return phase + theta * (1.0 - 2.0 * odd), None

    phase, _ = jax.lax.scan(add, jnp.zeros(index.shape), (masks, params))
    return phase


def _walsh_hadamard(state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
    """Apply ``H^n`` to a little-endian state."""
    for qubit in range(n_qubits):
        pairs = state.reshape(-1, 2, 2**qubit)
        state = jnp.stack(
            [pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]], axis=1
        ).reshape(-1)
    return state / 2 ** (n_qubits / 2)


def probabilities(model: ConditionalModel, params: np.ndarray) -> np.ndarray:
    """Return every condition's exact visible distribution, ``(C, 2**n_visible)``.

    Indices are little-endian, as in :meth:`ConditionalModel.probabilities`;
    ancillas are the highest qubits and are summed out.  The encoder must
    provide ``post_hadamard_amplitudes``.
    """
    encoder = model.encoder
    if not isinstance(encoder, ExactEncoder):
        raise TypeError("the encoder must provide post_hadamard_amplitudes")
    n_qubits = model.n_qubits
    phase = _phases(
        jnp.asarray(params, dtype=jnp.float64),
        np.asarray(model.generators, dtype=np.int64),
        n_qubits,
    )
    states = jnp.asarray(encoder.post_hadamard_amplitudes()) * jnp.exp(1j * phase)
    output = jax.vmap(lambda state: _walsh_hadamard(state, n_qubits))(states)
    probabilities = jnp.abs(output) ** 2
    hidden = 2 ** (n_qubits - model.visible)
    return np.asarray(probabilities.reshape(-1, hidden, 2**model.visible).sum(axis=1))
