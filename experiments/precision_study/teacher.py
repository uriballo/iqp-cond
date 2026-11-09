"""Teacher circuits of the precision study and their Pauli-Z moments.

The teacher is an IQP circuit on the sparse Erdos-Renyi graph of Shen et al.
(arXiv:2602.11042): every single-qubit generator, and each pair of qubits with
probability ``c ln n / n``.  Its angles are drawn from ``[-angle, angle]``,
narrower than the uniform ensemble so that low-weight moments stay above the
sampling noise up to 128 qubits.

For the input ``|+>^n``, the moment of mask ``a`` is ``E_z cos(2 sum_{j in J_a}
theta_j chi_j(z))`` over uniform ``z``, where ``J_a`` holds the generators that
anticommute with ``X^a`` (odd overlap with ``a``).  Only the qubits those
generators touch matter, so the expectation runs over ``2^k`` states for a
support of ``k`` qubits: exactly when ``k <= exact_limit``, and by Monte Carlo
otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_CHUNK = 2**20


def teacher(
    n_qubits: int,
    seed: int,
    *,
    c: float = 2.5,
    angle: float = np.pi / 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the teacher's binary generators and angles."""
    rng = np.random.default_rng(seed)
    probability = min(1.0, c * np.log(n_qubits) / n_qubits)
    rows = [np.eye(n_qubits)]
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < probability:
                row = np.zeros(n_qubits)
                row[[i, j]] = 1.0
                rows.append(row[None])
    generators = np.concatenate(rows)
    return generators, rng.uniform(-angle, angle, size=len(generators))


def _mean_cosine(z: jnp.ndarray, local: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    characters = 1.0 - 2.0 * jnp.mod(z @ local.T, 2.0)
    return jnp.mean(jnp.cos(2.0 * characters @ theta))


def teacher_moments(
    generators: np.ndarray,
    params: np.ndarray,
    masks: np.ndarray,
    *,
    exact_limit: int = 20,
    n_samples: int = 10**7,
    key: jax.Array | None = None,
) -> np.ndarray:
    """Return the teacher's moment for each mask, in float64."""
    moments = np.empty(len(masks))
    for index, mask in enumerate(masks):
        odd = (generators @ mask) % 2 == 1
        support = np.flatnonzero(generators[odd].any(axis=0))
        local = jnp.asarray(generators[odd][:, support])
        theta = jnp.asarray(params[odd])
        k = support.size
        if k <= exact_limit:
            z = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
            moments[index] = float(_mean_cosine(jnp.asarray(z, float), local, theta))
            continue
        mask_key = jax.random.fold_in(key, index)
        total = 0.0
        for start in range(0, n_samples, _CHUNK):
            size = min(_CHUNK, n_samples - start)
            z = jax.random.bernoulli(
                jax.random.fold_in(mask_key, start), shape=(size, k)
            )
            total += float(_mean_cosine(z.astype(float), local, theta)) * size
        moments[index] = total / n_samples
    return moments
