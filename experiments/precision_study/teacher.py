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

_CHUNK = 2**18  # Monte Carlo samples per mask and pass
_BUDGET = 2**24  # array elements per compiled batch


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


def _ceil(value: int, step: int) -> int:
    return -(-max(value, 1) // step) * step


@jax.jit
def _exact_means(local: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Exact ``E_z cos(2 chi(z) . theta)`` for a batch of padded masks."""
    k = local.shape[-1]
    z = ((jnp.arange(2**k)[:, None] >> jnp.arange(k)) & 1).astype(local.dtype)
    characters = 1.0 - 2.0 * jnp.mod(jnp.einsum("zk,bjk->bzj", z, local), 2.0)
    return jnp.mean(jnp.cos(2.0 * jnp.einsum("bzj,bj->bz", characters, theta)), axis=1)


@jax.jit
def _sampled_sums(keys, local, theta) -> jnp.ndarray:
    """Sums of ``cos(2 chi(z) . theta)`` over one chunk of uniform ``z`` per mask."""

    def one(key, local, theta):
        z = jax.random.bernoulli(key, shape=(_CHUNK, local.shape[-1]))
        characters = 1.0 - 2.0 * jnp.mod(z.astype(local.dtype) @ local.T, 2.0)
        return jnp.sum(jnp.cos(2.0 * characters @ theta))

    return jax.vmap(one)(keys, local, theta)


def teacher_moments(
    generators: np.ndarray,
    params: np.ndarray,
    masks: np.ndarray,
    *,
    exact_limit: int = 20,
    n_samples: int = 10**7,
    key: jax.Array | None = None,
) -> np.ndarray:
    """Return the teacher's moment for each mask, in float64.

    The Monte Carlo path draws ``n_samples`` rounded up to a whole number of
    chunks of ``2^18`` per mask, with an independent key per mask.
    Masks are grouped by their support ``k`` and anticommuting-generator count,
    padded to a few shared sizes (zero-angle generators and unused support bits
    leave the expectation unchanged), and evaluated a batch at a time.
    """
    problems = []
    for mask in masks:
        odd = (generators @ mask) % 2 == 1
        support = np.flatnonzero(generators[odd].any(axis=0))
        problems.append((generators[odd][:, support], params[odd]))
    sampled = [local.shape[1] > exact_limit for local, _ in problems]
    if any(sampled) and key is None:
        raise ValueError("supports beyond exact_limit need a key for Monte Carlo")

    groups: dict[tuple[bool, int, int], list[int]] = {}
    for index, (local, _) in enumerate(problems):
        k = _ceil(local.shape[1], 4)
        if not sampled[index]:
            k = min(k, max(exact_limit, local.shape[1]))
        groups.setdefault((sampled[index], k, _ceil(local.shape[0], 8)), []).append(
            index
        )

    moments = np.empty(len(masks))
    for (is_sampled, k, width), members in groups.items():
        local = np.zeros((len(members), width, k))
        theta = np.zeros((len(members), width))
        for row, index in enumerate(members):
            block, angles = problems[index]
            local[row, : block.shape[0], : block.shape[1]] = block
            theta[row, : angles.size] = angles
        per_mask = (2**k if not is_sampled else _CHUNK) * width
        batch = max(1, _BUDGET // per_mask)
        for start in range(0, len(members), batch):
            rows = slice(start, start + batch)
            indices = members[rows]
            if not is_sampled:
                moments[indices] = np.asarray(
                    _exact_means(jnp.asarray(local[rows]), jnp.asarray(theta[rows]))
                )
                continue
            total = np.zeros(len(indices))
            for chunk in range(0, n_samples, _CHUNK):
                keys = jnp.stack(
                    [
                        jax.random.fold_in(jax.random.fold_in(key, i), chunk)
                        for i in indices
                    ]
                )
                total += np.asarray(
                    _sampled_sums(
                        keys, jnp.asarray(local[rows]), jnp.asarray(theta[rows])
                    )
                )
            moments[indices] = total / (-(-n_samples // _CHUNK) * _CHUNK)
    return moments
