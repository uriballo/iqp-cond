"""The phase-dressed matrix-product-state encoder.

:class:`MPSEncoder` prepares a matrix product state ``A_c`` with a sequential
circuit and multiplies it by a diagonal phase over the complete one- and
two-body Z generators, the same bank as :meth:`iqp_cond.Ansatz.complete`.  The
post-Hadamard reference is::

    psi_c(z) = exp(i f_c(z)) A_c(z),    f_c(z) = sum_j lambda_cj (-1)^(g_j . z).

The encoder stays tractable because the circuit bounds the bond dimension of
``A_c``, not because of a permutation symmetry as in :mod:`.symmetric`.  The
dressing has unit modulus, so sampling reads the MPS alone, and its ratio
factor has a closed form in the generator characters.

The circuit slides a window of ``k + 1`` qubits along the register.  Window
``w`` acts on qubits ``w, ..., w + k`` with layers of ``RY`` and ``RZ`` on every
window qubit followed by a CNOT ladder.  Qubit ``w`` is never touched after
window ``w``, so every cut carries bond dimension at most ``2**k``.  Each
window then contributes one site tensor ``A_w[l, z, r] = U_w[(z, r), (l, 0)]``,
and the last ``k`` qubits are read out of the final window's right index.
Shifting any rotation angle keeps this structure, so shifted encoders remain
tractable.  Qubit ``q`` is bit ``q`` of a little-endian index and PennyLane
wire ``q``, like the other encoders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .base import _ConditionBatch, _safe_ratio

__all__ = ["MPSEncoder"]

_CNOT = jnp.asarray(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=jnp.complex128,
).reshape(2, 2, 2, 2)


def _complete_generators(n_qubits: int) -> np.ndarray:
    """Return the one- and two-body masks in :meth:`Ansatz.complete` order."""
    left, right = np.triu_indices(n_qubits, k=1)
    pairs = np.zeros((left.size, n_qubits))
    pairs[np.arange(left.size), left] = 1.0
    pairs[np.arange(left.size), right] = 1.0
    return np.concatenate([np.eye(n_qubits), pairs])


def _apply(
    unitary: jnp.ndarray, gate: jnp.ndarray, axes: tuple[int, ...]
) -> jnp.ndarray:
    """Apply a ``(2,) * 2k`` gate tensor to the output axes of ``unitary``."""
    n = len(axes)
    out = jnp.tensordot(gate, unitary, axes=(tuple(range(n, 2 * n)), axes))
    return jnp.moveaxis(out, tuple(range(n)), axes)


def _window_unitary(angles: jnp.ndarray) -> jnp.ndarray:
    """Return one window's ``(2**(k+1), 2**(k+1))`` unitary.

    ``angles`` is ``(n_layers, k + 1, 2)`` holding the ``RY`` and ``RZ`` angles.
    The window's first qubit is the most significant index bit.
    """
    width = angles.shape[1]
    dimension = 2**width
    unitary = jnp.eye(dimension, dtype=jnp.complex128).reshape((2,) * width + (-1,))
    cosine = jnp.cos(angles[..., 0] / 2.0)
    sine = jnp.sin(angles[..., 0] / 2.0)
    phase = jnp.exp(0.5j * angles[..., 1])
    for layer in range(angles.shape[0]):
        for qubit in range(width):
            c, s, p = cosine[layer, qubit], sine[layer, qubit], phase[layer, qubit]
            # RZ(b) RY(a) = [[c e^{-ib/2}, -s e^{-ib/2}], [s e^{ib/2}, c e^{ib/2}]].
            gate = jnp.stack([jnp.stack([c / p, -s / p]), jnp.stack([s * p, c * p])])
            unitary = _apply(unitary, gate, (qubit,))
        for qubit in range(width - 1):
            unitary = _apply(unitary, _CNOT, (qubit, qubit + 1))
    return unitary.reshape(dimension, dimension)


def _site_tensors(angles: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
    """Return ``(C, n_qubits, chi, 2, chi)`` site tensors ``[left, bit, right]``.

    Both boundary vectors are the first basis vector.  The trailing ``k`` sites
    copy one bit at a time out of the last window's right index and pad the
    unused bond entries with zeros.
    """
    width = angles.shape[3]
    bond = 2 ** (width - 1)
    unitaries = jax.vmap(jax.vmap(_window_unitary))(angles)
    # Column (l, 0) feeds the previous window's qubits in and a fresh |0>.
    columns = unitaries[..., 0::2].reshape(*unitaries.shape[:2], 2, bond, bond)
    windows = jnp.transpose(columns, (0, 1, 4, 2, 3))
    tail = np.zeros((width - 1, bond, 2, bond))
    for site in range(width - 1):
        span = 2 ** (width - 2 - site)
        for bit in range(2):
            for right in range(span):
                tail[site, bit * span + right, bit, right] = 1.0
    tail = jnp.broadcast_to(
        jnp.asarray(tail, dtype=jnp.complex128), (angles.shape[0], *tail.shape)
    )
    return jnp.concatenate([windows, tail], axis=1)


def _boundary(shape: tuple[int, ...]) -> jnp.ndarray:
    """Return the first basis vector along the last axis, broadcast to ``shape``."""
    return jnp.zeros(shape, dtype=jnp.complex128).at[..., 0].set(1.0)


@partial(jax.jit, static_argnums=2)
def _mps_sample(tensors: jnp.ndarray, key: jax.Array, n_samples: int) -> jnp.ndarray:
    """Draw ``(C, n_samples, n)`` bitstrings from ``|A_c(z)|**2`` site by site."""
    n_conditions, n_qubits, bond = tensors.shape[:3]
    sites = jnp.moveaxis(tensors, 1, 0)

    def environment(right: jnp.ndarray, tensor: jnp.ndarray) -> tuple:
        update = jnp.einsum("clzr,crs,ctzs->clt", tensor, right, tensor.conj())
        return update, right

    boundary = _boundary((n_conditions, bond))
    _, beyond = jax.lax.scan(
        environment,
        boundary[:, :, None] * boundary[:, None, :],
        sites,
        reverse=True,
    )
    uniforms = jax.random.uniform(key, (n_qubits, n_conditions, n_samples))

    def site(left: jnp.ndarray, inputs: tuple) -> tuple:
        tensor, right, uniform = inputs
        branches = jnp.einsum("csl,clzr->cszr", left, tensor)
        weights = jnp.einsum(
            "cszr,crt,cszt->csz", branches, right, branches.conj()
        ).real
        bit = uniform * (weights[..., 0] + weights[..., 1]) < weights[..., 1]
        chosen = jnp.where(bit[..., None], branches[..., 1, :], branches[..., 0, :])
        norm = jnp.sqrt(jnp.where(bit, weights[..., 1], weights[..., 0]))
        return chosen / norm[..., None], bit

    left = _boundary((n_conditions, n_samples, bond))
    _, bits = jax.lax.scan(site, left, (sites, beyond, uniforms))
    return jnp.moveaxis(bits, 0, -1).astype(jnp.float64)


@jax.jit
def _mps_ratios(
    tensors: jnp.ndarray,
    dressing: jnp.ndarray,
    samples: jnp.ndarray,
    masks: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``(C, S, R)`` ratios ``psi_c(z xor a) / psi_c(z)``.

    An empty mask is prepended so the denominator ``A_c(z)`` comes out of the
    same contraction as the flipped amplitudes.
    """
    n_qubits, bond = tensors.shape[1:3]
    masks = jnp.asarray(masks, dtype=jnp.float64)
    queries = jnp.concatenate([jnp.zeros((1, n_qubits)), masks])

    def site(left: jnp.ndarray, inputs: tuple) -> tuple:
        tensor, sample_bits, query_bits = inputs
        flipped = sample_bits[:, :, None, None] != query_bits[None, None, :, None]
        branches = jnp.einsum("csrl,clzt->csrzt", left, tensor)
        return jnp.where(flipped, branches[..., 1, :], branches[..., 0, :]), None

    left = _boundary((*samples.shape[:2], queries.shape[0], bond))
    left, _ = jax.lax.scan(
        site,
        left,
        (jnp.moveaxis(tensors, 1, 0), jnp.moveaxis(samples, -1, 0), queries.T),
    )
    amplitudes = left[..., 0]
    ratios = _safe_ratio(amplitudes[..., 1:], amplitudes[..., :1])
    generators = jnp.asarray(_complete_generators(n_qubits))
    characters = 1.0 - 2.0 * jnp.mod(samples @ generators.T, 2.0)
    flips = jnp.mod(masks @ generators.T, 2.0)
    # f(z xor a) - f(z) = -2 sum_j lambda_j chi_j(z) [g_j . a odd].
    delta = -2.0 * jnp.einsum("csg,rg->csr", characters * dressing[:, None, :], flips)
    return ratios * jnp.exp(1j * delta)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class MPSEncoder(_ConditionBatch):
    """Matrix-product-state reference with a diagonal phase dressing.

    ``angles`` is ``(n_conditions, n_qubits - k, n_layers, k + 1, 2)`` for
    ``bond_dimension = 2**k``: window ``w`` holds ``n_layers`` layers of
    ``(RY, RZ)`` angles for qubits ``w, ..., w + k``.  ``bond_dimension = 1``
    is a product state.  ``dressing`` is ``(n_conditions, n_generators)`` over
    the generators of ``Ansatz.complete(n_qubits)``; leaving it ``None`` sets
    every coefficient to zero, the undressed MPS.  A dressing ``lambda`` is
    equivalent to adding ``lambda`` to the model parameters.
    """

    angles: jnp.ndarray
    dressing: jnp.ndarray
    tensors: jnp.ndarray

    _data_fields: ClassVar[tuple[str, ...]] = ("angles", "dressing", "tensors")

    def __init__(
        self,
        angles: jnp.ndarray,
        dressing: jnp.ndarray | None = None,
        *,
        n_qubits: int,
        bond_dimension: int,
    ) -> None:
        if (
            isinstance(n_qubits, bool)
            or not isinstance(n_qubits, (int, np.integer))
            or n_qubits < 1
        ):
            raise ValueError("n_qubits must be a positive integer")
        if (
            isinstance(bond_dimension, bool)
            or not isinstance(bond_dimension, (int, np.integer))
            or bond_dimension < 1
            or bond_dimension & (bond_dimension - 1)
            or 2 * bond_dimension > 2**n_qubits
        ):
            raise ValueError(
                f"bond_dimension must be a power of two at most 2**{n_qubits - 1}",
            )
        width = int(bond_dimension).bit_length()
        if np.iscomplexobj(angles):
            raise ValueError("angles must be real")
        values = np.asarray(angles, dtype=np.float64)
        if (
            values.ndim != 5
            or values.shape[0] < 1
            or values.shape[2] < 1
            or values.shape[1] != n_qubits - width + 1
            or values.shape[3:] != (width, 2)
        ):
            raise ValueError(
                "angles must have shape (n_conditions, n_qubits - k, n_layers, "
                f"k + 1, 2) with k = {width - 1}; got {values.shape}",
            )
        n_generators = n_qubits * (n_qubits + 1) // 2
        if dressing is None:
            dressing = np.zeros((values.shape[0], n_generators))
        if np.iscomplexobj(dressing):
            raise ValueError("dressing must be real")
        phases = np.asarray(dressing, dtype=np.float64)
        if phases.shape != (values.shape[0], n_generators):
            raise ValueError(
                f"dressing must have shape ({values.shape[0]}, {n_generators})",
            )
        if not (np.isfinite(values).all() and np.isfinite(phases).all()):
            raise ValueError("angles and dressing must be finite")
        object.__setattr__(self, "angles", jnp.asarray(values))
        object.__setattr__(self, "dressing", jnp.asarray(phases))
        object.__setattr__(
            self, "tensors", _site_tensors(jnp.asarray(values), n_qubits)
        )

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return int(self.tensors.shape[1])

    @property
    def bond_dimension(self) -> int:
        """Largest bond dimension of the MPS."""
        return int(self.tensors.shape[2])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(n_conditions, n_samples, n_qubits)`` bitstrings.

        Sampling sweeps the sites left to right and draws each bit from its
        conditional given the earlier ones, contracting precomputed right
        environments.  The dressing has unit modulus and is ignored.
        """
        return _mps_sample(self.tensors, key, n_samples)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_conditions, n_samples, n_masks)`` dressed MPS ratios.

        Each flipped amplitude is one ``O(n chi^2)`` contraction, so the
        intermediate is ``(samples, masks, chi)``; set ``sample_chunk`` on the
        loss to bound it.
        """
        return _mps_ratios(self.tensors, self.dressing, samples, masks)

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return ``(n_conditions, 2**n_qubits)`` little-endian amplitudes."""
        tensors = np.asarray(self.tensors)
        n = self.n_qubits
        state = np.zeros(
            (self.n_conditions, 1, self.bond_dimension), dtype=np.complex128
        )
        state[:, 0, 0] = 1.0
        for site in range(n):
            # Stacking the branches makes this site's bit the next index bit.
            state = np.concatenate(
                [state @ tensors[:, site, :, 0, :], state @ tensors[:, site, :, 1, :]],
                axis=1,
            )
        bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
        characters = 1.0 - 2.0 * ((bits @ _complete_generators(n).T) % 2)
        phase = np.exp(1j * np.asarray(self.dressing) @ characters.T)
        return jnp.asarray(state[..., 0] * phase)

    def preparation(self, condition: int) -> Callable[[], None]:
        """Return a callable queuing the window circuit, the dressing and ``H^n``.

        The dressing goes through the simulation module's diagonal layer, so it
        matches the model's own phases gate for gate.  The trailing Hadamards
        make the queued state ``|phi(c)> = H^n D_lambda |A_c>``, so the model's
        Hadamard layer returns ``D_lambda |A_c>``.
        """
        self._check_condition(condition)
        import pennylane as qml

        from ..simulation import _diagonal_layer

        angles = np.asarray(self.angles[condition])
        dressing = np.asarray(self.dressing[condition])
        generators = _complete_generators(self.n_qubits)
        n_windows, n_layers, width = angles.shape[:3]

        def queue() -> None:
            for window in range(n_windows):
                for layer in range(n_layers):
                    for offset in range(width):
                        wire = window + offset
                        qml.RY(float(angles[window, layer, offset, 0]), wires=wire)
                        qml.RZ(float(angles[window, layer, offset, 1]), wires=wire)
                    for offset in range(width - 1):
                        qml.CNOT(wires=[window + offset, window + offset + 1])
            _diagonal_layer(dressing, generators)
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

        return queue
