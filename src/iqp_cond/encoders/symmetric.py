"""Entangled encoders whose amplitudes depend only on Hamming weights.

Each encoder treats some qubits identically.  For the parity encoders these are
the leaves ``0, ..., n - 2``, and the last qubit is a distinguished root.  The
collective phase encoder treats every qubit identically.  The post-Hadamard
amplitudes ``psi_c(z)`` then depend only on the leaf Hamming weight and the
root bit, so sampling and ratio queries reduce to a table with one entry per
weight shell, even though the states are entangled.

:class:`ParityEncoder`
    ``RX(t)`` on every leaf followed by a CNOT from each leaf to the root.
    It has closed-form amplitudes, so it scales to thousands of qubits.
:class:`TwistedParityEncoder`
    A one-axis twist and a collective turn on the leaves, followed by the
    parity encoder's leaf--root couplings.
:class:`CollectivePhaseEncoder`
    Linear and quadratic phases in the collective spin ``J_z`` applied to the
    uniform superposition.

The shell-table encoders find their reference amplitudes with a Krawtchouk
transform.  That transform is an alternating sum whose absolute rounding error
is near machine precision.  Shells with appreciable probability stay accurate,
and so do ratio queries at sampled strings.  Beyond about 24 qubits, however,
shells with negligible probability lose their relative accuracy, so
:meth:`post_hadamard_amplitudes` is exact only on small registers.
Construction checks normalisation and raises :class:`FloatingPointError` if it
fails.  All three encoders store qubit ``q`` as bit ``q`` of a little-endian
index and on PennyLane wire ``q``, like :class:`~iqp_cond.AngleEncoder`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb, lgamma
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .base import _safe_ratio, _select_condition

__all__ = [
    "CollectivePhaseEncoder",
    "ParityEncoder",
    "TwistedParityEncoder",
]

#: Largest normalisation error accepted for a shell table.
_NORM_TOLERANCE = 1e-10


def _condition_values(name: str, values: object) -> np.ndarray:
    """Return a validated nonempty, finite, real ``(n_conditions,)`` array."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_conditions,) with at least one")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _register_size(n_qubits: object, minimum: int) -> int:
    """Return ``n_qubits`` as an integer of at least ``minimum``."""
    if (
        isinstance(n_qubits, bool)
        or not isinstance(n_qubits, (int, np.integer))
        or n_qubits < minimum
    ):
        raise ValueError(f"n_qubits must be an integer >= {minimum}")
    return int(n_qubits)


def _krawtchouk(n: int) -> np.ndarray:
    """Return ``K[w, k] = sum over |x| = k of (-1)**(x . z)`` for any ``|z| = w``.

    Entries are summed exactly in integers before conversion to float.
    """
    return np.asarray(
        [
            [
                sum(
                    (-1) ** ell * comb(w, ell) * comb(n - w, k - ell)
                    for ell in range(max(0, k - n + w), min(w, k) + 1)
                )
                for k in range(n + 1)
            ]
            for w in range(n + 1)
        ],
        dtype=np.float64,
    )


def _log_binomials(n: int) -> np.ndarray:
    """Return ``log C(n, w)`` for ``w = 0, ..., n``."""
    return np.asarray(
        [lgamma(n + 1) - lgamma(w + 1) - lgamma(n - w + 1) for w in range(n + 1)],
    )


def _collective_spin(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(J_z, J_x)`` on ``n`` spins in the Dicke basis.

    Dicke state ``w`` is the normalised symmetric state of Hamming weight ``w``,
    so ``J_z`` has eigenvalue ``n/2 - w`` on it.
    """
    j = n / 2
    jz = np.diag(j - np.arange(n + 1)).astype(np.complex128)
    m = j - np.arange(n)
    lower = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), -1).astype(np.complex128)
    return jz, (lower + lower.T) / 2


def _check_normalised(shells: np.ndarray, n_leaves: int) -> None:
    """Raise when a ``(C, roots, n_leaves + 1)`` table is not normalised."""
    counts = np.exp(_log_binomials(n_leaves))
    norms = (np.abs(shells) ** 2 * counts).sum(axis=(1, 2))
    if np.max(np.abs(norms - 1.0)) > _NORM_TOLERANCE:
        raise FloatingPointError(
            "reference amplitudes are not normalised; the Krawtchouk transform "
            f"loses precision at {n_leaves + shells.shape[1] - 1} qubits",
        )


class _ConditionBatch:
    """Pytree and selection support for encoders batched over conditions.

    Subclasses list their per-condition arrays in ``_data_fields``, each with a
    leading condition axis, and their static attributes in ``_meta_fields``.
    """

    _data_fields: ClassVar[tuple[str, ...]]
    _meta_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _from_fields(cls, **fields: object) -> _ConditionBatch:
        instance = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(instance, name, value)
        return instance

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], tuple[object, ...]]:
        """Return dynamic arrays and static attributes for JAX transformations."""
        return (
            tuple(getattr(self, name) for name in self._data_fields),
            tuple(getattr(self, name) for name in self._meta_fields),
        )

    @classmethod
    def tree_unflatten(
        cls,
        meta: tuple[object, ...],
        data: tuple[jnp.ndarray, ...],
    ) -> _ConditionBatch:
        """Rebuild a traced encoder without recomputing its amplitudes."""
        return cls._from_fields(
            **dict(zip(cls._data_fields, data, strict=True)),
            **dict(zip(cls._meta_fields, meta, strict=True)),
        )

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""
        return int(getattr(self, self._data_fields[0]).shape[0])

    def _check_condition(self, condition: int) -> None:
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")

    def select(self, condition: int) -> _ConditionBatch:
        """Return this encoder restricted to one condition."""
        self._check_condition(condition)
        return _select_condition(self, condition)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class ParityEncoder(_ConditionBatch):
    """Parity-constrained encoder built from ``RX(t)`` and a CNOT star.

    ``angles`` is ``(n_conditions,)``.  Before the model's Hadamard layer the
    circuit applies ``RX(t)`` to each leaf and a CNOT from each leaf to the
    root, so the post-Hadamard leaves are independent with
    ``P(1) = sin(t/2)**2`` and the root holds their parity.  The amplitude is
    ``cos(t/2)**(n-1-w) (-i sin(t/2))**w`` on even-parity strings with leaf
    weight ``w``, and zero elsewhere.  Every reference string therefore has
    even parity, so the model output is invariant under a global bit flip.  The
    closed form needs no shell table, so this encoder has no register limit.
    """

    angles: jnp.ndarray
    width: int

    _data_fields: ClassVar[tuple[str, ...]] = ("angles",)
    _meta_fields: ClassVar[tuple[str, ...]] = ("width",)

    def __init__(self, angles: jnp.ndarray, *, n_qubits: int) -> None:
        values = _condition_values("angles", angles)
        object.__setattr__(self, "angles", jnp.asarray(values))
        object.__setattr__(self, "width", _register_size(n_qubits, 2))

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return self.width

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(n_conditions, n_samples, n_qubits)`` bitstrings."""
        probability = jnp.sin(self.angles / 2.0) ** 2
        leaves = jax.random.bernoulli(
            key,
            p=probability[:, None, None],
            shape=(self.n_conditions, n_samples, self.width - 1),
        ).astype(jnp.float64)
        root = jnp.mod(jnp.sum(leaves, axis=-1, keepdims=True), 2.0)
        return jnp.concatenate([leaves, root], axis=-1)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_conditions, n_samples, n_masks)`` ratios.

        A mask that flips ``d`` more leaves from 0 to 1 than from 1 to 0 has
        ratio ``(-i tan(t/2))**d``, and odd-weight masks leave the support.
        The base is inverted when ``|tan(t/2)| > 1`` so that powers of large
        registers stay bounded.
        """
        odd = jnp.mod(jnp.sum(masks, axis=-1), 2.0) == 1.0
        leaf_masks = masks[:, :-1]
        delta = jnp.sum(leaf_masks, axis=-1)[None, None, :] - 2.0 * jnp.einsum(
            "csn,mn->csm",
            samples[..., :-1],
            leaf_masks,
        )
        half = self.angles / 2.0
        cosine, sine = jnp.cos(half), jnp.sin(half)
        small = jnp.abs(sine) <= jnp.abs(cosine)
        base = jnp.where(
            small,
            -1j * sine / jnp.where(small, cosine, 1.0),
            1j * cosine / jnp.where(small, 1.0, sine),
        )[:, None, None]
        exponent = jnp.where(small[:, None, None], delta, -delta)
        ratios = jnp.where(
            base == 0.0,
            jnp.where(exponent == 0.0, 1.0 + 0.0j, 0.0j),
            jnp.power(jnp.where(base == 0.0, 1.0, base), exponent),
        )
        return jnp.where(odd[None, None, :], 0.0j, ratios)

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return ``(n_conditions, 2**n_qubits)`` little-endian amplitudes."""
        n = self.width
        index = np.arange(1 << n)
        bits = (index[:, None] >> np.arange(n)) & 1
        weight = bits[:, :-1].sum(axis=1)
        even = bits[:, -1] == weight % 2
        half = np.asarray(self.angles)[:, None] / 2.0
        amplitudes = np.cos(half) ** (n - 1 - weight) * (-1j * np.sin(half)) ** weight
        return jnp.asarray(np.where(even[None, :], amplitudes, 0.0j))

    def preparation(self, condition: int) -> Callable[[], None]:
        """Return a callable queuing ``RX``, the CNOT star and a Hadamard layer.

        The trailing Hadamards make the queued state ``|phi(c)>``, so the
        model's own Hadamard layer returns the ``RX``--CNOT state.
        """
        self._check_condition(condition)
        import pennylane as qml

        angle = float(self.angles[condition])
        root = self.width - 1

        def queue() -> None:
            for leaf in range(root):
                qml.RX(angle, wires=leaf)
                qml.CNOT(wires=[leaf, root])
            for wire in range(self.width):
                qml.Hadamard(wires=wire)

        return queue


class _ShellEncoder(_ConditionBatch):
    """Encoder whose amplitudes are indexed by root bit and leaf weight.

    ``shells`` is ``(n_conditions, roots, n_leaves + 1)``.  With two roots the
    last qubit is the root.  With one root every qubit is a leaf.
    """

    shells: jnp.ndarray

    @property
    def _n_leaves(self) -> int:
        return int(self.shells.shape[2]) - 1

    @property
    def _rooted(self) -> bool:
        return int(self.shells.shape[1]) == 2

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return self._n_leaves + int(self._rooted)

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(n_conditions, n_samples, n_qubits)`` bitstrings.

        A shell is drawn from its total probability, and then a uniformly
        random set of leaves with that weight is set to one.
        """
        key_shell, key_leaves = jax.random.split(key)
        n_leaves = self._n_leaves
        law = jnp.abs(self.shells) ** 2
        logits = jnp.where(
            law > 0.0,
            jnp.log(jnp.where(law > 0.0, law, 1.0)) + _log_binomials(n_leaves),
            -jnp.inf,
        ).reshape(self.n_conditions, 1, -1)
        cell = jax.random.categorical(
            key_shell,
            logits,
            shape=(self.n_conditions, n_samples),
        )
        root, weight = cell // (n_leaves + 1), cell % (n_leaves + 1)
        scores = jax.random.uniform(
            key_leaves,
            (self.n_conditions, n_samples, n_leaves),
        )
        ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
        leaves = (ranks < weight[..., None]).astype(jnp.float64)
        if not self._rooted:
            return leaves
        return jnp.concatenate([leaves, root[..., None].astype(jnp.float64)], -1)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_conditions, n_samples, n_masks)`` shell-table ratios."""
        n_leaves = self._n_leaves
        leaf_samples, leaf_masks = samples[..., :n_leaves], masks[:, :n_leaves]
        weight = jnp.sum(leaf_samples, axis=-1).astype(jnp.int32)
        flipped_weight = (
            weight[..., None]
            + jnp.sum(leaf_masks, axis=-1)[None, None, :]
            - 2.0 * jnp.einsum("csn,mn->csm", leaf_samples, leaf_masks)
        ).astype(jnp.int32)
        if self._rooted:
            root = samples[..., -1].astype(jnp.int32)
            flipped_root = jnp.bitwise_xor(
                root[..., None],
                masks[:, -1].astype(jnp.int32)[None, None, :],
            )
        else:
            root = jnp.zeros_like(weight)
            flipped_root = jnp.zeros_like(flipped_weight)
        rows = jnp.arange(self.n_conditions)
        denominator = self.shells[rows[:, None], root, weight]
        numerator = self.shells[rows[:, None, None], flipped_root, flipped_weight]
        return _safe_ratio(numerator, denominator[..., None])

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return ``(n_conditions, 2**n_qubits)`` little-endian amplitudes."""
        n_leaves = self._n_leaves
        index = np.arange(1 << self.n_qubits)
        bits = (index[:, None] >> np.arange(self.n_qubits)) & 1
        weight = bits[:, :n_leaves].sum(axis=1)
        root = bits[:, -1] if self._rooted else np.zeros_like(weight)
        return self.shells[:, root, weight]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class TwistedParityEncoder(_ShellEncoder):
    """Parity encoder preceded by a one-axis twist and a turn of the leaves.

    ``angles`` and ``twists`` are ``(n_conditions,)``.  With ``m = n - 1``
    leaves and their collective spin ``J``, the input state is::

        P(t) R_x(turn)^{(x) m} T(mu) |+>^n,
        T(mu) = exp[-i (mu / m) sum_{i<j<n} Z_i Z_j],
        P(t) = exp[-i (t / 2) sum_{i<n} Z_i Z_n].

    The twist and the collective turn ``exp(-i turn J_x)`` form the
    one-axis-twisting spin-squeezing protocol of Kitagawa and Ueda, and
    ``P(t)`` is the leaf--root coupling of :class:`ParityEncoder`.  At
    ``mu = 0`` the state is the parity encoder's up to a global phase.
    Every factor commutes with ``X^n``, so the reference has even parity, and
    the squeeze makes the input probabilities depend on the condition.
    """

    angles: jnp.ndarray
    twists: jnp.ndarray
    shells: jnp.ndarray
    turn: float

    _data_fields: ClassVar[tuple[str, ...]] = ("angles", "twists", "shells")
    _meta_fields: ClassVar[tuple[str, ...]] = ("turn",)

    def __init__(
        self,
        angles: jnp.ndarray,
        twists: jnp.ndarray,
        *,
        n_qubits: int,
        turn: float = -np.pi / 4,
    ) -> None:
        star = _condition_values("angles", angles)
        twist = _condition_values("twists", twists)
        if star.shape != twist.shape:
            raise ValueError("angles and twists must have the same shape")
        n = _register_size(n_qubits, 2)
        turn = float(turn)
        if not np.isfinite(turn):
            raise ValueError("turn must be finite")
        shells = self._shells(self._input_table(star, twist, n, turn), n)
        object.__setattr__(self, "angles", jnp.asarray(star))
        object.__setattr__(self, "twists", jnp.asarray(twist))
        object.__setattr__(self, "shells", jnp.asarray(shells))
        object.__setattr__(self, "turn", turn)

    @staticmethod
    def _input_table(
        angles: np.ndarray,
        twists: np.ndarray,
        n: int,
        turn: float,
    ) -> np.ndarray:
        """Return input amplitudes indexed by ``(condition, root bit, leaf weight)``."""
        leaves = n - 1
        weight = np.arange(leaves + 1)
        norm = np.sqrt([comb(leaves, w) for w in weight])
        jz, jx = _collective_spin(leaves)
        # sum_{i<j} Z_i Z_j = 2 J_z^2 - m/2 is diagonal in the Dicke basis.
        pairs = 2.0 * np.diag(jz).real ** 2 - leaves / 2
        eigenvalues, eigenvectors = np.linalg.eigh(jx)
        rotation = (eigenvectors * np.exp(-1j * turn * eigenvalues)) @ (
            eigenvectors.conj().T
        )
        star = np.stack([leaves - 2 * weight, -(leaves - 2 * weight)])
        rows = []
        for angle, twist in zip(angles, twists, strict=True):
            twisted = np.exp(-1j * twist / leaves * pairs) * norm / 2 ** (leaves / 2)
            dicke = rotation @ twisted
            rows.append(
                np.exp(-0.5j * angle * star) * (dicke / norm)[None, :] / np.sqrt(2.0),
            )
        return np.asarray(rows)

    @staticmethod
    def _shells(inputs: np.ndarray, n: int) -> np.ndarray:
        """Apply ``H^n`` to an input table and zero the odd-parity shells."""
        leaves = inputs @ _krawtchouk(n - 1).T / 2 ** ((n - 1) / 2)
        shells = np.stack(
            [leaves[:, 0] + leaves[:, 1], leaves[:, 0] - leaves[:, 1]],
            axis=1,
        ) / np.sqrt(2.0)
        _check_normalised(shells, n - 1)
        # X^n invariance makes the odd-parity amplitudes exactly zero.
        parity = (np.arange(2)[:, None] + np.arange(n)[None, :]) % 2
        return np.where(parity[None] == 1, 0.0, shells)

    def preparation(self, condition: int) -> Callable[[], None]:
        """Return a callable queuing the twist, the turn and the star coupling."""
        self._check_condition(condition)
        import pennylane as qml

        angle = float(self.angles[condition])
        twist = float(self.twists[condition])
        root = self.n_qubits - 1
        leaves = range(root)

        def queue() -> None:
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            for first in leaves:
                for second in range(first + 1, root):
                    qml.IsingZZ(2.0 * twist / root, wires=[first, second])
            for leaf in leaves:
                qml.RX(self.turn, wires=leaf)
            for leaf in leaves:
                qml.IsingZZ(angle, wires=[leaf, root])

        return queue


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CollectivePhaseEncoder(_ShellEncoder):
    """Phases linear and quadratic in the collective spin on ``|+>^n``.

    ``linear`` and ``quadratic`` are ``(n_conditions,)`` coefficients ``a`` and
    ``b`` of the input state ``exp[-i (a J_z + b J_z^2)] |+>^n``, where
    ``J_z = sum_j Z_j / 2``.  The input probabilities are uniform, but the
    phases make the reference magnitudes depend on the condition, so the
    conditioning cannot be absorbed into the model's diagonal.  At
    ``a = b = 0`` the reference is exactly ``|0...0>``.
    """

    linear: jnp.ndarray
    quadratic: jnp.ndarray
    shells: jnp.ndarray

    _data_fields: ClassVar[tuple[str, ...]] = ("linear", "quadratic", "shells")

    def __init__(
        self,
        linear: jnp.ndarray,
        quadratic: jnp.ndarray,
        *,
        n_qubits: int,
    ) -> None:
        first = _condition_values("linear", linear)
        second = _condition_values("quadratic", quadratic)
        if first.shape != second.shape:
            raise ValueError("linear and quadratic must have the same shape")
        n = _register_size(n_qubits, 1)
        spin = n / 2 - np.arange(n + 1)
        inputs = np.exp(
            -1j * (first[:, None] * spin + second[:, None] * spin**2),
        ) / 2 ** (n / 2)
        shells = inputs @ _krawtchouk(n).T / 2 ** (n / 2)
        # Pin the exact support of |+>^n instead of keeping rounding noise.
        uniform = (first == 0.0) & (second == 0.0)
        shells[uniform] = 0.0
        shells[uniform, 0] = 1.0
        shells = shells[:, None, :]
        _check_normalised(shells, n)
        object.__setattr__(self, "linear", jnp.asarray(first))
        object.__setattr__(self, "quadratic", jnp.asarray(second))
        object.__setattr__(self, "shells", jnp.asarray(shells))

    def preparation(self, condition: int) -> Callable[[], None]:
        """Return a callable queuing ``H``, ``RZ`` and all-pairs ``IsingZZ``.

        ``exp(-i b J_z^2)`` is ``IsingZZ(b)`` on every pair times the global
        phase ``exp(-i b n / 4)``, which is queued too so the state is exact.
        """
        self._check_condition(condition)
        import pennylane as qml

        first = float(self.linear[condition])
        second = float(self.quadratic[condition])
        n = self.n_qubits

        def queue() -> None:
            for wire in range(n):
                qml.Hadamard(wires=wire)
                qml.RZ(first, wires=wire)
            for left in range(n):
                for right in range(left + 1, n):
                    qml.IsingZZ(second, wires=[left, right])
            qml.GlobalPhase(second * n / 4)

        return queue
