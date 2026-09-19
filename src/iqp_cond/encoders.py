"""The encoder interface and the product-state angle encoder.

A conditional IQP model applies one shared diagonal unitary ``U_theta`` to a
condition-dependent input state ``|phi(c)>``.  An encoder provides four
capabilities:

``sample``
    Draw bitstrings from ``|psi_c(z)|^2`` in polynomial time, where
    ``psi_c(z) = <z|H^n|phi(c)>`` is the post-Hadamard reference amplitude.
``amplitude_ratios``
    Evaluate ``psi_c(z xor a) / psi_c(z)`` for sampled ``z`` and Fourier masks ``a``.
``preparation_ops``
    Describe how ``|phi(c)>`` is prepared as named gates, so the model can be
    compiled to a circuit and sampled on a simulator or backend.
``select``
    Restrict the condition batch to one condition for memory-bounded loss
    evaluation.

The first two operations make training classically tractable and keep the
trainer independent of any circuit simulator.  The third is used only by
:mod:`iqp_cond.simulation`.  Encoder families are ordinary Python objects;
custom implementations are instantiated directly and need no registration::

    import iqp_cond as ciqp

    states = MyEncoder(angles)
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(states.n_qubits), states)

The built-in family is :class:`AngleEncoder`, a product state built from any
number of fixed-axis single-qubit rotation layers.  Implementations are batched
over conditions.  ``angles`` always has a leading condition axis, and every
method keeps that axis, because the estimator samples all conditions from a
single random key.  Exact small-system work needs a further capability,
``post_hadamard_amplitudes``, declared separately by :class:`ExactEncoder`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AngleEncoder",
    "Encoder",
    "ExactEncoder",
    "Operation",
    "layered_amplitudes",
]

_ROTATION_AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class Operation:
    """One backend-agnostic gate in an encoder's state preparation.

    The simulation bridge maps the name to a PennyLane gate; the whitelist is
    ``RX``, ``RY``, and ``RZ``, each acting on one wire with one angle, and
    ``CNOT``, acting on two wires with no parameters.
    """

    name: str
    wires: tuple[int, ...]
    parameters: tuple[float, ...] = ()


@runtime_checkable
class Encoder(Protocol):
    """A condition-parameterised encoder evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    To save memory, the library evaluates one condition at a time through
    :meth:`select`.  Implementations may also register as JAX pytrees when
    useful, but that is not part of the public extension contract.
    """

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(C, S, n)`` bitstrings distributed as ``|psi_c(z)|^2``."""

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(C, S, R)`` ratios ``psi_c(z xor a) / psi_c(z)``."""

    def preparation_ops(self, condition: int) -> tuple[Operation, ...]:
        """Return operations preparing ``|phi(condition)>`` before ``H^n``."""

    def select(self, condition: int) -> Encoder:
        """Return an encoder containing only ``condition``."""


@runtime_checkable
class ExactEncoder(Encoder, Protocol):
    """A training encoder that can also materialise its exact post-Hadamard state.

    Required only for small-system exact checks (the full ``2**n`` state and the
    ratio-identity tests); training and simulation never call it.
    """

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return normalised ``(C, 2**n)`` amplitudes ``<z|H^n|phi(c)>``."""


def _safe_ratio(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Divide, mapping division by zero to zero."""
    return jnp.where(denominator != 0.0, numerator / denominator, 0.0)


def _local_amplitudes(
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the post-Hadamard ``(amp_zero, amp_one)`` of ``RY(a)|0>``.

    Writing ``c = cos(a/2)`` and ``s = sin(a/2)`` the state is
    ``(c + s)/sqrt(2) |0> + (c - s)/sqrt(2) |1>``.  The amplitudes are real, so
    the function returns them in float64.  The ratio query then runs on real
    arithmetic, which halves the memory traffic of its
    ``(samples, masks, qubits)`` intermediate.
    """
    angles = jnp.asarray(angles, dtype=jnp.float64)
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    zero = jnp.ones_like(angles, dtype=jnp.complex128)
    one = jnp.zeros_like(angles, dtype=jnp.complex128)
    pre_zero = cosine * zero - sine * one
    pre_one = sine * zero + cosine * one
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    # XLA divides complex by real differently from real by real, a reciprocal
    # multiply, so the code extracts the components from the complex expression
    # the regression fixtures pin rather than recomputing them in real
    # arithmetic.
    return (
        jnp.real((pre_zero + pre_one) / scale),
        jnp.real((pre_zero - pre_one) / scale),
    )


def _normalize_axes(
    axes: Sequence[Sequence[str]],
    n_layers: int,
    n_qubits: int,
) -> tuple[tuple[str, ...], ...]:
    """Return ``axes`` as a validated ``(n_layers, n_qubits)`` tuple.

    A layer may be any sequence of labels, so ``("YYY", ("X", "Y", "X"))`` is
    valid for three qubits.  The first layer must not contain ``Z``, because
    ``R_Z`` on ``|0>`` is a global phase, which cancels from every probability
    and ratio, so accepting it would silently ignore its angle.
    """
    if isinstance(axes, str):
        raise TypeError(
            "axes must be a sequence of layers, not a single string; pass "
            'e.g. ("Y" * n_qubits,) for one layer',
        )
    normalized = tuple(tuple(layer) for layer in axes)
    if len(normalized) != n_layers:
        raise ValueError(
            f"axes has {len(normalized)} layers but angles have {n_layers}",
        )
    if n_layers < 1 or n_qubits < 1:
        raise ValueError("angles must have at least one layer and one qubit")
    for index, layer in enumerate(normalized):
        if len(layer) != n_qubits:
            raise ValueError(
                f"axes layer {index} has {len(layer)} labels but the register "
                f"has {n_qubits} qubits",
            )
        for label in layer:
            if label not in _ROTATION_AXES:
                raise ValueError(
                    f"unknown rotation axis {label!r}; expected one of "
                    f"{', '.join(_ROTATION_AXES)}",
                )
    if any(label == "Z" for label in normalized[0]):
        raise ValueError(
            "the first rotation layer must be X or Y: R_Z on |0> is a global "
            "phase and cancels from every probability and ratio",
        )
    return normalized


def layered_amplitudes(
    angles: jnp.ndarray,
    axes: Sequence[Sequence[str]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the post-Hadamard amplitudes of a layered product state.

    ``angles`` is ``(n_conditions, n_layers, n_qubits)`` and ``axes`` the
    matching rotation labels, layer 0 applied first.  The amplitudes are real
    when every layer is ``Y``, including the one-layer case, which uses the
    pinned ``RY`` expression, and complex whenever an ``X`` or ``Z`` rotation
    is present.
    """
    values = jnp.asarray(angles, dtype=jnp.float64)
    if values.ndim != 3:
        raise ValueError(
            "angles must have shape (n_conditions, n_layers, n_qubits)",
        )
    n_layers, n_qubits = int(values.shape[1]), int(values.shape[2])
    normalized = _normalize_axes(axes, n_layers, n_qubits)
    all_y = all(label == "Y" for layer in normalized for label in layer)
    if n_layers == 1 and all_y:
        return _local_amplitudes(values[:, 0, :])

    cosine = jnp.cos(values / 2.0)
    sine = jnp.sin(values / 2.0)
    dtype = jnp.float64 if all_y else jnp.complex128
    zero_state = jnp.ones(values.shape[:1] + values.shape[2:], dtype=dtype)
    one_state = jnp.zeros_like(zero_state)
    for layer, labels in enumerate(normalized):
        layer_cosine = cosine[:, layer, :]
        layer_sine = sine[:, layer, :]
        if all_y:
            # R_Y(a) = [[c, -s], [s, c]].
            new_zero = layer_cosine * zero_state - layer_sine * one_state
            new_one = layer_sine * zero_state + layer_cosine * one_state
        else:
            z_mask = jnp.asarray([label == "Z" for label in labels])[None, :]
            x_mask = jnp.asarray([label == "X" for label in labels])[None, :]
            # R_Z(a) = diag(e^{-ia/2}, e^{+ia/2}), R_X(a) = [[c, -is], [-is, c]].
            phase = jnp.exp(0.5j * values[:, layer, :])
            rotation_00 = jnp.where(z_mask, jnp.conj(phase), layer_cosine)
            rotation_01 = jnp.where(
                z_mask,
                jnp.asarray(0.0, dtype=jnp.complex128),
                jnp.where(x_mask, -1j * layer_sine, -layer_sine),
            )
            rotation_10 = jnp.where(
                z_mask,
                jnp.asarray(0.0, dtype=jnp.complex128),
                jnp.where(x_mask, -1j * layer_sine, layer_sine),
            )
            rotation_11 = jnp.where(z_mask, phase, layer_cosine)
            new_zero = rotation_00 * zero_state + rotation_01 * one_state
            new_one = rotation_10 * zero_state + rotation_11 * one_state
        zero_state, one_state = new_zero, new_one
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (zero_state + one_state) / scale, (zero_state - one_state) / scale


@dataclass(frozen=True)
class _ProductEncoder:
    """Batched product-state encoder with per-qubit local amplitudes.

    Internal implementation shared by the public angle encoder methods.
    """

    angles: jnp.ndarray
    amp_zero: jnp.ndarray
    amp_one: jnp.ndarray

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return int(self.angles.shape[1])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(n_conditions, n_samples, n_qubits)`` bitstrings."""
        probabilities_one = jnp.abs(self.amp_one) ** 2
        return jax.random.bernoulli(
            key,
            p=probabilities_one[:, None, :],
            shape=(self.n_conditions, n_samples, self.n_qubits),
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_conditions, n_samples, n_masks)`` product ratios."""
        zero_to_one = _safe_ratio(self.amp_one, self.amp_zero)[:, None, None, :]
        one_to_zero = _safe_ratio(self.amp_zero, self.amp_one)[:, None, None, :]
        local_flip_ratios = jnp.where(
            samples[:, :, None, :] == 0.0,
            zero_to_one,
            one_to_zero,
        )
        selected = jnp.where(masks[None, None, :, :] == 1.0, local_flip_ratios, 1.0)
        return jnp.prod(selected, axis=-1)

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return ``(n_conditions, 2**n_qubits)`` little-endian amplitudes."""
        zero_reversed = np.asarray(self.amp_zero, dtype=np.complex128)[:, ::-1]
        one_reversed = np.asarray(self.amp_one, dtype=np.complex128)[:, ::-1]
        n_qubits = self.n_qubits
        out = np.empty((self.n_conditions, 2**n_qubits), dtype=np.complex128)
        for condition in range(self.n_conditions):
            state = np.array(
                [zero_reversed[condition, 0], one_reversed[condition, 0]],
                dtype=np.complex128,
            )
            for qubit in range(1, n_qubits):
                state = np.outer(
                    state,
                    [zero_reversed[condition, qubit], one_reversed[condition, qubit]],
                ).ravel()
            out[condition] = state
        return out


jax.tree_util.register_dataclass(
    _ProductEncoder,
    data_fields=["angles", "amp_zero", "amp_one"],
    meta_fields=[],
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class AngleEncoder(_ProductEncoder):
    """Product-state encoder with stacked single-qubit rotation layers.

    ``angles`` is ``(n_conditions, n_layers, n_qubits)`` and ``axes`` the
    static ``(n_layers, n_qubits)`` rotation labels, with layer 0 applied
    first.  A single all-``Y`` layer is the plain ``RY`` product state;
    anything else composes rotations, which makes the local amplitudes complex
    as soon as ``X`` or ``Z`` appears, since only ``Y`` keeps the single-qubit
    matrices real.  Construction validates the axes and caches the local
    post-Hadamard amplitudes used by sampling and ratio queries.
    """

    axes: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        angles: jnp.ndarray,
        *,
        axes: Sequence[Sequence[str]],
    ) -> None:
        if np.iscomplexobj(angles):
            raise ValueError("angles must be real")
        values = jnp.asarray(angles, dtype=jnp.float64)
        if values.ndim != 3:
            raise ValueError(
                "angles must have shape (n_conditions, n_layers, n_qubits)",
            )
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("angles must be finite")
        normalized = _normalize_axes(
            axes,
            int(values.shape[1]),
            int(values.shape[2]),
        )
        amp_zero, amp_one = layered_amplitudes(values, normalized)
        object.__setattr__(self, "angles", values)
        object.__setattr__(self, "amp_zero", jnp.asarray(amp_zero))
        object.__setattr__(self, "amp_one", jnp.asarray(amp_one))
        object.__setattr__(self, "axes", normalized)

    @classmethod
    def _from_components(
        cls,
        angles: jnp.ndarray,
        amp_zero: jnp.ndarray,
        amp_one: jnp.ndarray,
        axes: tuple[tuple[str, ...], ...],
    ) -> AngleEncoder:
        instance = object.__new__(cls)
        object.__setattr__(instance, "angles", angles)
        object.__setattr__(instance, "amp_zero", amp_zero)
        object.__setattr__(instance, "amp_one", amp_one)
        object.__setattr__(instance, "axes", axes)
        return instance

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[tuple[str, ...], ...],
    ]:
        """Return dynamic arrays and static axes for JAX transformations."""
        return (self.angles, self.amp_zero, self.amp_one), self.axes

    @classmethod
    def tree_unflatten(
        cls,
        axes: tuple[tuple[str, ...], ...],
        values: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> AngleEncoder:
        """Rebuild a traced encoder without recomputing its amplitudes."""
        angles, amp_zero, amp_one = values
        return cls._from_components(angles, amp_zero, amp_one, axes)

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return int(self.angles.shape[-1])

    def select(self, condition: int) -> AngleEncoder:
        """Return this encoder restricted to one condition."""
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        selection = slice(condition, condition + 1)
        return self._from_components(
            self.angles[selection],
            self.amp_zero[selection],
            self.amp_one[selection],
            self.axes,
        )

    def preparation_ops(self, condition: int) -> tuple[Operation, ...]:
        """Return this condition's rotation layers, layer 0 first."""
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        return tuple(
            Operation(
                name=f"R{label}",
                wires=(qubit,),
                parameters=(float(self.angles[condition, layer, qubit]),),
            )
            for layer, labels in enumerate(self.axes)
            for qubit, label in enumerate(labels)
        )
