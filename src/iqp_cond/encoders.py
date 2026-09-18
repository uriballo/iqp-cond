"""The encoder interface and the product-state angle family.

A conditional IQP model applies one shared diagonal unitary ``U_theta`` to a
condition-dependent input state ``|phi(c)>``.  A registered encoder provides
three capabilities:

``sample``
    Draw bitstrings from ``|phi(c)|^2`` in polynomial time.
``amplitude_ratios``
    Evaluate ``phi(z xor a) / phi(z)`` for sampled ``z`` and Fourier masks ``a``.
``preparation_ops``
    Describe how ``|phi(c)>`` is prepared as single-qubit rotations, so the
    model can be compiled to a circuit and sampled on a simulator or backend.

The first two are the whole classical-tractability requirement for training and
keep the trainer independent of any circuit simulator.  The third is required
at registration time and used only by :mod:`iqp_cond.simulation`.  Encoders are
looked up by name and can be added without touching this package::

    import iqp_cond as ciqp

    ciqp.register_encoder("my_family", my_encoder_factory)
    states = ciqp.encoder("my_family", angles)

The built-in family is :class:`AngleEncoder`, a product state built from any
number of fixed-axis single-qubit rotation layers.  Implementations are batched
over conditions: ``angles`` always carries a leading condition axis, and every
method keeps that axis, because the estimator samples all conditions from a
single random key.  Exact small-system work needs a further capability,
``post_hadamard_amplitudes``, declared separately by :class:`ExactEncoder`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ANGLE",
    "ENCODERS",
    "AngleEncoder",
    "CircuitEncoder",
    "Encoder",
    "ExactEncoder",
    "Operation",
    "angle_encoder",
    "available_encoders",
    "encoder",
    "layered_amplitudes",
    "register_encoder",
    "select_encoder",
]

ANGLE = "angle"

_ROTATION_AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class Operation:
    """One backend-agnostic gate in an encoder's state preparation.

    The simulation bridge maps the name to a PennyLane gate; the whitelist is
    ``RX``, ``RY``, and ``RZ``, each acting on one wire with one angle.
    """

    name: str
    wires: tuple[int, ...]
    parameters: tuple[float, ...] = ()


@runtime_checkable
class Encoder(Protocol):
    """A condition-parameterised encoder evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    The library evaluates one condition at a time for memory; it does that with
    :func:`select_encoder`, which slices every condition-batched leaf of a JAX
    pytree or falls back to an optional ``select`` method.  Families therefore
    either register as pytrees or implement ``select``; nothing else about them
    is required.
    """

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(C, S, n)`` bitstrings distributed as ``|phi(c)|^2``."""

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(C, S, R)`` ratios ``phi(z xor a) / phi(z)``."""


@runtime_checkable
class ExactEncoder(Encoder, Protocol):
    """A training encoder that can also materialise its exact post-Hadamard state.

    Required only for small-system exact checks (the full ``2**n`` state and the
    ratio-identity tests); training and simulation never call it.
    """

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return normalised ``(C, 2**n)`` amplitudes ``<z|H^n|phi(c)>``."""


@runtime_checkable
class CircuitEncoder(Encoder, Protocol):
    """The registered encoder interface: training plus a circuit representation.

    ``preparation_ops`` describes ``|phi(c)>`` as single-qubit rotations,
    without reference to any simulator.  :mod:`iqp_cond.simulation` compiles
    those operations plus the shared diagonal into a PennyLane circuit, so any
    registered family can be sampled on a classical simulator or a backend.
    Registered families are validated against this protocol.
    """

    def preparation_ops(self, condition: int) -> tuple[Operation, ...]:
        """Return the operations preparing ``|phi(condition)>`` before ``H^n``."""


def select_encoder(states: Encoder, condition: int) -> Encoder:
    """Restrict a batched encoder to a single condition.

    Every condition-batched leaf of a pytree is sliced to a one-condition
    batch, which is what keeps the estimator's sample/mask buffer at one
    condition.  An encoder that is not a pytree may implement ``select``
    instead; the library then uses it.
    """
    if not 0 <= condition < states.n_conditions:
        raise IndexError(f"condition {condition} outside the batch")
    own = getattr(states, "select", None)
    if callable(own):
        return own(condition)
    try:
        return jax.tree_util.tree_map(
            lambda value: value[condition : condition + 1],
            states,
        )
    except (TypeError, AttributeError, IndexError) as error:
        raise TypeError(
            "encoder must be a JAX pytree of condition-batched arrays or "
            "implement select()",
        ) from error


_REGISTRY: dict[str, object] = {}
#: Read-only view of the registered families.  Mutate it only through
#: :func:`register_encoder`, so a typo or a plugin cannot silently shadow a
#: built-in.
ENCODERS: Mapping[str, object] = MappingProxyType(_REGISTRY)


def register_encoder(
    family: str,
    factory: object,
    *,
    replace: bool = False,
) -> None:
    """Register an encoder factory under ``family``.

    The factory is called as ``factory(angles, **params)`` with family-specific
    angles.  Registering an existing name raises unless ``replace=True``, so a
    typo or a plugin cannot silently shadow a built-in family.
    """
    if not family or not isinstance(family, str):
        raise ValueError("family must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if family in _REGISTRY and not replace:
        raise ValueError(
            f"encoder family {family!r} is already registered; pass "
            "replace=True to overwrite it",
        )
    _REGISTRY[family] = factory


def available_encoders() -> tuple[str, ...]:
    """Return the registered family names, sorted."""
    return tuple(sorted(_REGISTRY))


def encoder(family: str, angles: jnp.ndarray, **params: object) -> Encoder:
    """Build the encoder of ``family`` for every supplied condition.

    The factory's result is checked against the :class:`Encoder` protocol, so a
    mistyped plugin fails here instead of silently returning something the
    estimator cannot use.
    """
    if family not in _REGISTRY:
        raise ValueError(
            f"unknown encoder family: {family!r}; "
            f"available: {', '.join(available_encoders())}",
        )
    states = _REGISTRY[family](angles, **params)  # type: ignore[operator]
    if not isinstance(states, CircuitEncoder):
        raise TypeError(
            f"encoder family {family!r} returned {type(states).__name__}, which "
            "does not implement the CircuitEncoder protocol: register families "
            "that provide sampling, amplitude ratios, and preparation_ops",
        )
    return states


def _safe_ratio(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Divide, mapping division by zero to zero."""
    return jnp.where(denominator != 0.0, numerator / denominator, 0.0)


def _local_amplitudes(
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the post-Hadamard ``(amp_zero, amp_one)`` of ``RY(a)|0>``.

    Writing ``c = cos(a/2)`` and ``s = sin(a/2)`` the state is
    ``(c + s)/sqrt(2) |0> + (c - s)/sqrt(2) |1>``.  The amplitudes are real, so
    they are returned in float64: the ratio query then runs on real arithmetic,
    which halves the memory traffic of its ``(samples, masks, qubits)``
    intermediate.
    """
    angles = jnp.asarray(angles, dtype=jnp.float64)
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    zero = jnp.ones_like(angles, dtype=jnp.complex128)
    one = jnp.zeros_like(angles, dtype=jnp.complex128)
    pre_zero = cosine * zero - sine * one
    pre_one = sine * zero + cosine * one
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    # XLA divides complex by real differently from real by real (reciprocal
    # multiply), so the components are extracted from the complex expression
    # the regression fixtures pin rather than recomputed in real arithmetic.
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
    valid for three qubits.  The first layer must not contain ``Z``: ``R_Z``
    on ``|0>`` is a global phase, which cancels from every probability and
    ratio, so accepting it would silently ignore its angle.
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
    when every layer is ``Y`` (including the one-layer case, which defers to
    the pinned ``RY`` expression) and complex whenever an ``X`` or ``Z``
    rotation is present.
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
class ProductEncoder:
    """Batched product-state encoder with per-qubit local amplitudes.

    The one-layer ``RY`` base of :class:`AngleEncoder`: subclasses store the
    condition-batched local amplitudes and inherit sampling, ratio queries, and
    the exact state expansion.  Registered as a JAX pytree so
    :func:`select_encoder` can slice it to one condition.
    """

    name: str
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
    ProductEncoder,
    data_fields=["angles", "amp_zero", "amp_one"],
    meta_fields=["name"],
)


@dataclass(frozen=True)
class AngleEncoder(ProductEncoder):
    """Product-state encoder with stacked single-qubit rotation layers.

    ``angles`` is ``(n_conditions, n_layers, n_qubits)`` and ``axes`` the
    static ``(n_layers, n_qubits)`` rotation labels, with layer 0 applied
    first.  A single all-``Y`` layer is the plain ``RY`` product state;
    anything else composes rotations, which makes the local amplitudes complex
    as soon as ``X`` or ``Z`` appears, since only ``Y`` keeps the single-qubit
    matrices real.  Sampling and ratio queries are inherited unchanged.
    """

    axes: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        # Static checks only: JAX rebuilds this dataclass while tracing.
        if self.angles.ndim != 3:
            raise ValueError(
                "angles must have shape (n_conditions, n_layers, n_qubits)",
            )
        normalized = _normalize_axes(
            self.axes,
            int(self.angles.shape[1]),
            int(self.angles.shape[2]),
        )
        object.__setattr__(self, "axes", normalized)

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return int(self.angles.shape[-1])

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


jax.tree_util.register_dataclass(
    AngleEncoder,
    data_fields=["angles", "amp_zero", "amp_one"],
    meta_fields=["name", "axes"],
)


def angle_encoder(
    angles: jnp.ndarray,
    axes: Sequence[Sequence[str]],
) -> AngleEncoder:
    """Build a layered product-state encoder.

    ``angles`` has shape ``(n_conditions, n_layers, n_qubits)``; ``axes`` gives
    one rotation label (``"X"``, ``"Y"``, or ``"Z"``) per layer and qubit, with
    layer 0 applied first.  A layer may be any sequence of labels, so
    ``("Y" * n, ("X", "Z") * (n // 2))`` is valid.

    The first layer must not contain ``Z``: ``R_Z`` on ``|0>`` is a global
    phase, which cancels from probabilities and ratios, so accepting it would
    silently ignore its angle.  Later ``Z`` layers are allowed; all-``Y``
    sequences keep the real-amplitude path.
    """
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
    return AngleEncoder(
        name=ANGLE,
        angles=values,
        amp_zero=jnp.asarray(amp_zero),
        amp_one=jnp.asarray(amp_one),
        axes=normalized,
    )


register_encoder(ANGLE, angle_encoder)
