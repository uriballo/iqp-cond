"""The product-state angle encoder.

:class:`AngleEncoder` builds a product state from any number of fixed-axis
single-qubit rotation layers.  ``angles`` has a leading condition axis, and
every method keeps that axis, because the estimator samples all conditions from
a single random key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .base import _safe_ratio, _select_condition

__all__ = ["AngleEncoder", "layered_amplitudes"]

_ROTATION_AXES = ("X", "Y", "Z")


def _local_amplitudes(
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the post-Hadamard ``(amp_zero, amp_one)`` of ``RY(a)|0>``.

    Writing ``c = cos(a/2)`` and ``s = sin(a/2)`` the state is
    ``(c + s)/sqrt(2) |0> + (c - s)/sqrt(2) |1>``.  The amplitudes are real, so
    the function returns them in float64.  The ratio query then runs on real
    arithmetic, which halves the memory traffic of its ``(samples, masks,
    qubits)`` intermediate.
    """
    angles = jnp.asarray(angles, dtype=jnp.float64)
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    zero = jnp.ones_like(angles, dtype=jnp.complex128)
    one = jnp.zeros_like(angles, dtype=jnp.complex128)
    pre_zero = cosine * zero - sine * one
    pre_one = sine * zero + cosine * one
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    # XLA divides complex by real differently from real by real, using a
    # reciprocal multiply, so the code takes the components from the complex
    # expression the regression fixtures pin instead of recomputing them in
    # real arithmetic.
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
    ``R_Z`` on ``|0>`` is a global phase that cancels from every probability
    and ratio, and accepting it would silently ignore its angle.
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


@jax.jit
def _product_log_ratios(
    amp_zero: jnp.ndarray,
    amp_one: jnp.ndarray,
    samples: jnp.ndarray,
    masks: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(C, S, R)`` log-magnitudes and arguments of the product ratios.

    A product-state ratio factorizes over the flipped qubits, so its logarithm
    is a contraction of the per-sample local log-ratios with the masks.
    Running that contraction as one matrix product avoids the ``(samples,
    masks, qubits)`` intermediate of the direct product.  The magnitudes, the
    arguments and the vanishing factors contract separately, so a zero local
    factor gives an exact ``-inf`` log-magnitude.  For real amplitudes the
    argument is the count of negative factors, whose parity is the exact sign.
    Compiled once so eager calls dispatch once.
    """
    zero_to_one = _safe_ratio(amp_one, amp_zero)[:, None, :]
    one_to_zero = _safe_ratio(amp_zero, amp_one)[:, None, :]
    local = jnp.where(samples == 0.0, zero_to_one, one_to_zero)
    vanishing = local == 0.0
    log_magnitude = jnp.log(jnp.where(vanishing, 1.0, jnp.abs(local)))
    is_complex = jnp.iscomplexobj(local)
    # Real ratios contract a negative-factor count, whose parity is the
    # exact sign; complex ratios contract their arguments.
    angle = jnp.angle(local) if is_complex else (local < 0.0)
    features = jnp.stack(
        [log_magnitude, angle.astype(jnp.float64), vanishing.astype(jnp.float64)],
    )
    totals = jnp.einsum("kcsn,rn->kcsr", features, masks.astype(jnp.float64))
    return jnp.where(totals[2] > 0.0, -jnp.inf, totals[0]), totals[1]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class AngleEncoder:
    """Product-state encoder with stacked single-qubit rotation layers.

    ``angles`` is ``(n_conditions, n_layers, n_qubits)`` and ``axes`` the
    static ``(n_layers, n_qubits)`` rotation labels, with layer 0 applied
    first.  A single all-``Y`` layer is the plain ``RY`` product state.
    Anything else composes rotations, which makes the local amplitudes complex
    as soon as ``X`` or ``Z`` appears, because only ``Y`` keeps the
    single-qubit matrices real.  The local post-Hadamard amplitudes are
    derived from the angles wherever sampling and ratio queries need them.
    """

    angles: jnp.ndarray
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
        object.__setattr__(self, "angles", values)
        object.__setattr__(self, "axes", normalized)

    @classmethod
    def _from_components(
        cls,
        angles: jnp.ndarray,
        axes: tuple[tuple[str, ...], ...],
    ) -> AngleEncoder:
        instance = object.__new__(cls)
        object.__setattr__(instance, "angles", angles)
        object.__setattr__(instance, "axes", axes)
        return instance

    def tree_flatten(
        self,
    ) -> tuple[tuple[jnp.ndarray], tuple[tuple[str, ...], ...]]:
        """Return the dynamic angles and static axes for JAX transformations."""
        return (self.angles,), self.axes

    @classmethod
    def tree_unflatten(
        cls,
        axes: tuple[tuple[str, ...], ...],
        values: tuple[jnp.ndarray],
    ) -> AngleEncoder:
        """Rebuild a traced encoder without revalidating its angles."""
        return cls._from_components(values[0], axes)

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        """Number of qubits the encoder state lives on."""
        return int(self.angles.shape[-1])

    @property
    def amp_zero(self) -> jnp.ndarray:
        """``(n_conditions, n_qubits)`` post-Hadamard amplitudes of ``|0>``."""
        return layered_amplitudes(self.angles, self.axes)[0]

    @property
    def amp_one(self) -> jnp.ndarray:
        """``(n_conditions, n_qubits)`` post-Hadamard amplitudes of ``|1>``."""
        return layered_amplitudes(self.angles, self.axes)[1]

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(n_conditions, n_samples, n_qubits)`` bitstrings."""
        probabilities_one = jnp.abs(self.amp_one) ** 2
        return jax.random.bernoulli(
            key,
            p=probabilities_one[:, None, :],
            shape=(self.n_conditions, n_samples, self.n_qubits),
        ).astype(jnp.float64)

    @property
    def _complex(self) -> bool:
        """Whether an ``X`` or ``Z`` rotation makes the ratios complex."""
        return any(label != "Y" for layer in self.axes for label in layer)

    def _log_amplitude_ratios(
        self,
        samples: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the log-magnitudes and arguments of :meth:`amplitude_ratios`."""
        amp_zero, amp_one = layered_amplitudes(self.angles, self.axes)
        return _product_log_ratios(amp_zero, amp_one, samples, masks)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_conditions, n_samples, n_masks)`` product ratios.

        See :func:`_product_log_ratios`; a ``-inf`` log-magnitude gives an exact
        zero ratio.
        """
        log_magnitude, argument = self._log_amplitude_ratios(samples, masks)
        if self._complex:
            return jnp.exp(log_magnitude + 1j * argument)
        return jnp.exp(log_magnitude) * (1.0 - 2.0 * jnp.mod(argument, 2.0))

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return ``(n_conditions, 2**n_qubits)`` little-endian amplitudes."""
        amp_zero, amp_one = layered_amplitudes(self.angles, self.axes)
        zero_reversed = np.asarray(amp_zero, dtype=np.complex128)[:, ::-1]
        one_reversed = np.asarray(amp_one, dtype=np.complex128)[:, ::-1]
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

    def select(self, condition: int) -> AngleEncoder:
        """Return this encoder restricted to one condition."""
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        return _select_condition(self, condition)

    def preparation(self, condition: int) -> Callable[[], None]:
        """Return a callable queuing this condition's rotation layers.

        Layer 0 is queued first.  PennyLane is imported here so that importing
        the package does not import it.
        """
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        import pennylane as qml

        rotations = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
        operations = tuple(
            (rotations[label], qubit, float(self.angles[condition, layer, qubit]))
            for layer, labels in enumerate(self.axes)
            for qubit, label in enumerate(labels)
        )

        def queue() -> None:
            for rotation, qubit, angle in operations:
                rotation(angle, wires=qubit)

        return queue
