"""Single-qubit product-state encoders.

Each qubit carries an independent one-qubit state built from that condition's
angles, so the encoder is a product state and the amplitude-ratio query
factorizes over qubits.  The base :class:`ProductEncoder` is one ``RY``
rotation per qubit (the translated Gaussian benchmark); :class:`AngleEncoder`
generalizes it to a stack of fixed-axis rotations per qubit, still a product
state and therefore still classically tractable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .kernels import safe_ratio
from .protocol import as_condition_batch

__all__ = [
    "ANGLE",
    "PRODUCT",
    "AngleEncoder",
    "ProductEncoder",
    "angle_encoder",
    "layered_amplitudes",
    "local_amplitudes",
    "product_encoder",
]

PRODUCT = "product"
ANGLE = "angle"

_ROTATION_AXES = ("X", "Y", "Z")


def local_amplitudes(
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the post-Hadamard ``(amp_zero, amp_one)`` of ``RY(a)|0>``.

    Writing ``c = cos(a/2)`` and ``s = sin(a/2)`` the state is
    ``(c + s)/sqrt(2) |0> + (c - s)/sqrt(2) |1>``.  The amplitudes are real, so
    they are returned in float64: the ratio query then runs on real arithmetic,
    which halves the memory traffic of its ``(samples, masks, qubits)``
    intermediate.  The float values are bit-for-bit the real parts of the
    complex amplitudes the golden fixtures pin.
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
    # that the golden fixtures pin rather than recomputed in real arithmetic.
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
    when every layer is ``Y`` (including the one-layer base case, which defers
    to :func:`local_amplitudes` so its pinned floats are preserved) and complex
    whenever an ``X`` or ``Z`` rotation is present.
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
        return local_amplitudes(values[:, 0, :])

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

    Registered as a JAX pytree so :func:`iqp_cond.select_encoder` can slice it
    to one condition by slicing its arrays; ``name`` is static metadata.
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
        zero_to_one = safe_ratio(self.amp_one, self.amp_zero)[:, None, None, :]
        one_to_zero = safe_ratio(self.amp_zero, self.amp_one)[:, None, None, :]
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


def product_encoder(angles: jnp.ndarray) -> ProductEncoder:
    """Build the product-state encoder for every supplied condition."""
    angles = as_condition_batch(angles)
    amp_zero, amp_one = local_amplitudes(angles)
    return ProductEncoder(
        name=PRODUCT,
        angles=angles,
        amp_zero=jnp.asarray(amp_zero),
        amp_one=jnp.asarray(amp_one),
    )


@dataclass(frozen=True)
class AngleEncoder(ProductEncoder):
    """Product-state encoder with stacked single-qubit rotation layers.

    A generalization of :class:`ProductEncoder`: ``angles`` is
    ``(n_conditions, n_layers, n_qubits)`` and ``axes`` the static
    ``(n_layers, n_qubits)`` rotation labels, with layer 0 applied first.  The
    one-layer all-``Y`` case is the base family; anything else composes
    rotations, which makes the local amplitudes complex as soon as ``X`` or
    ``Z`` appears, since only ``Y`` keeps the single-qubit matrices real.
    Sampling and ratio queries are inherited unchanged.
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
