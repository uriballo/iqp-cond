"""Single-qubit product-state references.

Each qubit carries an independent one-qubit state built from that condition's
angle, so the reference is a product state and the amplitude-ratio query
factorizes over qubits.  The plain ``RY`` family is the one used for the
translated Gaussian benchmark; the reupload variants enrich the same interface
with noncommuting single-qubit layers and remain classically tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .kernels import safe_ratio
from .protocol import as_condition_batch

__all__ = [
    "CALIBRATED_REUPLOAD",
    "PRODUCT",
    "REUPLOAD_L2",
    "REUPLOAD_L3",
    "SYMMETRIC_REUPLOAD",
    "ProductStates",
    "family_local_amplitudes",
    "product_states",
    "reupload_local_amplitudes",
]

PRODUCT = "product"
SYMMETRIC_REUPLOAD = "symmetric_reupload"
CALIBRATED_REUPLOAD = "calibrated_reupload"
REUPLOAD_L2 = "reupload_l2"
REUPLOAD_L3 = "reupload_l3"

REUPLOAD_LAYERS = {REUPLOAD_L2: 2, REUPLOAD_L3: 3}


def _ry(
    zero: jnp.ndarray,
    one: jnp.ndarray,
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    return cosine * zero - sine * one, sine * zero + cosine * one


def _pre_hadamard_amplitudes(
    angles: jnp.ndarray,
    family: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local amplitudes immediately before the first Hadamards."""
    angles = jnp.asarray(angles, dtype=jnp.float64)
    zero = jnp.ones_like(angles, dtype=jnp.complex128)
    one = jnp.zeros_like(angles, dtype=jnp.complex128)
    if family == PRODUCT:
        return _ry(zero, one, angles)
    if family == SYMMETRIC_REUPLOAD:
        zero, one = _ry(zero, one, angles / 2.0)
        phase = (jnp.pi / 2.0) * jnp.sin(4.0 * angles)
        zero = zero * jnp.exp(-0.5j * phase)
        one = one * jnp.exp(0.5j * phase)
        return _ry(zero, one, angles / 2.0)
    if family == CALIBRATED_REUPLOAD:
        zero, one = _ry(zero, one, angles / 2.0)
        phase = (jnp.pi / 4.0) * jnp.sin(2.0 * angles)
        zero = zero * jnp.exp(-0.5j * phase)
        one = one * jnp.exp(0.5j * phase)
        return _ry(zero, one, angles / 2.0)
    raise ValueError(f"unknown product family: {family!r}")


def reupload_local_amplitudes(
    angles: jnp.ndarray,
    layers: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Noncommuting L-layer data reupload ``RY(a) [RZ(pi/2) RY(a)]^{L-1}`` then H.

    Same-axis repeats collapse to one rotation, so the fixed ``RZ(pi/2)`` gates
    between the data rotations make the layers genuinely noncommuting.  The map
    stays linear in the broadcast angle.
    """
    a = jnp.asarray(angles)
    cosine = jnp.cos(a / 2.0)
    sine = jnp.sin(a / 2.0)
    zero = jnp.ones_like(a, dtype=jnp.complex128)
    one = jnp.zeros_like(a, dtype=jnp.complex128)
    rz_zero = jnp.exp(-0.25j * jnp.pi)
    rz_one = jnp.exp(0.25j * jnp.pi)
    zero, one = cosine * zero - sine * one, sine * zero + cosine * one
    for _ in range(layers - 1):
        zero, one = rz_zero * zero, rz_one * one
        zero, one = cosine * zero - sine * one, sine * zero + cosine * one
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (zero + one) / scale, (zero - one) / scale


def family_local_amplitudes(
    angles: jnp.ndarray,
    family: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(amp_zero, amp_one)`` local amplitudes after the first Hadamards."""
    if family in REUPLOAD_LAYERS:
        return reupload_local_amplitudes(angles, REUPLOAD_LAYERS[family])
    zero, one = _pre_hadamard_amplitudes(angles, family)
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (zero + one) / scale, (zero - one) / scale


@dataclass(frozen=True)
class ProductStates:
    """Batched product-state reference with per-qubit local amplitudes."""

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
        """Number of qubits the reference state lives on."""
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

    def select(self, condition: int) -> ProductStates:
        """Return this family restricted to one condition, recomputed.

        Built directly rather than through :func:`product_states`: this runs
        inside the traced loss, where even a slice of a constant array is a
        tracer and the constructor's Python-level validation cannot run.  The
        parent object was validated when it was built.
        """
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        row = self.angles[condition : condition + 1]
        amp_zero, amp_one = family_local_amplitudes(row, self.name)
        return ProductStates(
            name=self.name,
            angles=row,
            amp_zero=jnp.asarray(amp_zero),
            amp_one=jnp.asarray(amp_one),
        )

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


def product_states(angles: jnp.ndarray, family: str = PRODUCT) -> ProductStates:
    """Build the product-state reference for every supplied condition."""
    angles = as_condition_batch(angles)
    amp_zero, amp_one = family_local_amplitudes(angles, family)
    return ProductStates(
        name=family,
        angles=angles,
        amp_zero=jnp.asarray(amp_zero),
        amp_one=jnp.asarray(amp_one),
    )
