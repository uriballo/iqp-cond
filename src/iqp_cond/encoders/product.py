"""Single-qubit product-state encoders.

Each qubit carries an independent one-qubit state built from that condition's
angle, so the encoder is a product state and the amplitude-ratio query
factorizes over qubits.  This is the family used for the translated Gaussian
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .kernels import safe_ratio
from .protocol import as_condition_batch

__all__ = [
    "PRODUCT",
    "ProductEncoder",
    "local_amplitudes",
    "product_encoder",
]

PRODUCT = "product"


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
