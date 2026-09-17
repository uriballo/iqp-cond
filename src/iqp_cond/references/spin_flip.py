"""Spin-flip-symmetric radial references.

The two-branch construction

    ``|phi(c)> = |p>^n + X^n |p>^n``,   ``|p> = RY(a(c))|0>``,

imposes the global spin-flip symmetry of the zero-field Curie--Weiss target
without prescribing its magnetization profile.  After the first Hadamards every
computational-basis amplitude depends only on the Hamming weight, so the state
is radial: sampling reduces to drawing a weight from
``q(k) = C(n, k) |phi_k|^2``, and the amplitude ratio is a table lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .kernels import (
    radial_shell_ratios,
    radial_shell_samples,
    radial_shell_state,
)
from .protocol import as_condition_batch

__all__ = [
    "SPIN_FLIP",
    "SpinFlipStates",
    "spin_flip_states",
    "two_branch_shell_amplitudes",
]

SPIN_FLIP = "spin_flip"


def two_branch_shell_amplitudes(
    angles: jnp.ndarray,
    n_qubits: int,
) -> jnp.ndarray:
    """Post-Hadamard Hamming-shell amplitudes of the two-branch state.

    With ``A = (c + s)/sqrt(2)``, ``B = (c - s)/sqrt(2)`` and
    ``c = cos(a/2)``, ``s = sin(a/2)``, the shell amplitude is

        ``phi_k = A^(n-k) B^k (1 + (-1)^k)``,

    so odd-weight amplitudes vanish and the state is spin-flip symmetric.  The
    broadcast angle is read from the first qubit (``angles[..., 0]``).
    """
    a = jnp.asarray(angles)
    angle = a if a.ndim == 0 else a[..., 0]
    c = jnp.cos(angle / 2.0)
    s = jnp.sin(angle / 2.0)
    big_a = (c + s) / jnp.sqrt(2.0)
    big_b = (c - s) / jnp.sqrt(2.0)
    k = jnp.arange(n_qubits + 1)
    amp = (big_a[..., None] ** (n_qubits - k)) * (big_b[..., None] ** k)
    parity = jnp.where(k % 2 == 0, 1.0, -1.0)
    return amp * (1.0 + parity)


@dataclass(frozen=True)
class SpinFlipStates:
    """Batched spin-flip reference stored as Hamming-shell amplitudes."""

    name: str
    angles: jnp.ndarray
    shell_amplitudes: jnp.ndarray

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
        return radial_shell_samples(key, self.shell_amplitudes, n_samples)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_conditions, n_samples, n_masks)`` shell-table ratios."""
        return radial_shell_ratios(samples, masks, self.shell_amplitudes)

    def select(self, condition: int) -> SpinFlipStates:
        """Return this family restricted to one condition, recomputed.

        Built directly rather than through :func:`spin_flip_states`: this runs
        inside the traced loss, where even a slice of a constant array is a
        tracer and the constructor's Python-level validation cannot run.  The
        parent object was validated when it was built.
        """
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        row = self.angles[condition : condition + 1]
        shell = two_branch_shell_amplitudes(row, self.n_qubits)
        return SpinFlipStates(name=self.name, angles=row, shell_amplitudes=shell)

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return ``(n_conditions, 2**n_qubits)`` little-endian amplitudes."""
        return radial_shell_state(self.shell_amplitudes)


def spin_flip_states(angles: jnp.ndarray) -> SpinFlipStates:
    """Build the spin-flip reference for every supplied condition."""
    angles = as_condition_batch(angles)
    shell = two_branch_shell_amplitudes(angles, angles.shape[1])
    return SpinFlipStates(name=SPIN_FLIP, angles=angles, shell_amplitudes=shell)
