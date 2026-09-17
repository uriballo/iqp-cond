"""The reference-state interface: classical sampling plus amplitude ratios.

A conditional IQP model applies one shared diagonal unitary ``U_theta`` to a
condition-dependent input state ``|phi(c)>``.  Training needs three things from
that input state, and only these three:

``sample``
    Draw bitstrings from ``|phi(c)|^2`` in polynomial time.
``amplitude_ratios``
    Evaluate ``phi(z xor a) / phi(z)`` for sampled ``z`` and Fourier masks ``a``.
``post_hadamard_amplitudes``
    Return the normalised amplitudes ``<z|H^n|phi(c)>`` for every ``z``, for
    exact small-system work.

Any family providing them is trainable with the same estimator, loss, and
trainer.  Families are looked up by name and can be added without touching this
package::

    import iqp_cond as iqc

    iqc.register_reference("my_family", my_states_factory)
    states = iqc.reference("my_family", angles)

Implementations are batched over conditions: ``angles`` always carries a leading
condition axis, and every method keeps that axis, because the estimator samples
all conditions from a single random key.  Only ``sample`` and
``amplitude_ratios`` are needed for training; ``post_hadamard_amplitudes``
materialises the full ``2**n`` state and is the exact-evaluation hook.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

__all__ = [
    "FAMILIES",
    "ReferenceStates",
    "as_condition_batch",
    "available_families",
    "reference",
    "register_reference",
]


@runtime_checkable
class ReferenceStates(Protocol):
    """A condition-parameterised family evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    """

    name: str
    angles: jnp.ndarray

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""

    @property
    def n_qubits(self) -> int:
        """Number of qubits the reference state lives on."""

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw ``(C, S, n)`` bitstrings distributed as ``|phi(c)|^2``."""

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        """Return ``(C, S, R)`` ratios ``phi(z xor a) / phi(z)``."""

    def select(self, condition: int) -> ReferenceStates:
        """Return the same family restricted to a single condition.

        Used by the fixed-operator estimator, which evaluates one condition at a
        time so that the sample/mask tensor never carries every condition at
        once.  Implementations must recompute rather than slice the batched
        amplitudes, or the result differs from per-condition evaluation in the
        last bits.

        This method runs *inside* the jitted loss, where a slice of a constant
        array is itself a tracer.  Implementations must therefore build the
        child object directly instead of calling their validating constructor:
        the parent was already validated when the user built it.
        """

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return normalised ``(C, 2**n)`` amplitudes ``<z|H^n|phi(c)>``."""


FAMILIES: dict[str, object] = {}


def register_reference(family: str, factory: object) -> None:
    """Register a reference-state factory under ``family``.

    The factory is called as ``factory(angles, **params)`` with ``angles`` of
    shape ``(n_conditions, n_qubits)``.
    """
    if not family or not isinstance(family, str):
        raise ValueError("family must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    FAMILIES[family] = factory


def available_families() -> tuple[str, ...]:
    """Return the registered family names, sorted."""
    return tuple(sorted(FAMILIES))


def reference(family: str, angles: jnp.ndarray, **params: object) -> ReferenceStates:
    """Build the reference states of ``family`` for every supplied condition."""
    if family not in FAMILIES:
        raise ValueError(
            f"unknown reference family: {family!r}; "
            f"available: {', '.join(available_families())}",
        )
    return FAMILIES[family](angles, **params)  # type: ignore[operator]


def as_condition_batch(angles: jnp.ndarray, n_qubits: int | None = None) -> jnp.ndarray:
    """Return ``angles`` as ``(n_conditions, n_qubits)`` float64."""
    values = jnp.asarray(angles, dtype=jnp.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("angles must have shape (n_conditions, n_qubits)")
    if n_qubits is not None and values.shape[1] != n_qubits:
        raise ValueError(
            f"angles must have {n_qubits} columns, got {values.shape[1]}",
        )
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("angles must be finite")
    return values
