"""The encoder interface: classical sampling plus amplitude ratios.

A conditional IQP model applies one shared diagonal unitary ``U_theta`` to a
condition-dependent input state ``|phi(c)>``.  Training needs two things from
that input state, and only these two:

``sample``
    Draw bitstrings from ``|phi(c)|^2`` in polynomial time.
``amplitude_ratios``
    Evaluate ``phi(z xor a) / phi(z)`` for sampled ``z`` and Fourier masks ``a``.

That pair is the whole classical-tractability requirement.  An encoder
providing it, plus ``select`` and the two size properties, is trainable with the
same estimator, loss, and trainer.  Encoders are looked up by name and can be
added without touching this package::

    import iqp_cond as iqc

    iqc.register_encoder("my_family", my_encoder_factory)
    states = iqc.encoder("my_family", angles)

Implementations are batched over conditions: ``angles`` always carries a leading
condition axis, and every method keeps that axis, because the estimator samples
all conditions from a single random key.  Exact small-system work needs a
further capability, ``post_hadamard_amplitudes``, declared separately by
:class:`ExactEncoder`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

__all__ = [
    "ENCODERS",
    "Encoder",
    "ExactEncoder",
    "as_condition_batch",
    "available_encoders",
    "encoder",
    "register_encoder",
]


@runtime_checkable
class Encoder(Protocol):
    """A condition-parameterised encoder evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    """

    name: str
    angles: jnp.ndarray

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

    def select(self, condition: int) -> Encoder:
        """Return the same encoder restricted to a single condition.

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


@runtime_checkable
class ExactEncoder(Encoder, Protocol):
    """A training encoder that can also materialise its exact post-Hadamard state.

    Required only for small-system exact evaluation (the full ``2**n`` state,
    TVD, and the ratio-identity tests); training never calls it.
    """

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        """Return normalised ``(C, 2**n)`` amplitudes ``<z|H^n|phi(c)>``."""


ENCODERS: dict[str, object] = {}


def register_encoder(
    family: str,
    factory: object,
    *,
    replace: bool = False,
) -> None:
    """Register an encoder factory under ``family``.

    The factory is called as ``factory(angles, **params)`` with ``angles`` of
    shape ``(n_conditions, n_qubits)``.  Registering an existing name raises
    unless ``replace=True``, so a typo or a plugin cannot silently shadow a
    built-in family.
    """
    if not family or not isinstance(family, str):
        raise ValueError("family must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if family in ENCODERS and not replace:
        raise ValueError(
            f"encoder family {family!r} is already registered; pass "
            "replace=True to overwrite it",
        )
    ENCODERS[family] = factory


def available_encoders() -> tuple[str, ...]:
    """Return the registered family names, sorted."""
    return tuple(sorted(ENCODERS))


def encoder(family: str, angles: jnp.ndarray, **params: object) -> Encoder:
    """Build the encoder of ``family`` for every supplied condition."""
    if family not in ENCODERS:
        raise ValueError(
            f"unknown encoder family: {family!r}; "
            f"available: {', '.join(available_encoders())}",
        )
    return ENCODERS[family](angles, **params)  # type: ignore[operator]


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
