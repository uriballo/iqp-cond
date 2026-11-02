"""The encoder contract shared by every encoder family.

:class:`Encoder` is the structural protocol that training, simulation and
verification depend on.  :class:`ExactEncoder` adds the optional exact state
used by small-system checks.  Helpers shared between encoder families live
here too, so implementation modules never import from each other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

__all__ = ["Encoder", "ExactEncoder"]


@runtime_checkable
class Encoder(Protocol):
    """A condition-parameterised encoder evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    The loss and trainer require a JAX pytree whose array leaves all carry the
    leading condition axis: they index those leaves to vectorize over
    conditions and to pick one condition inside a compiled update.  A
    single-condition encoder may have no array leaves.  The
    preparation callable runs inside a PennyLane queuing context while a
    circuit is built and may queue any PennyLane operation; training never
    calls it.
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
        """Return ``(C, S, R)`` ratios ``psi_c(z xor a) / psi_c(z)``.

        Values where ``psi_c(z)`` vanishes are unspecified: the estimator
        samples ``z`` from ``|psi_c(z)|**2``, so it never queries them.
        """

    def preparation(self, condition: int) -> Callable[[], None]:
        """Return a callable queuing ``|phi(condition)>`` before ``H^n``."""

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
