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
import numpy as np

__all__ = ["Encoder", "ExactEncoder"]


@runtime_checkable
class Encoder(Protocol):
    """A condition-parameterised encoder evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    The loss, trainer and :func:`~iqp_cond.check_encoder` require a JAX
    pytree whose array leaves all carry the leading condition axis: row ``c``
    of every leaf is condition ``c``.  Indexing those rows restricts the
    encoder to one condition, which vectorizes the loss over conditions and
    picks the condition inside a compiled update.  ``n_conditions`` must
    therefore follow the leaves' leading axis rather than a stored value, and
    rebuilding the pytree from traced leaves must not check their values.  A
    single-condition encoder may have no array leaves.

    The preparation callable runs inside a PennyLane queuing context while a
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


def _check_condition_pytree(encoder: Encoder) -> None:
    """Raise unless ``encoder`` restricts to one condition by its leaf rows."""
    name = type(encoder).__name__
    leaves = jax.tree_util.tree_leaves(encoder)
    if (not leaves and encoder.n_conditions != 1) or any(
        not isinstance(leaf, (jax.Array, np.ndarray))
        or leaf.ndim == 0
        or leaf.shape[0] != encoder.n_conditions
        for leaf in leaves
    ):
        raise TypeError(
            f"{name} must be a JAX pytree whose array leaves share a leading "
            "condition axis",
        )
    if _select_condition(encoder, 0).n_conditions != 1:
        raise TypeError(
            f"{name}.n_conditions must follow the leading axis of its array "
            "leaves, so that one row restricts it to one condition",
        )


def _select_condition(encoder: Encoder, condition: int | jax.Array) -> Encoder:
    """Return ``encoder`` restricted to ``condition``, which may be traced.

    Every leaf keeps a size-1 condition axis, so the result is an encoder over
    one condition.  A traced index is not bounds-checked.
    """
    return jax.tree_util.tree_map(lambda leaf: leaf[condition][None], encoder)


def _condition_rows(encoder: Encoder) -> Encoder:
    """Return ``encoder`` with a size-1 axis after the leading condition axis.

    Mapping over the leading axis then hands every condition the encoder
    :func:`_select_condition` would return, so the conditions run as one
    vectorized program.  Callers validate the encoder at their boundary with
    :func:`_check_condition_pytree`.
    """
    return jax.tree_util.tree_map(lambda leaf: leaf[:, None], encoder)
