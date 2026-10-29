"""The encoder contract shared by every encoder family.

:class:`Encoder` is the structural protocol that training, simulation and
verification depend on.  :class:`ExactEncoder` adds the optional exact state
used by small-system checks.  Helpers shared between encoder families live
here too, so implementation modules never import from each other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

__all__ = ["Encoder", "ExactEncoder"]


@runtime_checkable
class Encoder(Protocol):
    """A condition-parameterised encoder evaluated over a batch of conditions.

    Shapes are ``C`` conditions, ``S`` samples, ``R`` masks, ``n`` qubits.
    To save memory, the library evaluates one condition at a time by calling
    :meth:`select`.  Implementations may also register as JAX pytrees when
    useful, but that is not part of the public extension contract.  The
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


class _ConditionBatch:
    """Pytree and selection support for encoders batched over conditions.

    Subclasses list their per-condition arrays in ``_data_fields``, each with a
    leading condition axis, and their static attributes in ``_meta_fields``.
    """

    _data_fields: ClassVar[tuple[str, ...]]
    _meta_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _from_fields(cls, **fields: object) -> _ConditionBatch:
        instance = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(instance, name, value)
        return instance

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], tuple[object, ...]]:
        """Return dynamic arrays and static attributes for JAX transformations."""
        return (
            tuple(getattr(self, name) for name in self._data_fields),
            tuple(getattr(self, name) for name in self._meta_fields),
        )

    @classmethod
    def tree_unflatten(
        cls,
        meta: tuple[object, ...],
        data: tuple[jnp.ndarray, ...],
    ) -> _ConditionBatch:
        """Rebuild a traced encoder without recomputing its amplitudes."""
        return cls._from_fields(
            **dict(zip(cls._data_fields, data, strict=True)),
            **dict(zip(cls._meta_fields, meta, strict=True)),
        )

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the batch."""
        return int(getattr(self, self._data_fields[0]).shape[0])

    def _check_condition(self, condition: int) -> None:
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")

    def select(self, condition: int) -> _ConditionBatch:
        """Return this encoder restricted to one condition."""
        self._check_condition(condition)
        selection = slice(condition, condition + 1)
        return self._from_fields(
            **{name: getattr(self, name)[selection] for name in self._data_fields},
            **{name: getattr(self, name) for name in self._meta_fields},
        )
