"""The conditional IQP model: a shared diagonal unitary and an encoder.

The paper's model applies one learned IQP block to a condition-dependent input
state, ``q_theta(x | c) = |<x| U_theta |phi(c)>|^2``.  This module ties the two
pieces together so the estimator, the loss, and the trainer share one validated
description of the register, in particular the visible/ancilla split.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .ansatz import Ansatz
from .encoders import Encoder
from .estimator import ExpectationEstimate, estimate_pauli_z
from .loss import ConditionalMMDResult, mmd
from .operators import OperatorBank

__all__ = ["ConditionalModel"]


@dataclass(frozen=True)
class ConditionalModel:
    """A shared IQP generator bank acting on a condition-dependent encoder.

    ``n_visible`` declares how many of the encoder's qubits are measured;
    the remaining qubits are ancillas, ordered last and marginalized.  Leaving
    it ``None`` means every qubit is visible.
    """

    ansatz: Ansatz
    encoders: Encoder
    n_visible: int | None = None

    def __post_init__(self) -> None:
        if self.ansatz.n_qubits != self.encoders.n_qubits:
            raise ValueError(
                f"ansatz acts on {self.ansatz.n_qubits} qubits but the encoder "
                f"acts on {self.encoders.n_qubits}",
            )
        if not 1 <= self.visible <= self.ansatz.n_qubits:
            raise ValueError(
                f"n_visible must be between 1 and {self.ansatz.n_qubits}",
            )

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the model register."""
        return self.ansatz.n_qubits

    @property
    def visible(self) -> int:
        """Number of measured (visible) qubits."""
        if self.n_visible is None:
            return self.ansatz.n_qubits
        return int(self.n_visible)

    @property
    def n_ancillas(self) -> int:
        """Number of marginalized qubits."""
        return self.n_qubits - self.visible

    @property
    def generators(self) -> jnp.ndarray:
        """The shared generator bank."""
        return self.ansatz.generators

    def estimate_pauli_z(
        self,
        params: jnp.ndarray,
        masks: jnp.ndarray,
        *,
        key: jax.Array | None = None,
        samples: jnp.ndarray | None = None,
        n_samples: int | None = None,
        gemm_dtype: jnp.dtype | None = None,
        return_samples: bool = False,
    ) -> ExpectationEstimate:
        """Estimate visible Pauli-Z moments for every condition and mask."""
        return estimate_pauli_z(
            params,
            self.generators,
            self.encoders,
            masks,
            key=key,
            samples=samples,
            n_samples=n_samples,
            gemm_dtype=gemm_dtype,
            return_samples=return_samples,
        )

    def mmd(
        self,
        params: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
        *,
        bank: OperatorBank | None = None,
        sigmas: tuple[float, ...] | None = None,
        n_ops: int | None = None,
        n_samples: int,
        shared_samples: bool = False,
        gemm_dtype: jnp.dtype | None = None,
        sample_chunk: int | None = None,
    ) -> ConditionalMMDResult:
        """Condition-balanced multiscale MMD against visible-width targets."""
        if targets.shape[-1] != self.visible:
            raise ValueError(
                f"targets have {targets.shape[-1]} visible qubits but the model "
                f"declares {self.visible}",
            )
        return mmd(
            params,
            self.generators,
            self.encoders,
            targets,
            key,
            bank=bank,
            sigmas=sigmas,
            n_ops=n_ops,
            n_samples=n_samples,
            shared_samples=shared_samples,
            gemm_dtype=gemm_dtype,
            sample_chunk=sample_chunk,
        )
