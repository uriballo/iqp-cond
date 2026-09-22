"""Generator banks and the conditional model.

A generator row is a binary mask over the register; the diagonal unitary is
``exp(i sum_j theta_j (-1)^{g_j . z})``.  :meth:`Ansatz.complete` builds the
default two-local bank, while the constructor accepts any binary bank.

The model applies that shared diagonal to a condition-dependent encoder state,
``q_theta(x | c) = |<x| U_theta |phi(c)>|^2``.  Training uses the classically
computable estimator; :mod:`iqp_cond.simulation` compiles the same model into a
circuit for sampling on a simulator or backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .encoders import Encoder

if TYPE_CHECKING:
    import pennylane as qml

    from .estimator import ExpectationEstimate
    from .loss import ConditionalMMDResult
    from .operators import OperatorBank

__all__ = [
    "Ansatz",
    "ConditionalModel",
    "ancilla_support",
    "pad_visible_masks",
]


def _validated_generators(generators: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Return binary generator rows as float64, or raise.

    The diagonal phase is defined for bitmasks only.  The estimator reads rows
    modulo two, so a non-binary row would silently describe a different
    generator.  The constructor therefore rejects it.
    """
    rows = np.asarray(generators)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("generators must have shape (n_parameters, n_qubits)")
    if not np.isfinite(rows).all() or not np.isin(rows, (0, 1)).all():
        raise ValueError("generators must contain only zeros and ones")
    return jnp.asarray(rows, dtype=jnp.float64)


@dataclass(frozen=True)
class Ansatz:
    """A bank of diagonal IQP generator masks.

    ``generators`` is ``(n_parameters, n_qubits)`` and binary.  The
    constructor validates the rows before the estimator reads them mod two.
    """

    generators: jnp.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        object.__setattr__(self, "generators", _validated_generators(self.generators))

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return int(self.generators.shape[1])

    @property
    def n_parameters(self) -> int:
        """Number of independently parameterized generators."""
        return int(self.generators.shape[0])

    @classmethod
    def complete(cls, n_qubits: int) -> Ansatz:
        """Return the complete one- and two-body ansatz on ``n_qubits``."""
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        left, right = np.triu_indices(n_qubits, k=1)
        two_body = np.zeros((left.size, n_qubits), dtype=np.float64)
        two_body[np.arange(left.size), left] = 1.0
        two_body[np.arange(left.size), right] = 1.0
        generators = np.concatenate([np.eye(n_qubits, dtype=np.float64), two_body])
        return cls(generators=generators, name="complete")


def ancilla_support(
    generators: jnp.ndarray | np.ndarray,
    n_visible: int,
) -> np.ndarray:
    """Return the parameter mask whose generators act on an ancilla qubit.

    Under the visible-first ordering this is the set of parameters that have no
    data-based initialization and receive seeded noise instead; pass it to
    :func:`iqp_cond.parameters_from_data`.
    """
    rows = np.asarray(generators)
    if rows.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if not 0 <= n_visible <= rows.shape[1]:
        raise ValueError("n_visible must be between zero and the register width")
    return np.any(rows[:, n_visible:] != 0, axis=1)


def pad_visible_masks(
    masks: jnp.ndarray | np.ndarray,
    n_ancillas: int,
) -> jnp.ndarray:
    """Extend visible-qubit masks to the model register by zero padding.

    ``masks`` has shape ``(n_masks, n_visible)``; the result has shape
    ``(n_masks, n_visible + n_ancillas)``.
    """
    masks = jnp.asarray(masks)
    if n_ancillas < 0:
        raise ValueError("n_ancillas must be nonnegative")
    if masks.ndim == 1:
        masks = masks[None, :]
    if masks.ndim != 2:
        raise ValueError("masks must have shape (n_masks, n_visible)")
    if n_ancillas == 0:
        return masks
    zeros = jnp.zeros((masks.shape[0], n_ancillas), dtype=masks.dtype)
    return jnp.concatenate([masks, zeros], axis=1)


@dataclass(frozen=True)
class ConditionalModel:
    """A shared IQP generator bank acting on a condition-dependent encoder.

    ``n_visible`` declares how many of the encoder's qubits are measured;
    the remaining qubits are ancillas, ordered last and marginalized.  Leaving
    it ``None`` means every qubit is visible.
    """

    ansatz: Ansatz
    encoder: Encoder
    n_visible: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, Encoder):
            raise TypeError(
                f"{type(self.encoder).__name__} does not implement the Encoder "
                "protocol",
            )
        if self.ansatz.n_qubits != self.encoder.n_qubits:
            raise ValueError(
                f"ansatz acts on {self.ansatz.n_qubits} qubits but the encoder "
                f"acts on {self.encoder.n_qubits}",
            )
        if self.n_visible is not None and not isinstance(
            self.n_visible,
            (int, np.integer),
        ):
            raise ValueError(
                f"n_visible must be an integer or None, got "
                f"{type(self.n_visible).__name__}",
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
        estimator_dtype: jnp.dtype | None = None,
        return_samples: bool = False,
    ) -> ExpectationEstimate:
        """Estimate visible Pauli-Z moments for every condition and mask."""
        from .estimator import estimate_pauli_z

        return estimate_pauli_z(
            params,
            self.generators,
            self.encoder,
            masks,
            key=key,
            samples=samples,
            n_samples=n_samples,
            gemm_dtype=gemm_dtype,
            estimator_dtype=estimator_dtype,
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
        estimator_dtype: jnp.dtype | None = None,
        sample_chunk: int | None = None,
    ) -> ConditionalMMDResult:
        """Condition-balanced multiscale MMD against visible-width targets."""
        from .loss import mmd

        if targets.shape[-1] != self.visible:
            raise ValueError(
                f"targets have {targets.shape[-1]} visible qubits but the model "
                f"declares {self.visible}",
            )
        return mmd(
            params,
            self.generators,
            self.encoder,
            targets,
            key,
            bank=bank,
            sigmas=sigmas,
            n_ops=n_ops,
            n_samples=n_samples,
            shared_samples=shared_samples,
            gemm_dtype=gemm_dtype,
            estimator_dtype=estimator_dtype,
            sample_chunk=sample_chunk,
        )

    def sample(
        self,
        params: np.ndarray,
        n_samples: int,
        *,
        seed: int | None = None,
        device: str | qml.devices.Device = "lightning.qubit",
    ) -> np.ndarray:
        """Draw bitstrings from every condition through a PennyLane device."""
        from .simulation import _sample_model

        return _sample_model(
            self,
            params,
            n_samples,
            seed=seed,
            device=device,
        )

    def probabilities(
        self,
        params: np.ndarray,
        *,
        device: str | qml.devices.Device = "lightning.qubit",
        seed: int | None = None,
    ) -> np.ndarray:
        """Return every condition's exact visible probability distribution."""
        from .simulation import _probabilities

        return _probabilities(self, params, device=device, seed=seed)
