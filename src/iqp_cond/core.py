"""Generator banks, the conditional model, and exact small-system evaluation.

A generator row is a binary mask over the register; the diagonal unitary is
``exp(i sum_j theta_j (-1)^{g_j . z})``.  The complete two-local bank built by
:func:`complete_ansatz` is the production choice, and any binary bank can be
used through :func:`ansatz_from_generators`.

The model applies that shared diagonal to a condition-dependent encoder state,
``q_theta(x | c) = |<x| U_theta |phi(c)>|^2``.  For registers small enough to
materialize the full ``2**n`` state, :func:`exact_probabilities` and
:func:`sample_exact` evaluate the model distribution exactly; training and
inference otherwise rely on the classically computable estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .encoders import Encoder, ExactEncoder

if TYPE_CHECKING:
    from .estimator import ExpectationEstimate
    from .loss import ConditionalMMDResult
    from .operators import OperatorBank

__all__ = [
    "Ansatz",
    "ConditionalModel",
    "ancilla_support",
    "ansatz_from_generators",
    "complete_ansatz",
    "exact_probabilities",
    "pad_visible_masks",
    "sample_exact",
]


def _validated_generators(generators: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Return binary generator rows as float64, or raise.

    The diagonal phase is defined for bitmasks only: the estimator reads rows
    modulo two, so a non-binary row would silently describe a different
    generator.  Construction is therefore the place to reject it.
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

    ``generators`` is ``(n_parameters, n_qubits)`` and binary; the constructor
    validates the rows, so no path can build an ansatz whose generators the
    estimator would read mod two.
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


def ansatz_from_generators(
    generators: jnp.ndarray | np.ndarray,
    name: str = "custom",
) -> Ansatz:
    """Build an ansatz directly from a binary generator bank."""
    return Ansatz(generators=generators, name=name)


def complete_ansatz(n_qubits: int) -> Ansatz:
    """Return the complete two-local ansatz on ``n_qubits`` qubits.

    The bank holds every one-body row followed by one row per unordered pair,
    so it has ``n + n(n-1)/2`` parameters.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    left, right = np.triu_indices(n_qubits, k=1)
    two_body = np.zeros((left.size, n_qubits), dtype=np.float64)
    two_body[np.arange(left.size), left] = 1.0
    two_body[np.arange(left.size), right] = 1.0
    generators = np.concatenate([np.eye(n_qubits, dtype=np.float64), two_body])
    return Ansatz(generators=generators, name="complete")


def ancilla_support(
    generators: jnp.ndarray | np.ndarray,
    n_visible: int,
) -> np.ndarray:
    """Return the parameter mask whose generators touch an ancilla qubit.

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
        from .estimator import estimate_pauli_z

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
        from .loss import mmd

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


def _hadamard_transform(amplitudes: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply ``H^{tensor n}`` along the last axis of ``amplitudes``."""
    blocks = amplitudes.reshape(amplitudes.shape[:-1] + (2,) * n_qubits)
    scale = np.sqrt(2.0)
    for axis in range(-n_qubits, 0):
        zero = blocks.take(0, axis=axis)
        one = blocks.take(1, axis=axis)
        blocks = np.stack([(zero + one) / scale, (zero - one) / scale], axis=axis)
    return blocks.reshape(amplitudes.shape)


def exact_probabilities(
    model: ConditionalModel,
    params: np.ndarray,
) -> np.ndarray:
    """Return the exact visible distribution ``(conditions, 2**n_visible)``.

    Materializes the state entering the diagonal, applies the shared phases and
    the final Hadamard layer, and marginalizes the ancillas.  Cost is
    ``O(conditions * 2**n_qubits)`` time and memory, so this is a small-system
    evaluation path (a few hundred megabytes at roughly 26 qubits); training
    never needs it.  The encoder must provide ``post_hadamard_amplitudes``
    (:class:`~iqp_cond.encoders.ExactEncoder`).
    """
    if not isinstance(model.encoders, ExactEncoder):
        raise TypeError(
            "exact evaluation needs an encoder with post_hadamard_amplitudes; "
            f"{type(model.encoders).__name__} does not provide it",
        )
    params = np.asarray(params, dtype=np.float64)
    generators = np.asarray(model.generators, dtype=np.float64)
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if not np.isfinite(params).all():
        raise ValueError("params must contain only finite values")

    n_qubits = model.n_qubits
    amplitudes = np.asarray(
        model.encoders.post_hadamard_amplitudes(),
        dtype=np.complex128,
    )
    if amplitudes.shape != (model.encoders.n_conditions, 2**n_qubits):
        raise ValueError(
            "post_hadamard_amplitudes must have shape (conditions, 2**n_qubits)",
        )

    codes = np.arange(2**n_qubits, dtype=np.int64)
    weights = 1 << np.arange(n_qubits, dtype=np.int64)
    generator_codes = np.asarray(generators, dtype=np.int64) @ weights
    overlaps = codes[:, None] & generator_codes[None, :]
    characters = 1.0 - 2.0 * (np.bitwise_count(overlaps) % 2)
    phases = characters @ params
    state = amplitudes * np.exp(1j * phases)[None, :]
    state = _hadamard_transform(state, n_qubits)

    n_visible = model.visible
    densities = (
        np.abs(
            state.reshape(
                model.encoders.n_conditions,
                2**model.n_ancillas,
                2**n_visible,
            ),
        )
        ** 2
    )
    # Ancillas are measured and traced out: sum probabilities, not amplitudes.
    return densities.sum(axis=1)


def sample_exact(
    model: ConditionalModel,
    params: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``(conditions, n_samples, n_visible)`` bitstrings from the model.

    Exact classical sampling via :func:`exact_probabilities`, so it is limited
    to registers small enough to materialize the full state.  Uses NumPy's
    generator, independent of the JAX keys used for training.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    probabilities = exact_probabilities(model, params)
    rng = np.random.default_rng(seed)
    codes = np.stack(
        [
            rng.choice(probabilities.shape[1], size=n_samples, p=condition)
            for condition in probabilities
        ],
    )
    return ((codes[..., None] >> np.arange(model.visible)) & 1).astype(np.float64)
