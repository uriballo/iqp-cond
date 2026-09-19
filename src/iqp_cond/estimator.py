"""Monte Carlo Pauli-Z estimation for conditional IQP circuits.

The estimator is the paper's conditional Pauli estimator: for a Fourier mask
``a`` it averages ``Re[R_c(z, a) exp(i Delta_a f_theta(z))]`` over basis samples
``z`` drawn from the condition state, where ``R_c(z, a) = phi(z xor a)/phi(z)``
comes from the encoder and ``Delta_a f`` from the diagonal phase bank.  Both
pieces are classically computable, so the estimator is unbiased with bounded
variance and its parameter gradient is available in closed form.

The module exposes two entry points.  :func:`estimate_pauli_z` validates its
inputs and pads visible-width masks with identity on the ancillas.
:func:`estimate_pauli_z_unchecked` skips both checks and is safe to trace under
``jax.jit``.  Either function accepts a random key or pre-drawn reference
samples, so several mask banks can share one draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import pad_visible_masks
from .encoders import Encoder

__all__ = [
    "ExpectationEstimate",
    "estimate_pauli_z",
    "estimate_pauli_z_unchecked",
]


@dataclass(frozen=True)
class ExpectationEstimate:
    """Pauli-Z estimates for every condition and mask.

    ``values`` has shape ``(n_conditions, n_masks)``.  ``sample_values`` holds
    the ``(conditions, samples, masks)`` per-sample values and is ``None``
    unless ``return_samples=True``.  It needs a large allocation, so only the
    loss and callers computing their own variance ask for it.
    ``standard_errors`` derives from it on demand.
    """

    values: jnp.ndarray
    n_samples: int
    sample_values: jnp.ndarray | None = None

    @property
    def standard_errors(self) -> jnp.ndarray:
        """Monte Carlo standard error of each ``values`` entry."""
        if self.sample_values is None:
            raise ValueError(
                "standard_errors needs per-sample values; estimate with "
                "return_samples=True",
            )
        return jnp.std(self.sample_values, axis=1, ddof=1) / jnp.sqrt(self.n_samples)


jax.tree_util.register_dataclass(
    ExpectationEstimate,
    data_fields=["values", "sample_values"],
    meta_fields=["n_samples"],
)


def _validate_binary(name: str, values: jnp.ndarray) -> None:
    if not bool(jnp.all((values == 0) | (values == 1))):
        raise ValueError(f"{name} must contain only zeros and ones")


def _validate_sources(
    key: jax.Array | None,
    samples: jnp.ndarray | None,
    n_samples: int | None,
) -> None:
    if (key is None) == (samples is None):
        raise ValueError("pass exactly one of key or samples")
    if key is not None:
        if n_samples is None:
            raise ValueError("n_samples is required when sampling from a key")
        if n_samples < 2:
            raise ValueError("n_samples must be at least two")
    elif n_samples is not None and samples is not None:
        if int(n_samples) != samples.shape[1]:
            raise ValueError("n_samples must match the drawn samples")


def _validate_phase_weighting(phase_weighting: str) -> None:
    if phase_weighting not in ("operators", "samples"):
        raise ValueError(
            f"unknown phase_weighting: {phase_weighting!r}; "
            "expected 'operators' or 'samples'",
        )


def _phase_differences(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
    gemm_dtype: jnp.dtype | None = None,
    phase_weighting: str = "operators",
) -> jnp.ndarray:
    """Return ``Delta_a f_theta(z)`` for every sample and mask.

    ``gemm_dtype`` runs only the ``(samples, generators) x (masks, generators)``
    contraction in a reduced dtype and casts the result back to float64, so the
    exponentiation and every reduction stay precise.  This is the mixed
    precision mode that gives large speedups on GPUs, whose float64 throughput
    is a small fraction of float32.  On CPU it ranges from no change to a small
    speedup.
    """
    sample_parities = jnp.einsum("bsn,mn->bsm", samples, generators)
    sample_characters = 1.0 - 2.0 * jnp.mod(sample_parities, 2.0)
    op_parities = jnp.mod(jnp.einsum("rn,mn->rm", masks, generators), 2.0)
    # Weighting either operand gives the same phase, but changes the work in
    # its parameter gradient. Keep this choice static when tracing the kernel.
    if phase_weighting == "samples":
        sample_characters = -2.0 * sample_characters * params[None, None, :]
        weighted_flips = op_parities
    else:
        weighted_flips = -2.0 * op_parities * params[None, :]
    if gemm_dtype is not None:
        phases = jnp.einsum(
            "bsm,rm->bsr",
            sample_characters.astype(gemm_dtype),
            weighted_flips.astype(gemm_dtype),
        )
        return phases.astype(jnp.float64)
    return jnp.einsum("bsm,rm->bsr", sample_characters, weighted_flips)


def _estimate(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
    gemm_dtype: jnp.dtype | None,
    return_samples: bool,
    phase_weighting: str = "operators",
) -> ExpectationEstimate:
    ratios = encoders.amplitude_ratios(samples, masks)
    phases = _phase_differences(
        params, generators, masks, samples, gemm_dtype, phase_weighting
    )
    sample_values = jnp.real(ratios * jnp.exp(1j * phases))
    values = jnp.mean(sample_values, axis=1)
    return ExpectationEstimate(
        values,
        int(samples.shape[1]),
        sample_values if return_samples else None,
    )


def _validate_sample_axes(encoders: Encoder, drawn: jnp.ndarray) -> None:
    """Reject draws whose condition or qubit axis does not match the encoder.

    A size-1 axis would otherwise broadcast through the encoder and the phase
    contraction, silently mixing reference states or qubits.
    """
    if drawn.shape[0] != encoders.n_conditions:
        raise ValueError(
            f"sample condition axis has {drawn.shape[0]} entries but the encoder "
            f"covers {encoders.n_conditions} conditions",
        )
    if drawn.shape[2] != encoders.n_qubits:
        raise ValueError(
            f"sample qubit axis has {drawn.shape[2]} entries but the encoder "
            f"acts on {encoders.n_qubits} qubits",
        )


def _draw(
    encoders: Encoder,
    key: jax.Array | None,
    samples: jnp.ndarray | None,
    n_samples: int | None,
) -> jnp.ndarray:
    if samples is not None:
        if samples.ndim == 2:
            if encoders.n_conditions != 1:
                raise ValueError(
                    "a rank-two samples draw covers a single condition; pass "
                    "one reference set per condition with shape "
                    "(conditions, samples, qubits)",
                )
            return samples[None, :, :]
        if samples.ndim != 3:
            raise ValueError(
                "samples must have shape (conditions, samples, qubits)",
            )
        return samples
    assert key is not None and n_samples is not None
    return encoders.sample(key, n_samples)


def estimate_pauli_z(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    masks: jnp.ndarray,
    *,
    key: jax.Array | None = None,
    samples: jnp.ndarray | None = None,
    n_samples: int | None = None,
    gemm_dtype: jnp.dtype | None = None,
    return_samples: bool = False,
    phase_weighting: str = "operators",
) -> ExpectationEstimate:
    """Estimate Pauli-Z expectations for one or more conditions.

    Pass ``key`` with ``n_samples`` to draw fresh reference samples, or
    ``samples`` to reuse a ``(conditions, samples, qubits)`` draw; the draw
    must match the encoder's condition and qubit axes, because a size-1 axis
    would otherwise broadcast silently.  A rank-two ``(samples, qubits)`` draw
    is accepted only for a single-condition encoder, because sharing one draw
    across different reference states would bias every condition but the one
    it was sampled from.  ``masks`` has shape
    ``(n_masks, n_qubits)``; visible-width masks are accepted and padded with
    identity on the ancilla positions, following the visible-first convention
    of :mod:`iqp_cond.core`.

    ``phase_weighting="samples"`` multiplies parameters into sample characters
    instead of operator parities. This can accelerate unchunked float64
    gradients, but is workload-dependent and can change rounding. Both choices
    use the same reference draws and compute the same mathematical expectation
    estimates and parameter gradients. Neither constructs a statevector or
    samples the IQP output distribution. The default
    ``"operators"`` retains the original contraction. Under ``jax.jit``, keep
    this string static, for example by closing over it.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    masks = jnp.asarray(masks, dtype=jnp.float64)
    _validate_phase_weighting(phase_weighting)
    _validate_sources(key, samples, n_samples)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if masks.ndim == 1:
        masks = masks[None, :]
    if masks.ndim != 2:
        raise ValueError("masks must have shape (n_masks, n_qubits)")
    if masks.shape[1] > generators.shape[1]:
        raise ValueError("masks cannot cover more qubits than the generators")
    if encoders.n_qubits != generators.shape[1]:
        raise ValueError(
            f"encoder acts on {encoders.n_qubits} qubits but generators "
            f"cover {generators.shape[1]}",
        )
    if masks.shape[1] < encoders.n_qubits:
        masks = pad_visible_masks(masks, encoders.n_qubits - masks.shape[1])
    _validate_binary("generators", generators)
    _validate_binary("masks", masks)

    drawn = _draw(encoders, key, samples, n_samples)
    _validate_sample_axes(encoders, drawn)
    _validate_binary("samples", drawn)
    return _estimate(
        params,
        generators,
        encoders,
        masks,
        drawn,
        gemm_dtype,
        return_samples,
        phase_weighting,
    )


def estimate_pauli_z_unchecked(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    masks: jnp.ndarray,
    *,
    key: jax.Array | None = None,
    samples: jnp.ndarray | None = None,
    n_samples: int | None = None,
    gemm_dtype: jnp.dtype | None = None,
    return_samples: bool = False,
    phase_weighting: str = "operators",
) -> ExpectationEstimate:
    """Traceable estimator kernel with no validation and model-width masks.

    The loss calls this function under ``jax.jit``.
    """
    drawn = _draw(encoders, key, samples, n_samples)
    return _estimate(
        params,
        generators,
        encoders,
        masks,
        drawn,
        gemm_dtype,
        return_samples,
        phase_weighting,
    )
