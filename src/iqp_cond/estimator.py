"""Monte Carlo Pauli-Z estimation for conditional IQP circuits.

For a Fourier mask ``a`` the estimator averages
``Re[R_c(z, a) exp(i Delta_a f_theta(z))]`` over basis samples ``z`` drawn
from the post-Hadamard reference state.  The encoder supplies
``R_c(z, a) = psi_c(z xor a)/psi_c(z)``, where
``psi_c(z) = <z|H^n|phi(c)>``; the diagonal phase bank supplies ``Delta_a f``.
Both pieces are classically computable, so the estimator is unbiased with
bounded variance and its parameter gradient is available in closed form.

:func:`estimate_pauli_z` validates its inputs and pads visible-width masks
with identity on the ancillas.  :func:`estimate_pauli_z_unchecked` skips both
checks and is safe to trace under ``jax.jit``.  Either function accepts a
random key or pre-drawn reference samples, so several mask banks can share
one draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._layout import pad_visible_masks
from ._precision import resolve_estimator_dtype
from .encoders import AngleEncoder, Encoder

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


def _phase_differences(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
    gemm_dtype: jnp.dtype | None = None,
    estimator_dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    """Return ``Delta_a f_theta(z)`` for every sample and mask.

    ``gemm_dtype`` runs only the final contraction in a reduced dtype and casts
    the result back to float64. ``estimator_dtype=float32`` instead runs parity
    construction and the phase contraction in float32; the caller then keeps
    the complex estimator arithmetic in single precision before returning to
    float64 for statistics and loss accumulation.
    """
    resolved_dtype = resolve_estimator_dtype(gemm_dtype, estimator_dtype)
    if resolved_dtype is not None:
        samples = samples.astype(resolved_dtype)
        generators = generators.astype(resolved_dtype)
        masks = masks.astype(resolved_dtype)
        params = params.astype(resolved_dtype)
    # Flattening the leading (condition, sample) axes keeps both contractions
    # plain matrix products; under the loss's condition vmap, batched einsums
    # with the unit condition axis measured about a quarter slower on CPU.
    leading = samples.shape[:-1]
    sample_parities = samples.reshape(-1, samples.shape[-1]) @ generators.T
    sample_characters = 1.0 - 2.0 * jnp.mod(sample_parities, 2.0)
    op_parities = jnp.mod(masks @ generators.T, 2.0)
    weighted_flips = -2.0 * op_parities * params[None, :]
    if gemm_dtype is not None:
        characters = sample_characters.astype(gemm_dtype)
        flips = weighted_flips.astype(gemm_dtype)
        phases = (characters @ flips.T).astype(jnp.float64)
    else:
        phases = sample_characters @ weighted_flips.T
    return phases.reshape(*leading, masks.shape[0])


def _sample_values(
    ratios: jnp.ndarray,
    phases: jnp.ndarray,
    resolved_dtype: jnp.dtype | None,
) -> jnp.ndarray:
    """Return the float64 per-sample values ``Re[R exp(i Delta f)]``.

    ``resolved_dtype`` comes from :func:`resolve_estimator_dtype`; float32
    evaluates the product in single precision before returning to float64.
    """
    # Re[R exp(i phi)] = Re R cos phi - Im R sin phi, which skips the complex
    # exponential and, for real ratios, the sine and the imaginary part.
    if resolved_dtype is not None:
        phases = phases.astype(jnp.float32)
        if jnp.iscomplexobj(ratios):
            ratios = ratios.astype(jnp.complex64)
        else:
            ratios = ratios.astype(jnp.float32)
    if jnp.iscomplexobj(ratios):
        values = jnp.real(ratios) * jnp.cos(phases) - jnp.imag(ratios) * jnp.sin(phases)
    else:
        values = ratios * jnp.cos(phases)
    return values.astype(jnp.float64)


def _ratio_sample_values(
    encoder: Encoder,
    samples: jnp.ndarray,
    masks: jnp.ndarray,
    phases: jnp.ndarray,
    resolved_dtype: jnp.dtype | None,
) -> jnp.ndarray:
    """Return the float64 per-sample values ``Re[R exp(i Delta f)]``.

    Complex angle-encoder ratios are evaluated from their log-polar form as
    ``|R| cos(arg R + Delta f)``, one exponential and one cosine per entry
    instead of a complex exponential and complex product; that measured about
    a third faster.  It was slower for real ratios and the parity encoder, so
    every other encoder supplies ``R``.
    """
    if isinstance(encoder, AngleEncoder) and encoder._complex:
        log_magnitudes, arguments = encoder._log_amplitude_ratios(samples, masks)
        if resolved_dtype is not None:
            log_magnitudes = log_magnitudes.astype(jnp.float32)
            arguments = arguments.astype(jnp.float32)
            phases = phases.astype(jnp.float32)
        values = jnp.exp(log_magnitudes) * jnp.cos(arguments + phases)
        return values.astype(jnp.float64)
    return _sample_values(
        encoder.amplitude_ratios(samples, masks),
        phases,
        resolved_dtype,
    )


def _estimate(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
    gemm_dtype: jnp.dtype | None,
    estimator_dtype: jnp.dtype | None,
    return_samples: bool,
) -> ExpectationEstimate:
    resolved_dtype = resolve_estimator_dtype(gemm_dtype, estimator_dtype)
    phases = _phase_differences(
        params,
        generators,
        masks,
        samples,
        gemm_dtype,
        resolved_dtype,
    )
    sample_values = _ratio_sample_values(
        encoders,
        samples,
        masks,
        phases,
        resolved_dtype,
    )
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
    estimator_dtype: jnp.dtype | None = None,
    return_samples: bool = False,
) -> ExpectationEstimate:
    """Estimate Pauli-Z expectations for one or more conditions.

    Pass ``key`` with ``n_samples`` to draw fresh reference samples, or
    ``samples`` to reuse a ``(conditions, samples, qubits)`` draw; the draw
    must match the encoder's condition and qubit axes, because a size-1 axis
    would otherwise broadcast silently.  A rank-two ``(samples, qubits)`` draw
    is accepted only for a single-condition encoder, because sharing one draw
    across different reference states would bias every condition but the one
    it was sampled from.  ``masks`` has shape ``(n_masks, n_qubits)``.
    Visible-width masks are accepted and padded with identity on the ancilla
    positions.  The padding follows the visible-first convention of
    :mod:`iqp_cond.model`.

    ``gemm_dtype`` changes only the final phase contraction. Set
    ``estimator_dtype=jnp.float32`` to also construct parities and phases in
    float32 and evaluate the ratio-phase product in single precision. The
    estimator casts per-sample values back to float64 before statistics. The
    two dtype options are mutually exclusive.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    masks = jnp.asarray(masks, dtype=jnp.float64)
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
        estimator_dtype,
        return_samples,
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
    estimator_dtype: jnp.dtype | None = None,
    return_samples: bool = False,
) -> ExpectationEstimate:
    """Traceable estimator kernel with no array validation and model-width masks.

    The loss calls this function under ``jax.jit``. Precision-option validation
    is retained because dtype choices are static Python values.
    """
    drawn = _draw(encoders, key, samples, n_samples)
    return _estimate(
        params,
        generators,
        encoders,
        masks,
        drawn,
        gemm_dtype,
        estimator_dtype,
        return_samples,
    )
