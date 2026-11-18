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
from ._precision import Precision, _Arithmetic
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
    arithmetic: _Arithmetic,
) -> jnp.ndarray:
    """Return ``Delta_a f_theta(z)`` for every sample and mask.

    Parities and characters are exact integers, built in the contraction's
    dtype; the contraction rounds its inputs and accumulates in the GEMM
    knobs' formats, and the phases are stored in the ``phases`` format.
    """
    dtype = arithmetic.dtype("gemm_accumulate")
    samples = samples.astype(dtype)
    generators = generators.astype(dtype)
    masks = masks.astype(dtype)
    params = params.astype(dtype)
    # Flattening the leading (condition, sample) axes keeps both contractions
    # plain matrix products; under the loss's condition vmap, batched einsums
    # with the unit condition axis measured about a quarter slower on CPU.
    leading = samples.shape[:-1]
    sample_parities = samples.reshape(-1, samples.shape[-1]) @ generators.T
    sample_characters = 1.0 - 2.0 * jnp.mod(sample_parities, 2.0)
    op_parities = jnp.mod(masks @ generators.T, 2.0)
    weighted_flips = -2.0 * op_parities * params[None, :]
    phases = arithmetic.gemm(sample_characters, weighted_flips)
    phases = arithmetic.round(phases, "phases")
    return phases.reshape(*leading, masks.shape[0])


def _trig(phases: jnp.ndarray, arithmetic: _Arithmetic, function) -> jnp.ndarray:
    """Evaluate ``function`` on the phases and store it in the ``phases`` format."""
    return arithmetic.round(function(phases), "phases")


def _sample_values(
    ratios: jnp.ndarray,
    phases: jnp.ndarray,
    arithmetic: _Arithmetic,
) -> jnp.ndarray:
    """Return the per-sample values ``Re[R exp(i Delta f)]``.

    The ratios are rounded to the ``ratios`` format, the cosine and sine to the
    ``phases`` format, and the values to the ``statistics`` format.
    """
    # Re[R exp(i phi)] = Re R cos phi - Im R sin phi, which skips the complex
    # exponential and, for real ratios, the sine and the imaginary part.
    if jnp.iscomplexobj(ratios):
        real = arithmetic.round(jnp.real(ratios), "ratios")
        imaginary = arithmetic.round(jnp.imag(ratios), "ratios")
        values = real * _trig(phases, arithmetic, jnp.cos) - imaginary * _trig(
            phases, arithmetic, jnp.sin
        )
    else:
        values = arithmetic.round(ratios, "ratios") * _trig(phases, arithmetic, jnp.cos)
    return arithmetic.round(values, "statistics")


def _ratio_sample_values(
    encoder: Encoder,
    samples: jnp.ndarray,
    masks: jnp.ndarray,
    phases: jnp.ndarray,
    arithmetic: _Arithmetic,
) -> jnp.ndarray:
    """Return the per-sample values ``Re[R exp(i Delta f)]``.

    Complex angle-encoder ratios are evaluated from their log-polar form as
    ``|R| cos(arg R + Delta f)``, one exponential and one cosine per entry
    instead of a complex exponential and complex product; that measured about
    a third faster.  It was slower for real ratios and the parity encoder, so
    every other encoder supplies ``R``.  Below fp64 ``ratios``, the encoder's
    floating leaves, the samples and the masks are cast to float32 first, so
    the query runs in float32 before its outputs are rounded.
    """
    if arithmetic.dtype("ratios") != jnp.dtype(jnp.float64):

        def cast(leaf: jnp.ndarray) -> jnp.ndarray:
            if not jnp.issubdtype(leaf.dtype, jnp.inexact):
                return leaf
            return leaf.astype(jnp.complex64 if jnp.iscomplexobj(leaf) else jnp.float32)

        encoder = jax.tree_util.tree_map(cast, encoder)
        samples = samples.astype(jnp.float32)
        masks = masks.astype(jnp.float32)
    if isinstance(encoder, AngleEncoder) and encoder._complex:
        log_magnitudes, arguments = encoder._log_amplitude_ratios(samples, masks)
        magnitudes = arithmetic.round(jnp.exp(log_magnitudes), "ratios")
        angles = arithmetic.round(
            arithmetic.round(arguments, "ratios") + phases, "phases"
        )
        values = magnitudes * _trig(angles, arithmetic, jnp.cos)
        return arithmetic.round(values, "statistics")
    return _sample_values(encoder.amplitude_ratios(samples, masks), phases, arithmetic)


def _gradient_terms(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoder: Encoder,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one condition's per-sample, per-mask gradient terms in float64.

    ``encoder`` covers one condition, ``samples`` has shape ``(S, qubits)`` and
    visible-width ``masks`` are padded with identity on the ancillas.  The
    values ``v = Re[R exp(i phi)]`` have the derivative
    ``d v / d theta = s C * F`` with ``s = 2 Im[R exp(i phi)]``, the sample
    characters ``C`` of shape ``(S, generators)`` and the mask flips ``F`` of
    shape ``(R, generators)``; ``v`` and ``s`` have shape ``(S, R)``.
    """
    masks = pad_visible_masks(masks, encoder.n_qubits - masks.shape[1])
    characters = 1.0 - 2.0 * jnp.mod(samples @ generators.T, 2.0)
    flips = jnp.mod(masks @ generators.T, 2.0)
    phases = characters @ (-2.0 * flips * params[None, :]).T
    terms = encoder.amplitude_ratios(samples[None], masks)[0] * jnp.exp(1j * phases)
    return jnp.real(terms), 2.0 * jnp.imag(terms), characters, flips


def _estimate(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
    precision: str | Precision,
    return_samples: bool,
) -> ExpectationEstimate:
    arithmetic = _Arithmetic(precision)
    phases = _phase_differences(params, generators, masks, samples, arithmetic)
    sample_values = _ratio_sample_values(encoders, samples, masks, phases, arithmetic)
    if arithmetic.precision.statistics == "fp64":
        values = jnp.mean(sample_values, axis=1)
    else:
        values = arithmetic.round(
            arithmetic.sum(sample_values, axis=1) / sample_values.shape[1],
            "statistics",
        )
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
    precision: str | Precision = "fp64",
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

    ``precision`` is a :class:`iqp_cond.Precision` or the name of a preset in
    :data:`iqp_cond.PRECISIONS`; values come back in float64 when the
    statistics are fp64 and in float32 otherwise.
    """
    _Arithmetic(precision)
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
        precision,
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
    precision: str | Precision = "fp64",
    return_samples: bool = False,
) -> ExpectationEstimate:
    """Traceable estimator kernel with no array validation and model-width masks.

    Callable under ``jax.jit``. The precision name is still validated because
    it is a static Python value.
    """
    drawn = _draw(encoders, key, samples, n_samples)
    return _estimate(
        params,
        generators,
        encoders,
        masks,
        drawn,
        precision,
        return_samples,
    )
