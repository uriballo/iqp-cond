"""Monte Carlo Pauli-Z estimation for conditional IQP circuits.

The estimator is the paper's conditional Pauli estimator: for a Fourier mask
``a`` it averages ``Re[R_c(z, a) exp(i Delta_a f_theta(z))]`` over basis samples
``z`` drawn from the condition state, where ``R_c(z, a) = phi(z xor a)/phi(z)``
comes from the encoder and ``Delta_a f`` from the diagonal phase bank.  Both
pieces are classically computable, so the estimator is unbiased with bounded
variance and its parameter gradient is available in closed form.

Two entry points share one kernel.  :func:`estimate_pauli_z` validates its
inputs and pads visible-width masks with identity on the ancillas;
:func:`estimate_pauli_z_unchecked` skips both, is safe to trace under
``jax.jit``, and is what the loss calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .ancilla import pad_visible_masks
from .encoders import Encoder

__all__ = [
    "ExpectationEstimate",
    "estimate_pauli_z",
    "estimate_pauli_z_from_samples",
    "estimate_pauli_z_unchecked",
]


@dataclass(frozen=True)
class ExpectationEstimate:
    """Pauli-Z estimates and per-sample values for every condition and mask.

    ``values`` has shape ``(n_conditions, n_masks)`` and ``sample_values``
    ``(n_conditions, n_samples, n_masks)``.  ``standard_errors`` is derived on
    demand from ``sample_values``; training never reads it, so computing it
    eagerly would tax every step.
    """

    values: jnp.ndarray
    sample_values: jnp.ndarray
    n_samples: int

    @property
    def standard_errors(self) -> jnp.ndarray:
        """Monte Carlo standard error of each ``values`` entry."""
        return jnp.std(self.sample_values, axis=1, ddof=1) / jnp.sqrt(self.n_samples)


jax.tree_util.register_dataclass(
    ExpectationEstimate,
    data_fields=["values", "sample_values"],
    meta_fields=["n_samples"],
)


def _validate_binary(name: str, values: jnp.ndarray) -> None:
    if not bool(jnp.all((values == 0) | (values == 1))):
        raise ValueError(f"{name} must contain only zeros and ones")


def _phase_differences(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``Delta_a f_theta(z)`` for every sample and mask."""
    sample_parities = jnp.einsum("bsn,mn->bsm", samples, generators)
    sample_characters = 1.0 - 2.0 * jnp.mod(sample_parities, 2.0)
    op_parities = jnp.mod(jnp.einsum("rn,mn->rm", masks, generators), 2.0)
    weighted_flips = -2.0 * op_parities * params[None, :]
    return jnp.einsum("bsm,rm->bsr", sample_characters, weighted_flips)


def estimate_pauli_z(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    encoders: Encoder,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    """Estimate Pauli-Z expectations for one or more conditions.

    ``masks`` has shape ``(n_masks, n_qubits)``.  Visible-width masks are
    accepted and padded with identity on the ancilla positions, following the
    visible-first convention of :mod:`iqp_cond.ancilla`; model-width masks pass
    through unchanged.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    masks = jnp.asarray(masks, dtype=jnp.float64)
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
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if encoders.n_qubits != generators.shape[1]:
        raise ValueError(
            f"encoder acts on {encoders.n_qubits} qubits but generators "
            f"cover {generators.shape[1]}",
        )
    if masks.shape[1] < encoders.n_qubits:
        masks = pad_visible_masks(masks, encoders.n_qubits - masks.shape[1])
    _validate_binary("generators", generators)
    _validate_binary("masks", masks)

    return estimate_pauli_z_unchecked(
        params,
        generators,
        masks,
        encoders,
        key,
        n_samples,
    )


def estimate_pauli_z_unchecked(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    encoders: Encoder,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    """Unchecked estimator kernel, safe to trace under ``jax.jit``.

    Skips the input validation and the visible-mask padding of
    :func:`estimate_pauli_z`: ``masks`` must already cover the model register
    and the caller is responsible for the array shapes.  The loss uses this
    entry point because Python-level checks cannot run on tracers.
    """
    samples = encoders.sample(key, n_samples)
    return estimate_pauli_z_from_samples(
        params,
        generators,
        masks,
        encoders,
        samples,
    )


def estimate_pauli_z_from_samples(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    encoders: Encoder,
    samples: jnp.ndarray,
) -> ExpectationEstimate:
    """Estimator kernel over pre-drawn reference samples.

    Lets several mask banks share one reference draw per condition — the
    paper's estimator evaluates every mask on the same samples, and sharing
    avoids resampling (and re-sorting, for radial encoders) once per bandwidth.
    ``samples`` is ``(n_conditions, n_samples, n_qubits)`` as produced by
    :meth:`Encoder.sample`.
    """
    ratios = encoders.amplitude_ratios(samples, masks)
    phases = _phase_differences(params, generators, masks, samples)
    sample_values = jnp.real(ratios * jnp.exp(1j * phases))
    values = jnp.mean(sample_values, axis=1)
    return ExpectationEstimate(values, sample_values, samples.shape[1])
