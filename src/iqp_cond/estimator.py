"""Monte Carlo Pauli-Z estimation for conditional IQP circuits.

The estimator is the paper's conditional Pauli estimator: for a Fourier mask
``a`` it averages ``Re[R_c(z, a) exp(i Delta_a f_theta(z))]`` over basis samples
``z`` drawn from the condition state, where ``R_c(z, a) = phi(z xor a)/phi(z)``
comes from the reference and ``Delta_a f`` from the diagonal phase bank.  Both
pieces are classically computable, so the estimator is unbiased with bounded
variance and its parameter gradient is available in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .references import ReferenceStates

__all__ = ["ExpectationEstimate", "estimate_pauli_z"]


@dataclass(frozen=True)
class ExpectationEstimate:
    """Pauli-Z estimates and uncertainty for every condition and observable."""

    values: jnp.ndarray
    standard_errors: jnp.ndarray
    sample_values: jnp.ndarray


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
    references: ReferenceStates,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    """Estimate Pauli-Z expectations for one or more conditions.

    ``masks`` has shape ``(n_masks, n_qubits)`` and must cover the whole model
    register; visible-only masks are padded with
    :func:`iqp_cond.ancilla.pad_visible_masks` first.
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
    if masks.ndim != 2 or masks.shape[1] != generators.shape[1]:
        raise ValueError("masks must have shape (n_masks, n_qubits)")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if references.n_qubits != generators.shape[1]:
        raise ValueError(
            f"reference acts on {references.n_qubits} qubits but generators "
            f"cover {generators.shape[1]}",
        )
    _validate_binary("generators", generators)
    _validate_binary("masks", masks)

    return _estimate_pauli_z(params, generators, masks, references, key, n_samples)


def _estimate_pauli_z(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    masks: jnp.ndarray,
    references: ReferenceStates,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    """Unchecked estimator kernel, safe to trace under ``jax.jit``."""
    samples = references.sample(key, n_samples)
    ratios = references.amplitude_ratios(samples, masks)
    phases = _phase_differences(params, generators, masks, samples)
    sample_values = jnp.real(ratios * jnp.exp(1j * phases))
    values = jnp.mean(sample_values, axis=1)
    standard_errors = jnp.std(sample_values, axis=1, ddof=1) / jnp.sqrt(n_samples)
    return ExpectationEstimate(values, standard_errors, sample_values)
