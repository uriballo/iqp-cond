"""Condition-balanced multiscale Hamming-RBF MMD.

The kernel ``k(x, y) = exp(-d_H(x, y) / (2 sigma^2))`` has a product character
expansion, so the squared MMD is a weighted sum over Fourier masks whose
probability under each bandwidth is Bernoulli with
``p = (1 - exp(-1/(2 sigma^2))) / 2``.  The loss averages over conditions and
bandwidths, uses the unbiased ``(B-1)`` normalization for both target and model
moments, and consumes only quantities the encoder makes classically computable:
sampled masks, sampled bitstrings, and amplitude ratios.

Two entry points share one condition-level evaluator:

:func:`sampled_mmd`
    Draws nonidentity masks afresh from a JAX key, one draw per condition and
    bandwidth, exactly as ``iqpopt`` redraws its observables every call.  It
    needs no host loop but cannot collapse duplicate masks, so at equal draw
    budgets it evaluates more mask columns than :func:`banked_mmd`.
:func:`banked_mmd`
    Consumes a prebuilt compressed :class:`~iqp_cond.operators.OperatorBank`
    (duplicates collapsed to multiplicities).  The trainer builds one per step
    and passes it into the compiled step; this is also the evaluation and
    campaign-reproduction path, and the one the golden fixtures pin.

Both estimators skip the identity mask exactly: it contributes zero to the
kernel MMD, so spending draws on it only adds variance.  Mask law, samplers,
and banks live in :mod:`iqp_cond.operators`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .ancilla import pad_visible_masks
from .encoders import Encoder
from .estimator import estimate_pauli_z_from_samples
from .operators import OperatorBank, sample_nonidentity_masks_jax

__all__ = [
    "ConditionalMMDResult",
    "banked_mmd",
    "hamming_rbf",
    "sampled_mmd",
    "target_moments",
]


@dataclass(frozen=True)
class ConditionalMMDResult:
    """Conditional MMD estimate with condition- and bandwidth-level diagnostics."""

    loss: jnp.ndarray
    per_condition: jnp.ndarray
    per_bandwidth_condition: jnp.ndarray


jax.tree_util.register_dataclass(
    ConditionalMMDResult,
    data_fields=["loss", "per_condition", "per_bandwidth_condition"],
    meta_fields=[],
)


def hamming_rbf(x: jnp.ndarray, y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Evaluate ``exp(-d_H(x,y)/(2 sigma^2))``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return jnp.exp(-jnp.sum(jnp.not_equal(x, y), axis=-1) / (2.0 * sigma**2))


def target_moments(
    targets: jnp.ndarray,
    masks: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the target mean character and its unbiased squared mean.

    Shapes are ``(conditions, samples, qubits)`` for ``targets`` and
    ``(masks, qubits)`` for ``masks``.
    """
    target_characters = 1.0 - 2.0 * jnp.mod(
        jnp.einsum("btn,rn->btr", targets, masks),
        2.0,
    )
    n_targets = targets.shape[1]
    means = jnp.mean(target_characters, axis=1)
    if n_targets < 2:
        raise ValueError("each condition needs at least two target samples")
    sums = jnp.sum(target_characters, axis=1)
    unbiased_squares = (sums**2 - n_targets) / (n_targets * (n_targets - 1))
    return means, unbiased_squares


def _validate_loss_inputs(
    targets: jnp.ndarray,
    encoders: Encoder,
    n_samples: int,
) -> int:
    """Shared shape checks; returns the number of target (visible) qubits."""
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape (n_conditions, n_targets, n_visible)",
        )
    n_conditions, _, n_data_qubits = targets.shape
    if encoders.n_conditions != n_conditions:
        raise ValueError(
            f"encoder covers {encoders.n_conditions} conditions but targets "
            f"have {n_conditions}",
        )
    if encoders.n_qubits < n_data_qubits:
        raise ValueError(
            "encoder qubit count cannot be smaller than the target width",
        )
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    return n_data_qubits


def _condition_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoder: Encoder,
    target_condition: jnp.ndarray,
    visible_masks: jnp.ndarray,
    weights: jnp.ndarray,
    n_ancillas: int,
    n_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Weighted sum of per-mask squared moment differences for one condition.

    ``target_condition`` has shape ``(1, n_targets, n_visible)`` and
    ``visible_masks`` ``(n_masks, n_visible)``; the U-statistic corrections for
    both the model and target terms are the paper's finite-sample estimator.
    """
    samples = encoder.sample(key, n_samples)
    return _condition_mmd_from_samples(
        params,
        generators,
        encoder,
        target_condition,
        visible_masks,
        weights,
        n_ancillas,
        samples,
    )


def _condition_mmd_from_samples(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoder: Encoder,
    target_condition: jnp.ndarray,
    visible_masks: jnp.ndarray,
    weights: jnp.ndarray,
    n_ancillas: int,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    """Same weighted sum over one already-drawn reference sample set."""
    n_samples = samples.shape[1]
    model_masks = pad_visible_masks(visible_masks, n_ancillas)
    model = estimate_pauli_z_from_samples(
        params,
        generators,
        model_masks,
        encoder,
        samples,
    )
    target_means, target_squares = target_moments(target_condition, visible_masks)
    sums = jnp.sum(model.sample_values, axis=1)
    square_sums = jnp.sum(model.sample_values**2, axis=1)
    model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
    per_mask = model_squares - 2.0 * model.values * target_means + target_squares
    return jnp.sum(per_mask[0] * weights)


def sampled_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    targets: jnp.ndarray,
    key: jax.Array,
    *,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_samples: int,
) -> ConditionalMMDResult:
    """Estimate the multiscale squared MMD with masks drawn from ``key``.

    One nonidentity mask bank is drawn per condition and bandwidth at every
    call, so a fresh key per optimization step refreshes the objective exactly
    as ``iqpopt`` does.  Shapes are static, so the whole estimator traces into
    the compiled step; no bank is built on the host.  ``n_ops`` is the number
    of mask draws per entry.
    """
    targets = jnp.asarray(targets, dtype=jnp.float64)
    if not sigmas:
        raise ValueError("sigmas must not be empty")
    if n_ops < 1:
        raise ValueError("n_ops must be positive")
    n_data_qubits = _validate_loss_inputs(targets, encoders, n_samples)
    n_ancillas = encoders.n_qubits - n_data_qubits
    n_conditions = targets.shape[0]

    keys = jax.random.split(key, len(sigmas) * n_conditions * 2)
    bandwidth_values = []
    cursor = 0
    for sigma in sigmas:
        condition_values = []
        for condition in range(n_conditions):
            visible_masks, weights = sample_nonidentity_masks_jax(
                keys[cursor],
                n_ops,
                n_data_qubits,
                float(sigma),
            )
            cursor += 1
            condition_values.append(
                _condition_mmd(
                    params,
                    generators,
                    encoders.select(condition),
                    targets[condition : condition + 1],
                    visible_masks,
                    weights,
                    n_ancillas,
                    n_samples,
                    keys[cursor],
                ),
            )
            cursor += 1
        bandwidth_values.append(jnp.stack(condition_values))

    per_bandwidth_condition = jnp.stack(bandwidth_values)
    return ConditionalMMDResult(
        loss=jnp.mean(per_bandwidth_condition),
        per_condition=jnp.mean(per_bandwidth_condition, axis=0),
        per_bandwidth_condition=per_bandwidth_condition,
    )


def banked_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    targets: jnp.ndarray,
    bank: OperatorBank,
    n_samples: int,
    key: jax.Array,
    *,
    shared_samples: bool = False,
) -> ConditionalMMDResult:
    """Estimate the multiscale squared MMD with a fixed compressed operator bank.

    One condition is evaluated at a time, so the ``(samples, masks)`` buffer
    covers a single condition instead of the whole batch; combined with the
    compressed bank this keeps the fixed-bank evaluation inside a small memory
    budget and makes checkpoint selection comparable across steps.

    With ``shared_samples=True`` each condition draws one reference sample set
    and evaluates every bandwidth's masks on it, which is the paper's estimator
    (all masks see the same samples) and avoids resampling once per bandwidth.
    The default draws one sample set per (bandwidth, condition), the layout the
    published campaign and the golden fixtures pin.
    """
    targets = jnp.asarray(targets, dtype=jnp.float64)
    n_data_qubits = _validate_loss_inputs(targets, encoders, n_samples)
    n_conditions = targets.shape[0]
    if bank.n_conditions != n_conditions:
        raise ValueError(
            f"operator bank covers {bank.n_conditions} conditions but targets "
            f"have {n_conditions}",
        )
    if bank.n_visible != n_data_qubits:
        raise ValueError(
            f"operator bank is built for {bank.n_visible} visible qubits but "
            f"targets have {n_data_qubits}",
        )
    n_ancillas = encoders.n_qubits - n_data_qubits

    if shared_samples:
        bandwidth_values = []
        for condition in range(n_conditions):
            condition_encoder = encoders.select(condition)
            samples = condition_encoder.sample(
                jax.random.fold_in(key, condition),
                n_samples,
            )
            condition_values = [
                _condition_mmd_from_samples(
                    params,
                    generators,
                    condition_encoder,
                    targets[condition : condition + 1],
                    bank.entry(bandwidth, condition)[0],
                    bank.entry(bandwidth, condition)[1],
                    n_ancillas,
                    samples,
                )
                for bandwidth in range(bank.n_bandwidths)
            ]
            bandwidth_values.append(jnp.stack(condition_values))
        per_bandwidth_condition = jnp.stack(bandwidth_values, axis=1)
    else:
        keys = jax.random.split(key, bank.n_bandwidths * n_conditions)
        bandwidth_values = []
        cursor = 0
        for bandwidth in range(bank.n_bandwidths):
            condition_values = []
            for condition in range(n_conditions):
                visible_masks, weights = bank.entry(bandwidth, condition)
                condition_values.append(
                    _condition_mmd(
                        params,
                        generators,
                        encoders.select(condition),
                        targets[condition : condition + 1],
                        visible_masks,
                        weights,
                        n_ancillas,
                        n_samples,
                        keys[cursor],
                    ),
                )
                cursor += 1
            bandwidth_values.append(jnp.stack(condition_values))
        per_bandwidth_condition = jnp.stack(bandwidth_values)

    return ConditionalMMDResult(
        loss=jnp.mean(per_bandwidth_condition),
        per_condition=jnp.mean(per_bandwidth_condition, axis=0),
        per_bandwidth_condition=per_bandwidth_condition,
    )
