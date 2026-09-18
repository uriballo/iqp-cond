"""Condition-balanced multiscale Hamming-RBF MMD.

The kernel ``k(x, y) = exp(-d_H(x, y) / (2 sigma^2))`` has a product character
expansion, so the squared MMD is a weighted sum over Fourier masks whose
probability under each bandwidth is Bernoulli with
``p = (1 - exp(-1/(2 sigma^2))) / 2``.  The loss averages over conditions and
bandwidths, uses the unbiased ``(B-1)`` normalization for both target and model
moments, and needs only the quantities the encoder makes classically
computable: sampled masks, sampled bitstrings, and amplitude ratios.

:func:`mmd` is the single objective; its masks come from exactly one of two
sources, which differ in cost and statistics, not in target:

``key``
    Draw nonidentity masks afresh, one draw per condition and bandwidth, exactly
    as ``iqpopt`` redraws its observables every call.  This path has no host
    loop, but it does not compress duplicates either, so at equal draw budgets
    it evaluates more mask columns.
``bank``
    Evaluate a prebuilt compressed
    :class:`~iqp_cond.operators.OperatorBank` with duplicates collapsed to
    multiplicities.  This is the evaluation and paper-reproduction path, and the
    one the golden fixtures pin.

Both sources skip the identity mask exactly.  It contributes zero to the kernel
MMD, so spending draws on it only adds variance.  Mask law, samplers, and banks
are in :mod:`iqp_cond.operators`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import pad_visible_masks
from .encoders import Encoder, select_encoder
from .estimator import estimate_pauli_z_unchecked
from .operators import OperatorBank, sample_nonidentity_masks_jax

__all__ = [
    "ConditionalMMDResult",
    "hamming_rbf",
    "mmd",
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
    params: jnp.ndarray,
    generators: jnp.ndarray,
    targets: jnp.ndarray,
    encoders: Encoder,
    n_samples: int,
) -> int:
    """Shared shape checks; returns the number of target visible qubits.

    The function inspects only static properties, so the checks stay valid
    while the trainer traces ``mmd`` under ``jax.jit``.  The messages mirror
    the checked estimator's, because the loss itself must call the unchecked
    kernel.
    """
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape (n_conditions, n_targets, n_visible)",
        )
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if encoders.n_qubits != generators.shape[1]:
        raise ValueError(
            f"encoder acts on {encoders.n_qubits} qubits but generators "
            f"cover {generators.shape[1]}",
        )
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
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


def _condition_value(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoder: Encoder,
    target_condition: jnp.ndarray,
    visible_masks: jnp.ndarray,
    weights: jnp.ndarray,
    n_ancillas: int,
    samples: jnp.ndarray,
    gemm_dtype: jnp.dtype | None,
    sample_chunk: int | None,
) -> jnp.ndarray:
    """Weighted sum of per-mask squared moment differences for one condition.

    ``target_condition`` has shape ``(1, n_targets, n_visible)`` and
    ``visible_masks`` ``(n_masks, n_visible)``; the U-statistic corrections for
    both the model and target terms come from the paper's finite-sample
    estimator.
    ``sample_chunk`` streams the sample axis and accumulates the per-mask
    moments, bounding the live ``(samples, masks, qubits)`` ratio tensor.
    """
    n_samples = samples.shape[1]
    model_masks = pad_visible_masks(visible_masks, n_ancillas)
    if sample_chunk is not None and 0 < sample_chunk < n_samples:
        sums = None
        square_sums = None
        for start in range(0, n_samples, sample_chunk):
            chunk = estimate_pauli_z_unchecked(
                params,
                generators,
                encoder,
                model_masks,
                samples=samples[:, start : start + sample_chunk],
                gemm_dtype=gemm_dtype,
                return_samples=True,
            ).sample_values
            assert chunk is not None
            chunk_sums = jnp.sum(chunk, axis=1)
            chunk_squares = jnp.sum(chunk**2, axis=1)
            sums = chunk_sums if sums is None else sums + chunk_sums
            square_sums = (
                chunk_squares if square_sums is None else square_sums + chunk_squares
            )
        assert sums is not None
        assert square_sums is not None
        values = sums / n_samples
    else:
        model = estimate_pauli_z_unchecked(
            params,
            generators,
            encoder,
            model_masks,
            samples=samples,
            gemm_dtype=gemm_dtype,
            return_samples=True,
        )
        assert model.sample_values is not None
        values = model.values
        sums = jnp.sum(model.sample_values, axis=1)
        square_sums = jnp.sum(model.sample_values**2, axis=1)
    target_means, target_squares = target_moments(target_condition, visible_masks)
    model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
    per_mask = model_squares - 2.0 * values * target_means + target_squares
    return jnp.sum(per_mask[0] * weights)


def mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
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
    """Estimate the condition-balanced multiscale squared MMD.

    ``key`` always supplies the reference-state randomness.  Pass ``sigmas``
    and ``n_ops`` to draw fresh nonidentity masks from that key, or pass
    ``bank`` to evaluate a prebuilt compressed bank instead; the two mask
    sources are exclusive.  The function evaluates one condition at a time, so
    the sample/mask buffer covers a single condition instead of the whole
    batch.

    With ``shared_samples=True`` each condition draws one reference sample set
    and evaluates every bandwidth's masks on it, regardless of the mask source;
    this is the paper's estimator form and avoids resampling once per
    bandwidth.  The default resamples per (bandwidth, condition) entry.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    targets = jnp.asarray(targets, dtype=jnp.float64)
    if sample_chunk is not None and sample_chunk < 1:
        raise ValueError("sample_chunk must be positive")
    if bank is None:
        if not sigmas:
            raise ValueError(
                "pass sigmas and n_ops for fresh masks, or a prebuilt bank",
            )
        if n_ops is None or n_ops < 1:
            raise ValueError("n_ops must be positive")
    elif sigmas is not None or n_ops is not None:
        raise ValueError("sigmas and n_ops belong to the key mask source only")
    n_data_qubits = _validate_loss_inputs(
        params,
        generators,
        targets,
        encoders,
        n_samples,
    )
    n_conditions = targets.shape[0]
    n_ancillas = encoders.n_qubits - n_data_qubits

    if bank is not None:
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
        return _mmd_from_bank(
            params,
            generators,
            encoders,
            targets,
            bank,
            key,
            n_samples,
            shared_samples,
            gemm_dtype,
            sample_chunk,
        )

    assert sigmas is not None and n_ops is not None
    n_bandwidths = len(sigmas)
    if shared_samples:
        # The paper's estimator form. One reference draw per condition covers
        # every bandwidth's masks.
        keys = jax.random.split(key, n_bandwidths * n_conditions + n_conditions)
        mask_keys = keys[: n_bandwidths * n_conditions]
        sample_keys = keys[n_bandwidths * n_conditions :]
        bandwidth_values = []
        cursor = 0
        for condition in range(n_conditions):
            encoder = select_encoder(encoders, condition)
            samples = encoder.sample(sample_keys[condition], n_samples)
            condition_values = []
            for sigma in sigmas:
                visible_masks, weights = sample_nonidentity_masks_jax(
                    mask_keys[cursor],
                    n_ops,
                    n_data_qubits,
                    float(sigma),
                )
                cursor += 1
                condition_values.append(
                    _condition_value(
                        params,
                        generators,
                        encoder,
                        targets[condition : condition + 1],
                        visible_masks,
                        weights,
                        n_ancillas,
                        samples,
                        gemm_dtype,
                        sample_chunk,
                    ),
                )
            bandwidth_values.append(jnp.stack(condition_values))
        return _result(jnp.stack(bandwidth_values, axis=1))

    n_keys = n_bandwidths * n_conditions * 2
    keys = jax.random.split(key, max(n_keys, 1))
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
            encoder = select_encoder(encoders, condition)
            samples = encoder.sample(keys[cursor], n_samples)
            cursor += 1
            condition_values.append(
                _condition_value(
                    params,
                    generators,
                    encoder,
                    targets[condition : condition + 1],
                    visible_masks,
                    weights,
                    n_ancillas,
                    samples,
                    gemm_dtype,
                    sample_chunk,
                ),
            )
        bandwidth_values.append(jnp.stack(condition_values))
    return _result(jnp.stack(bandwidth_values))


def _mmd_from_bank(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    targets: jnp.ndarray,
    bank: OperatorBank,
    key: jax.Array,
    n_samples: int,
    shared_samples: bool,
    gemm_dtype: jnp.dtype | None,
    sample_chunk: int | None,
) -> ConditionalMMDResult:
    n_data_qubits = targets.shape[2]
    n_conditions = targets.shape[0]
    n_ancillas = encoders.n_qubits - n_data_qubits

    if shared_samples:
        bandwidth_values = []
        for condition in range(n_conditions):
            encoder = select_encoder(encoders, condition)
            samples = encoder.sample(
                jax.random.fold_in(key, condition),
                n_samples,
            )
            condition_values = [
                _condition_value(
                    params,
                    generators,
                    encoder,
                    targets[condition : condition + 1],
                    bank.entry(bandwidth, condition)[0],
                    bank.entry(bandwidth, condition)[1],
                    n_ancillas,
                    samples,
                    gemm_dtype,
                    sample_chunk,
                )
                for bandwidth in range(bank.n_bandwidths)
            ]
            bandwidth_values.append(jnp.stack(condition_values))
        return _result(jnp.stack(bandwidth_values, axis=1))

    keys = jax.random.split(key, bank.n_bandwidths * n_conditions)
    bandwidth_values = []
    cursor = 0
    for bandwidth in range(bank.n_bandwidths):
        condition_values = []
        for condition in range(n_conditions):
            encoder = select_encoder(encoders, condition)
            samples = encoder.sample(keys[cursor], n_samples)
            cursor += 1
            visible_masks, weights = bank.entry(bandwidth, condition)
            condition_values.append(
                _condition_value(
                    params,
                    generators,
                    encoder,
                    targets[condition : condition + 1],
                    visible_masks,
                    weights,
                    n_ancillas,
                    samples,
                    gemm_dtype,
                    sample_chunk,
                ),
            )
        bandwidth_values.append(jnp.stack(condition_values))
    return _result(jnp.stack(bandwidth_values))


def _result(per_bandwidth_condition: jnp.ndarray) -> ConditionalMMDResult:
    return ConditionalMMDResult(
        loss=jnp.mean(per_bandwidth_condition),
        per_condition=jnp.mean(per_bandwidth_condition, axis=0),
        per_bandwidth_condition=per_bandwidth_condition,
    )
