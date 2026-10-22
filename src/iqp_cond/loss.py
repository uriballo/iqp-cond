"""Condition-balanced multiscale Hamming-RBF MMD.

The kernel ``k(x, y) = exp(-d_H(x, y) / (2 sigma^2))`` has a product character
expansion, so the squared MMD is a weighted sum over Fourier masks whose
probability under each bandwidth is Bernoulli with
``p = (1 - exp(-1/(2 sigma^2))) / 2``.  The loss averages over conditions and
bandwidths and uses the unbiased ``(B-1)`` normalization for both target and
model moments.  The remaining inputs are sampled masks, sampled bitstrings,
and amplitude ratios, all classically computable by the encoder.

:func:`mmd` is the single objective; its masks come from exactly one of two
sources, which differ in cost and statistics, not in target:

``key``
    Draw nonidentity masks afresh, one draw per condition and bandwidth.  This
    path has no host loop, but it does not compress duplicates either, so at
    equal draw budgets it evaluates more mask columns.
``bank``
    Evaluate a prebuilt compressed
    :class:`~iqp_cond.operators.OperatorBank` with duplicates collapsed to
    multiplicities.  The regression fixtures pin this evaluation path.

Both sources skip the identity mask exactly.  It contributes zero to the kernel
MMD, so spending draws on it only adds variance.  See
:mod:`iqp_cond.operators` for the mask law, samplers, and banks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._layout import pad_visible_masks
from ._precision import resolve_estimator_dtype
from .encoders import Encoder
from .estimator import (
    _phase_differences,
    _sample_values,
    estimate_pauli_z_unchecked,
)
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
    """Validate shared shapes and return the number of target visible qubits.

    The checks read only static shapes, so they stay valid while the trainer
    traces ``mmd`` under ``jax.jit``.  The messages mirror the checked
    estimator's, because the loss calls the unchecked kernel.
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
    estimator_dtype: jnp.dtype | None,
    sample_chunk: int | None,
) -> jnp.ndarray:
    """Weighted sum of per-mask squared moment differences for one condition.

    ``target_condition`` has shape ``(1, n_targets, n_visible)`` and
    ``visible_masks`` ``(n_masks, n_visible)``; the U-statistic corrections for
    both the model and target terms use the unbiased finite-sample estimator.
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
                estimator_dtype=estimator_dtype,
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
            estimator_dtype=estimator_dtype,
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
    estimator_dtype: jnp.dtype | None = None,
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
    and evaluates every bandwidth's masks on it, whatever the mask source.
    This avoids resampling once per bandwidth.  The default resamples once per
    (bandwidth, condition) entry.

    ``estimator_dtype=jnp.float32`` selects the wider mixed-precision estimator
    and returns to float64 before the MMD reductions. It is mutually exclusive
    with the narrower ``gemm_dtype`` option.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    targets = jnp.asarray(targets, dtype=jnp.float64)
    estimator_dtype = resolve_estimator_dtype(gemm_dtype, estimator_dtype)
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
            estimator_dtype,
            sample_chunk,
        )

    assert sigmas is not None and n_ops is not None
    n_bandwidths = len(sigmas)

    def condition_values(
        encoder: Encoder,
        target: jnp.ndarray,
        samples: jnp.ndarray,
        masks_and_weights: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        visible_masks, weights = masks_and_weights
        return _condition_value(
            params,
            generators,
            encoder,
            target[None],
            visible_masks,
            weights,
            n_ancillas,
            samples,
            gemm_dtype,
            estimator_dtype,
            sample_chunk,
        )

    def fresh_masks(mask_key: jax.Array, sigma: float | jax.Array):
        return sample_nonidentity_masks_jax(mask_key, n_ops, n_data_qubits, sigma)

    if shared_samples:
        # One reference draw per condition covers every bandwidth's masks.
        keys = jax.random.split(key, n_bandwidths * n_conditions + n_conditions)
        mask_keys = keys[: n_bandwidths * n_conditions].reshape(
            (n_conditions, n_bandwidths, *keys.shape[1:]),
        )
        sample_keys = keys[n_bandwidths * n_conditions :]

        def shared_condition(
            encoder: Encoder,
            target: jnp.ndarray,
            condition_mask_keys: jax.Array,
            sample_key: jax.Array,
        ) -> jnp.ndarray:
            samples = encoder.sample(sample_key, n_samples)
            return jnp.stack(
                [
                    condition_values(
                        encoder,
                        target,
                        samples,
                        fresh_masks(condition_mask_keys[bandwidth], sigma),
                    )
                    for bandwidth, sigma in enumerate(sigmas)
                ],
            )

        per_condition = _per_condition(
            encoders,
            shared_condition,
            targets,
            mask_keys,
            sample_keys,
        )
        return _result(per_condition.T)

    keys = jax.random.split(key, max(n_bandwidths * n_conditions * 2, 1))
    keys = keys.reshape((n_bandwidths, n_conditions, 2, *keys.shape[1:]))
    bandwidth_values = []
    for bandwidth, sigma in enumerate(sigmas):

        def fresh_condition(
            encoder: Encoder,
            target: jnp.ndarray,
            entry_keys: jax.Array,
            sigma: float | jax.Array = sigma,
        ) -> jnp.ndarray:
            masks_and_weights = fresh_masks(entry_keys[0], sigma)
            samples = encoder.sample(entry_keys[1], n_samples)
            return condition_values(encoder, target, samples, masks_and_weights)

        bandwidth_values.append(
            _per_condition(encoders, fresh_condition, targets, keys[bandwidth]),
        )
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
    estimator_dtype: jnp.dtype | None,
    sample_chunk: int | None,
) -> ConditionalMMDResult:
    n_data_qubits = targets.shape[2]
    n_conditions = targets.shape[0]
    n_ancillas = encoders.n_qubits - n_data_qubits
    # Condition-major copies, so the per-condition map slices one bank entry
    # row per condition.
    condition_masks = jnp.swapaxes(bank.masks, 0, 1)
    condition_weights = jnp.swapaxes(bank.weights, 0, 1)

    def entry_value(
        encoder: Encoder,
        target: jnp.ndarray,
        samples: jnp.ndarray,
        masks: jnp.ndarray,
        weights: jnp.ndarray,
        bandwidth: int,
    ) -> jnp.ndarray:
        width = bank.width(bandwidth)
        return _condition_value(
            params,
            generators,
            encoder,
            target[None],
            masks[bandwidth, :width],
            weights[bandwidth, :width],
            n_ancillas,
            samples,
            gemm_dtype,
            estimator_dtype,
            sample_chunk,
        )

    if shared_samples:
        sample_keys = jnp.stack(
            [jax.random.fold_in(key, condition) for condition in range(n_conditions)],
        )

        def shared_condition(
            encoder: Encoder,
            target: jnp.ndarray,
            masks: jnp.ndarray,
            weights: jnp.ndarray,
            sample_key: jax.Array,
        ) -> jnp.ndarray:
            samples = encoder.sample(sample_key, n_samples)
            return jnp.stack(
                [
                    entry_value(encoder, target, samples, masks, weights, bandwidth)
                    for bandwidth in range(bank.n_bandwidths)
                ],
            )

        per_condition = _per_condition(
            encoders,
            shared_condition,
            targets,
            condition_masks,
            condition_weights,
            sample_keys,
        )
        return _result(per_condition.T)

    keys = jax.random.split(key, bank.n_bandwidths * n_conditions)
    keys = keys.reshape((bank.n_bandwidths, n_conditions, *keys.shape[1:]))
    bandwidth_values = []
    for bandwidth in range(bank.n_bandwidths):

        def bank_condition(
            encoder: Encoder,
            target: jnp.ndarray,
            masks: jnp.ndarray,
            weights: jnp.ndarray,
            sample_key: jax.Array,
            bandwidth: int = bandwidth,
        ) -> jnp.ndarray:
            samples = encoder.sample(sample_key, n_samples)
            return entry_value(encoder, target, samples, masks, weights, bandwidth)

        bandwidth_values.append(
            _per_condition(
                encoders,
                bank_condition,
                targets,
                condition_masks,
                condition_weights,
                keys[bandwidth],
            ),
        )
    return _result(jnp.stack(bandwidth_values))


# Largest cached ratio tensor a fixed-randomness plan may hold on the device.
_FIXED_PLAN_BYTES = 1 << 30


@dataclass(frozen=True)
class _FixedMMDPlan:
    """Parameter-independent inputs of :func:`mmd` for one fixed key.

    Each array has a leading entry axis that runs bandwidth-major over the
    ``(bandwidth, condition)`` entries.  Masks are model-width, samples and
    ratios keep the encoder's size-1 condition axis, and the target moments
    are evaluated once on the visible masks.
    """

    masks: jnp.ndarray
    weights: jnp.ndarray
    samples: jnp.ndarray
    ratios: jnp.ndarray
    target_means: jnp.ndarray
    target_squares: jnp.ndarray
    n_bandwidths: int
    n_conditions: int


jax.tree_util.register_dataclass(
    _FixedMMDPlan,
    data_fields=[
        "masks",
        "weights",
        "samples",
        "ratios",
        "target_means",
        "target_squares",
    ],
    meta_fields=["n_bandwidths", "n_conditions"],
)


def _fixed_mmd_plan(
    encoders: Encoder,
    targets: jnp.ndarray,
    key: jax.Array,
    *,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_samples: int,
) -> _FixedMMDPlan | None:
    """Precompute every parameter-independent input of a fresh-mask ``mmd``.

    With a fixed key, :func:`mmd` redraws the same masks and reference samples
    and recomputes the same amplitude ratios and target moments on every call;
    only the phases depend on the parameters.  The plan reproduces the
    fresh-mask, per-entry key layout of :func:`mmd`, so :func:`_fixed_mmd`
    estimates the same quantity.  It returns ``None`` when the cached ratios
    would exceed ``_FIXED_PLAN_BYTES``.
    """
    targets = jnp.asarray(targets, dtype=jnp.float64)
    n_conditions, _, n_visible = targets.shape
    n_ancillas = encoders.n_qubits - n_visible
    n_bandwidths = len(sigmas)
    ratio_bytes = n_bandwidths * n_conditions * n_samples * n_ops * 16
    if ratio_bytes > _FIXED_PLAN_BYTES:
        return None
    keys = jax.random.split(key, max(n_bandwidths * n_conditions * 2, 1))
    keys = keys.reshape((n_bandwidths, n_conditions, 2, *keys.shape[1:]))
    entries = []
    for bandwidth, sigma in enumerate(sigmas):
        for condition in range(n_conditions):
            visible_masks, weights = sample_nonidentity_masks_jax(
                keys[bandwidth, condition, 0],
                n_ops,
                n_visible,
                float(sigma),
            )
            encoder = encoders.select(condition)
            samples = encoder.sample(keys[bandwidth, condition, 1], n_samples)
            masks = pad_visible_masks(visible_masks, n_ancillas)
            target_means, target_squares = target_moments(
                targets[condition : condition + 1],
                visible_masks,
            )
            entries.append(
                (
                    masks,
                    weights,
                    samples,
                    encoder.amplitude_ratios(samples, masks),
                    target_means[0],
                    target_squares[0],
                ),
            )
    stacked = [jnp.stack(column) for column in zip(*entries, strict=True)]
    return _FixedMMDPlan(*stacked, n_bandwidths, n_conditions)


def _fixed_mmd(
    plan: _FixedMMDPlan,
    params: jnp.ndarray,
    generators: jnp.ndarray,
    *,
    gemm_dtype: jnp.dtype | None = None,
    estimator_dtype: jnp.dtype | None = None,
) -> ConditionalMMDResult:
    """Evaluate a :class:`_FixedMMDPlan` at ``params``.

    Bandwidths run one at a time and their conditions are vectorized, so the
    live phase tensor covers one bandwidth's conditions.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    resolved_dtype = resolve_estimator_dtype(gemm_dtype, estimator_dtype)

    def entry_value(
        inputs: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        masks, weights, samples, ratios, target_means, target_squares = inputs
        phases = _phase_differences(
            params,
            generators,
            masks,
            samples,
            gemm_dtype,
            resolved_dtype,
        )
        sample_values = _sample_values(ratios, phases, resolved_dtype)
        n_samples = samples.shape[1]
        values = jnp.mean(sample_values, axis=1)[0]
        sums = jnp.sum(sample_values, axis=1)[0]
        square_sums = jnp.sum(sample_values**2, axis=1)[0]
        model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
        per_mask = model_squares - 2.0 * values * target_means + target_squares
        return jnp.sum(per_mask * weights)

    entries = (
        plan.masks,
        plan.weights,
        plan.samples,
        plan.ratios,
        plan.target_means,
        plan.target_squares,
    )
    by_bandwidth = [
        entry.reshape(plan.n_bandwidths, plan.n_conditions, *entry.shape[1:])
        for entry in entries
    ]
    values = jnp.stack(
        [
            jax.vmap(lambda *inputs: entry_value(inputs))(
                *(entry[bandwidth] for entry in by_bandwidth),
            )
            for bandwidth in range(plan.n_bandwidths)
        ],
    )
    return _result(values)


def _pytree_encoder_signature(
    encoder: object,
) -> tuple[object, tuple[tuple[tuple[int, ...], str], ...]] | None:
    """Return a stable JAX signature, or ``None`` for opaque encoders.

    Only array leaves are accepted, because passing an arbitrary Python object
    through a JAX transformation would fail during tracing.  The fallback
    matters for user-defined encoders that are deliberately not registered as
    pytrees.
    """
    leaves, structure = jax.tree_util.tree_flatten(encoder)
    if not leaves:
        return None
    signatures = []
    for leaf in leaves:
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            return None
        signatures.append((tuple(leaf.shape), np.dtype(leaf.dtype).str))
    return structure, tuple(signatures)


def _dynamic_encoder_support(encoders: Encoder) -> bool:
    """Whether every selected encoder shares one pytree signature.

    Such encoders can be stacked and traced once for all conditions instead
    of once per condition.
    """
    signatures = [
        _pytree_encoder_signature(encoders.select(i))
        for i in range(encoders.n_conditions)
    ]
    return signatures[0] is not None and all(
        signature == signatures[0] for signature in signatures[1:]
    )


def _per_condition(
    encoders: Encoder,
    body: Callable[..., jnp.ndarray],
    *per_condition_inputs: jnp.ndarray,
) -> jnp.ndarray:
    """Stack ``body(encoder_c, *inputs[c])`` over the conditions.

    Encoders whose selections share one pytree signature are stacked and
    vectorized with ``vmap``, so the program is traced once whatever the
    condition count and the conditions run in parallel.  The product
    encoder's ratio contraction keeps one condition's live buffer at
    ``(samples, masks)``, so the batch stays below the former
    ``(samples, masks, qubits)`` buffer of a single condition whenever the
    conditions do not outnumber the qubits.  Opaque encoders fall back to a
    Python loop.
    """
    n_conditions = encoders.n_conditions
    selected = [encoders.select(condition) for condition in range(n_conditions)]
    if n_conditions == 1 or not _dynamic_encoder_support(encoders):
        return jnp.stack(
            [
                body(encoder, *(values[condition] for values in per_condition_inputs))
                for condition, encoder in enumerate(selected)
            ],
        )
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *selected)
    return jax.vmap(body)(stacked, *per_condition_inputs)


def _result(per_bandwidth_condition: jnp.ndarray) -> ConditionalMMDResult:
    return ConditionalMMDResult(
        loss=jnp.mean(per_bandwidth_condition),
        per_condition=jnp.mean(per_bandwidth_condition, axis=0),
        per_bandwidth_condition=per_bandwidth_condition,
    )
