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

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._precision import resolve_precision
from .encoders import Encoder
from .encoders.base import _check_condition_pytree, _condition_rows
from .estimator import _phase_differences, _ratio_sample_values
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
    estimator's.
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


@dataclass(frozen=True)
class _Entries:
    """One bandwidth's parameter-independent MMD inputs, one row per condition.

    ``masks`` are model-width ``(C, M, n)`` with matching ``(C, M)`` weights
    and target moments; ``samples`` are ``(C, 1, S, n)``, keeping the size-1
    condition axis of each condition's encoder.
    """

    masks: jnp.ndarray
    weights: jnp.ndarray
    samples: jnp.ndarray
    target_means: jnp.ndarray
    target_squares: jnp.ndarray


jax.tree_util.register_dataclass(
    _Entries,
    data_fields=[
        "masks",
        "weights",
        "samples",
        "target_means",
        "target_squares",
    ],
    meta_fields=[],
)


def _mmd_entries(
    encoders: Encoder,
    targets: jnp.ndarray,
    key: jax.Array,
    *,
    bank: OperatorBank | None,
    sigmas: tuple[float, ...] | None,
    n_ops: int | None,
    n_samples: int,
    shared_samples: bool,
) -> tuple[_Entries, ...]:
    """Draw every parameter-independent input of :func:`mmd`, per bandwidth.

    Fresh masks use one key per (bandwidth, condition) entry; a bank supplies
    its own entries.  Reference samples come from one key per entry, or with
    ``shared_samples`` from one key per condition reused by every bandwidth.
    """
    rows = _condition_rows(encoders)
    n_conditions, _, n_visible = targets.shape
    n_ancillas = encoders.n_qubits - n_visible
    if bank is None:
        assert sigmas is not None and n_ops is not None
        n_bandwidths = len(sigmas)
        if shared_samples:
            keys = jax.random.split(key, n_bandwidths * n_conditions + n_conditions)
            mask_keys = jnp.swapaxes(
                keys[: n_bandwidths * n_conditions].reshape(
                    (n_conditions, n_bandwidths, *keys.shape[1:]),
                ),
                0,
                1,
            )
            sample_keys = keys[n_bandwidths * n_conditions :]
        else:
            keys = jax.random.split(key, max(n_bandwidths * n_conditions * 2, 1))
            keys = keys.reshape((n_bandwidths, n_conditions, 2, *keys.shape[1:]))
            mask_keys, sample_keys = keys[:, :, 0], keys[:, :, 1]
        mask_sets = [
            jax.vmap(
                lambda mask_key, sigma=sigma: sample_nonidentity_masks_jax(
                    mask_key,
                    n_ops,
                    n_visible,
                    sigma,
                ),
            )(mask_keys[bandwidth])
            for bandwidth, sigma in enumerate(sigmas)
        ]
    else:
        n_bandwidths = bank.n_bandwidths
        if shared_samples:
            sample_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
                key,
                jnp.arange(n_conditions),
            )
        else:
            keys = jax.random.split(key, n_bandwidths * n_conditions)
            sample_keys = keys.reshape((n_bandwidths, n_conditions, *keys.shape[1:]))
        mask_sets = [
            (
                bank.masks[bandwidth, :, : bank.width(bandwidth)],
                bank.weights[bandwidth, :, : bank.width(bandwidth)],
            )
            for bandwidth in range(n_bandwidths)
        ]

    def draw(keys: jax.Array) -> jnp.ndarray:
        return jax.vmap(lambda encoder, k: encoder.sample(k, n_samples))(rows, keys)

    if shared_samples:
        samples = draw(sample_keys)
        sample_sets = [samples] * n_bandwidths
    else:
        sample_sets = [draw(keys) for keys in sample_keys]

    entries = []
    for (visible_masks, weights), samples in zip(mask_sets, sample_sets, strict=True):
        target_means, target_squares = jax.vmap(target_moments)(
            targets[:, None],
            visible_masks,
        )
        entries.append(
            _Entries(
                jnp.pad(visible_masks, ((0, 0), (0, 0), (0, n_ancillas))),
                weights,
                samples,
                target_means[:, 0],
                target_squares[:, 0],
            ),
        )
    return tuple(entries)


def _mmd_from_entries(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: Encoder,
    entries: tuple[_Entries, ...],
    *,
    precision: str = "fp64",
    sample_chunk: int | None = None,
) -> ConditionalMMDResult:
    """Evaluate drawn entries at ``params``; only the phases depend on them.

    Each bandwidth vectorizes its conditions.  ``sample_chunk`` streams the
    sample axis and accumulates the per-mask moments, bounding the live
    ``(samples, masks)`` buffers; both model and target terms use the unbiased
    finite-sample estimator, accumulated in the policy's ``accumulate`` dtype.
    """
    policy = resolve_precision(precision)
    params = jnp.asarray(params, dtype=policy.accumulate)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    rows = _condition_rows(encoders)

    def condition_value(encoder: Encoder, entry: _Entries) -> jnp.ndarray:
        n_samples = entry.samples.shape[1]
        chunk = n_samples if sample_chunk is None else min(sample_chunk, n_samples)
        sums = square_sums = 0.0
        for start in range(0, n_samples, chunk):
            samples = entry.samples[:, start : start + chunk]
            phases = _phase_differences(
                params,
                generators,
                entry.masks,
                samples,
                policy,
            )
            values = _ratio_sample_values(
                encoder,
                samples,
                entry.masks,
                phases,
                policy,
            )[0]
            sums = sums + jnp.sum(values, axis=0)
            square_sums = square_sums + jnp.sum(values**2, axis=0)
        model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
        per_mask = (
            model_squares
            - 2.0 * (sums / n_samples) * entry.target_means.astype(policy.accumulate)
            + entry.target_squares.astype(policy.accumulate)
        )
        return jnp.sum(per_mask * entry.weights.astype(policy.accumulate))

    per_bandwidth_condition = jnp.stack(
        [jax.vmap(condition_value)(rows, entry) for entry in entries],
    )
    return ConditionalMMDResult(
        loss=jnp.mean(per_bandwidth_condition),
        per_condition=jnp.mean(per_bandwidth_condition, axis=0),
        per_bandwidth_condition=per_bandwidth_condition,
    )


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
    precision: str = "fp64",
    sample_chunk: int | None = None,
) -> ConditionalMMDResult:
    """Estimate the condition-balanced multiscale squared MMD.

    ``key`` always supplies the reference-state randomness.  Pass ``sigmas``
    and ``n_ops`` to draw fresh nonidentity masks from that key, or pass
    ``bank`` to evaluate a prebuilt compressed bank instead; the two mask
    sources are exclusive.  ``encoders`` must be a JAX pytree whose array
    leaves share a leading condition axis; the conditions are vectorized.

    With ``shared_samples=True`` each condition draws one reference sample set
    and evaluates every bandwidth's masks on it, whatever the mask source.
    This avoids resampling once per bandwidth.  The default resamples once per
    (bandwidth, condition) entry.

    ``precision`` names a policy from :data:`iqp_cond.PRECISIONS`; the loss
    comes back in the policy's ``accumulate`` dtype.
    """
    params = jnp.asarray(params, dtype=resolve_precision(precision).accumulate)
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
    _check_condition_pytree(encoders)
    if bank is not None:
        if bank.n_conditions != targets.shape[0]:
            raise ValueError(
                f"operator bank covers {bank.n_conditions} conditions but targets "
                f"have {targets.shape[0]}",
            )
        if bank.n_visible != n_data_qubits:
            raise ValueError(
                f"operator bank is built for {bank.n_visible} visible qubits but "
                f"targets have {n_data_qubits}",
            )
    entries = _mmd_entries(
        encoders,
        targets,
        key,
        bank=bank,
        sigmas=sigmas,
        n_ops=n_ops,
        n_samples=n_samples,
        shared_samples=shared_samples,
    )
    return _mmd_from_entries(
        params,
        generators,
        encoders,
        entries,
        precision=precision,
        sample_chunk=sample_chunk,
    )
