"""Reference samples and masks per update, set from the update's own noise.

The trainer's gradient for one (condition, bandwidth) cell, at ``S`` reference
samples and ``R`` masks, has the Monte Carlo variance
``A / R + sigma_1^2 / S + B_int / (R S)``: a mask part, a sample part and
their interaction.  :func:`measure` estimates the three parts and the matching
ratio ``gamma`` of every cell, and from them the step's settings:

``S = gamma B``
    ``B`` is ``TrainingConfig.batch_size``.  ``gamma`` is the sample noise of
    one reference draw relative to the data noise of one batch row, so at
    ``S = gamma B`` the model side is as noisy as the data side.  It is the
    gradient-weighted model covariance over the gradient-weighted data
    covariance, ``t_{a xor a'} - t_a t_a'``, summed over pairs of distinct
    masks; the two mean gradients come from independent sample sets, so their
    cross product is unbiased, and the data side is exact over the training
    rows.
``R = R* = sum (A + B_int / S) / sum (sigma_1^2 / S)``
    The mask count at which the mask and sample terms are equal.  Each update
    draws one cell uniformly, so the sums run over the cells.

The parts come from a two-way analysis of variance of one samples x masks
table of per-sample, per-mask gradient terms with the centring correction; the
effective parts at ``S`` are ``1.5 v(S) - 0.5 v(S/2)``, which removes the
second-order term of the plug-in parts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ._layout import pad_visible_masks
from .encoders import Encoder
from .encoders.base import _select_condition
from .estimator import _gradient_terms
from .loss import TargetMoments
from .model import ConditionalModel
from .operators import sample_nonidentity_masks_jax
from .training import TrainingConfig, _validate_training_inputs

__all__ = ["Measurement", "measure"]

Flag = Literal["estimated", "capped", "kept"]

_COVARIANCE_SAMPLES = 16_384
_GRADIENT_SAMPLES = 16_384
_CHUNK = 1_024
_MIN_SAMPLES = 4


@dataclass(frozen=True)
class Measurement:
    """The per-cell noise parts and the step's sample and mask settings.

    The per-cell arrays have shape ``(conditions, bandwidths)``.
    ``gamma_numerators`` and ``gamma_denominators`` are the gradient-weighted
    model and data covariances; ``mask_parts``, ``sample_parts`` and
    ``interaction_parts`` are ``A``, ``sigma_1^2`` and ``B_int``, all on the
    scale of the trainer's cell loss.  ``gamma`` is the step-level ratio of the
    summed numerators to the summed denominators and ``mask_ratio`` the
    unrounded ``R*``.  ``samples`` and ``masks`` are the settings to train
    with; ``flag`` is ``"estimated"``, ``"capped"`` when ``R*`` exceeded the
    cap, or ``"kept"`` when the estimate was invalid and the previous settings
    were kept.
    """

    gamma_numerators: np.ndarray
    gamma_denominators: np.ndarray
    mask_parts: np.ndarray
    sample_parts: np.ndarray
    interaction_parts: np.ndarray
    gamma: float
    mask_ratio: float
    samples: int
    masks: int
    flag: Flag


def _table_parts_from_terms(v, s, C, F, targets):
    """Return ``(A, sigma_1^2, B_int)`` of one samples x masks table.

    ``v`` and ``s`` are the ``(S, R)`` values and derivative factors of a model
    whose per-sample gradient is ``d v(z, a) / d theta = s(z, a) C(z) * F_a``,
    with characters ``C`` of shape ``(S, m)``, flips ``F`` of shape ``(R, m)``
    and the ``(R,)`` target moments.  The parts are those of the mean over the
    table's masks of ``(mu_a - t_a)^2``.
    """
    S, R = v.shape
    mu = v.mean(0)
    alpha = 2.0 * (mu - targets)
    beta = 2.0 * (v - mu[None, :])
    D = (s.T @ C) / S * F
    rows = alpha[:, None] * D
    cols = C * ((s * alpha[None, :]) @ F) / R + beta @ D / R
    grand = rows.mean(0)
    term1 = ((alpha[None, :] * s) ** 2 * F.sum(1)[None, :]).sum()
    cross = 2.0 * (alpha[None, :] * s * beta * (C @ (F * D).T)).sum()
    term3 = (beta**2 * (D**2).sum(1)[None, :]).sum()
    total = term1 + cross + term3 - R * S * grand @ grand
    ss_rows = S * ((rows - grand) ** 2).sum()
    ss_cols = R * ((cols - grand) ** 2).sum()
    ss_res = total - ss_rows - ss_cols
    ms_rows = ss_rows / (R - 1) + ((D**2).sum(1) * beta.var(0, ddof=1)).mean()
    ms_cols = ss_cols / (S - 1)
    ms_res = ss_res / ((R - 1) * (S - 1))
    return jnp.array([(ms_rows - ms_res) / S, (ms_cols - ms_res) / R, ms_res])


@jax.jit
def _gradient_sums(params, generators, encoder: Encoder, masks, key):
    samples = encoder.sample(key, _CHUNK)[0]
    _, s, C, _ = _gradient_terms(params, generators, encoder, masks, samples)
    return s.T @ C


@jax.jit
def _value_sums(params, generators, encoder: Encoder, masks, key, first, second):
    samples = encoder.sample(key, _CHUNK)[0]
    v = _gradient_terms(params, generators, encoder, masks, samples)[0]
    return v.sum(0), (v**2).sum(0), jnp.sum((v @ first) * (v @ second))


@jax.jit
def _character_sums(rows, masks, first, second):
    characters = 1.0 - 2.0 * jnp.mod(rows @ masks.T, 2.0)
    return characters.sum(0), jnp.sum((characters @ first) * (characters @ second))


def _cell_gamma(params, generators, encoder, masks, weights, rows, key):
    """Return one cell's gamma numerator and denominator over distinct masks."""
    keys = jax.random.split(
        key, (_COVARIANCE_SAMPLES + 2 * _GRADIENT_SAMPLES) // _CHUNK
    )
    n_gradient = _GRADIENT_SAMPLES // _CHUNK
    flips = jnp.mod(
        pad_visible_masks(masks, encoder.n_qubits - masks.shape[1]) @ generators.T,
        2.0,
    )
    first, second = (
        sum(_gradient_sums(params, generators, encoder, masks, k) for k in part)
        / _GRADIENT_SAMPLES
        * flips
        * weights[:, None]
        for part in (keys[:n_gradient], keys[n_gradient : 2 * n_gradient])
    )
    diagonal = jnp.sum(first * second, axis=1)

    totals, squares, products = 0.0, 0.0, 0.0
    for k in keys[2 * n_gradient :]:
        total, square, product = _value_sums(
            params, generators, encoder, masks, k, first, second
        )
        totals, squares, products = totals + total, squares + square, products + product
    n = _COVARIANCE_SAMPLES
    mean = totals / n
    variances = (squares - n * mean**2) / (n - 1)
    numerator = (products - n * (mean @ first) @ (mean @ second)) / (n - 1)
    numerator = numerator - diagonal @ variances

    totals, products = 0.0, 0.0
    for start in range(0, rows.shape[0], _CHUNK):
        total, product = _character_sums(
            rows[start : start + _CHUNK], masks, first, second
        )
        totals, products = totals + total, products + product
    t = totals / rows.shape[0]
    denominator = products / rows.shape[0] - (t @ first) @ (t @ second)
    denominator = denominator - diagonal @ (1.0 - t**2)
    return float(numerator), float(denominator)


@partial(jax.jit, static_argnames="n_samples")
def _table_parts(params, generators, encoder: Encoder, masks, targets, key, n_samples):
    samples = encoder.sample(key, n_samples)[0]
    terms = _gradient_terms(params, generators, encoder, masks, samples)
    return _table_parts_from_terms(*terms, targets)


def _cell_parts(params, generators, encoder, masks, weights, rows, key, n_samples):
    """Return one cell's effective ``(A, sigma_1^2, B_int)`` at ``n_samples``."""
    targets = jnp.mean(1.0 - 2.0 * jnp.mod(rows @ masks.T, 2.0), axis=0)
    full, half = (
        _table_parts(params, generators, encoder, masks, targets, k, size)
        for k, size in zip(
            jax.random.split(key), (n_samples, n_samples // 2), strict=True
        )
    )
    return np.asarray((1.5 * full - 0.5 * half) * jnp.sum(weights) ** 2)


def _power_of_two(value: float) -> int:
    return 2 ** max(round(math.log2(value)), 0)


def _measure(
    model: ConditionalModel,
    params: jnp.ndarray,
    training_targets: np.ndarray,
    config: TrainingConfig,
    key: jax.Array,
    masks: int,
    r_cap: int,
    previous: tuple[int, int] | None,
) -> Measurement:
    """Measure every cell and choose the settings.

    An invalid estimate returns ``previous`` flagged ``"kept"``, or raises
    without one.
    """
    sigmas = config.resolved_sigmas(model.visible)
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(model.generators, dtype=jnp.float64)
    shape = (model.encoder.n_conditions, len(sigmas))
    cells = []
    gammas = np.zeros((2, *shape))
    for condition, bandwidth in np.ndindex(shape):
        mask_key, gamma_key, table_key = jax.random.split(
            jax.random.fold_in(jax.random.fold_in(key, condition), bandwidth), 3
        )
        cell = (
            _select_condition(model.encoder, condition),
            *sample_nonidentity_masks_jax(
                mask_key, masks, model.visible, sigmas[bandwidth]
            ),
            jnp.asarray(training_targets[condition], dtype=jnp.float64),
        )
        gammas[:, condition, bandwidth] = _cell_gamma(
            params, generators, *cell, gamma_key
        )
        cells.append((cell, table_key))
    gamma = float(gammas[0].sum() / gammas[1].sum())

    def kept(reason: str, *parts: np.ndarray, ratio: float = math.nan) -> Measurement:
        if previous is None:
            raise ValueError(reason)
        nan = np.full(shape, math.nan)
        return Measurement(
            *gammas, *(parts or (nan, nan, nan)), gamma, ratio, *previous, "kept"
        )

    if not (math.isfinite(gamma) and gamma > 0.0):
        return kept(f"gamma must be positive and finite, not {gamma}")
    samples = max(_MIN_SAMPLES, _power_of_two(gamma * config.batch_size))
    parts = np.stack(
        [
            _cell_parts(params, generators, *cell, table_key, samples)
            for cell, table_key in cells
        ],
        axis=-1,
    ).reshape(3, *shape)
    mask_parts, sample_parts, interaction_parts = parts
    sample_noise = sample_parts.sum()
    if not (math.isfinite(sample_noise) and sample_noise > 0.0):
        return kept(
            f"summed sigma_1^2 must be positive and finite, not {sample_noise}", *parts
        )
    ratio = float(
        (mask_parts.sum() + interaction_parts.sum() / samples) / sample_noise * samples
    )
    if not (math.isfinite(ratio) and ratio > 0.0):
        return kept(f"R* must be positive and finite, not {ratio}", *parts, ratio=ratio)
    n_masks = _power_of_two(ratio)
    flag = "capped" if n_masks > r_cap else "estimated"
    return Measurement(
        *gammas, *parts, gamma, ratio, samples, min(n_masks, r_cap), flag
    )


def measure(
    model: ConditionalModel,
    params: np.ndarray,
    training_targets: np.ndarray,
    config: TrainingConfig,
    key: jax.Array,
    *,
    masks: int = 4096,
    r_cap: int = 65536,
) -> Measurement:
    """Measure the step noise at ``params`` and set the samples and masks.

    ``training_targets`` are the raw samples :func:`iqp_cond.train` takes,
    ``(conditions, examples, visible qubits)``; the data side uses every row.
    ``config`` supplies the batch size and the bandwidths.  Every (condition,
    bandwidth) cell draws ``masks`` masks from the library's mask law,
    whatever ``config.n_ops``, and its own reference samples from ``key``; the
    measurement runs in float64 whatever ``config.precision``.  The returned
    ``samples`` is ``gamma * batch_size`` rounded to a power of two, at least
    4, and ``masks`` is ``R*`` rounded to a power of two and capped at
    ``r_cap``.  A non-positive or non-finite ``gamma``, summed ``sigma_1^2`` or
    ``R*`` raises :class:`ValueError`.

    The noise changes as the model trains: at initialization ``gamma`` is
    typically 0.03-0.2 against about 0.5 for trained parameters, and ``R*`` is
    near 1e4 against hundreds later.  Fixed settings should therefore be
    measured at trained or warm parameters; :class:`iqp_cond.Adaptation`,
    which remeasures during training, is the recommended route.  Each update
    draws one cell, so the variance between the cells' gradients is outside
    the parts.
    """
    if isinstance(training_targets, TargetMoments):
        raise ValueError("measure needs raw-sample training targets")
    training_targets = np.asarray(training_targets, dtype=np.float64)
    _validate_training_inputs(model, training_targets, training_targets)
    if masks < 2:
        raise ValueError("masks must be at least two")
    if r_cap < 1:
        raise ValueError("r_cap must be positive")
    return _measure(model, params, training_targets, config, key, masks, r_cap, None)
