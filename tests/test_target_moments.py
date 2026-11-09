"""Training targets given as exact moments instead of samples."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from fp64_golden import inputs

import iqp_cond as ciqp
from iqp_cond.loss import TargetMoments, mmd
from iqp_cond.operators import operator_bank


def test_moment_targets_drop_only_the_finite_sample_correction():
    data = inputs()
    model, targets = data["model"], data["targets"]
    bank = operator_bank((1.0, 2.0), 16, targets.shape[0], targets.shape[2], seed=4)
    # Empirical moments of the target samples, one per bank mask.
    parities = np.einsum("ctn,wcmn->wctm", targets, np.asarray(bank.masks)) % 2
    characters = 1.0 - 2.0 * parities
    moments = characters.mean(axis=2)
    n = targets.shape[1]

    def loss(target_source, **options):
        return float(
            mmd(
                data["params"],
                model.ansatz.generators,
                model.encoder,
                target_source,
                jax.random.key(3),
                n_samples=64,
                **options,
            ).loss
        )

    from_moments = loss(TargetMoments(bank, moments))
    from_samples = loss(targets, bank=bank)
    # Exact moments enter as mu^2; samples as the unbiased (n mu^2 - 1)/(n - 1).
    # The difference, (1 - mu^2) / (n - 1) per mask, is weighted and averaged
    # over the (bandwidth, condition) cells.
    weights = np.asarray(bank.weights)
    correction = np.sum(weights * (1.0 - moments**2) / (n - 1), axis=-1).mean()
    np.testing.assert_allclose(from_moments - from_samples, correction, rtol=1e-10)


def exact_moments(model, params, bank):
    """Teacher moments from PennyLane's exact output distribution."""
    probabilities = model.probabilities(params)  # (conditions, 2^n)
    n = model.visible
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1  # little-endian
    masks = np.asarray(bank.masks)  # (bandwidths, conditions, masks, n)
    characters = 1.0 - 2.0 * (np.einsum("xn,wcmn->wcmx", bits, masks) % 2)
    return np.einsum("cx,wcmx->wcm", probabilities, characters)


@pytest.mark.parametrize("refresh", ["step", "cache"])
def test_training_on_teacher_moments_approaches_the_teacher(refresh):
    data = inputs()
    model = data["model"]
    teacher = np.random.default_rng(9).normal(0.0, 0.5, model.ansatz.n_parameters)
    pool = operator_bank((1.0,), 256, model.encoder.n_conditions, model.visible, seed=6)
    targets = TargetMoments(pool, exact_moments(model, teacher, pool))

    def distance(params):
        return float(
            mmd(
                params,
                model.ansatz.generators,
                model.encoder,
                targets,
                jax.random.key(0),
                n_samples=4096,
            ).loss
        )

    result = ciqp.train(
        model,
        targets,
        targets,
        seed=2,
        initial_params=data["params"],
        config=ciqp.TrainingConfig(
            steps=60,
            learning_rate=0.05,
            final_learning_rate=0.01,
            validation_interval=30,
            n_ops=32,
            mc_samples=64,
            validation_mc_samples=64,
            mask_refresh=refresh,
        ),
    )
    assert distance(result.final_params) < 0.5 * distance(data["params"])


def test_numpy_moments_work_under_jit_with_reduced_statistics():
    data = inputs()
    model = data["model"]
    bank = operator_bank((1.0,), 16, model.encoder.n_conditions, model.visible, seed=4)
    targets = TargetMoments(bank, np.full(bank.weights.shape, 0.25))

    @jax.jit
    def loss(params):
        return mmd(
            params,
            model.ansatz.generators,
            model.encoder,
            targets,
            jax.random.key(0),
            n_samples=16,
            precision=ciqp.precision(statistics="bf16"),
        ).loss

    assert np.isfinite(float(loss(data["params"])))
