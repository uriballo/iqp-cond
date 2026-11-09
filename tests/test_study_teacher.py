"""Teacher circuits of the precision study and their moments."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import iqp_cond as ciqp
from experiments.precision_study.teacher import (
    teacher,
    teacher_moments,
    teacher_samples,
)
from iqp_cond.exact import probabilities

N = 10


def moments_from_distribution(model, params, masks):
    distribution = probabilities(model, params)[0]
    bits = (np.arange(2**N)[:, None] >> np.arange(N)) & 1
    characters = 1.0 - 2.0 * ((bits @ masks.T) % 2)
    return distribution @ characters


@pytest.fixture(scope="module")
def setup():
    generators, params = teacher(N, seed=3)
    model = ciqp.ConditionalModel(
        ciqp.Ansatz(generators), ciqp.AngleEncoder(np.zeros((1, 1, N)), axes=("Y" * N,))
    )
    masks = np.random.default_rng(4).integers(0, 2, size=(40, N)).astype(float)
    masks[masks.sum(axis=1) == 0, 0] = 1.0
    return generators, params, masks, moments_from_distribution(model, params, masks)


def test_exact_teacher_moments_match_the_exact_distribution(setup):
    generators, params, masks, expected = setup
    moments = teacher_moments(generators, params, masks, exact_limit=N)
    np.testing.assert_allclose(moments, expected, atol=1e-12)


def test_sampled_teacher_moments_are_unbiased(setup):
    generators, params, masks, expected = setup
    n_samples = 200_000
    moments = teacher_moments(
        generators,
        params,
        masks,
        exact_limit=0,
        n_samples=n_samples,
        key=jax.random.key(5),
    )
    # Each moment is a mean of cosines, so its standard error is at most
    # 1 / sqrt(n_samples).
    assert np.max(np.abs(moments - expected)) < 5 / np.sqrt(n_samples)


def test_teacher_samples_follow_the_teacher_moments(setup):
    generators, params, masks, expected = setup
    n_samples = 50_000
    samples = teacher_samples(generators, params, n_samples, key=jax.random.key(8))
    assert samples.shape == (n_samples, N)
    characters = 1.0 - 2.0 * ((samples @ masks.T) % 2)
    # Each empirical moment is a mean of +-1 values: standard error at most
    # 1 / sqrt(n_samples).
    assert np.max(np.abs(characters.mean(axis=0) - expected)) < 5 / np.sqrt(n_samples)
