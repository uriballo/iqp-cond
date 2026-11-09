"""Trajectory metrics of the precision study."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.precision_study.metrics import (
    classify,
    iteration_lag,
    noise_ratio,
    parameter_deviation,
)


def test_parameter_deviation_ignores_shifts_by_pi():
    reference = np.zeros((1, 3))
    other = np.array([[np.pi, np.pi + 0.1, -0.2]])
    np.testing.assert_allclose(
        parameter_deviation(other, reference), [np.hypot(0.1, 0.2)], atol=1e-12
    )


STEPS = np.arange(7) * 25  # checkpoints every 25 steps


def test_iteration_lag_measures_a_shifted_loss_curve_in_steps():
    reference = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0])
    delayed = np.array([5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    # Both start at 5, so there is no lag at the start; each later reference
    # value is reached two checkpoints, 50 steps, late.  The reference stops
    # falling at checkpoint 4, after which the lag is undefined.
    np.testing.assert_array_equal(
        iteration_lag(delayed, reference, STEPS, window=1),
        [0, 50, 50, 50, np.nan, np.nan, np.nan],
    )


def test_iteration_lag_is_negative_for_a_run_ahead_of_fp64():
    reference = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0])
    ahead = np.array([5.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    # From checkpoint 2 the run reaches each reference value one checkpoint,
    # 25 steps, earlier.
    np.testing.assert_array_equal(
        iteration_lag(ahead, reference, STEPS, window=1),
        [0, 0, -25, -25, np.nan, np.nan, np.nan],
    )


BASELINE = np.array([[0.0, 0.1, 0.2], [0.0, 0.2, 0.3], [0.0, 0.1, 0.4]])


@pytest.mark.parametrize(
    ("deviation", "distribution_differs", "expected"),
    [
        (np.array([0.0, 0.1, 0.2]), False, "tracks"),
        (np.array([0.0, 0.9, 0.2]), False, "delayed"),
        (np.array([0.0, 0.9, 2.0]), True, "diverges"),
        # An endpoint beyond the baseline whose distribution still agrees
        # counts as delayed, not diverged.
        (np.array([0.0, 0.9, 2.0]), False, "delayed"),
    ],
)
def test_classification_against_the_chaos_baseline(
    deviation, distribution_differs, expected
):
    assert classify(deviation, BASELINE, distribution_differs) == expected


REFERENCE = np.array([1.0, 3.0, 1.0, 3.0])  # sample standard deviation 2/sqrt(3)
SIGMA = 2 / np.sqrt(3)


@pytest.mark.parametrize(
    ("error", "bias", "spread"),
    [
        (np.full(4, 0.5), 0.5 / SIGMA, 0.0),
        (np.array([0.2, -0.2, 0.2, -0.2]), 0.0, 0.2 / SIGMA),
        (np.array([0.7, 0.3, 0.7, 0.3]), 0.5 / SIGMA, 0.2 / SIGMA),
    ],
    ids=["constant offset", "zero-mean noise", "both"],
)
def test_noise_ratio_splits_rounding_error_into_bias_and_spread(error, bias, spread):
    ratio = noise_ratio(REFERENCE + error, REFERENCE)
    np.testing.assert_allclose(
        [ratio.bias, ratio.spread, ratio.rho],
        [bias, spread, np.hypot(bias, spread)],
        atol=1e-12,
    )


def test_noise_ratio_rejects_a_reference_without_noise():
    with pytest.raises(ValueError):
        noise_ratio(np.ones(4) + 0.1, np.ones(4))
