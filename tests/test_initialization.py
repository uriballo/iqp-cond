"""Data-based initialization: what the parameters are pinned to."""

from __future__ import annotations

import numpy as np
import pytest

import iqp_cond as iqc


@pytest.fixture
def data() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 2, size=(200, 4)).astype(np.float64)


@pytest.fixture
def generators() -> np.ndarray:
    return np.asarray(iqc.complete_ansatz(4).generators)


def test_single_body_parameters_follow_the_observed_marginals(
    data: np.ndarray,
    generators: np.ndarray,
) -> None:
    params = iqc.initialize_parameters_from_data(data, generators)
    means = data.mean(axis=0)
    for qubit in range(data.shape[1]):
        row = np.zeros(data.shape[1])
        row[qubit] = 1.0
        index = int(np.flatnonzero(np.all(generators == row, axis=1))[0])
        assert params[index] == pytest.approx(np.arcsin(np.sqrt(means[qubit])))


def test_two_body_parameters_track_the_observed_covariance(
    data: np.ndarray,
    generators: np.ndarray,
) -> None:
    params = iqc.initialize_parameters_from_data(data, generators)
    pair = np.array([1.0, 1.0, 0.0, 0.0])
    index = int(np.flatnonzero(np.all(generators == pair, axis=1))[0])
    signed = 2.0 * data - 1.0
    covariance = float(np.cov(signed, rowvar=False, bias=True)[0, 1])
    assert covariance != 0.0
    assert np.sign(params[index]) == np.sign(covariance)


def test_unsupported_rows_start_at_zero(generators: np.ndarray) -> None:
    data = np.random.default_rng(4).integers(0, 2, size=(50, 4)).astype(np.float64)
    generators = np.vstack([generators, np.array([[1.0, 1.0, 1.0, 0.0]])])
    params = iqc.initialize_parameters_from_data(data, generators)
    assert params[-1] == 0.0


def test_ancilla_initialization_confines_noise_to_the_ancilla_support(
    data: np.ndarray,
    generators: np.ndarray,
) -> None:
    ancilla_generators = np.asarray(iqc.complete_ansatz(6).generators)
    support = np.any(ancilla_generators[:, 4:] == 1.0, axis=1)
    params = iqc.initialize_parameters_with_ancillas(
        data,
        ancilla_generators,
        support,
        11,
    )
    visible_only = iqc.initialize_parameters_from_data(
        data,
        ancilla_generators[:, :4],
    )
    assert np.array_equal(params[~support], visible_only[~support])
    assert np.any(params[support] != 0.0)
    # Seeded noise of width 0.01/sqrt(P) applies only on ancilla supports.
    assert np.max(np.abs(params[support])) < 0.05


def test_ancilla_initialization_validates_the_support_shape(
    data: np.ndarray,
) -> None:
    generators = np.asarray(iqc.complete_ansatz(5).generators)
    with pytest.raises(ValueError, match="ancilla_support must match"):
        iqc.initialize_parameters_with_ancillas(
            data,
            generators,
            np.zeros(3, dtype=bool),
            0,
        )
