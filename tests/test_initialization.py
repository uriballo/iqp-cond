"""Data-based initialization and the exact parameter values it assigns."""

from __future__ import annotations

import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.initialization import parameters_from_data, random_parameters
from iqp_cond.model import ancilla_support


@pytest.fixture
def data() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 2, size=(200, 4)).astype(np.float64)


@pytest.fixture
def generators() -> np.ndarray:
    return np.asarray(ciqp.Ansatz.complete(4).generators)


def test_single_body_parameters_follow_the_observed_marginals(
    data: np.ndarray,
    generators: np.ndarray,
) -> None:
    params = parameters_from_data(data, generators)
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
    params = parameters_from_data(data, generators)
    pair = np.array([1.0, 1.0, 0.0, 0.0])
    index = int(np.flatnonzero(np.all(generators == pair, axis=1))[0])
    signed = 2.0 * data - 1.0
    covariance = float(np.cov(signed, rowvar=False, bias=True)[0, 1])
    assert covariance != 0.0
    assert np.sign(params[index]) == np.sign(covariance)


def test_unsupported_rows_start_at_zero(generators: np.ndarray) -> None:
    data = np.random.default_rng(4).integers(0, 2, size=(50, 4)).astype(np.float64)
    generators = np.vstack([generators, np.array([[1.0, 1.0, 1.0, 0.0]])])
    params = parameters_from_data(data, generators)
    assert params[-1] == 0.0


def test_exactly_uncorrelated_pairs_stay_at_zero() -> None:
    """A balanced product design has no covariance direction to scale."""
    data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    generators = np.asarray(ciqp.Ansatz.complete(2).generators)
    one_body = np.arcsin(np.sqrt(0.5))
    expected = np.array([one_body, one_body, 0.0])
    assert np.array_equal(parameters_from_data(data, generators), expected)
    assert np.array_equal(
        parameters_from_data(data, generators, scale=1.0),
        expected,
    )


def test_ancilla_noise_is_confined_to_the_ancilla_support(
    data: np.ndarray,
) -> None:
    ancilla_generators = np.asarray(ciqp.Ansatz.complete(6).generators)
    support = ancilla_support(ancilla_generators, data.shape[1])
    params = parameters_from_data(
        data,
        ancilla_generators,
        n_visible=data.shape[1],
        seed=11,
    )
    visible_only = parameters_from_data(data, ancilla_generators[:, :4])
    assert np.array_equal(params[~support], visible_only[~support])
    assert np.any(params[support] != 0.0)
    # Seeded noise of width 0.01/sqrt(P) applies only on ancilla supports.
    assert np.max(np.abs(params[support])) < 0.05


def test_ancilla_initialization_is_deterministic_in_the_seed(
    data: np.ndarray,
) -> None:
    ancilla_generators = np.asarray(ciqp.Ansatz.complete(6).generators)
    first = parameters_from_data(data, ancilla_generators, n_visible=4, seed=7)
    second = parameters_from_data(data, ancilla_generators, n_visible=4, seed=7)
    other = parameters_from_data(data, ancilla_generators, n_visible=4, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_parameters_from_data_defaults_to_no_ancillas(
    data: np.ndarray,
    generators: np.ndarray,
) -> None:
    assert np.array_equal(
        parameters_from_data(data, generators),
        parameters_from_data(data, generators, n_visible=data.shape[1]),
    )


def test_parameters_from_data_validates_the_split(
    data: np.ndarray,
    generators: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="n_visible must be between"):
        parameters_from_data(data, generators, n_visible=0)
    with pytest.raises(ValueError, match="data width must equal n_visible"):
        parameters_from_data(data, generators, n_visible=3)


def test_parameters_from_data_validates_its_inputs(
    generators: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="non-empty rank-two"):
        parameters_from_data(np.zeros((0, 4)), generators)
    with pytest.raises(ValueError, match="data must contain only finite binary"):
        parameters_from_data(np.full((4, 4), 0.5), generators)
    with pytest.raises(ValueError, match="cover all visible data qubits"):
        parameters_from_data(np.zeros((4, 5)), generators)
    with pytest.raises(ValueError, match="generators must contain only finite"):
        parameters_from_data(np.zeros((4, 4)), generators * np.nan)
    with pytest.raises(ValueError, match="scale must be finite"):
        parameters_from_data(np.zeros((4, 4)), generators, scale=-1.0)


def test_random_parameters_are_normalized_gaussians() -> None:
    params = random_parameters(64, seed=2)
    assert params.shape == (64,)
    assert np.all(np.isfinite(params))
    assert np.std(params) == pytest.approx(0.1 / np.sqrt(64), rel=0.5)
    with pytest.raises(ValueError, match="n_parameters must be positive"):
        random_parameters(0, seed=2)
