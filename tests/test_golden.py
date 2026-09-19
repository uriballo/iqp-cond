"""Equivalence with the research implementation.

``fixtures/golden.npz`` was recorded from ``src/iqp_cond`` at commit 8b64193d
(plus the 2026-09-16 working-tree edits) by
``.scratch/library-extraction/record_golden.py``.

The port reorganises the code but must not change a computed number, so every
assertion below is exact equality.

``test_loss.py`` pins the compressed operator-bank loss, the training
objective, bit-for-bit against the campaign implementation.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.estimator import estimate_pauli_z
from iqp_cond.initialization import parameters_from_data


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit: the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


FIXTURES = Path(__file__).parent / "fixtures" / "golden.npz"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(FIXTURES) as data:
        return {name: data[name] for name in data.files}


def assert_identical(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    """Assert exact equality, including for complex arrays."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape}"
    if actual.dtype.kind == "c":
        assert_identical(actual.real, expected.real, f"{label}.real")
        assert_identical(actual.imag, expected.imag, f"{label}.imag")
    elif actual.dtype.kind == "f":
        deviation = np.max(np.abs(actual - expected)) if actual.size else 0.0
        assert deviation == 0.0, f"{label}: max abs deviation {deviation!r}"
    else:
        assert np.array_equal(actual, expected), f"{label}: values differ"


def test_estimate_pauli_z_matches_research(
    golden: dict[str, np.ndarray],
) -> None:
    prefix = "product_"
    angles = golden[prefix + "angles"]
    encoders = y_layer(angles)
    estimate = estimate_pauli_z(
        golden[prefix + "params"],
        golden[prefix + "generators"],
        encoders,
        golden[prefix + "masks"],
        key=jax.random.PRNGKey(11),
        n_samples=64,
        return_samples=True,
    )
    assert_identical(estimate.values, golden[prefix + "values"], f"{prefix}values")
    assert_identical(
        estimate.standard_errors,
        golden[prefix + "standard_errors"],
        f"{prefix}standard_errors",
    )
    assert_identical(
        estimate.sample_values,
        golden[prefix + "sample_values"],
        f"{prefix}sample_values",
    )


def test_post_hadamard_amplitudes_match_research(
    golden: dict[str, np.ndarray],
) -> None:
    prefix = "product_"
    angles = golden[prefix + "angles"]
    encoders = y_layer(angles)
    assert_identical(
        encoders.post_hadamard_amplitudes(),
        golden[prefix + "reference_state"],
        f"{prefix}reference_state",
    )


def test_parameters_from_data_matches_research(golden: dict[str, np.ndarray]) -> None:
    assert_identical(
        parameters_from_data(
            golden["init_data"],
            golden["init_generators"],
        ),
        golden["init_params"],
        "init_params",
    )
    assert_identical(
        parameters_from_data(
            golden["init_data"],
            golden["ancilla_generators"],
            n_visible=golden["init_data"].shape[1],
            seed=37,
        ),
        golden["ancilla_init_params"],
        "ancilla_init_params",
    )
