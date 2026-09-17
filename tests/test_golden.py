"""Equivalence with the research implementation.

``fixtures/golden.npz`` was recorded from ``src/iqp_cond`` at commit 8b64193d
(plus the 2026-09-16 working-tree edits) by
``.scratch/library-extraction/record_golden.py``.

The port reorganises the code but must not change a computed number, so almost
every assertion below is exact equality.  One comparison needs a tolerance, with
a stated numerical cause: the spin-flip exact state, because the encoder is now
evaluated for a batch of conditions where the research code evaluated one
condition at a time.

The compressed operator-bank loss — the training objective — is pinned bit-for-bit
against the campaign implementation in ``test_loss_sampled.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

import iqp_cond as iqc

FIXTURES = Path(__file__).parent / "fixtures" / "golden.npz"


def broadcast_row(angles: np.ndarray) -> np.ndarray:
    """Turn a recorded random angle matrix into a spin-flip broadcast row.

    The fixture stores the ``(n_conditions, n_qubits)`` matrix the research code
    handed in; spin-flip read only column 0, so broadcasting that column
    reproduces the recorded numbers exactly while satisfying the library's
    one-angle-per-condition contract.
    """
    return np.broadcast_to(angles[:, :1], angles.shape)


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


def assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    rtol: float,
    atol: float = 0.0,
) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape}"
    deviation = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    assert deviation <= atol + rtol * scale, (
        f"{label}: max abs deviation {deviation!r} exceeds "
        f"atol {atol} + rtol {rtol} * scale {scale}"
    )


@pytest.mark.parametrize("family", ["product", "spin_flip"])
def test_estimate_pauli_z_matches_research(
    golden: dict[str, np.ndarray],
    family: str,
) -> None:
    prefix = f"{family}_"
    angles = golden[prefix + "angles"]
    encoders = iqc.encoder(
        family, broadcast_row(angles) if family == "spin_flip" else angles
    )
    estimate = iqc.estimate_pauli_z(
        golden[prefix + "params"],
        golden[prefix + "generators"],
        encoders,
        golden[prefix + "masks"],
        key=jax.random.PRNGKey(11 if family == "product" else 12),
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


@pytest.mark.parametrize("family", ["product", "spin_flip"])
def test_post_hadamard_amplitudes_match_research(
    golden: dict[str, np.ndarray],
    family: str,
) -> None:
    prefix = f"{family}_"
    angles = golden[prefix + "angles"]
    encoders = iqc.encoder(
        family, broadcast_row(angles) if family == "spin_flip" else angles
    )
    expected = golden[prefix + "reference_state"]
    if family == "spin_flip":
        # The research shell table for the two-branch family is not normalised
        # (norms 2.276 and 3.684); sampling and ratio queries are invariant to
        # that scale, and the library normalises the state it returns.
        expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
        assert_close(
            encoders.post_hadamard_amplitudes(),
            expected,
            f"{prefix}reference_state",
            rtol=0.0,
            atol=1e-14,
        )
    else:
        assert_identical(
            encoders.post_hadamard_amplitudes(),
            expected,
            f"{prefix}reference_state",
        )


def test_parameters_from_data_matches_research(golden: dict[str, np.ndarray]) -> None:
    assert_identical(
        iqc.parameters_from_data(
            golden["init_data"],
            golden["init_generators"],
        ),
        golden["init_params"],
        "init_params",
    )
    assert_identical(
        iqc.parameters_from_data(
            golden["init_data"],
            golden["ancilla_generators"],
            n_visible=golden["init_data"].shape[1],
            seed=37,
        ),
        golden["ancilla_init_params"],
        "ancilla_init_params",
    )
