"""Equivalence with the research implementation.

``fixtures/golden.npz`` was recorded from ``src/iqp_cond`` at commit 8b64193d
(plus the 2026-09-16 working-tree edits) by
``.scratch/library-extraction/record_golden.py``.

The port reorganises the code but must not change a computed number, so almost
every assertion below is exact equality.  Two comparisons need a tolerance, and
both have a stated numerical cause:

- the spin-flip exact state, because the reference is now evaluated for a batch
  of conditions where the research code evaluated one condition at a time;
- the training loop, because reference constants are now folded into the traced
  program instead of being computed inside it.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

import iqp_cond as iqc

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
    references = iqc.reference(family, golden[prefix + "angles"])
    estimate = iqc.estimate_pauli_z(
        golden[prefix + "params"],
        golden[prefix + "generators"],
        golden[prefix + "masks"],
        references,
        jax.random.PRNGKey(11 if family == "product" else 12),
        64,
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
    references = iqc.reference(family, golden[prefix + "angles"])
    expected = golden[prefix + "reference_state"]
    if family == "spin_flip":
        # The research shell table for the two-branch family is not normalised
        # (norms 2.276 and 3.684); sampling and ratio queries are invariant to
        # that scale, and the library normalises the state it returns.
        expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
        assert_close(
            references.post_hadamard_amplitudes(),
            expected,
            f"{prefix}reference_state",
            rtol=0.0,
            atol=1e-14,
        )
    else:
        assert_identical(
            references.post_hadamard_amplitudes(),
            expected,
            f"{prefix}reference_state",
        )


def test_conditional_mmd_matches_research(golden: dict[str, np.ndarray]) -> None:
    ansatz = iqc.complete_ansatz(4)
    references = iqc.reference("product", golden["plain_angles"])
    result = iqc.conditional_mmd(
        golden["plain_params"],
        ansatz.generators,
        references,
        golden["plain_targets"],
        (1.0, 2.0, 4.0),
        8,
        64,
        jax.random.PRNGKey(27),
    )
    assert_identical(result.loss, golden["plain_loss"], "plain loss")


def test_conditional_mmd_with_ancillas_matches_research(
    golden: dict[str, np.ndarray],
) -> None:
    ansatz = iqc.complete_ansatz(6)
    references = iqc.reference("spin_flip", golden["ancilla_angles"])
    result = iqc.conditional_mmd(
        golden["ancilla_params"],
        ansatz.generators,
        references,
        golden["ancilla_targets"],
        (1.0, 2.0, 4.0),
        8,
        64,
        jax.random.PRNGKey(24),
    )
    assert_identical(result.loss, golden["ancilla_loss"], "ancilla loss")
    assert_identical(
        result.per_condition,
        golden["ancilla_per_condition"],
        "ancilla per_condition",
    )
    assert_identical(
        result.per_bandwidth_condition,
        golden["ancilla_per_bandwidth_condition"],
        "ancilla per_bandwidth_condition",
    )


def test_loss_gradient_matches_research(golden: dict[str, np.ndarray]) -> None:
    ansatz = iqc.complete_ansatz(4)
    references = iqc.reference("product", golden["plain_angles"])

    def loss(params: jax.Array) -> jax.Array:
        return iqc.conditional_mmd(
            params,
            ansatz.generators,
            references,
            golden["plain_targets"],
            (1.0, 2.0, 4.0),
            8,
            64,
            jax.random.PRNGKey(27),
        ).loss

    gradient = jax.grad(loss)(golden["plain_params"])
    assert_identical(gradient, golden["plain_gradient"], "plain gradient")


def test_initialize_parameters_matches_research(golden: dict[str, np.ndarray]) -> None:
    assert_identical(
        iqc.initialize_parameters_from_data(
            golden["init_data"],
            golden["init_generators"],
        ),
        golden["init_params"],
        "init_params",
    )
    assert_identical(
        iqc.initialize_parameters_with_ancillas(
            golden["init_data"],
            golden["ancilla_generators"],
            golden["ancilla_support"],
            37,
        ),
        golden["ancilla_init_params"],
        "ancilla_init_params",
    )


def test_training_matches_research(golden: dict[str, np.ndarray]) -> None:
    ansatz = iqc.complete_ansatz(4)
    references = iqc.reference("product", golden["train_angles"])
    config = iqc.TrainingConfig(
        steps=20,
        batch_size=8,
        validation_interval=10,
        sigmas=(1.0, 2.0),
        n_ops=8,
        n_iqp_samples=32,
        validation_n_ops=8,
        validation_n_iqp_samples=32,
    )
    result = iqc.train(
        ansatz,
        references,
        golden["train_targets"],
        golden["train_validation_targets"],
        44,
        config,
        initial_params=golden["train_initial_params"],
        estimator=iqc.PLAIN,
    )
    # Reference amplitudes are constants of the traced loss here, so XLA folds
    # them differently than the research loop did; the drift is ~1e-16 and
    # amplifies through the optimizer, hence a relative bound rather than
    # equality for the trained parameters and their histories.
    assert_close(
        result.best_params,
        golden["train_best_params"],
        "best_params",
        rtol=1e-9,
        atol=1e-14,
    )
    assert_close(
        result.final_params,
        golden["train_final_params"],
        "final_params",
        rtol=1e-9,
        atol=1e-14,
    )
    assert int(result.best_step) == int(golden["train_best_step"])
    assert_close(
        np.asarray(result.best_validation_loss),
        golden["train_best_validation_loss"],
        "best_validation_loss",
        rtol=1e-9,
        atol=1e-14,
    )
    assert_close(result.train_losses, golden["train_losses"], "train_losses", rtol=1e-9)
    assert_identical(
        result.validation_steps,
        golden["train_validation_steps"],
        "validation_steps",
    )
    assert_close(
        result.validation_losses,
        golden["train_validation_losses"],
        "validation_losses",
        rtol=1e-9,
    )
