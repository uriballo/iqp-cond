"""Training behaviour: the loop must reduce held-out MMD and respect shapes."""

from __future__ import annotations

import numpy as np
import pytest

import iqp_cond as iqc

N_QUBITS = 3
N_CONDITIONS = 2


@pytest.fixture
def targets() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    training = rng.integers(0, 2, size=(N_CONDITIONS, 64, N_QUBITS)).astype(np.float64)
    validation = rng.integers(0, 2, size=(N_CONDITIONS, 32, N_QUBITS)).astype(
        np.float64
    )
    return training, validation


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(6).uniform(0.4, 2.2, size=(N_CONDITIONS, N_QUBITS))


@pytest.fixture
def config() -> iqc.TrainingConfig:
    return iqc.TrainingConfig(
        steps=25,
        batch_size=16,
        validation_interval=5,
        sigmas=(1.0, 2.0),
        n_ops=8,
        n_iqp_samples=32,
        validation_n_ops=8,
        validation_n_iqp_samples=32,
    )


def test_training_reduces_the_validation_objective(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    references = iqc.reference("product", angles)
    result = iqc.train(
        iqc.complete_ansatz(N_QUBITS),
        references,
        training,
        validation,
        seed=9,
        config=config,
    )
    assert result.best_validation_loss < result.validation_losses[0]
    assert result.best_step >= 1
    assert result.train_losses.shape == (config.steps,)
    assert result.family == "product"


def test_ancilla_reference_trains_against_visible_targets(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A five-qubit model with two ancillas trains against three-bit targets."""
    training, validation = targets
    angles = np.random.default_rng(7).uniform(0.4, 2.2, size=(N_CONDITIONS, 5))
    references = iqc.reference("spin_flip", angles)
    config = iqc.TrainingConfig(
        steps=10,
        batch_size=8,
        validation_interval=5,
        sigmas=(1.0,),
        n_ops=4,
        n_iqp_samples=16,
        validation_n_ops=4,
        validation_n_iqp_samples=16,
    )
    result = iqc.train(
        iqc.complete_ansatz(5),
        references,
        training,
        validation,
        seed=10,
        config=config,
    )
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_params.shape == (iqc.complete_ansatz(5).n_parameters,)
    assert 0 <= result.best_step <= config.steps


def test_qubit_count_mismatch_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    references = iqc.reference("product", angles)
    with pytest.raises(ValueError, match="reference acts on"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS + 1),
            references,
            training,
            validation,
            seed=1,
            config=config,
        )


def test_unknown_estimator_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    with pytest.raises(ValueError, match="unknown estimator"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS),
            iqc.reference("product", angles),
            training,
            validation,
            seed=1,
            config=config,
            estimator="not_an_estimator",
        )


def test_plain_estimator_still_trains(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    """The reference estimator remains usable end to end."""
    training, validation = targets
    result = iqc.train(
        iqc.complete_ansatz(N_QUBITS),
        iqc.reference("product", angles),
        training,
        validation,
        seed=9,
        config=config,
        estimator=iqc.PLAIN,
    )
    assert np.all(np.isfinite(result.train_losses))
    assert result.best_validation_loss < result.validation_losses[0]


def test_epoch_refresh_rebuilds_the_bank_and_fit_does_not(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    """The default redraws the bank each epoch; 'fit' retains one bank."""
    training, validation = targets
    ansatz = iqc.complete_ansatz(N_QUBITS)
    references = iqc.reference("product", angles)
    steps_per_epoch = max(1, training.shape[1] // config.batch_size)
    expected_epochs = -(-config.steps // steps_per_epoch)

    epoch_result = iqc.train(
        ansatz,
        references,
        training,
        validation,
        seed=4,
        config=config,
        mask_refresh=iqc.EPOCH,
    )
    assert epoch_result.n_banks == expected_epochs > 1

    fit_result = iqc.train(
        ansatz,
        references,
        training,
        validation,
        seed=4,
        config=config,
        mask_refresh=iqc.FIT,
    )
    assert fit_result.n_banks == 1
    # Different banks are different objectives, so the runs must diverge.
    assert not np.array_equal(epoch_result.final_params, fit_result.final_params)
    assert np.all(np.isfinite(epoch_result.train_losses))


def test_epoch_refresh_trains_the_banked_estimator(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    """The default must train the compressed objective, not fall back to plain.

    Both estimators see the same seed, batches and samples, so identical
    training losses would mean the epoch path quietly trained `conditional_mmd`
    while the validation pass scored the bank.
    """
    training, validation = targets
    ansatz = iqc.complete_ansatz(N_QUBITS)
    references = iqc.reference("product", angles)
    epoch_result = iqc.train(
        ansatz,
        references,
        training,
        validation,
        seed=4,
        config=config,
        estimator=iqc.COMPRESSED,
        mask_refresh=iqc.EPOCH,
    )
    plain_result = iqc.train(
        ansatz,
        references,
        training,
        validation,
        seed=4,
        config=config,
        estimator=iqc.PLAIN,
    )
    assert not np.array_equal(epoch_result.train_losses, plain_result.train_losses)


def test_unknown_mask_refresh_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    with pytest.raises(ValueError, match="unknown mask_refresh"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS),
            iqc.reference("product", angles),
            training,
            validation,
            seed=1,
            config=config,
            mask_refresh="sometimes",
        )


def test_condition_count_mismatch_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    references = iqc.reference(
        "product",
        np.zeros((N_CONDITIONS + 1, N_QUBITS)),
    )
    with pytest.raises(ValueError, match="conditions but targets"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS),
            references,
            training,
            validation,
            seed=1,
            config=config,
        )
