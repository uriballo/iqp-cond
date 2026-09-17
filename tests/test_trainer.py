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
    encoders = iqc.encoder("product", angles)
    result = iqc.train(
        iqc.complete_ansatz(N_QUBITS),
        encoders,
        training,
        validation,
        seed=9,
        config=config,
    )
    assert result.best_validation_loss < result.validation_losses[0]
    assert result.best_step >= 1
    assert result.train_losses.shape == (config.steps,)
    assert result.family == "product"


def test_ancilla_encoder_trains_against_visible_targets(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A five-qubit model with two ancillas trains against three-bit targets."""
    training, validation = targets
    angles = np.random.default_rng(7).uniform(0.4, 2.2, size=(N_CONDITIONS, 5))
    angles = np.broadcast_to(angles[:, :1], angles.shape)
    encoders = iqc.encoder("spin_flip", angles)
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
        encoders,
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
    encoders = iqc.encoder("product", angles)
    with pytest.raises(ValueError, match="encoder acts on"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS + 1),
            encoders,
            training,
            validation,
            seed=1,
            config=config,
        )


def test_step_refresh_is_the_default_and_fit_retains_one_bank(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    """The default redraws masks every step; 'cache' retains one bank."""
    training, validation = targets
    ansatz = iqc.complete_ansatz(N_QUBITS)
    encoders = iqc.encoder("product", angles)

    step_result = iqc.train(
        ansatz,
        encoders,
        training,
        validation,
        seed=4,
        config=config,
    )
    assert step_result.n_banks == config.steps

    explicit = iqc.train(
        ansatz,
        encoders,
        training,
        validation,
        seed=4,
        config=config,
        mask_refresh=iqc.STEP,
    )
    # The default path must be the step path, not a silent fallback.
    assert np.array_equal(step_result.train_losses, explicit.train_losses)

    cache_result = iqc.train(
        ansatz,
        encoders,
        training,
        validation,
        seed=4,
        config=config,
        mask_refresh=iqc.CACHE,
    )
    assert cache_result.n_banks == 1
    # Different mask protocols are different objectives, so the runs diverge.
    assert not np.array_equal(step_result.final_params, cache_result.final_params)
    assert np.all(np.isfinite(step_result.train_losses))


def test_unknown_mask_refresh_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    angles: np.ndarray,
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    with pytest.raises(ValueError, match="unknown mask_refresh"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS),
            iqc.encoder("product", angles),
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
    encoders = iqc.encoder(
        "product",
        np.zeros((N_CONDITIONS + 1, N_QUBITS)),
    )
    with pytest.raises(ValueError, match="conditions but targets"):
        iqc.train(
            iqc.complete_ansatz(N_QUBITS),
            encoders,
            training,
            validation,
            seed=1,
            config=config,
        )
