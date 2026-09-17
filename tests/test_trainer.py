"""Training behaviour: the loop must reduce held-out MMD and respect shapes."""

from __future__ import annotations

import dataclasses

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
def model(angles: np.ndarray) -> iqc.ConditionalModel:
    return iqc.ConditionalModel(
        iqc.complete_ansatz(N_QUBITS),
        iqc.encoder("product", angles),
    )


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
    model: iqc.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    result = iqc.train(model, training, validation, seed=9, config=config)
    assert result.best_validation_loss < result.validation_losses[0]
    assert result.best_step >= 1
    assert result.train_losses.shape == (config.steps,)
    assert result.best_params.shape == (model.ansatz.n_parameters,)


def test_ancilla_model_trains_against_visible_targets(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A five-qubit model with two ancillas trains against three-bit targets."""
    training, validation = targets
    angles = np.random.default_rng(7).uniform(0.4, 2.2, size=(N_CONDITIONS, 5))
    angles = np.broadcast_to(angles[:, :1], angles.shape)
    model = iqc.ConditionalModel(
        iqc.complete_ansatz(5),
        iqc.encoder("spin_flip", angles),
        n_visible=N_QUBITS,
    )
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
    result = iqc.train(model, training, validation, seed=10, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_params.shape == (model.ansatz.n_parameters,)
    assert 0 <= result.best_step <= config.steps


def test_model_rejects_mismatched_widths() -> None:
    with pytest.raises(ValueError, match="ansatz acts on 4 qubits"):
        iqc.ConditionalModel(
            iqc.complete_ansatz(4),
            iqc.encoder("product", np.zeros((2, 3))),
        )
    with pytest.raises(ValueError, match="n_visible must be between"):
        iqc.ConditionalModel(
            iqc.complete_ansatz(3),
            iqc.encoder("product", np.zeros((2, 3))),
            n_visible=4,
        )


def test_visible_width_mismatch_is_rejected(
    model: iqc.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    wide = np.concatenate([training, np.zeros_like(training[:, :, :1])], axis=2)
    with pytest.raises(ValueError, match="declares 3 visible qubits"):
        iqc.train(model, wide, validation, seed=1, config=config)


def test_step_refresh_is_the_default_and_cache_changes_the_objective(
    model: iqc.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: iqc.TrainingConfig,
) -> None:
    """The default redraws masks every step; 'cache' retains one bank."""
    training, validation = targets
    assert config.mask_refresh == iqc.STEP

    step_result = iqc.train(model, training, validation, seed=4, config=config)
    explicit = iqc.train(
        model,
        training,
        validation,
        seed=4,
        config=dataclasses.replace(config, mask_refresh=iqc.STEP),
    )
    assert np.array_equal(step_result.train_losses, explicit.train_losses)

    cache_result = iqc.train(
        model,
        training,
        validation,
        seed=4,
        config=dataclasses.replace(config, mask_refresh=iqc.CACHE),
    )
    assert not np.array_equal(step_result.final_params, cache_result.final_params)
    assert np.all(np.isfinite(step_result.train_losses))


def test_condition_count_mismatch_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    config: iqc.TrainingConfig,
) -> None:
    training, validation = targets
    model = iqc.ConditionalModel(
        iqc.complete_ansatz(N_QUBITS),
        iqc.encoder("product", np.zeros((N_CONDITIONS + 1, N_QUBITS))),
    )
    with pytest.raises(ValueError, match="conditions but targets"):
        iqc.train(model, training, validation, seed=1, config=config)


def test_sample_chunk_training_runs(
    model: iqc.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """The streaming option works through the trainer."""
    training, validation = targets
    config = iqc.TrainingConfig(
        steps=10,
        batch_size=8,
        validation_interval=5,
        sigmas=(1.0,),
        n_ops=8,
        n_iqp_samples=32,
        validation_n_ops=8,
        validation_n_iqp_samples=32,
        sample_chunk=16,
    )
    result = iqc.train(model, training, validation, seed=3, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_step >= 0


def test_training_config_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        iqc.TrainingConfig(steps=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        iqc.TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning rates must be positive"):
        iqc.TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="sigmas must be positive"):
        iqc.TrainingConfig(sigmas=())
    with pytest.raises(ValueError, match="sigmas must be positive"):
        iqc.TrainingConfig(sigmas=(1.0, -2.0))
    with pytest.raises(ValueError, match="n_ops and validation_n_ops"):
        iqc.TrainingConfig(n_ops=0)
    with pytest.raises(ValueError, match="at least two"):
        iqc.TrainingConfig(n_iqp_samples=1)
    with pytest.raises(ValueError, match="sample_chunk must be positive"):
        iqc.TrainingConfig(sample_chunk=0)
    with pytest.raises(ValueError, match="unknown mask_refresh"):
        iqc.TrainingConfig(mask_refresh="sometimes")
