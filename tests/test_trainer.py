"""The training loop must reduce the held-out MMD and respect shapes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import iqp_cond as ciqp


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit: the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.angle_encoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


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
def model(angles: np.ndarray) -> ciqp.ConditionalModel:
    return ciqp.ConditionalModel(
        ciqp.complete_ansatz(N_QUBITS),
        y_layer(angles),
    )


@pytest.fixture
def config() -> ciqp.TrainingConfig:
    return ciqp.TrainingConfig(
        steps=25,
        batch_size=16,
        validation_interval=5,
        sigmas=(1.0, 2.0),
        n_ops=8,
        mc_samples=32,
        validation_n_ops=8,
        validation_mc_samples=32,
    )


def test_training_reduces_the_validation_objective(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    result = ciqp.train(model, training, validation, seed=9, config=config)
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
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(5),
        y_layer(angles),
        n_visible=N_QUBITS,
    )
    config = ciqp.TrainingConfig(
        steps=10,
        batch_size=8,
        validation_interval=5,
        sigmas=(1.0,),
        n_ops=4,
        mc_samples=16,
        validation_n_ops=4,
        validation_mc_samples=16,
    )
    result = ciqp.train(model, training, validation, seed=10, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_params.shape == (model.ansatz.n_parameters,)
    assert 0 <= result.best_step <= config.steps


def test_model_rejects_mismatched_widths() -> None:
    with pytest.raises(ValueError, match="ansatz acts on 4 qubits"):
        ciqp.ConditionalModel(
            ciqp.complete_ansatz(4),
            y_layer(np.zeros((2, 3))),
        )
    with pytest.raises(ValueError, match="n_visible must be between"):
        ciqp.ConditionalModel(
            ciqp.complete_ansatz(3),
            y_layer(np.zeros((2, 3))),
            n_visible=4,
        )


def test_visible_width_mismatch_is_rejected(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    wide = np.concatenate([training, np.zeros_like(training[:, :, :1])], axis=2)
    with pytest.raises(ValueError, match="declares 3 visible qubits"):
        ciqp.train(model, wide, validation, seed=1, config=config)


def test_step_refresh_is_the_default_and_cache_changes_the_objective(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    """The default redraws masks every step; 'cache' retains one bank."""
    training, validation = targets
    assert config.mask_refresh == ciqp.STEP

    step_result = ciqp.train(model, training, validation, seed=4, config=config)
    explicit = ciqp.train(
        model,
        training,
        validation,
        seed=4,
        config=dataclasses.replace(config, mask_refresh=ciqp.STEP),
    )
    assert np.array_equal(step_result.train_losses, explicit.train_losses)

    cache_result = ciqp.train(
        model,
        training,
        validation,
        seed=4,
        config=dataclasses.replace(config, mask_refresh=ciqp.CACHE),
    )
    assert not np.array_equal(step_result.final_params, cache_result.final_params)
    assert np.all(np.isfinite(step_result.train_losses))


def test_condition_count_mismatch_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(N_QUBITS),
        y_layer(np.zeros((N_CONDITIONS + 1, N_QUBITS))),
    )
    with pytest.raises(ValueError, match="conditions but targets"):
        ciqp.train(model, training, validation, seed=1, config=config)


def test_sample_chunk_training_runs(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """The streaming option works through the trainer."""
    training, validation = targets
    config = ciqp.TrainingConfig(
        steps=10,
        batch_size=8,
        validation_interval=5,
        sigmas=(1.0,),
        n_ops=8,
        mc_samples=32,
        validation_n_ops=8,
        validation_mc_samples=32,
        sample_chunk=16,
    )
    result = ciqp.train(model, training, validation, seed=3, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_step >= 0


def test_training_config_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        ciqp.TrainingConfig(steps=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        ciqp.TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning rates must be positive"):
        ciqp.TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="sigmas must be positive"):
        ciqp.TrainingConfig(sigmas=())
    with pytest.raises(ValueError, match="sigmas must be positive"):
        ciqp.TrainingConfig(sigmas=(1.0, -2.0))
    with pytest.raises(ValueError, match="n_ops and validation_n_ops"):
        ciqp.TrainingConfig(n_ops=0)
    with pytest.raises(ValueError, match="at least two"):
        ciqp.TrainingConfig(mc_samples=1)
    with pytest.raises(ValueError, match="sample_chunk must be positive"):
        ciqp.TrainingConfig(sample_chunk=0)
    with pytest.raises(ValueError, match="unknown mask_refresh"):
        ciqp.TrainingConfig(mask_refresh="sometimes")


def test_training_config_defaults_follow_the_library_protocol() -> None:
    config = ciqp.TrainingConfig()
    assert config.steps == 2000
    assert config.batch_size == 256
    assert config.n_ops == 512
    assert config.validation_n_ops == 512
    assert config.mc_samples == 1000
    assert config.validation_mc_samples == 1000
    assert config.mask_refresh == ciqp.STEP
    assert config.sigmas is None
    n_visible = 12
    assert config.resolved_sigmas(n_visible) == ciqp.expected_bodyness_sigmas(n_visible)
    explicit = ciqp.TrainingConfig(sigmas=(1.0, 2.0, 4.0))
    assert explicit.resolved_sigmas(n_visible) == (1.0, 2.0, 4.0)


def test_default_sigmas_resolve_and_train(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    training, validation = targets
    config = ciqp.TrainingConfig(
        steps=5,
        batch_size=8,
        validation_interval=1,
        n_ops=16,
        mc_samples=32,
        validation_n_ops=16,
        validation_mc_samples=32,
    )
    assert config.sigmas is None
    result = ciqp.train(model, training, validation, seed=12, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))


def test_train_initializes_from_the_pooled_training_data(
    monkeypatch: pytest.MonkeyPatch,
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """Without initial_params the trainer pools the conditions for initialization."""
    training, validation = targets
    from iqp_cond.initialization import parameters_from_data as original

    seen: dict[str, object] = {}

    def spy(data: np.ndarray, generators: np.ndarray, **kwargs: object) -> np.ndarray:
        seen["data"] = np.asarray(data)
        seen["kwargs"] = kwargs
        return original(data, generators, **kwargs)

    monkeypatch.setattr("iqp_cond.trainer.parameters_from_data", spy)
    config = ciqp.TrainingConfig(
        steps=1,
        batch_size=8,
        validation_interval=1,
        sigmas=(1.0,),
        n_ops=8,
        mc_samples=16,
        validation_n_ops=8,
        validation_mc_samples=16,
    )
    ciqp.train(model, training, validation, seed=11, config=config)
    pooled = seen["data"]
    assert isinstance(pooled, np.ndarray)
    assert pooled.shape == (training.shape[0] * training.shape[1], N_QUBITS)
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["n_visible"] == N_QUBITS
    assert kwargs["seed"] == 11


def test_train_rejects_non_binary_targets(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    fractional = training.copy()
    fractional[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="only finite binary values"):
        ciqp.train(model, fractional, validation, seed=1, config=config)


def test_train_rejects_splits_with_one_sample_per_condition(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    with pytest.raises(ValueError, match="at least two target samples"):
        ciqp.train(model, training[:, :1, :], validation, seed=1, config=config)


def test_angle_encoder_trains_end_to_end(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A layered encoder survives selection and jit inside the trainer."""
    training, validation = targets
    angles = np.random.default_rng(16).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 2, N_QUBITS),
    )
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(N_QUBITS),
        ciqp.angle_encoder(angles, axes=("XYX", "ZYZ")),
    )
    config = ciqp.TrainingConfig(
        steps=5,
        batch_size=8,
        validation_interval=5,
        n_ops=8,
        mc_samples=16,
        validation_n_ops=8,
        validation_mc_samples=16,
    )
    result = ciqp.train(model, training, validation, seed=13, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_params.shape == (model.ansatz.n_parameters,)
