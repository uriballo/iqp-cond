"""The training loop must reduce the held-out MMD and respect shapes."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.initialization import parameters_from_data
from iqp_cond.operators import expected_bodyness_sigmas
from iqp_cond.training import CACHE, STEP


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit: the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


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
        ciqp.Ansatz.complete(N_QUBITS),
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
        ciqp.Ansatz.complete(5),
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
            ciqp.Ansatz.complete(4),
            y_layer(np.zeros((2, 3))),
        )
    with pytest.raises(ValueError, match="n_visible must be between"):
        ciqp.ConditionalModel(
            ciqp.Ansatz.complete(3),
            y_layer(np.zeros((2, 3))),
            n_visible=4,
        )


def test_model_requires_an_integer_visible_width() -> None:
    """A float would be truncated silently, so it is rejected."""
    encoders = y_layer(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="n_visible must be an integer"):
        ciqp.ConditionalModel(ciqp.Ansatz.complete(3), encoders, n_visible=2.5)
    integral = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(3),
        encoders,
        n_visible=np.int64(2),
    )
    assert integral.visible == 2


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
    assert config.mask_refresh == STEP

    step_result = ciqp.train(model, training, validation, seed=4, config=config)
    explicit = ciqp.train(
        model,
        training,
        validation,
        seed=4,
        config=dataclasses.replace(config, mask_refresh=STEP),
    )
    assert np.array_equal(step_result.train_losses, explicit.train_losses)

    cache_result = ciqp.train(
        model,
        training,
        validation,
        seed=4,
        config=dataclasses.replace(config, mask_refresh=CACHE),
    )
    assert not np.array_equal(step_result.final_params, cache_result.final_params)
    assert np.all(np.isfinite(step_result.train_losses))


def test_condition_count_mismatch_is_rejected(
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(N_QUBITS),
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
    with pytest.raises(ValueError, match="validation_interval must be positive"):
        ciqp.TrainingConfig(validation_interval=0)
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
    assert config.mask_refresh == STEP
    assert config.sigmas is None
    n_visible = 12
    assert config.resolved_sigmas(n_visible) == expected_bodyness_sigmas(n_visible)
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


def test_train_defaults_to_the_pooled_data_initialization(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    """The implicit start equals the pooled mean-covariance prescription."""
    training, validation = targets
    default = ciqp.train(model, training, validation, seed=11, config=config)
    explicit = ciqp.train(
        model,
        training,
        validation,
        seed=11,
        config=config,
        initial_params=parameters_from_data(
            training.reshape(-1, N_QUBITS),
            model.ansatz.generators,
            n_visible=N_QUBITS,
            seed=11,
        ),
    )
    assert np.array_equal(default.final_params, explicit.final_params)
    assert np.array_equal(default.train_losses, explicit.train_losses)


def test_train_validates_initial_params(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    with pytest.raises(ValueError, match="initial_params must have shape"):
        ciqp.train(
            model,
            training,
            validation,
            seed=1,
            config=config,
            initial_params=np.zeros(2),
        )
    bad = np.zeros(model.ansatz.n_parameters)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="initial_params must contain only finite"):
        ciqp.train(
            model,
            training,
            validation,
            seed=1,
            config=config,
            initial_params=bad,
        )


def test_train_validates_target_shapes(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    with pytest.raises(ValueError, match="training_targets must be rank three"):
        ciqp.train(model, training[0], validation, seed=1, config=config)
    with pytest.raises(ValueError, match="validation_targets must be rank three"):
        ciqp.train(model, training, validation[0], seed=1, config=config)
    with pytest.raises(ValueError, match="conditions must align"):
        ciqp.train(model, training, validation[:1], seed=1, config=config)
    with pytest.raises(ValueError, match="validation targets"):
        ciqp.train(model, training, validation[:, :, :1], seed=1, config=config)


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
        ciqp.Ansatz.complete(N_QUBITS),
        ciqp.AngleEncoder(angles, axes=("XYX", "ZYZ")),
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


def test_training_is_deterministic_in_the_seed(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
) -> None:
    training, validation = targets
    first = ciqp.train(model, training, validation, seed=21, config=config)
    repeat = ciqp.train(model, training, validation, seed=21, config=config)
    other = ciqp.train(model, training, validation, seed=22, config=config)
    assert np.array_equal(first.final_params, repeat.final_params)
    assert np.array_equal(first.train_losses, repeat.train_losses)
    assert not np.array_equal(first.final_params, other.final_params)


def test_complex_ancilla_model_trains(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A complex layered encoder on an ancilla register trains end to end."""
    training, validation = targets
    angles = np.random.default_rng(27).uniform(
        0.3,
        1.2,
        size=(N_CONDITIONS, 2, N_QUBITS + 2),
    )
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(N_QUBITS + 2),
        ciqp.AngleEncoder(angles, axes=("XYXYX", "ZYZYZ")),
        n_visible=N_QUBITS,
    )
    config = ciqp.TrainingConfig(
        steps=6,
        batch_size=8,
        learning_rate=0.015,
        final_learning_rate=0.002,
        validation_interval=3,
        sigmas=(0.8, 1.3),
        n_ops=8,
        mc_samples=17,
        validation_n_ops=8,
        validation_mc_samples=19,
    )
    result = ciqp.train(model, training, validation, seed=31, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_params.shape == (model.ansatz.n_parameters,)


def test_mixed_precision_and_chunked_training_run(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    training, validation = targets
    config = ciqp.TrainingConfig(
        steps=6,
        batch_size=8,
        learning_rate=0.015,
        final_learning_rate=0.002,
        validation_interval=3,
        sigmas=(0.8, 1.3),
        n_ops=8,
        mc_samples=17,
        validation_n_ops=8,
        validation_mc_samples=19,
        gemm_dtype=jnp.float32,
        sample_chunk=8,
    )
    result = ciqp.train(model, training, validation, seed=31, config=config)
    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert result.best_params.shape == (model.ansatz.n_parameters,)


def _exact_objective(
    probabilities: np.ndarray,
    targets: np.ndarray,
    sigma: float,
) -> float:
    """Exact conditional MMD from model probabilities and target samples.

    Independent of :mod:`iqp_cond.loss`: the Hamming-RBF character expansion
    gives ``k(x, y) = sum_a P(a) chi_a(x) chi_a(y)`` with
    ``P(a) = prod_i p^{a_i} (1-p)^{1-a_i}`` and
    ``p = (1 - exp(-1/(2 sigma^2))) / 2``.
    """
    n_visible = targets.shape[-1]
    indices = np.arange(2**n_visible)
    masks = ((indices[:, None] >> np.arange(n_visible)) & 1).astype(np.float64)
    p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma**2)))
    weights = np.prod(np.where(masks == 1.0, p, 1.0 - p), axis=1)
    characters = 1.0 - 2.0 * np.bitwise_count(indices[:, None] & indices[None, :])

    terms = []
    for condition in range(targets.shape[0]):
        target_mean = targets[condition].mean(axis=0)
        mu_p = np.prod(
            np.where(masks == 1.0, 1.0 - 2.0 * target_mean, 1.0),
            axis=1,
        )
        mu_q = probabilities[condition] @ characters
        terms.append(float(np.sum(weights * (mu_q - mu_p) ** 2)))
    return float(np.mean(terms))


@pytest.mark.slow
def test_training_reduces_the_exact_objective() -> None:
    """Training lowers the exact MMD, not only its sampled estimate."""
    probabilities = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])
    rng = np.random.default_rng(23)
    training = np.stack(
        [
            (rng.uniform(size=(128, N_QUBITS)) < probabilities[c]).astype(float)
            for c in range(N_CONDITIONS)
        ],
    )
    validation = np.stack(
        [
            (rng.uniform(size=(64, N_QUBITS)) < probabilities[c]).astype(float)
            for c in range(N_CONDITIONS)
        ],
    )
    angles = np.random.default_rng(24).uniform(0.4, 2.2, size=(N_CONDITIONS, N_QUBITS))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(N_QUBITS), y_layer(angles))
    sigma = 2.0
    initial = np.random.default_rng(25).normal(
        0.0,
        0.2,
        size=model.ansatz.n_parameters,
    )
    config = ciqp.TrainingConfig(
        steps=200,
        batch_size=64,
        learning_rate=0.05,
        final_learning_rate=0.005,
        validation_interval=200,
        sigmas=(sigma,),
        n_ops=64,
        mc_samples=128,
        validation_n_ops=64,
        validation_mc_samples=128,
    )
    result = ciqp.train(
        model,
        training,
        validation,
        seed=26,
        config=config,
        initial_params=initial,
    )
    before = _exact_objective(model.probabilities(initial), training, sigma)
    after = _exact_objective(model.probabilities(result.best_params), training, sigma)
    assert after < before
