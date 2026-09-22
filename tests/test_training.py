"""The training loop must reduce the held-out MMD and respect shapes."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond import training as training_module
from iqp_cond.initialization import parameters_from_data
from iqp_cond.loss import _condition_value
from iqp_cond.operators import expected_bodyness_sigmas, operator_bank
from iqp_cond.training import CACHE, STEP


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit, the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


N_QUBITS = 3
N_CONDITIONS = 2


@dataclasses.dataclass(frozen=True)
class WrappedEncoder:
    """Custom encoder deliberately left unregistered as a JAX pytree."""

    inner: ciqp.AngleEncoder

    @property
    def n_conditions(self) -> int:
        return self.inner.n_conditions

    @property
    def n_qubits(self) -> int:
        return self.inner.n_qubits

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        return self.inner.sample(key, n_samples)

    def amplitude_ratios(
        self,
        samples: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.inner.amplitude_ratios(samples, masks)

    def preparation(self, condition: int) -> Callable[[], None]:
        return self.inner.preparation(condition)

    def select(self, condition: int) -> WrappedEncoder:
        return WrappedEncoder(self.inner.select(condition))


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
    """A float would be truncated silently, so the model rejects it."""
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


def test_step_refresh_is_the_default_and_cache_changes_the_trajectory(
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


def test_step_refresh_selects_one_uniform_bandwidth_per_update(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training samples one bandwidth while pilot and validation cover all."""
    training, validation = targets
    config = dataclasses.replace(config, steps=5, validation_interval=5)
    calls: list[tuple[tuple[float, ...], int]] = []
    original = training_module.operator_bank

    def recording_bank(sigmas, n_ops, n_conditions, *args, **kwargs):
        calls.append((tuple(sigmas), n_conditions))
        return original(sigmas, n_ops, n_conditions, *args, **kwargs)

    monkeypatch.setattr(training_module, "operator_bank", recording_bank)
    ciqp.train(model, training, validation, seed=14, config=config)

    expected = training_module._bandwidth_schedule(len(config.sigmas), config.steps, 14)
    assert calls[:2] == [(config.sigmas, N_CONDITIONS)] * 2
    assert calls[2:] == [((config.sigmas[index],), 1) for index in expected]


def test_cache_refresh_builds_one_full_multiscale_bank(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
    config: ciqp.TrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache mode builds one full training bank and one validation bank."""
    training, validation = targets
    config = dataclasses.replace(
        config,
        steps=5,
        validation_interval=5,
        mask_refresh=CACHE,
    )
    calls: list[tuple[float, ...]] = []
    original = training_module.operator_bank

    def recording_bank(sigmas, *args, **kwargs):
        calls.append(tuple(sigmas))
        return original(sigmas, *args, **kwargs)

    monkeypatch.setattr(training_module, "operator_bank", recording_bank)
    ciqp.train(model, training, validation, seed=14, config=config)

    assert calls == [config.sigmas, config.sigmas]


def test_condition_schedule_is_uniform_and_reproducible() -> None:
    first = training_module._condition_schedule(5, 10_000, seed=17)
    second = training_module._condition_schedule(5, 10_000, seed=17)

    assert np.array_equal(first, second)
    assert set(first) == set(range(5))


def test_cached_condition_slice_preserves_exact_bank_entry() -> None:
    bank = operator_bank((0.8, 1.6), 16, N_CONDITIONS, N_QUBITS, seed=70)
    selected = training_module._select_condition(bank, 1)

    assert selected.n_conditions == 1
    assert selected.sigmas == bank.sigmas
    assert selected.widths == bank.widths
    assert jnp.array_equal(selected.masks[:, 0], bank.masks[:, 1])
    assert jnp.array_equal(selected.weights[:, 0], bank.weights[:, 1])


def test_selected_bandwidth_gradient_is_unbiased_for_multiscale_loss(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A uniform average of selected bands equals the full bank evaluation."""
    training, _ = targets
    sigmas = (0.8, 1.6, 2.4)
    bank = operator_bank(sigmas, 16, N_CONDITIONS, N_QUBITS, seed=71)
    params = jnp.linspace(-0.2, 0.2, model.ansatz.n_parameters)
    target_batch = jnp.asarray(training[:, :16])
    key = jax.random.PRNGKey(72)

    def objective(current_params, current_bank):
        return model.mmd(
            current_params,
            target_batch,
            key,
            bank=current_bank,
            n_samples=32,
            shared_samples=True,
        ).loss

    full_loss, full_gradient = jax.value_and_grad(objective)(params, bank)
    selected = []
    for bandwidth in range(bank.n_bandwidths):
        selected_bank = training_module._select_bandwidth(bank, bandwidth)
        selected.append(
            jax.value_and_grad(objective)(params, selected_bank),
        )

    selected_loss = jnp.mean(jnp.stack([loss for loss, _ in selected]))
    selected_gradient = jnp.mean(
        jnp.stack([gradient for _, gradient in selected]), axis=0
    )
    assert jnp.allclose(selected_loss, full_loss, atol=1e-15, rtol=0.0)
    assert jnp.allclose(selected_gradient, full_gradient, atol=1e-15, rtol=0.0)


def test_selected_condition_gradient_is_unbiased_for_balanced_loss(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """A uniform average of condition cells equals the balanced objective."""
    training, _ = targets
    bank = operator_bank((1.2,), 16, N_CONDITIONS, N_QUBITS, seed=73)
    params = jnp.linspace(-0.2, 0.2, model.ansatz.n_parameters)
    target_batch = jnp.asarray(training[:, :16])
    key = jax.random.PRNGKey(74)

    def full_objective(current_params):
        return model.mmd(
            current_params,
            target_batch,
            key,
            bank=bank,
            n_samples=32,
            shared_samples=True,
        ).loss

    def selected_objective(current_params):
        values = []
        for condition in range(N_CONDITIONS):
            encoder = model.encoder.select(condition)
            samples = encoder.sample(jax.random.fold_in(key, condition), 32)
            masks, weights = bank.entry(0, condition)
            values.append(
                _condition_value(
                    current_params,
                    model.generators,
                    encoder,
                    target_batch[condition : condition + 1],
                    masks,
                    weights,
                    model.n_ancillas,
                    samples,
                    None,
                    None,
                    None,
                )
            )
        return jnp.mean(jnp.stack(values))

    full_loss, full_gradient = jax.value_and_grad(full_objective)(params)
    selected_loss, selected_gradient = jax.value_and_grad(selected_objective)(params)
    assert jnp.allclose(selected_loss, full_loss, atol=1e-15, rtol=0.0)
    assert jnp.allclose(selected_gradient, full_gradient, atol=1e-15, rtol=0.0)


def test_training_supports_non_pytree_custom_encoder(
    angles: np.ndarray,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """Condition-specific closures keep custom encoders outside JIT arguments."""
    training, validation = targets
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(N_QUBITS),
        WrappedEncoder(y_layer(angles)),
    )
    config = ciqp.TrainingConfig(
        steps=4,
        batch_size=8,
        validation_interval=2,
        sigmas=(1.0,),
        n_ops=4,
        mc_samples=16,
        validation_n_ops=4,
        validation_mc_samples=16,
    )

    result = ciqp.train(model, training, validation, seed=75, config=config)

    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))


def test_encoder_path_uses_shared_pytree_step_for_angle_encoder(
    angles: np.ndarray,
) -> None:
    encoder = y_layer(angles)
    assert training_module._dynamic_encoder_support(encoder)
    assert not training_module._dynamic_encoder_support(WrappedEncoder(encoder))


def test_dynamic_encoder_step_preserves_training_trajectory(
    angles: np.ndarray,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """The shared pytree step has the same deterministic result as the fallback."""
    training, validation = targets
    config = ciqp.TrainingConfig(
        steps=3,
        batch_size=8,
        validation_interval=3,
        sigmas=(1.0,),
        n_ops=4,
        mc_samples=16,
        validation_n_ops=4,
        validation_mc_samples=16,
    )
    dynamic = ciqp.train(
        ciqp.ConditionalModel(ciqp.Ansatz.complete(N_QUBITS), y_layer(angles)),
        training,
        validation,
        seed=76,
        config=config,
    )
    fallback = ciqp.train(
        ciqp.ConditionalModel(
            ciqp.Ansatz.complete(N_QUBITS), WrappedEncoder(y_layer(angles))
        ),
        training,
        validation,
        seed=76,
        config=config,
    )
    np.testing.assert_array_equal(dynamic.train_losses, fallback.train_losses)
    np.testing.assert_array_equal(dynamic.validation_losses, fallback.validation_losses)
    np.testing.assert_array_equal(dynamic.best_params, fallback.best_params)
    np.testing.assert_array_equal(dynamic.final_params, fallback.final_params)


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
    """The streaming option runs through the trainer."""
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
    with pytest.raises(ValueError, match="mutually exclusive"):
        ciqp.TrainingConfig(
            gemm_dtype=jnp.float32,
            estimator_dtype=jnp.float32,
        )
    with pytest.raises(ValueError, match="must be float32, float64, or None"):
        ciqp.TrainingConfig(estimator_dtype=jnp.bfloat16)


def test_training_config_defaults_follow_the_library_protocol() -> None:
    config = ciqp.TrainingConfig()
    assert config.steps == 2000
    assert config.batch_size == 256
    assert config.n_ops == 512
    assert config.validation_n_ops == 512
    assert config.mc_samples == 1000
    assert config.validation_mc_samples == 1000
    assert config.gemm_dtype is None
    assert config.estimator_dtype is None
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
    """A layered encoder runs through selection and jit inside the trainer."""
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


def test_external_selection_metric_selects_checkpoint_and_keeps_mmd_history(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    training, validation = targets
    config = ciqp.TrainingConfig(
        steps=4,
        batch_size=8,
        validation_interval=2,
        n_ops=8,
        mc_samples=16,
        validation_n_ops=8,
        validation_mc_samples=16,
    )
    calls: list[float] = []

    def selection(params: np.ndarray) -> float:
        score = float(len(calls))
        calls.append(float(np.linalg.norm(params)))
        return -score

    result = ciqp.train(
        model,
        training,
        validation,
        seed=21,
        config=config,
        selection_metric=selection,
    )

    assert len(calls) == 3
    assert result.best_step == 4
    assert result.best_selection_score == -2.0
    assert result.selection_scores.tolist() == [0.0, -1.0, -2.0]
    assert result.validation_losses.shape == result.selection_scores.shape
    assert np.all(np.isfinite(result.validation_losses))


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


def test_float32_estimator_training_stays_close_to_default(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """Training and validation both receive the estimator dtype option."""
    training, validation = targets
    base = ciqp.TrainingConfig(
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
        sample_chunk=8,
    )
    initial = np.linspace(-0.05, 0.05, model.ansatz.n_parameters)
    reference = ciqp.train(
        model,
        training,
        validation,
        seed=31,
        config=base,
        initial_params=initial,
    )
    mixed = ciqp.train(
        model,
        training,
        validation,
        seed=31,
        config=dataclasses.replace(base, estimator_dtype=jnp.float32),
        initial_params=initial,
    )
    assert np.max(np.abs(mixed.final_params - reference.final_params)) < 2e-6
    assert np.max(np.abs(mixed.validation_losses - reference.validation_losses)) < 1e-7
    assert mixed.final_params.dtype == np.float64


def test_float32_estimator_chunking_matches_full_samples(
    model: ciqp.ConditionalModel,
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    training, _validation = targets
    params = jnp.linspace(-0.2, 0.2, model.ansatz.n_parameters)
    bank = operator_bank((1.2,), 16, N_CONDITIONS, N_QUBITS, seed=81)
    arguments = (params, jnp.asarray(training[:, :16]), jax.random.PRNGKey(82))
    full = model.mmd(
        *arguments,
        bank=bank,
        n_samples=32,
        shared_samples=True,
        estimator_dtype=jnp.float32,
    )
    chunked = model.mmd(
        *arguments,
        bank=bank,
        n_samples=32,
        shared_samples=True,
        estimator_dtype=jnp.float32,
        sample_chunk=8,
    )
    assert jnp.allclose(chunked.loss, full.loss, atol=1e-14, rtol=0.0)

    def objective(current_params, estimator_dtype=None):
        return model.mmd(
            current_params,
            arguments[1],
            arguments[2],
            bank=bank,
            n_samples=32,
            shared_samples=True,
            estimator_dtype=estimator_dtype,
        ).loss

    reference_gradient = jax.grad(objective)(params)
    mixed_gradient = jax.grad(lambda current: objective(current, jnp.float32))(params)
    relative_error = float(
        jnp.linalg.norm(mixed_gradient - reference_gradient)
        / jnp.linalg.norm(reference_gradient)
    )
    assert relative_error < 2e-6


def _exact_objective(
    probabilities: np.ndarray,
    targets: np.ndarray,
    sigma: float,
) -> float:
    """Exact conditional MMD from model probabilities and target samples.

    Independent of :mod:`iqp_cond.loss`. The Hamming-RBF character expansion
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
