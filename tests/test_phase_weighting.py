"""Regression coverage for the opt-in sample-side phase contraction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_fused_training import training_case

import iqp_cond as ciqp
from iqp_cond.operators import operator_bank


@dataclass(frozen=True)
class ControlledPhaseEncoder:
    """Small non-product, non-pytree encoder for general estimator coverage."""

    n_conditions: int = 2
    n_qubits: int = 6
    visible: int = 4

    def sample(self, key, n_samples):
        return jax.random.bernoulli(
            key, 0.5, (self.n_conditions, n_samples, self.n_qubits)
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples, masks):
        # The phase couples the first visible qubit to the last ancilla.
        base = samples[..., 0] * samples[..., -1]
        flipped = jnp.mod(samples[..., None, :] + masks[None, None, :, :], 2.0)
        changed = flipped[..., 0] * flipped[..., -1] - base[..., None]
        return jnp.exp(1j * (0.37 * changed))

    def select(self, condition):
        return ControlledPhaseEncoder(1, self.n_qubits, self.visible)


def _case():
    rng = np.random.default_rng(917)
    encoder = ControlledPhaseEncoder()
    generators = rng.integers(0, 2, (20, 6)).astype(float)
    generators[0] = 0
    masks = rng.integers(0, 2, (9, 6)).astype(float)
    masks[0] = 0
    masks[1] = masks[2]
    samples = rng.integers(0, 2, (2, 31, 6)).astype(float)
    params = rng.normal(0.0, 0.2, generators.shape[0])
    return params, generators, encoder, masks, samples


@pytest.mark.parametrize("gemm_dtype, atol", [(None, 1e-11), (jnp.float32, 1e-5)])
def test_sample_weighting_matches_operator_weighting(gemm_dtype, atol):
    params, generators, encoder, masks, samples = _case()

    def values(theta, weighting):
        result = ciqp.estimate_pauli_z(
            theta,
            generators,
            encoder,
            masks,
            samples=samples,
            gemm_dtype=gemm_dtype,
            phase_weighting=weighting,
            return_samples=True,
        )
        return result.sample_values, result.standard_errors

    operators = values(params, "operators")
    sampled = values(params, "samples")
    np.testing.assert_allclose(operators[0][:, :, 0], 1.0, atol=atol)
    np.testing.assert_allclose(operators[0], sampled[0], rtol=atol, atol=atol)
    np.testing.assert_allclose(operators[1], sampled[1], rtol=atol, atol=atol)

    for weighting in ("operators", "samples"):
        grad = jax.grad(
            lambda theta, mode=weighting: jnp.sum(values(theta, mode)[0] ** 2)
        )(jnp.asarray(params))
        assert grad[0] == 0
        if weighting == "operators":
            reference = grad
        else:
            np.testing.assert_allclose(reference, grad, rtol=atol, atol=atol)

    tangent = jnp.linspace(-0.1, 0.1, params.size)
    for weighting in ("operators", "samples"):
        _, jvp = jax.jvp(
            lambda theta, mode=weighting: values(theta, mode)[0],
            (jnp.asarray(params),),
            (tangent,),
        )
        if weighting == "operators":
            reference_jvp = jvp
        else:
            np.testing.assert_allclose(reference_jvp, jvp, rtol=atol, atol=atol)

    zero_grad = jax.grad(lambda theta: jnp.sum(values(theta, "samples")[0][:, :, 0]))(
        jnp.asarray(params)
    )
    np.testing.assert_allclose(zero_grad, 0.0, atol=atol)


@pytest.mark.parametrize(
    "bank, shared, chunk, dtype",
    [
        (False, False, None, None),
        (False, True, 8, jnp.float32),
        (True, False, 8, None),
        (True, True, None, jnp.float32),
    ],
)
def test_mmd_weighting_matches_for_all_sampling_paths(bank, shared, chunk, dtype):
    params, generators, encoder, _masks, _samples = _case()
    targets = np.random.default_rng(29).integers(0, 2, (2, 9, 4)).astype(float)
    kwargs = dict(
        n_samples=31,
        shared_samples=shared,
        sample_chunk=chunk,
        gemm_dtype=dtype,
    )
    if bank:
        source = {"bank": operator_bank((0.8, 1.2), 8, 2, 4, seed=11)}
    else:
        source = {"sigmas": (0.8, 1.2), "n_ops": 8}

    def objective(theta, weighting):
        result = ciqp.mmd(
            theta,
            generators,
            encoder,
            targets,
            jax.random.PRNGKey(77),
            phase_weighting=weighting,
            **source,
            **kwargs,
        )
        return result.loss

    op = jax.jit(jax.value_and_grad(lambda theta: objective(theta, "operators")))
    sample = jax.jit(jax.value_and_grad(lambda theta: objective(theta, "samples")))
    op_value, op_grad = op(jnp.asarray(params))
    sample_value, sample_grad = sample(jnp.asarray(params))
    atol = 1e-5 if dtype is not None else 1e-11
    np.testing.assert_allclose(op_value, sample_value, rtol=atol, atol=atol)
    np.testing.assert_allclose(op_grad, sample_grad, rtol=atol, atol=atol)


@pytest.mark.parametrize(
    "case", ["real_step", "mixed_chunk", "complex_ancilla", "custom"]
)
def test_training_sample_weighting_preserves_trajectory(case):
    model, train, valid, config, initial = training_case(case)
    sampled_config = replace(config, phase_weighting="samples")
    operators = ciqp.train(
        model, train, valid, seed=31, config=config, initial_params=initial
    )
    sampled = ciqp.train(
        model, train, valid, seed=31, config=sampled_config, initial_params=initial
    )
    atol = 1e-5 if config.gemm_dtype is not None else 1e-11
    assert sampled.best_step == operators.best_step
    np.testing.assert_allclose(sampled.best_params, operators.best_params, atol=atol)
    np.testing.assert_allclose(sampled.train_losses, operators.train_losses, atol=atol)
    np.testing.assert_allclose(
        sampled.validation_losses, operators.validation_losses, atol=atol
    )
    np.testing.assert_allclose(sampled.final_params, operators.final_params, atol=atol)


@pytest.mark.parametrize("call", ["estimator", "mmd", "config"])
def test_phase_weighting_rejects_unknown_values(call):
    params, generators, encoder, masks, samples = _case()
    if call == "estimator":
        with pytest.raises(ValueError, match="phase_weighting"):
            ciqp.estimate_pauli_z(
                params,
                generators,
                encoder,
                masks,
                samples=samples,
                phase_weighting="bad",
            )
    elif call == "mmd":
        targets = np.zeros((2, 8, 4))
        with pytest.raises(ValueError, match="phase_weighting"):
            ciqp.mmd(
                params,
                generators,
                encoder,
                targets,
                jax.random.PRNGKey(3),
                sigmas=(0.8,),
                n_ops=4,
                n_samples=8,
                phase_weighting="bad",
            )
    else:
        with pytest.raises(ValueError, match="phase_weighting"):
            ciqp.TrainingConfig(phase_weighting="bad")


def test_model_estimator_forwards_sample_weighting():
    _params, _generators, encoder, masks, samples = _case()
    model = ciqp.ConditionalModel(ciqp.complete_ansatz(6), encoder, n_visible=4)
    params = jnp.linspace(-0.1, 0.1, model.ansatz.n_parameters)
    result = model.estimate_pauli_z(
        params,
        masks[:, :4],
        samples=samples,
        phase_weighting="samples",
        return_samples=True,
    )
    direct = ciqp.estimate_pauli_z(
        params,
        model.generators,
        encoder,
        masks[:, :4],
        samples=samples,
        phase_weighting="samples",
        return_samples=True,
    )
    np.testing.assert_allclose(result.values, direct.values)


def test_public_paths_forward_phase_weighting(monkeypatch):
    params, generators, encoder, masks, samples = _case()
    import iqp_cond.estimator as estimator

    seen = []
    sample_counts = []
    original = estimator._phase_differences

    def spy(*args, **kwargs):
        seen.append(kwargs.get("phase_weighting", args[-1]))
        sample_counts.append(args[3].shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator, "_phase_differences", spy)
    ciqp.estimate_pauli_z(
        params, generators, encoder, masks, samples=samples, phase_weighting="samples"
    )
    ciqp.mmd(
        params,
        generators,
        encoder,
        np.zeros((2, 8, 4)),
        jax.random.PRNGKey(2),
        sigmas=(0.8,),
        n_ops=4,
        n_samples=8,
        phase_weighting="samples",
    )
    model, train, valid, config, initial = training_case("custom")
    sample_counts.clear()
    ciqp.train(
        model,
        train,
        valid,
        seed=31,
        config=replace(config, phase_weighting="samples"),
        initial_params=initial,
    )
    assert {17, 19}.issubset(sample_counts)  # Training and validation draws.
    assert seen and set(seen) == {"samples"}
