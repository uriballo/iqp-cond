"""The step-noise measurement must predict the trainer's gradient noise."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond import guidelines
from iqp_cond.encoders.base import _select_condition
from iqp_cond.estimator import _gradient_terms
from iqp_cond.exact import probabilities
from iqp_cond.loss import _Entries, _mmd_from_entries, target_moments
from iqp_cond.operators import sample_nonidentity_masks_jax

N_QUBITS = 4


def zero_state_model() -> ciqp.ConditionalModel:
    """Plain IQP: one condition whose input state is ``|0>``."""
    encoder = ciqp.AngleEncoder(np.zeros((1, 1, N_QUBITS)), axes=("Y" * N_QUBITS,))
    return ciqp.ConditionalModel(ciqp.Ansatz.complete(N_QUBITS), encoder)


def ancilla_model() -> ciqp.ConditionalModel:
    """Two conditions of a complex angle encoder with two ancillas."""
    angles = np.random.default_rng(1).uniform(0.3, 2.0, (2, 2, N_QUBITS + 2))
    encoder = ciqp.AngleEncoder(
        angles, axes=("Y" * (N_QUBITS + 2), "X" * (N_QUBITS + 2))
    )
    return ciqp.ConditionalModel(
        ciqp.Ansatz.complete(N_QUBITS + 2), encoder, n_visible=N_QUBITS
    )


def model_params(model: ciqp.ConditionalModel) -> jnp.ndarray:
    return jnp.asarray(
        np.random.default_rng(0).normal(0.0, 0.4, model.ansatz.n_parameters)
    )


@pytest.mark.parametrize("build", [zero_state_model, ancilla_model])
def test_summed_terms_give_the_cell_loss_gradient(build) -> None:
    """The per-sample terms sum to the gradient of the trainer's cell loss."""
    model = build()
    params = model_params(model)
    generators = jnp.asarray(model.generators, dtype=jnp.float64)
    encoder = _select_condition(model.encoder, model.encoder.n_conditions - 1)
    masks, weights = sample_nonidentity_masks_jax(
        jax.random.PRNGKey(2), 16, N_QUBITS, 1.2
    )
    samples = encoder.sample(jax.random.PRNGKey(3), 64)[0]
    batch = jnp.asarray(
        np.random.default_rng(4).integers(0, 2, (1, 32, N_QUBITS)), dtype=jnp.float64
    )
    means, squares = target_moments(batch, masks)
    entry = _Entries(
        jnp.pad(masks, ((0, 0), (0, model.n_qubits - N_QUBITS)))[None],
        weights[None],
        samples[None, None],
        means,
        squares,
    )
    expected = jax.grad(
        lambda p: _mmd_from_entries(p, generators, encoder, (entry,)).loss
    )(params)

    v, s, C, F = _gradient_terms(params, generators, encoder, masks, samples)
    n = samples.shape[0]
    sums = s.T @ C
    model_squares = 2.0 * (v.sum(0)[:, None] * sums - (v * s).T @ C) / (n * (n - 1))
    per_mask = (model_squares - 2.0 * means[0][:, None] * sums / n) * F
    np.testing.assert_allclose(weights @ per_mask, expected, rtol=1e-10, atol=1e-13)


def test_parts_predict_the_trainer_gradient_variance() -> None:
    """``A/R + sigma_1^2/S + B_int/(RS)`` matches the trainer's cell gradient.

    The empirical trace over 4000 keys has a relative standard error of at
    most ``sqrt(2/4000)``, about 2%; the mean of eight 4096-mask tables
    measured within 5% at ``S = 32`` and less at larger ``S``.  A 15%
    tolerance is three combined standard errors.
    """
    model = zero_state_model()
    params = model_params(model)
    generators = jnp.asarray(model.generators, dtype=jnp.float64)
    encoder = _select_condition(model.encoder, 0)
    pool = jnp.asarray(
        np.random.default_rng(5).integers(0, 2, (1, 300, N_QUBITS)), dtype=jnp.float64
    )
    sigma = 1.0
    for n_samples, n_masks in ((32, 8), (128, 32), (512, 4)):
        predictions = []
        for index in range(8):
            masks, weights = sample_nonidentity_masks_jax(
                jax.random.PRNGKey(100 + index), 4096, N_QUBITS, sigma
            )
            A, sigma_1, B = guidelines._cell_parts(
                params,
                generators,
                encoder,
                masks,
                weights,
                pool[0],
                jax.random.PRNGKey(200 + index),
                n_samples,
            )
            predictions.append(
                A / n_masks + sigma_1 / n_samples + B / (n_masks * n_samples)
            )
        gradients = jax.jit(
            jax.vmap(
                jax.grad(
                    lambda p, key, n_samples=n_samples, n_masks=n_masks: (
                        model.mmd(
                            p,
                            pool,
                            key,
                            sigmas=(sigma,),
                            n_ops=n_masks,
                            n_samples=n_samples,
                            shared_samples=True,
                        ).loss
                    )
                ),
                in_axes=(None, 0),
            )
        )(params, jax.random.split(jax.random.PRNGKey(7), 4000))
        empirical = float(jnp.var(gradients, axis=0, ddof=1).sum())
        assert np.mean(predictions) == pytest.approx(empirical, rel=0.15)


def test_gamma_matches_the_dense_off_diagonal_ratio() -> None:
    """The cell's gamma equals the dense ratio over pairs of distinct masks.

    The data are drawn from the model.  The dense reference uses its own
    16 384 covariance and two 16 384-sample gradient sets; repeated draws of
    either estimate spread by about 1%, so 5% is a loose bound.
    """
    model = zero_state_model()
    params = model_params(model)
    generators = jnp.asarray(model.generators, dtype=jnp.float64)
    encoder = _select_condition(model.encoder, 0)
    rng = np.random.default_rng(6)
    states = rng.choice(2**N_QUBITS, size=20_000, p=probabilities(model, params)[0])
    rows = jnp.asarray((states[:, None] >> np.arange(N_QUBITS)) & 1, jnp.float64)
    masks, weights = sample_nonidentity_masks_jax(
        jax.random.PRNGKey(8), 128, N_QUBITS, 1.0
    )
    numerator, denominator = guidelines._cell_gamma(
        params, generators, encoder, masks, weights, rows, jax.random.PRNGKey(9)
    )

    def terms(key):
        samples = encoder.sample(key, 16_384)[0]
        return _gradient_terms(params, generators, encoder, masks, samples)

    keys = jax.random.split(jax.random.PRNGKey(10), 3)
    covariance = jnp.cov(terms(keys[0])[0].T)
    first, second = (
        s.T @ C / C.shape[0] * F for _, s, C, F in (terms(keys[1]), terms(keys[2]))
    )
    products = first @ second.T
    characters = 1.0 - 2.0 * jnp.mod(rows @ masks.T, 2.0)
    data = jnp.cov(characters.T, bias=True)
    distinct = 1.0 - jnp.eye(masks.shape[0])
    dense = jnp.sum(products * covariance * distinct) / jnp.sum(
        products * data * distinct
    )
    assert numerator / denominator == pytest.approx(float(dense), rel=0.05)


def test_settings_round_cap_and_keep(monkeypatch: pytest.MonkeyPatch) -> None:
    """S and R round to powers of two, R is capped, invalid estimates keep."""
    model = zero_state_model()
    targets = np.random.default_rng(11).integers(0, 2, (1, 64, N_QUBITS))
    config = ciqp.TrainingConfig(batch_size=100, sigmas=(1.0, 2.0))
    cells = {"gamma": (0.3, 1.0), "parts": (3.0, 1.0, 0.0)}
    monkeypatch.setattr(guidelines, "_cell_gamma", lambda *_: cells["gamma"])
    monkeypatch.setattr(
        guidelines, "_cell_parts", lambda *_: np.asarray(cells["parts"])
    )

    def settings(**options):
        return guidelines.measure(
            model,
            model_params(model),
            targets,
            config,
            jax.random.PRNGKey(0),
            **options,
        )

    measured = settings()
    assert (measured.samples, measured.masks, measured.flag) == (32, 128, "estimated")
    assert measured.gamma == pytest.approx(0.3)
    assert measured.mask_ratio == pytest.approx(96.0)
    capped = settings(r_cap=100)
    assert (capped.masks, capped.flag) == (100, "capped")
    cells["gamma"] = (0.01, 1.0)
    assert settings().samples == 4

    cells["parts"] = (3.0, -1.0, 0.0)
    with pytest.raises(ValueError, match="sigma_1"):
        settings()
    kept = guidelines._measure(
        model,
        model_params(model),
        targets,
        config,
        jax.random.PRNGKey(0),
        64,
        100,
        (16, 8),
    )
    assert (kept.samples, kept.masks, kept.flag) == (16, 8, "kept")
