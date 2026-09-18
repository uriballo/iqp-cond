"""The conditional Pauli estimator: sources, ancilla padding, error paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as iqc

N_VISIBLE = 3
N_ANCILLAS = 2
N_CONDITIONS = 2


@pytest.fixture
def setup() -> tuple[iqc.Ansatz, iqc.Encoder, np.ndarray]:
    ansatz = iqc.complete_ansatz(N_VISIBLE + N_ANCILLAS)
    rng = np.random.default_rng(31)
    angles = np.broadcast_to(
        rng.uniform(0.4, 2.2, size=(N_CONDITIONS, 1)),
        (N_CONDITIONS, N_VISIBLE + N_ANCILLAS),
    )
    encoders = iqc.encoder(iqc.SPIN_FLIP, angles)
    return (
        ansatz,
        encoders,
        rng.integers(0, 2, size=(N_CONDITIONS, 4, N_VISIBLE)).astype(float),
    )


def test_visible_masks_are_padded_with_identity_on_ancillas(
    setup: tuple[iqc.Ansatz, iqc.Encoder, np.ndarray],
) -> None:
    ansatz, encoders, _targets = setup
    visible_masks = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(0)

    padded = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        visible_masks,
        key=key,
        n_samples=64,
    )
    model_masks = np.asarray(iqc.pad_visible_masks(visible_masks, N_ANCILLAS))
    direct = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        model_masks,
        key=key,
        n_samples=64,
    )
    assert padded.values.shape == (N_CONDITIONS, 2)
    assert np.array_equal(np.asarray(padded.values), np.asarray(direct.values))


def test_pre_drawn_samples_match_the_key_source(setup) -> None:
    """Passing samples evaluates exactly those strings."""
    ansatz, encoders, _targets = setup
    masks = np.ones((1, N_VISIBLE))
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(7)

    drawn = encoders.sample(key, 32)
    from_samples = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        samples=drawn,
    )
    unchecked = iqc.estimate_pauli_z_unchecked(
        params,
        ansatz.generators,
        encoders,
        iqc.pad_visible_masks(masks, N_ANCILLAS),
        samples=drawn,
    )
    assert np.array_equal(
        np.asarray(from_samples.values),
        np.asarray(unchecked.values),
    )


def test_rank_two_samples_require_a_single_condition() -> None:
    """One shared draw across different reference states would be biased."""
    ansatz = iqc.complete_ansatz(N_VISIBLE)
    generators = ansatz.generators
    masks = np.ones((1, N_VISIBLE))
    params = np.zeros(ansatz.n_parameters)
    rng = np.random.default_rng(32)
    draw = rng.integers(0, 2, size=(16, N_VISIBLE)).astype(np.float64)

    single = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(1, N_VISIBLE)))
    estimate = iqc.estimate_pauli_z(
        params,
        generators,
        single,
        masks,
        samples=draw,
    )
    assert estimate.values.shape == (1, 1)
    assert estimate.n_samples == 16

    paired = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, N_VISIBLE)))
    with pytest.raises(ValueError, match="single condition"):
        iqc.estimate_pauli_z(params, generators, paired, masks, samples=draw)
    with pytest.raises(ValueError, match="shape \\(conditions, samples, qubits\\)"):
        iqc.estimate_pauli_z(
            params,
            generators,
            single,
            masks,
            samples=draw.ravel(),
        )


def test_exactly_one_sample_source_is_required(setup) -> None:
    ansatz, encoders, _targets = setup
    params = np.zeros(ansatz.n_parameters)
    masks = np.ones((1, N_VISIBLE))
    with pytest.raises(ValueError, match="exactly one of key or samples"):
        iqc.estimate_pauli_z(params, ansatz.generators, encoders, masks)
    with pytest.raises(ValueError, match="exactly one of key or samples"):
        iqc.estimate_pauli_z(
            params,
            ansatz.generators,
            encoders,
            masks,
            key=jax.random.PRNGKey(0),
            samples=encoders.sample(jax.random.PRNGKey(0), 8),
            n_samples=8,
        )


def test_n_samples_is_required_with_a_key(setup) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="n_samples is required"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.ones((1, N_VISIBLE)),
            key=jax.random.PRNGKey(0),
        )


def test_standard_errors_need_per_sample_values(setup) -> None:
    ansatz, encoders, _targets = setup
    estimate = iqc.estimate_pauli_z(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        encoders,
        np.ones((1, N_VISIBLE)),
        key=jax.random.PRNGKey(0),
        n_samples=16,
    )
    assert estimate.sample_values is None
    with pytest.raises(ValueError, match="return_samples=True"):
        _ = estimate.standard_errors
    with_samples = iqc.estimate_pauli_z(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        encoders,
        np.ones((1, N_VISIBLE)),
        key=jax.random.PRNGKey(0),
        n_samples=16,
        return_samples=True,
    )
    assert with_samples.standard_errors.shape == (N_CONDITIONS, 1)


def test_masks_wider_than_the_register_are_rejected(setup) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="more qubits than the generators"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.ones((1, N_VISIBLE + N_ANCILLAS + 1)),
            key=jax.random.PRNGKey(0),
            n_samples=64,
        )


def test_non_binary_masks_are_rejected(setup) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="only zeros and ones"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.full((1, N_VISIBLE + N_ANCILLAS), 0.5),
            key=jax.random.PRNGKey(0),
            n_samples=64,
        )


def test_too_few_samples_are_rejected(setup) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="at least two"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.ones((1, N_VISIBLE)),
            key=jax.random.PRNGKey(0),
            n_samples=1,
        )


def test_encoder_width_mismatch_is_rejected(setup) -> None:
    ansatz, _encoders, _targets = setup
    smaller = iqc.encoder(iqc.PRODUCT, np.zeros((N_CONDITIONS, N_VISIBLE)))
    with pytest.raises(ValueError, match="encoder acts on"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            smaller,
            np.ones((1, N_VISIBLE + N_ANCILLAS)),
            key=jax.random.PRNGKey(0),
            n_samples=64,
        )


def test_unchecked_kernel_is_jittable(setup) -> None:
    ansatz, encoders, _targets = setup
    masks = np.ones((1, N_VISIBLE + N_ANCILLAS))
    params = np.zeros(ansatz.n_parameters)

    def estimate(current_params: jax.Array) -> jax.Array:
        return iqc.estimate_pauli_z_unchecked(
            current_params,
            ansatz.generators,
            encoders,
            masks,
            key=jax.random.PRNGKey(0),
            n_samples=64,
        ).values

    assert jax.jit(estimate)(params).shape == (N_CONDITIONS, 1)


def test_mixed_precision_gemm_stays_close_to_float64(setup) -> None:
    """Casting only the phase contraction to float32 keeps float64 accuracy."""
    ansatz, encoders, _targets = setup
    masks = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    params = np.linspace(-0.4, 0.4, ansatz.n_parameters)
    key = jax.random.PRNGKey(3)

    reference = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        key=key,
        n_samples=4096,
    )
    mixed = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        key=key,
        n_samples=4096,
        gemm_dtype=jnp.float32,
    )
    deviation = float(
        np.max(np.abs(np.asarray(mixed.values) - np.asarray(reference.values))),
    )
    # Nonzero pins the cast as active; the bound pins it as harmless.
    assert 0.0 < deviation < 1e-5
