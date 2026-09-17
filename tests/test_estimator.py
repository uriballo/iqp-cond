"""The conditional Pauli estimator: ancilla padding and public error paths."""

from __future__ import annotations

import jax
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
        visible_masks,
        encoders,
        key,
        64,
    )
    model_masks = np.asarray(iqc.pad_visible_masks(visible_masks, N_ANCILLAS))
    direct = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        model_masks,
        encoders,
        key,
        64,
    )
    assert padded.values.shape == (N_CONDITIONS, 2)
    assert np.array_equal(np.asarray(padded.values), np.asarray(direct.values))


def test_masks_wider_than_the_register_are_rejected(
    setup: tuple[iqc.Ansatz, iqc.Encoder, np.ndarray],
) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="more qubits than the generators"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            np.ones((1, N_VISIBLE + N_ANCILLAS + 1)),
            encoders,
            jax.random.PRNGKey(0),
            64,
        )


def test_non_binary_masks_are_rejected(
    setup: tuple[iqc.Ansatz, iqc.Encoder, np.ndarray],
) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="only zeros and ones"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            np.full((1, N_VISIBLE + N_ANCILLAS), 0.5),
            encoders,
            jax.random.PRNGKey(0),
            64,
        )


def test_too_few_samples_are_rejected(
    setup: tuple[iqc.Ansatz, iqc.Encoder, np.ndarray],
) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="at least two"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            np.ones((1, N_VISIBLE)),
            encoders,
            jax.random.PRNGKey(0),
            1,
        )


def test_encoder_width_mismatch_is_rejected(
    setup: tuple[iqc.Ansatz, iqc.Encoder, np.ndarray],
) -> None:
    ansatz, _encoders, _targets = setup
    smaller = iqc.encoder(iqc.PRODUCT, np.zeros((N_CONDITIONS, N_VISIBLE)))
    with pytest.raises(ValueError, match="encoder acts on"):
        iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            np.ones((1, N_VISIBLE + N_ANCILLAS)),
            smaller,
            jax.random.PRNGKey(0),
            64,
        )


def test_unchecked_kernel_is_jittable(
    setup: tuple[iqc.Ansatz, iqc.Encoder, np.ndarray],
) -> None:
    ansatz, encoders, _targets = setup
    masks = np.ones((1, N_VISIBLE + N_ANCILLAS))
    params = np.zeros(ansatz.n_parameters)

    def estimate(current_params: jax.Array) -> jax.Array:
        return iqc.estimate_pauli_z_unchecked(
            current_params,
            ansatz.generators,
            masks,
            encoders,
            jax.random.PRNGKey(0),
            64,
        ).values

    assert jax.jit(estimate)(params).shape == (N_CONDITIONS, 1)
