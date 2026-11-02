"""Sample sources, ancilla padding, and error paths of the Pauli estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.estimator import estimate_pauli_z, estimate_pauli_z_unchecked
from iqp_cond.model import pad_visible_masks


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit, the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


N_VISIBLE = 3
N_ANCILLAS = 2
N_CONDITIONS = 2


@pytest.fixture
def setup() -> tuple[ciqp.Ansatz, ciqp.Encoder, np.ndarray]:
    ansatz = ciqp.Ansatz.complete(N_VISIBLE + N_ANCILLAS)
    rng = np.random.default_rng(31)
    angles = rng.uniform(0.4, 2.2, size=(N_CONDITIONS, N_VISIBLE + N_ANCILLAS))
    encoders = y_layer(angles)
    return (
        ansatz,
        encoders,
        rng.integers(0, 2, size=(N_CONDITIONS, 4, N_VISIBLE)).astype(float),
    )


def test_visible_masks_are_padded_with_identity_on_ancillas(
    setup: tuple[ciqp.Ansatz, ciqp.Encoder, np.ndarray],
) -> None:
    ansatz, encoders, _targets = setup
    visible_masks = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(0)

    padded = estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        visible_masks,
        key=key,
        n_samples=64,
    )
    model_masks = np.asarray(pad_visible_masks(visible_masks, N_ANCILLAS))
    direct = estimate_pauli_z(
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
    """Passing samples evaluates exactly those bitstrings."""
    ansatz, encoders, _targets = setup
    masks = np.ones((1, N_VISIBLE))
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(7)

    drawn = encoders.sample(key, 32)
    from_samples = estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        samples=drawn,
    )
    unchecked = estimate_pauli_z_unchecked(
        params,
        ansatz.generators,
        encoders,
        pad_visible_masks(masks, N_ANCILLAS),
        samples=drawn,
    )
    assert np.array_equal(
        np.asarray(from_samples.values),
        np.asarray(unchecked.values),
    )


def test_rank_two_samples_require_a_single_condition() -> None:
    """One shared draw across different reference states would be biased."""
    ansatz = ciqp.Ansatz.complete(N_VISIBLE)
    generators = ansatz.generators
    masks = np.ones((1, N_VISIBLE))
    params = np.zeros(ansatz.n_parameters)
    rng = np.random.default_rng(32)
    draw = rng.integers(0, 2, size=(16, N_VISIBLE)).astype(np.float64)

    single = y_layer(rng.uniform(0.4, 2.4, size=(1, N_VISIBLE)))
    estimate = estimate_pauli_z(
        params,
        generators,
        single,
        masks,
        samples=draw,
    )
    assert estimate.values.shape == (1, 1)
    assert estimate.n_samples == 16

    paired = y_layer(rng.uniform(0.4, 2.4, size=(2, N_VISIBLE)))
    with pytest.raises(ValueError, match="single condition"):
        estimate_pauli_z(params, generators, paired, masks, samples=draw)
    with pytest.raises(ValueError, match="shape \\(conditions, samples, qubits\\)"):
        estimate_pauli_z(
            params,
            generators,
            single,
            masks,
            samples=draw.ravel(),
        )


def test_sample_condition_axis_must_match_the_encoder() -> None:
    """A mismatched condition axis must fail instead of broadcasting."""
    ansatz = ciqp.Ansatz.complete(N_VISIBLE)
    params = np.zeros(ansatz.n_parameters)
    masks = np.ones((1, N_VISIBLE))
    rng = np.random.default_rng(34)
    five_conditions = rng.integers(0, 2, size=(5, 8, N_VISIBLE)).astype(np.float64)

    single = y_layer(rng.uniform(0.4, 2.4, size=(1, N_VISIBLE)))
    with pytest.raises(ValueError, match="condition axis has 5 entries"):
        estimate_pauli_z(
            params,
            ansatz.generators,
            single,
            masks,
            samples=five_conditions,
        )
    paired = y_layer(rng.uniform(0.4, 2.4, size=(2, N_VISIBLE)))
    with pytest.raises(ValueError, match="condition axis has 5 entries"):
        estimate_pauli_z(
            params,
            ansatz.generators,
            paired,
            masks,
            samples=five_conditions,
        )


def test_sample_qubit_axis_must_match_the_encoder() -> None:
    """A size-1 qubit axis broadcasts in einsum, so it must be rejected."""
    ansatz = ciqp.Ansatz.complete(N_VISIBLE)
    params = np.zeros(ansatz.n_parameters)
    masks = np.ones((1, N_VISIBLE))
    rng = np.random.default_rng(35)
    narrow = rng.integers(0, 2, size=(1, 8, 1)).astype(np.float64)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(1, N_VISIBLE)))
    with pytest.raises(ValueError, match="qubit axis has 1 entries"):
        estimate_pauli_z(
            params,
            ansatz.generators,
            encoders,
            masks,
            samples=narrow,
        )


def test_exactly_one_sample_source_is_required(setup) -> None:
    ansatz, encoders, _targets = setup
    params = np.zeros(ansatz.n_parameters)
    masks = np.ones((1, N_VISIBLE))
    with pytest.raises(ValueError, match="exactly one of key or samples"):
        estimate_pauli_z(params, ansatz.generators, encoders, masks)
    with pytest.raises(ValueError, match="exactly one of key or samples"):
        estimate_pauli_z(
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
        estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.ones((1, N_VISIBLE)),
            key=jax.random.PRNGKey(0),
        )


def test_standard_errors_need_per_sample_values(setup) -> None:
    ansatz, encoders, _targets = setup
    estimate = estimate_pauli_z(
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
    with_samples = estimate_pauli_z(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        encoders,
        np.ones((1, N_VISIBLE)),
        key=jax.random.PRNGKey(0),
        n_samples=16,
        return_samples=True,
    )
    assert with_samples.standard_errors.shape == (N_CONDITIONS, 1)


def test_estimator_validates_shapes_and_supplied_sample_counts(setup) -> None:
    ansatz, encoders, _targets = setup
    params = np.zeros(ansatz.n_parameters)
    masks = np.ones((1, N_VISIBLE))
    key = jax.random.PRNGKey(0)

    with pytest.raises(ValueError, match="generators must have shape"):
        estimate_pauli_z(
            params,
            ansatz.generators[0],
            encoders,
            masks,
            key=key,
            n_samples=8,
        )
    with pytest.raises(ValueError, match="params must have shape"):
        estimate_pauli_z(
            np.zeros(3),
            ansatz.generators,
            encoders,
            masks,
            key=key,
            n_samples=8,
        )
    with pytest.raises(ValueError, match="masks must have shape"):
        estimate_pauli_z(
            params,
            ansatz.generators,
            encoders,
            np.ones((1, 2, 2)),
            key=key,
            n_samples=8,
        )

    flat = estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        np.ones(N_VISIBLE),
        key=key,
        n_samples=16,
    )
    matrix = estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        np.ones((1, N_VISIBLE)),
        key=key,
        n_samples=16,
    )
    assert np.array_equal(np.asarray(flat.values), np.asarray(matrix.values))

    drawn = encoders.sample(key, 8)
    with pytest.raises(ValueError, match="n_samples must match the drawn samples"):
        estimate_pauli_z(
            params,
            ansatz.generators,
            encoders,
            masks,
            samples=drawn,
            n_samples=7,
        )


def test_masks_wider_than_the_register_are_rejected(setup) -> None:
    ansatz, encoders, _targets = setup
    with pytest.raises(ValueError, match="more qubits than the generators"):
        estimate_pauli_z(
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
        estimate_pauli_z(
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
        estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.ones((1, N_VISIBLE)),
            key=jax.random.PRNGKey(0),
            n_samples=1,
        )


def test_encoder_width_mismatch_is_rejected(setup) -> None:
    ansatz, _encoders, _targets = setup
    smaller = y_layer(np.zeros((N_CONDITIONS, N_VISIBLE)))
    with pytest.raises(ValueError, match="encoder acts on"):
        estimate_pauli_z(
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
        return estimate_pauli_z_unchecked(
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

    reference = estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        key=key,
        n_samples=4096,
    )
    mixed = estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        key=key,
        n_samples=4096,
        precision="gemm_f32",
    )
    deviation = float(
        np.max(np.abs(np.asarray(mixed.values) - np.asarray(reference.values))),
    )
    # Nonzero deviation shows the cast is active; the bound shows it is harmless.
    assert 0.0 < deviation < 1e-5


@pytest.mark.parametrize("complex_encoder", [False, True])
def test_float32_estimator_keeps_float64_outputs_and_gradients_close(
    complex_encoder: bool,
) -> None:
    """The wider mixed path returns to float64 before statistics."""
    rng = np.random.default_rng(41)
    n_qubits = N_VISIBLE + N_ANCILLAS
    axes = ("X" * n_qubits, "Z" * n_qubits) if complex_encoder else ("Y" * n_qubits,)
    encoder = ciqp.AngleEncoder(
        rng.uniform(0.4, 2.2, size=(N_CONDITIONS, len(axes), n_qubits)),
        axes=axes,
    )
    ansatz = ciqp.Ansatz.complete(n_qubits)
    params = jnp.linspace(-0.3, 0.3, ansatz.n_parameters)
    masks = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    samples = encoder.sample(jax.random.PRNGKey(42), 1_024)

    def objective(current_params, precision="fp64"):
        result = estimate_pauli_z(
            current_params,
            ansatz.generators,
            encoder,
            masks,
            samples=samples,
            precision=precision,
            return_samples=True,
        )
        assert result.sample_values is not None
        return jnp.sum(result.values), result

    reference_loss, reference_gradient = jax.value_and_grad(
        lambda current: objective(current)[0]
    )(params)
    mixed_loss, mixed_gradient = jax.value_and_grad(
        lambda current: objective(current, "estimator_f32")[0]
    )(params)
    _, mixed = objective(params, "estimator_f32")
    _, single = objective(params, "fp32")

    relative_error = float(
        jnp.linalg.norm(mixed_gradient - reference_gradient)
        / jnp.linalg.norm(reference_gradient)
    )
    assert mixed.values.dtype == jnp.float64
    assert mixed.sample_values is not None
    assert mixed.sample_values.dtype == jnp.float64
    assert mixed.standard_errors.dtype == jnp.float64
    # A sum of raw moments has a small, cancellation-prone gradient. The MMD
    # test below applies the tighter relative gate to the training objective.
    assert 0.0 < relative_error < 2e-5
    assert abs(float(mixed_loss - reference_loss)) < 1e-6
    # The end-to-end policy differs only in returning float32 statistics.
    assert single.values.dtype == jnp.float32
    assert jnp.allclose(single.values, mixed.values, atol=1e-6, rtol=0.0)

    model_masks = pad_visible_masks(masks, N_ANCILLAS)

    @jax.jit
    def unchecked(current_params):
        return estimate_pauli_z_unchecked(
            current_params,
            ansatz.generators,
            encoder,
            model_masks,
            samples=samples,
            precision="estimator_f32",
        ).values

    # Compiled and eager evaluation may fuse the ratio contraction differently,
    # so they agree to single precision rather than bit for bit.
    assert jnp.allclose(unchecked(params), mixed.values, atol=1e-6, rtol=0.0)


def test_checked_estimator_rejects_non_binary_samples(setup) -> None:
    """A fractional sample matrix would be read mod two by the kernel."""
    ansatz, encoders, _targets = setup
    fractional = np.full(
        (N_CONDITIONS, 4, N_VISIBLE + N_ANCILLAS),
        0.5,
    )
    with pytest.raises(ValueError, match="samples must contain only zeros and ones"):
        estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.ones((1, N_VISIBLE)),
            samples=fractional,
        )
