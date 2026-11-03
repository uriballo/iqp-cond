"""The encoder contract and the built-in product-state implementation.

The ratio query must equal ``psi_c(z xor a) / psi_c(z)`` computed from the full
post-Hadamard state.  A custom class that satisfies the structural protocol
works without registration or inheritance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

import iqp_cond as ciqp
from iqp_cond.encoders import ExactEncoder, layered_amplitudes
from iqp_cond.estimator import estimate_pauli_z

N_QUBITS = 4
N_CONDITIONS = 3


def indices_of(bits: np.ndarray) -> np.ndarray:
    """Little-endian integer index of each bitstring."""
    shifts = 2 ** np.arange(bits.shape[-1], dtype=np.int64)
    return (bits.astype(np.int64) * shifts).sum(axis=-1)


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(7).uniform(0.3, 2.4, size=(N_CONDITIONS, N_QUBITS))


@pytest.fixture
def masks() -> np.ndarray:
    return (
        np.random.default_rng(8).integers(0, 2, size=(5, N_QUBITS)).astype(np.float64)
    )


def make_encoder(angles: np.ndarray) -> ciqp.AngleEncoder:
    """Build the angle encoder with one valid Y layer."""
    return ciqp.AngleEncoder(
        np.asarray(angles)[:, None, :],
        axes=("Y" * angles.shape[1],),
    )


def test_family_satisfies_protocol(
    angles: np.ndarray,
    masks: np.ndarray,
) -> None:
    states = make_encoder(angles)
    assert isinstance(states, ciqp.Encoder)
    assert states.n_qubits == N_QUBITS
    assert states.n_conditions == N_CONDITIONS

    samples = states.sample(jax.random.PRNGKey(0), 32)
    assert samples.shape == (N_CONDITIONS, 32, N_QUBITS)
    if not np.all(np.isin(np.asarray(samples), (0.0, 1.0))):
        raise AssertionError("encoder sampling must return bits")

    ratios = states.amplitude_ratios(samples, masks)
    assert ratios.shape == (N_CONDITIONS, 32, masks.shape[0])


def test_post_hadamard_state_is_normalised(angles: np.ndarray) -> None:
    state = np.asarray(make_encoder(angles).post_hadamard_amplitudes())
    assert state.shape == (N_CONDITIONS, 2**N_QUBITS)
    norm = np.sum(np.abs(state) ** 2, axis=-1)
    assert np.allclose(norm, 1.0, atol=1e-12)


def expected_ratios(
    states: ciqp.Encoder,
    samples: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """``psi_c(z xor a)/psi_c(z)`` from the exact state, zero where undefined."""
    state = np.asarray(states.post_hadamard_amplitudes())
    base = indices_of(samples)
    conditions = np.arange(samples.shape[0])
    denominator = state[conditions[:, None], base]
    out = np.empty(
        (samples.shape[0], samples.shape[1], masks.shape[0]),
        dtype=state.dtype,
    )
    for mask_index, mask in enumerate(masks.astype(np.int64)):
        flipped = indices_of(samples.astype(np.int64) ^ mask)
        out[:, :, mask_index] = np.where(
            np.abs(denominator) > 0.0,
            state[conditions[:, None], flipped]
            / np.where(denominator == 0.0, 1.0, denominator),
            0.0,
        )
    return out


def test_ratios_match_post_hadamard_state(
    angles: np.ndarray,
    masks: np.ndarray,
) -> None:
    """``R(z, a)`` must equal ``psi_c(z xor a)/psi_c(z)`` from the full state."""
    states = make_encoder(angles)
    samples = np.asarray(states.sample(jax.random.PRNGKey(1), 16))
    ratios = np.asarray(states.amplitude_ratios(jnp.asarray(samples), masks))
    assert np.allclose(
        ratios,
        expected_ratios(states, samples, masks),
        atol=1e-12,
    )


def test_angle_encoder_selects_one_condition() -> None:
    angles = np.random.default_rng(4).uniform(0.3, 2.4, size=(N_CONDITIONS, N_QUBITS))
    encoder = make_encoder(angles)
    for condition in range(N_CONDITIONS):
        selected = encoder.select(condition)
        assert selected.n_conditions == 1
        assert np.allclose(
            np.asarray(selected.angles)[0],
            np.asarray(encoder.angles)[condition],
        )
        samples = selected.sample(jax.random.PRNGKey(0), 8)
        assert samples.shape == (1, 8, N_QUBITS)
    with pytest.raises(IndexError, match="condition"):
        encoder.select(N_CONDITIONS)


def test_built_in_encoder_has_the_optional_exact_capability(
    angles: np.ndarray,
) -> None:
    states = make_encoder(angles)
    assert isinstance(states, ExactEncoder)


@dataclass(frozen=True)
class UniformBits:
    """A minimal frozen family with uniform bits and unit ratios."""

    angles: jnp.ndarray

    @property
    def n_conditions(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.angles.shape[1])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        return jax.random.bernoulli(
            key,
            p=0.5,
            shape=(self.n_conditions, n_samples, self.n_qubits),
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones(
            (self.n_conditions, samples.shape[1], masks.shape[0]),
            dtype=jnp.float64,
        )

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        return jnp.full(
            (self.n_conditions, 2**self.n_qubits),
            2.0 ** (-self.n_qubits / 2),
            dtype=jnp.complex128,
        )

    def preparation(self, condition: int) -> Callable[[], None]:
        # |+>^n is the image of |0>^n under the first Hadamard layer.
        return lambda: None


jax.tree_util.register_dataclass(UniformBits, data_fields=["angles"], meta_fields=[])


def test_custom_family_trains_through_the_public_api() -> None:
    """A pytree family defined outside the package trains and samples."""
    ansatz = ciqp.Ansatz.complete(3)
    encoder = UniformBits(jnp.zeros((2, 3)))
    model = ciqp.ConditionalModel(ansatz, encoder)
    estimate = estimate_pauli_z(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        encoder,
        np.array([[1.0, 0.0, 0.0]]),
        key=jax.random.PRNGKey(2),
        n_samples=64,
    )
    assert isinstance(model.encoder, ciqp.Encoder)
    # Zero parameters leave the phase bank at identity, so every sample value
    # is Re[R * 1] = 1 for this family's unit ratios.
    assert np.allclose(np.asarray(estimate.values), 1.0, atol=1e-12)

    targets = np.random.default_rng(19).integers(0, 2, size=(2, 8, 3)).astype(float)
    result = ciqp.train(
        model,
        targets,
        targets[:, :4],
        seed=3,
        config=ciqp.TrainingConfig(
            steps=1,
            batch_size=4,
            validation_interval=1,
            sigmas=(1.0,),
            n_ops=2,
            mc_samples=4,
            validation_n_ops=2,
            validation_mc_samples=4,
        ),
    )
    assert result.best_params.shape == (ansatz.n_parameters,)
    assert model.probabilities(result.best_params).shape == (2, 8)
    assert model.sample(result.best_params, 4, seed=5).shape == (2, 4, 3)


def test_angle_encoder_single_y_layer_matches_the_ry_formula() -> None:
    """One all-Y layer is the plain RY product state with real amplitudes."""
    angles = np.random.default_rng(9).uniform(0.3, 2.4, size=(N_CONDITIONS, N_QUBITS))
    states = ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * N_QUBITS,))
    cosine = np.cos(angles / 2.0)
    sine = np.sin(angles / 2.0)
    assert states.axes == (("Y",) * N_QUBITS,)
    amp_zero, amp_one = layered_amplitudes(states.angles, states.axes)
    assert np.asarray(amp_zero).dtype == np.float64
    assert np.allclose(np.asarray(amp_zero), (cosine + sine) / np.sqrt(2.0), atol=1e-15)
    assert np.allclose(np.asarray(amp_one), (cosine - sine) / np.sqrt(2.0), atol=1e-15)


def test_angle_encoder_ratios_match_post_hadamard_state() -> None:
    """The complex multi-layer path satisfies the same ratio identity."""
    angles = np.random.default_rng(11).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 3, N_QUBITS),
    )
    axes = ("X" * N_QUBITS, "ZY" * (N_QUBITS // 2), "Y" * N_QUBITS)
    states = ciqp.AngleEncoder(angles, axes=axes)
    assert np.asarray(layered_amplitudes(angles, axes)[0]).dtype == np.complex128
    samples = np.asarray(states.sample(jax.random.PRNGKey(12), 16))
    masks = (
        np.random.default_rng(13)
        .integers(
            0,
            2,
            size=(4, N_QUBITS),
        )
        .astype(np.float64)
    )
    ratios = np.asarray(states.amplitude_ratios(jnp.asarray(samples), masks))
    assert np.allclose(
        ratios,
        expected_ratios(states, samples, masks),
        atol=1e-12,
    )


def test_angle_encoder_composes_layers_in_order() -> None:
    """X then Y differs from Y then X, so layer 0 is applied first."""
    angles = np.random.default_rng(14).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 2, N_QUBITS),
    )
    first = ciqp.AngleEncoder(angles, axes=("X" * N_QUBITS, "Y" * N_QUBITS))
    swapped = ciqp.AngleEncoder(
        angles[:, ::-1, :],
        axes=("Y" * N_QUBITS, "X" * N_QUBITS),
    )
    assert not np.allclose(
        np.asarray(first.post_hadamard_amplitudes()),
        np.asarray(swapped.post_hadamard_amplitudes()),
    )


def test_layered_amplitudes_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="angles must have shape"):
        layered_amplitudes(np.zeros(3), axes=("Y",))
    with pytest.raises(ValueError, match="at least one layer and one qubit"):
        layered_amplitudes(np.zeros((1, 0, 3)), axes=())
    with pytest.raises(ValueError, match="at least one layer and one qubit"):
        layered_amplitudes(np.zeros((1, 1, 0)), axes=((),))


def test_angle_encoder_validates_axes_and_shapes() -> None:
    angles = np.zeros((N_CONDITIONS, 2, N_QUBITS))
    with pytest.raises(ValueError, match="first rotation layer"):
        ciqp.AngleEncoder(angles, axes=("Z" * N_QUBITS, "Y" * N_QUBITS))
    with pytest.raises(ValueError, match="unknown rotation axis"):
        ciqp.AngleEncoder(angles, axes=("W" * N_QUBITS, "Y" * N_QUBITS))
    with pytest.raises(ValueError, match="2 layers but angles have"):
        ciqp.AngleEncoder(angles[:, :1, :], axes=("Y", "Y"))
    with pytest.raises(ValueError, match="labels but the register"):
        ciqp.AngleEncoder(angles, axes=("Y" * (N_QUBITS - 1), "Y" * N_QUBITS))
    with pytest.raises(TypeError, match="sequence of layers"):
        ciqp.AngleEncoder(angles, axes="Y")
    with pytest.raises(ValueError, match="shape"):
        ciqp.AngleEncoder(np.zeros((N_CONDITIONS, N_QUBITS)), axes=("Y",))
    with pytest.raises(ValueError, match="angles must be real"):
        ciqp.AngleEncoder(
            np.zeros((N_CONDITIONS, 2, N_QUBITS), dtype=np.complex128),
            axes=("X" * N_QUBITS, "Y" * N_QUBITS),
        )
    bad = np.zeros((N_CONDITIONS, 2, N_QUBITS))
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="angles must be finite"):
        ciqp.AngleEncoder(bad, axes=("X" * N_QUBITS, "Y" * N_QUBITS))


def test_angle_encoder_is_a_registered_pytree() -> None:
    """JAX transformations treat the encoder as a pytree of dynamic arrays."""
    angles = np.random.default_rng(16).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 2, N_QUBITS),
    )
    encoders = ciqp.AngleEncoder(angles, axes=("X" * N_QUBITS, "Z" * N_QUBITS))
    leaves, treedef = jax.tree_util.tree_flatten(encoders)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, ciqp.AngleEncoder)
    assert np.array_equal(np.asarray(restored.angles), np.asarray(encoders.angles))
    assert restored.axes == encoders.axes
    assert np.allclose(
        np.asarray(restored.post_hadamard_amplitudes()),
        np.asarray(encoders.post_hadamard_amplitudes()),
    )

    samples = jnp.asarray(encoders.sample(jax.random.PRNGKey(0), 8))
    masks = jnp.ones((2, N_QUBITS))

    def ratios(encoder: ciqp.AngleEncoder) -> jnp.ndarray:
        return encoder.amplitude_ratios(samples, masks)

    jitted = jax.jit(ratios)(encoders)
    assert np.allclose(
        np.asarray(jitted),
        np.asarray(encoders.amplitude_ratios(samples, masks)),
    )


def test_angle_encoder_selects_conditions_as_a_pytree() -> None:
    angles = np.random.default_rng(15).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 2, N_QUBITS),
    )
    encoders = ciqp.AngleEncoder(angles, axes=("X" * N_QUBITS, "Z" * N_QUBITS))
    selected = encoders.select(1)
    assert selected.n_conditions == 1
    assert selected.n_qubits == N_QUBITS
    assert selected.axes == encoders.axes
    assert np.array_equal(np.asarray(selected.angles)[0], angles[1])


def test_preparation_queues_axes_and_layers_in_order() -> None:
    """The encoder queues rotations in layer order, then qubit order."""
    angles = np.array([[[0.1, 0.2], [0.3, 0.4]]])
    states = ciqp.AngleEncoder(angles, axes=("XY", "ZY"))
    script = qml.tape.make_qscript(states.preparation(0))()
    assert [op.name for op in script.operations] == ["RX", "RY", "RZ", "RY"]
    assert [tuple(int(wire) for wire in op.wires) for op in script.operations] == [
        (0,),
        (1,),
        (0,),
        (1,),
    ]
    assert [float(op.parameters[0]) for op in script.operations] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4],
    )
    with pytest.raises(IndexError, match="condition"):
        states.preparation(4)


def test_model_requires_the_complete_encoder_contract() -> None:
    """A family without circuit preparation or selection is rejected."""

    class TrainingOnly:
        n_conditions = 1
        n_qubits = 2

        def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
            return jnp.zeros((1, n_samples, 2))

        def amplitude_ratios(
            self,
            samples: jnp.ndarray,
            masks: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.ones((1, samples.shape[1], masks.shape[0]))

    with pytest.raises(TypeError, match="Encoder protocol"):
        ciqp.ConditionalModel(ciqp.Ansatz.complete(2), TrainingOnly())


@pytest.mark.parametrize("axes", [("YYYY",), ("YYYY", "ZXZX")])
def test_layered_amplitudes_keep_single_precision(axes) -> None:
    """Float32 angles give single-precision amplitudes close to float64."""
    angles = np.random.default_rng(46).uniform(0.3, 2.4, size=(2, len(axes), 4))
    single = layered_amplitudes(angles.astype(np.float32), axes)
    double = layered_amplitudes(angles, axes)
    for low, high in zip(single, double, strict=True):
        assert low.dtype in (jnp.float32, jnp.complex64)
        assert np.allclose(np.asarray(low), np.asarray(high), atol=1e-6)
