"""The encoder contract: what any family must satisfy to be trainable.

Each registered family is checked against the defining identity of its own
interface — the ratio query must equal ``phi(z xor a) / phi(z)`` computed from
the full post-Hadamard state — rather than against the code path that produced
it.  A custom family registered from outside the package must train too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as iqc
from iqp_cond.encoders import (
    ENCODERS,
    Encoder,
    ExactEncoder,
    available_encoders,
    register_encoder,
    select_encoder,
)
from iqp_cond.encoders.protocol import _REGISTRY

N_QUBITS = 4
N_CONDITIONS = 3


def indices_of(bits: np.ndarray) -> np.ndarray:
    """Little-endian integer index of each bitstring."""
    shifts = 2 ** np.arange(bits.shape[-1], dtype=np.int64)
    return (bits.astype(np.int64) * shifts).sum(axis=-1)


@pytest.fixture(params=sorted(ENCODERS))
def family(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(7).uniform(0.3, 2.4, size=(N_CONDITIONS, N_QUBITS))


@pytest.fixture
def masks() -> np.ndarray:
    return (
        np.random.default_rng(8).integers(0, 2, size=(5, N_QUBITS)).astype(np.float64)
    )


def broadcast_for(family: str, angles: np.ndarray) -> np.ndarray:
    """Angles valid for ``family``: spin-flip needs a constant-angle row."""
    if family == "spin_flip":
        return np.broadcast_to(angles[:, :1], angles.shape)
    return angles


def make_encoder(family: str, angles: np.ndarray) -> iqc.Encoder:
    """Build each registered family with angles valid for its interface."""
    if family == "angle":
        return iqc.encoder(
            family,
            np.asarray(angles)[:, None, :],
            axes=("Y" * angles.shape[1],),
        )
    return iqc.encoder(family, broadcast_for(family, angles))


def test_family_satisfies_protocol(
    family: str,
    angles: np.ndarray,
    masks: np.ndarray,
) -> None:
    states = make_encoder(family, angles)
    assert isinstance(states, Encoder)
    assert states.n_qubits == N_QUBITS
    assert states.n_conditions == N_CONDITIONS

    samples = states.sample(jax.random.PRNGKey(0), 32)
    assert samples.shape == (N_CONDITIONS, 32, N_QUBITS)
    if not np.all(np.isin(np.asarray(samples), (0.0, 1.0))):
        raise AssertionError("encoder sampling must return bits")

    ratios = states.amplitude_ratios(samples, masks)
    assert ratios.shape == (N_CONDITIONS, 32, masks.shape[0])


def test_post_hadamard_state_is_normalised(family: str, angles: np.ndarray) -> None:
    state = np.asarray(make_encoder(family, angles).post_hadamard_amplitudes())
    assert state.shape == (N_CONDITIONS, 2**N_QUBITS)
    norm = np.sum(np.abs(state) ** 2, axis=-1)
    assert np.allclose(norm, 1.0, atol=1e-12)


def expected_ratios(
    states: iqc.Encoder,
    samples: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """``phi(z xor a)/phi(z)`` from the exact state, zero where undefined."""
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
    family: str,
    angles: np.ndarray,
    masks: np.ndarray,
) -> None:
    """``R(z, a)`` must equal ``phi(z xor a)/phi(z)`` from the full state."""
    states = make_encoder(family, angles)
    samples = np.asarray(states.sample(jax.random.PRNGKey(1), 16))
    ratios = np.asarray(states.amplitude_ratios(jnp.asarray(samples), masks))
    assert np.allclose(
        ratios,
        expected_ratios(states, samples, masks),
        atol=1e-12,
    )


def test_select_encoder_slices_built_in_pytrees() -> None:
    """A registered family is a pytree, so selection is a generic slice."""
    encoders = iqc.encoder(
        iqc.PRODUCT,
        np.random.default_rng(4).uniform(0.3, 2.4, size=(N_CONDITIONS, N_QUBITS)),
    )
    for condition in range(N_CONDITIONS):
        selected = select_encoder(encoders, condition)
        assert selected.n_conditions == 1
        assert np.allclose(
            np.asarray(selected.angles)[0],
            np.asarray(encoders.angles)[condition],
        )
        samples = selected.sample(jax.random.PRNGKey(0), 8)
        assert samples.shape == (1, 8, N_QUBITS)


def test_select_encoder_uses_an_optional_select_method() -> None:
    """Encoders that are not pytrees can implement select themselves."""

    class NotAPytree:
        def __init__(self, n_conditions: int) -> None:
            self.n_conditions = n_conditions

        @property
        def n_qubits(self) -> int:
            return 2

        def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
            return jnp.zeros((1, n_samples, 2))

        def amplitude_ratios(
            self,
            samples: jnp.ndarray,
            masks: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.ones((1, samples.shape[1], masks.shape[0]))

        def select(self, condition: int) -> NotAPytree:
            return NotAPytree(1)

    states = NotAPytree(3)
    assert isinstance(states, Encoder)
    assert select_encoder(states, 2).n_conditions == 1


def test_select_encoder_rejects_a_plain_object() -> None:
    class Plain:
        n_conditions = 2
        n_qubits = 2

        def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
            return jnp.zeros((2, n_samples, 2))

        def amplitude_ratios(
            self,
            samples: jnp.ndarray,
            masks: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.ones((2, samples.shape[1], masks.shape[0]))

    with pytest.raises(TypeError, match="implement select"):
        select_encoder(Plain(), 0)


def test_spin_flip_encoder_vanishes_on_odd_shells(angles: np.ndarray) -> None:
    """The two-branch state is spin-flip symmetric, so odd weights cancel."""
    state = np.asarray(
        iqc.encoder(
            "spin_flip", broadcast_for("spin_flip", angles)
        ).post_hadamard_amplitudes()
    )
    weights = np.bitwise_count(np.arange(2**N_QUBITS)).astype(np.int64)
    assert np.allclose(state[:, weights % 2 == 1], 0.0, atol=1e-14)
    assert np.any(state[:, weights % 2 == 0] != 0.0)


def test_spin_flip_rejects_per_qubit_angles(angles: np.ndarray) -> None:
    """A differing column would be silently ignored, so it must raise."""
    per_qubit = np.asarray(angles)
    assert not np.allclose(per_qubit[:, :1], per_qubit[:, 1:])
    with pytest.raises(ValueError, match="broadcast angle"):
        iqc.encoder("spin_flip", per_qubit)


def test_spin_flip_accepts_zero_ancilla_columns(angles: np.ndarray) -> None:
    """Column 0 broadcasts while zero columns (ancillas) stay allowed."""
    matrix = np.broadcast_to(angles[:, :1], angles.shape).copy()
    matrix[:, 1] = 0.0
    states = iqc.encoder("spin_flip", matrix)
    assert states.n_qubits == N_QUBITS


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown encoder family"):
        iqc.encoder("not_a_family", np.zeros((1, N_QUBITS)))


def test_only_the_core_families_are_registered() -> None:
    """The shipped registry is exactly the core conditional-IQP surface."""
    assert available_encoders() == ("angle", "product", "spin_flip")


def test_the_built_in_families_are_distinct(angles: np.ndarray) -> None:
    """A registry alias must not collapse a family onto the product state."""
    product = np.asarray(iqc.encoder("product", angles).post_hadamard_amplitudes())
    spin_flip = np.asarray(
        iqc.encoder(
            "spin_flip", broadcast_for("spin_flip", angles)
        ).post_hadamard_amplitudes()
    )
    assert not np.allclose(product, spin_flip)


def test_built_in_families_are_exact(
    family: str,
    angles: np.ndarray,
) -> None:
    states = make_encoder(family, angles)
    assert isinstance(states, ExactEncoder)


def test_minimal_family_needs_only_sampling_and_ratios() -> None:
    """The training interface must not require metadata or exact amplitudes."""

    class Minimal:
        def __init__(self, angles: np.ndarray) -> None:
            self.angles = jnp.asarray(angles)

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

        def amplitude_ratios(
            self, samples: jnp.ndarray, masks: jnp.ndarray
        ) -> jnp.ndarray:
            return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    states = Minimal(np.zeros((2, 3)))
    assert isinstance(states, Encoder)
    assert not isinstance(states, ExactEncoder)


def test_register_encoder_refuses_to_shadow_a_builtin() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_encoder("product", lambda angles, **_: None)


def test_register_encoder_replace_overrides() -> None:
    original = _REGISTRY["product"]
    try:
        register_encoder(
            "product",
            lambda angles, **_: UniformBits("replaced", jnp.asarray(angles)),
            replace=True,
        )
        assert iqc.encoder("product", np.zeros((1, N_QUBITS))).name == "replaced"
    finally:
        _REGISTRY["product"] = original


def test_encoder_factory_result_is_validated() -> None:
    register_encoder("test_bad_factory", lambda angles, **_: 42)
    try:
        with pytest.raises(TypeError, match="does not implement the Encoder"):
            iqc.encoder("test_bad_factory", np.zeros((1, N_QUBITS)))
    finally:
        _REGISTRY.pop("test_bad_factory", None)


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        ENCODERS["product"] = lambda angles, **_: None  # type: ignore[index]


@dataclass(frozen=True)
class UniformBits:
    """A minimal frozen family: uniform bits with unit ratios."""

    name: str
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


def test_custom_family_trains_through_the_public_api() -> None:
    """A family defined outside the package must work without editing it."""
    register_encoder(
        "test_uniform_bits",
        lambda angles, **_: UniformBits("test_uniform_bits", jnp.asarray(angles)),
    )
    try:
        assert "test_uniform_bits" in available_encoders()
        ansatz = iqc.complete_ansatz(3)
        encoders = iqc.encoder("test_uniform_bits", np.zeros((2, 3)))
        estimate = iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            np.array([[1.0, 0.0, 0.0]]),
            key=jax.random.PRNGKey(2),
            n_samples=64,
        )
        # Zero parameters leave the phase bank at identity, so every sample
        # value is Re[R * 1] = 1 for this family's unit ratios.
        assert np.allclose(np.asarray(estimate.values), 1.0, atol=1e-12)
    finally:
        _REGISTRY.pop("test_uniform_bits", None)
    assert "test_uniform_bits" not in available_encoders()


def test_radial_sampling_follows_the_shell_law() -> None:
    """Drawn Hamming weights must follow ``q(k) = C(n,k) |phi_k|^2 / sum``.

    The sampler's contract is the weight law, not any particular construction:
    the bitstring of a drawn weight only has to be a valid bitstring of that
    weight.  A rewrite that changes the distribution (uniform bits, a wrong
    weighted-sampling step, a broken permutation) fails here.
    """
    n_qubits, n_samples = 6, 20_000
    angles = np.full((1, n_qubits), 1.0)
    states = iqc.encoder("spin_flip", angles)
    shell = np.asarray(states.shell_amplitudes)[0]

    binomial = np.array(
        [math.comb(n_qubits, k) for k in range(n_qubits + 1)], dtype=np.float64
    )
    probabilities = binomial * np.abs(shell) ** 2
    law = probabilities / probabilities.sum()

    draws = np.asarray(states.sample(jax.random.PRNGKey(0), n_samples))[0]
    assert np.isin(draws, (0.0, 1.0)).all()

    empirical = (
        np.bincount(
            draws.sum(axis=-1).astype(np.int64),
            minlength=n_qubits + 1,
        )
        / n_samples
    )
    tolerance = 3.0 * np.sqrt(law.max() * (1.0 - law.max()) / n_samples)
    assert np.max(np.abs(empirical - law)) < tolerance


def test_angle_encoder_single_y_layer_equals_the_product_encoder() -> None:
    """The base product family is the one-layer all-Y angle encoder."""
    angles = np.random.default_rng(9).uniform(0.3, 2.4, size=(N_CONDITIONS, N_QUBITS))
    product = iqc.product_encoder(angles)
    angle = iqc.angle_encoder(angles[:, None, :], axes=("Y" * N_QUBITS,))
    assert isinstance(angle, iqc.ProductEncoder)
    assert angle.axes == (("Y",) * N_QUBITS,)
    assert np.asarray(angle.amp_zero).dtype == np.float64
    assert np.array_equal(np.asarray(angle.amp_zero), np.asarray(product.amp_zero))
    assert np.array_equal(np.asarray(angle.amp_one), np.asarray(product.amp_one))
    key = jax.random.PRNGKey(10)
    assert np.array_equal(
        np.asarray(angle.sample(key, 32)),
        np.asarray(product.sample(key, 32)),
    )


def test_angle_encoder_ratios_match_post_hadamard_state() -> None:
    """The complex multi-layer path satisfies the same ratio identity."""
    angles = np.random.default_rng(11).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 3, N_QUBITS),
    )
    axes = ("X" * N_QUBITS, "ZY" * (N_QUBITS // 2), "Y" * N_QUBITS)
    states = iqc.angle_encoder(angles, axes)
    assert np.asarray(states.amp_zero).dtype == np.complex128
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
    first = iqc.angle_encoder(angles, axes=("X" * N_QUBITS, "Y" * N_QUBITS))
    swapped = iqc.angle_encoder(
        angles[:, ::-1, :],
        axes=("Y" * N_QUBITS, "X" * N_QUBITS),
    )
    assert not np.allclose(
        np.asarray(first.post_hadamard_amplitudes()),
        np.asarray(swapped.post_hadamard_amplitudes()),
    )


def test_angle_encoder_validates_axes_and_shapes() -> None:
    angles = np.zeros((N_CONDITIONS, 2, N_QUBITS))
    with pytest.raises(ValueError, match="first rotation layer"):
        iqc.angle_encoder(angles, axes=("Z" * N_QUBITS, "Y" * N_QUBITS))
    with pytest.raises(ValueError, match="unknown rotation axis"):
        iqc.angle_encoder(angles, axes=("W" * N_QUBITS, "Y" * N_QUBITS))
    with pytest.raises(ValueError, match="2 layers but angles have"):
        iqc.angle_encoder(angles[:, :1, :], axes=("Y", "Y"))
    with pytest.raises(ValueError, match="labels but the register"):
        iqc.angle_encoder(angles, axes=("Y" * (N_QUBITS - 1), "Y" * N_QUBITS))
    with pytest.raises(TypeError, match="sequence of layers"):
        iqc.angle_encoder(angles, axes="Y")
    with pytest.raises(ValueError, match="shape"):
        iqc.angle_encoder(np.zeros((N_CONDITIONS, N_QUBITS)), axes=("Y",))
    with pytest.raises(ValueError, match="angles must be real"):
        iqc.angle_encoder(
            np.zeros((N_CONDITIONS, 2, N_QUBITS), dtype=np.complex128),
            axes=("X" * N_QUBITS, "Y" * N_QUBITS),
        )
    bad = np.zeros((N_CONDITIONS, 2, N_QUBITS))
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="angles must be finite"):
        iqc.angle_encoder(bad, axes=("X" * N_QUBITS, "Y" * N_QUBITS))


def test_angle_encoder_selects_conditions_as_a_pytree() -> None:
    angles = np.random.default_rng(15).uniform(
        0.3,
        2.4,
        size=(N_CONDITIONS, 2, N_QUBITS),
    )
    encoders = iqc.angle_encoder(angles, axes=("X" * N_QUBITS, "Z" * N_QUBITS))
    selected = select_encoder(encoders, 1)
    assert selected.n_conditions == 1
    assert selected.n_qubits == N_QUBITS
    assert selected.axes == encoders.axes
    assert np.array_equal(np.asarray(selected.angles)[0], angles[1])
