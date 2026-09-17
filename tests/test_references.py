"""The reference-state contract: what any family must satisfy to be trainable.

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
from iqp_cond.references import (
    FAMILIES,
    ReferenceStates,
    available_families,
    register_reference,
)

N_QUBITS = 4
N_CONDITIONS = 3


def indices_of(bits: np.ndarray) -> np.ndarray:
    """Little-endian integer index of each bitstring."""
    shifts = 2 ** np.arange(bits.shape[-1], dtype=np.int64)
    return (bits.astype(np.int64) * shifts).sum(axis=-1)


@pytest.fixture(params=sorted(FAMILIES))
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


def test_family_satisfies_protocol(
    family: str,
    angles: np.ndarray,
    masks: np.ndarray,
) -> None:
    states = iqc.reference(family, angles)
    assert isinstance(states, ReferenceStates)
    assert states.n_qubits == N_QUBITS
    assert states.n_conditions == N_CONDITIONS

    samples = states.sample(jax.random.PRNGKey(0), 32)
    assert samples.shape == (N_CONDITIONS, 32, N_QUBITS)
    if not np.all(np.isin(np.asarray(samples), (0.0, 1.0))):
        raise AssertionError("reference sampling must return bits")

    ratios = states.amplitude_ratios(samples, masks)
    assert ratios.shape == (N_CONDITIONS, 32, masks.shape[0])


def test_post_hadamard_state_is_normalised(family: str, angles: np.ndarray) -> None:
    state = np.asarray(iqc.reference(family, angles).post_hadamard_amplitudes())
    assert state.shape == (N_CONDITIONS, 2**N_QUBITS)
    norm = np.sum(np.abs(state) ** 2, axis=-1)
    assert np.allclose(norm, 1.0, atol=1e-12)


def test_ratios_match_post_hadamard_state(
    family: str,
    angles: np.ndarray,
    masks: np.ndarray,
) -> None:
    """``R(z, a)`` must equal ``phi(z xor a)/phi(z)`` from the full state."""
    states = iqc.reference(family, angles)
    state = np.asarray(states.post_hadamard_amplitudes())
    samples = np.asarray(states.sample(jax.random.PRNGKey(1), 16))
    mask_bits = masks.astype(np.int64)
    base = indices_of(samples)
    ratios = np.asarray(states.amplitude_ratios(jnp.asarray(samples), masks))
    for mask_index, mask in enumerate(mask_bits):
        flipped = indices_of(samples.astype(np.int64) ^ mask)
        expected = np.where(
            np.abs(state[np.arange(N_CONDITIONS)[:, None], base]) > 0.0,
            state[np.arange(N_CONDITIONS)[:, None], flipped]
            / np.where(
                state[np.arange(N_CONDITIONS)[:, None], base] == 0.0,
                1.0,
                state[np.arange(N_CONDITIONS)[:, None], base],
            ),
            0.0,
        )
        assert np.allclose(ratios[:, :, mask_index], expected, atol=1e-12)


def test_spin_flip_reference_vanishes_on_odd_shells(angles: np.ndarray) -> None:
    """The two-branch state is spin-flip symmetric, so odd weights cancel."""
    state = np.asarray(iqc.reference("spin_flip", angles).post_hadamard_amplitudes())
    weights = np.bitwise_count(np.arange(2**N_QUBITS)).astype(np.int64)
    assert np.allclose(state[:, weights % 2 == 1], 0.0, atol=1e-14)
    assert np.any(state[:, weights % 2 == 0] != 0.0)


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reference family"):
        iqc.reference("not_a_family", np.zeros((1, N_QUBITS)))


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

    def select(self, condition: int) -> UniformBits:
        return UniformBits(self.name, self.angles[condition : condition + 1])

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        return jnp.full(
            (self.n_conditions, 2**self.n_qubits),
            2.0 ** (-self.n_qubits / 2),
            dtype=jnp.complex128,
        )


def test_custom_family_trains_through_the_public_api() -> None:
    """A family defined outside the package must work without editing it."""
    register_reference(
        "test_uniform_bits",
        lambda angles, **_: UniformBits("test_uniform_bits", jnp.asarray(angles)),
    )
    try:
        assert "test_uniform_bits" in available_families()
        ansatz = iqc.complete_ansatz(3)
        references = iqc.reference("test_uniform_bits", np.zeros((2, 3)))
        estimate = iqc.estimate_pauli_z(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            np.array([[1.0, 0.0, 0.0]]),
            references,
            jax.random.PRNGKey(2),
            64,
        )
        # Zero parameters leave the phase bank at identity, so every sample
        # value is Re[R * 1] = 1 for this family's unit ratios.
        assert np.allclose(np.asarray(estimate.values), 1.0, atol=1e-12)
    finally:
        FAMILIES.pop("test_uniform_bits", None)
    assert "test_uniform_bits" not in available_families()


def test_radial_sampling_follows_the_shell_law() -> None:
    """Drawn Hamming weights must follow ``q(k) = C(n,k) |phi_k|^2 / sum``.

    The sampler's contract is the weight law, not any particular construction:
    the bitstring of a drawn weight only has to be a valid bitstring of that
    weight.  A rewrite that changes the distribution (uniform bits, a wrong
    weighted-sampling step, a broken permutation) fails here.
    """
    n_qubits, n_samples = 6, 20_000
    angles = np.full((1, n_qubits), 1.0)
    states = iqc.reference("spin_flip", angles)
    shell = np.asarray(states.shell_amplitudes)[0]

    binomial = np.array(
        [math.comb(n_qubits, k) for k in range(n_qubits + 1)], dtype=np.float64
    )
    probabilities = binomial * np.abs(shell) ** 2
    law = probabilities / probabilities.sum()

    draws = np.asarray(states.sample(jax.random.PRNGKey(0), n_samples))[0]
    assert np.isin(draws, (0.0, 1.0)).all()

    empirical = np.bincount(
        draws.sum(axis=-1).astype(np.int64), minlength=n_qubits + 1,
    ) / n_samples
    tolerance = 3.0 * np.sqrt(law.max() * (1.0 - law.max()) / n_samples)
    assert np.max(np.abs(empirical - law)) < tolerance
