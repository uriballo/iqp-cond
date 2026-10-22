"""The parity, twisted parity and collective phase encoders.

Each encoder is compared against a dense statevector built gate by gate in
numpy, independently of its shell table or closed form, and against the
PennyLane conformance check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.encoders import ExactEncoder
from iqp_cond.verification import check_encoder

N_QUBITS = 5


def bits_of(n: int) -> np.ndarray:
    """Little-endian bits of every index, shape ``(2**n, n)``."""
    return (np.arange(1 << n)[:, None] >> np.arange(n)) & 1


def indices_of(bits: np.ndarray) -> np.ndarray:
    """Little-endian integer index of each bitstring."""
    return (bits.astype(np.int64) << np.arange(bits.shape[-1])).sum(axis=-1)


def hadamard_transform(state: np.ndarray) -> np.ndarray:
    """Apply ``H^n`` to a dense state."""
    n = int(np.log2(len(state)))
    out = np.asarray(state, dtype=np.complex128)
    for qubit in range(n):
        out = out.reshape(-1, 2, 1 << qubit)
        out = np.stack([out[:, 0] + out[:, 1], out[:, 0] - out[:, 1]], axis=1)
    return out.reshape(-1) / 2 ** (n / 2)


def apply_rx(state: np.ndarray, angle: float, qubit: int) -> np.ndarray:
    index = np.arange(len(state))
    return (
        np.cos(angle / 2) * state - 1j * np.sin(angle / 2) * state[index ^ (1 << qubit)]
    )


def parity_reference(angle: float, n: int) -> np.ndarray:
    """``RX(t)`` and a CNOT star on ``|0>^n``, which equals ``H^n |phi>``."""
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    index = np.arange(1 << n)
    for leaf in range(n - 1):
        state = apply_rx(state, angle, leaf)
        control = (index >> leaf) & 1 == 1
        state = np.where(control, state[index ^ (1 << (n - 1))], state)
    return state


def twisted_input(angle: float, twist: float, n: int, turn: float) -> np.ndarray:
    """Gate-level twisted parity input state; the root is qubit ``n - 1``."""
    spin = 1 - 2 * bits_of(n)
    leaves = n - 1
    state = np.full(1 << n, 2 ** (-n / 2), dtype=np.complex128)
    pairs = (spin[:, :leaves].sum(axis=1) ** 2 - leaves) / 2
    state = state * np.exp(-1j * twist / leaves * pairs)
    for leaf in range(leaves):
        state = apply_rx(state, turn, leaf)
    star = spin[:, :leaves].sum(axis=1) * spin[:, -1]
    return state * np.exp(-0.5j * angle * star)


def collective_input(linear: float, quadratic: float, n: int) -> np.ndarray:
    spin = n / 2 - bits_of(n).sum(axis=1)
    phase = np.exp(-1j * (linear * spin + quadratic * spin**2))
    return phase / 2 ** (n / 2)


def make_parity() -> ciqp.ParityEncoder:
    return ciqp.ParityEncoder([0.0, 0.7, 1.9, 2.8, np.pi], n_qubits=N_QUBITS)


def make_twisted() -> ciqp.TwistedParityEncoder:
    return ciqp.TwistedParityEncoder(
        [0.0, 0.4, 1.3],
        [0.0, 0.8, -0.5],
        n_qubits=N_QUBITS,
        turn=-np.pi / 4,
    )


def make_collective() -> ciqp.CollectivePhaseEncoder:
    return ciqp.CollectivePhaseEncoder(
        [0.0, 0.5, -0.3],
        [0.0, 0.0, 0.2],
        n_qubits=N_QUBITS,
    )


ENCODERS = {
    "parity": make_parity,
    "twisted": make_twisted,
    "collective": make_collective,
}


def dense_references(name: str) -> np.ndarray:
    """Independent post-Hadamard states for the encoders above."""
    if name == "parity":
        return np.stack(
            [parity_reference(t, N_QUBITS) for t in [0.0, 0.7, 1.9, 2.8, np.pi]],
        )
    if name == "twisted":
        return np.stack(
            [
                hadamard_transform(twisted_input(t, mu, N_QUBITS, -np.pi / 4))
                for t, mu in [(0.0, 0.0), (0.4, 0.8), (1.3, -0.5)]
            ],
        )
    return np.stack(
        [
            hadamard_transform(collective_input(a, b, N_QUBITS))
            for a, b in [(0.0, 0.0), (0.5, 0.0), (-0.3, 0.2)]
        ],
    )


@pytest.mark.parametrize("name", ENCODERS)
def test_reference_matches_a_gate_level_statevector(name: str) -> None:
    encoder = ENCODERS[name]()
    assert isinstance(encoder, ExactEncoder)
    np.testing.assert_allclose(
        np.asarray(encoder.post_hadamard_amplitudes()),
        dense_references(name),
        atol=1e-12,
    )


@pytest.mark.parametrize("name", ENCODERS)
def test_ratios_match_the_reference_for_every_mask(name: str) -> None:
    encoder = ENCODERS[name]()
    state = np.asarray(encoder.post_hadamard_amplitudes())
    samples = np.asarray(encoder.sample(jax.random.key(1), 64))
    masks = bits_of(N_QUBITS).astype(np.float64)
    ratios = np.asarray(encoder.amplitude_ratios(jnp.asarray(samples), masks))
    rows = np.arange(encoder.n_conditions)[:, None, None]
    base = indices_of(samples)[..., None]
    flipped = base ^ np.arange(1 << N_QUBITS)[None, None, :]
    expected = state[rows, flipped] / state[rows, base]
    np.testing.assert_allclose(ratios, expected, atol=1e-10)


@pytest.mark.parametrize("name", ENCODERS)
def test_sampling_matches_the_reference_law(name: str) -> None:
    encoder = ENCODERS[name]()
    samples = np.asarray(encoder.sample(jax.random.key(3), 40_000))
    assert samples.shape == (encoder.n_conditions, 40_000, N_QUBITS)
    exact = np.abs(np.asarray(encoder.post_hadamard_amplitudes())) ** 2
    for condition in range(encoder.n_conditions):
        counts = np.bincount(indices_of(samples[condition]), minlength=1 << N_QUBITS)
        assert 0.5 * np.abs(counts / 40_000 - exact[condition]).sum() < 0.02
    if name != "collective":
        assert np.all(samples.sum(axis=-1) % 2 == 0)


@pytest.mark.slow
@pytest.mark.parametrize("name", ENCODERS)
def test_encoder_passes_the_conformance_check(name: str) -> None:
    report = check_encoder(ENCODERS[name](), seed=5)
    assert report.ok, report.summary()
    assert report.exact_state_error is not None
    assert report.exact_state_error < 1e-10


def test_twisted_encoder_without_twist_is_the_parity_encoder() -> None:
    """At ``mu = 0`` the turn acts on ``|+>`` and leaves only a global phase."""
    angles = np.array([0.3, 1.1, 2.5])
    twisted = ciqp.TwistedParityEncoder(angles, np.zeros(3), n_qubits=N_QUBITS)
    parity = ciqp.ParityEncoder(angles, n_qubits=N_QUBITS)
    first = np.asarray(twisted.post_hadamard_amplitudes())
    second = np.asarray(parity.post_hadamard_amplitudes())
    phase = np.sum(first * second.conj(), axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs(phase), 1.0, atol=1e-12)
    np.testing.assert_allclose(first, phase * second, atol=1e-12)


def test_collective_encoder_pins_the_uniform_input() -> None:
    shells = np.asarray(make_collective().shells)[0, 0]
    assert shells[0] == 1.0
    assert np.all(shells[1:] == 0.0)


def test_parity_encoder_ratios_stay_bounded_on_large_registers() -> None:
    """The closed form needs no table and inverts its base past ``t = pi/2``."""
    n = 1024
    angles = np.array([0.2, 2.9])
    encoder = ciqp.ParityEncoder(angles, n_qubits=n)
    samples = encoder.sample(jax.random.key(4), 16)
    masks = np.zeros((8, n))
    for row, leaves in enumerate(np.random.default_rng(5).integers(0, n - 1, (8, 6))):
        masks[row, leaves] = 1.0
    masks[:, -1] = masks[:, :-1].sum(axis=1) % 2
    ratios = np.asarray(encoder.amplitude_ratios(samples, masks))
    delta = masks[:, :-1].sum(axis=1)[None, None, :] - 2 * np.einsum(
        "csn,mn->csm",
        np.asarray(samples)[..., :-1],
        masks[:, :-1],
    )
    log_magnitude = delta * np.log(np.abs(np.tan(angles / 2)))[:, None, None]
    assert np.all(np.isfinite(ratios))
    np.testing.assert_allclose(
        np.log(np.abs(ratios)),
        log_magnitude,
        rtol=1e-10,
        atol=1e-8,
    )


@pytest.mark.parametrize("name", ENCODERS)
def test_encoder_is_a_pytree_that_selects_conditions(name: str) -> None:
    encoder = ENCODERS[name]()
    leaves, treedef = jax.tree_util.tree_flatten(encoder)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(restored) is type(encoder)
    samples = encoder.sample(jax.random.key(0), 8)
    masks = jnp.asarray(bits_of(N_QUBITS)[:7], dtype=jnp.float64)

    def ratios(states: ciqp.Encoder) -> jnp.ndarray:
        return states.amplitude_ratios(samples, masks)

    np.testing.assert_allclose(
        np.asarray(jax.jit(ratios)(encoder)),
        np.asarray(encoder.amplitude_ratios(samples, masks)),
    )
    selected = encoder.select(1)
    assert selected.n_conditions == 1
    assert selected.n_qubits == N_QUBITS
    np.testing.assert_allclose(
        np.asarray(selected.post_hadamard_amplitudes())[0],
        np.asarray(encoder.post_hadamard_amplitudes())[1],
    )
    with pytest.raises(IndexError, match="condition"):
        encoder.select(encoder.n_conditions)
    with pytest.raises(IndexError, match="condition"):
        encoder.preparation(-1)


def test_encoders_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        ciqp.ParityEncoder([0.1], n_qubits=1)
    with pytest.raises(ValueError, match="n_qubits"):
        ciqp.CollectivePhaseEncoder([0.1], [0.1], n_qubits=True)
    with pytest.raises(ValueError, match="finite"):
        ciqp.ParityEncoder([np.nan], n_qubits=3)
    with pytest.raises(ValueError, match="real"):
        ciqp.ParityEncoder([1j], n_qubits=3)
    with pytest.raises(ValueError, match="shape"):
        ciqp.ParityEncoder([[0.1]], n_qubits=3)
    with pytest.raises(ValueError, match="same shape"):
        ciqp.TwistedParityEncoder([0.1, 0.2], [0.1], n_qubits=3)
    with pytest.raises(ValueError, match="turn"):
        ciqp.TwistedParityEncoder([0.1], [0.1], n_qubits=3, turn=np.inf)
    with pytest.raises(ValueError, match="same shape"):
        ciqp.CollectivePhaseEncoder([0.1], [0.1, 0.2], n_qubits=3)


def test_twisted_encoder_trains_through_the_public_api() -> None:
    ansatz = ciqp.Ansatz.complete(4)
    encoder = ciqp.TwistedParityEncoder([0.5, 1.2], [0.3, 0.6], n_qubits=4)
    model = ciqp.ConditionalModel(ansatz, encoder)
    targets = np.random.default_rng(19).integers(0, 2, size=(2, 8, 4)).astype(float)
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
    assert model.sample(result.best_params, 4, seed=5).shape == (2, 4, 4)
