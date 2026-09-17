"""Independent cross-check of the conditional IQP model against PennyLane.

PennyLane is not a package dependency; these tests skip when it is absent.  They
rebuild the encoder state and the full conditional circuit from scratch with
``default.qubit`` and compare the exact statevector expectations against the
library's classically computable estimator, so a shared convention bug cannot
hide behind the library's own exact path.

Wire order differs: PennyLane returns states with wire 0 as the most significant
bit, the library is little-endian.  State comparisons bit-reverse the index;
expectation comparisons build PennyLane observables on the same wires and are
order-independent.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import iqp_cond as iqc

pennylane = pytest.importorskip("pennylane")
qml = pennylane

N_VISIBLE = 3
N_ANCILLAS = 1
N_QUBITS = N_VISIBLE + N_ANCILLAS
N_CONDITIONS = 2


def reverse_bits(indices: np.ndarray, n_bits: int) -> np.ndarray:
    """Bit-reverse little-endian indices into PennyLane's wire ordering."""
    out = np.zeros_like(indices)
    for bit in range(n_bits):
        out |= ((indices >> bit) & 1) << (n_bits - 1 - bit)
    return out


def angles_for(family: str, seed: int = 3) -> np.ndarray:
    """Angles in the well-spread regime.

    The two-branch state is nearly deterministic at ``alpha = pi/2`` (the
    reference state collapses towards ``|0...0>``), where the estimator is still
    unbiased but heavy-tailed: rare sampled strings carry huge amplitude ratios,
    so a sample-standard-error tolerance is meaningless.  Staying away from that
    point keeps these checks about correctness, not about rare events.
    """
    rng = np.random.default_rng(seed)
    if family == "spin_flip":
        return np.broadcast_to(
            rng.uniform(0.1, 0.6, size=(N_CONDITIONS, 1)),
            (N_CONDITIONS, N_QUBITS),
        )
    return rng.uniform(0.4, 2.2, size=(N_CONDITIONS, N_QUBITS))


def prepare_encoder(family: str, angles: np.ndarray, n_qubits: int) -> None:
    """State preparation for one condition, in PennyLane's wire ordering."""
    if family == "product":
        for wire, angle in enumerate(angles):
            qml.RY(float(angle), wires=wire)
        return
    angle = float(angles[0])
    single = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0)])
    state = np.ones(1)
    for _ in range(n_qubits):
        state = np.kron(state, single)
    state = state + state[::-1]
    qml.StatePrep(state / np.linalg.norm(state), wires=range(n_qubits))


def apply_iqp(params: np.ndarray, generators: np.ndarray, n_qubits: int) -> None:
    """The shared IQP block: H, diagonal phases, H."""
    for wire in range(n_qubits):
        qml.Hadamard(wire)
    for generator, theta in zip(generators, params, strict=True):
        wires = [int(i) for i, bit in enumerate(generator) if bit]
        # MultiRZ(phi) = exp(-i phi/2 Z^g), so -2 theta gives exp(+i theta Z^g).
        qml.MultiRZ(-2.0 * float(theta), wires=wires)
    for wire in range(n_qubits):
        qml.Hadamard(wire)


def pl_state_after_first_hadamard(
    family: str,
    angles: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        prepare_encoder(family, angles, n_qubits)
        for wire in range(n_qubits):
            qml.Hadamard(wire)
        return qml.state()

    return np.asarray(circuit())


def pl_expectation(
    family: str,
    angles: np.ndarray,
    params: np.ndarray,
    generators: np.ndarray,
    visible_mask: np.ndarray,
    n_visible: int,
    n_qubits: int,
) -> float:
    """Exact ``<Z_a>`` of the visible marginal, with identity on ancillas."""
    wires = [wire for wire in range(n_visible) if visible_mask[wire] > 0.5]
    if not wires:
        return 1.0
    observable = (
        qml.PauliZ(wires[0])
        if len(wires) == 1
        else qml.prod(*[qml.PauliZ(wire) for wire in wires])
    )
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> float:
        prepare_encoder(family, angles, n_qubits)
        apply_iqp(params, generators, n_qubits)
        return qml.expval(observable)

    return float(circuit())


def pl_probabilities(
    family: str,
    angles: np.ndarray,
    params: np.ndarray,
    generators: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        prepare_encoder(family, angles, n_qubits)
        apply_iqp(params, generators, n_qubits)
        return qml.probs(wires=range(n_qubits))

    return np.asarray(circuit())


@pytest.mark.parametrize("family", ["product", "spin_flip"])
def test_encoder_state_matches_pennylane(family: str) -> None:
    angles = angles_for(family)
    encoders = iqc.encoder(family, angles)
    state = np.asarray(encoders.post_hadamard_amplitudes())

    for condition in range(N_CONDITIONS):
        expected = pl_state_after_first_hadamard(family, angles[condition], N_QUBITS)
        reordered = state[condition][reverse_bits(np.arange(2**N_QUBITS), N_QUBITS)]
        assert np.max(np.abs(reordered - expected)) < 1e-12


@pytest.mark.parametrize("family", ["product", "spin_flip"])
def test_amplitude_ratios_match_pennylane(family: str) -> None:
    angles = angles_for(family, seed=5)
    encoders = iqc.encoder(family, angles)
    state = np.asarray(encoders.post_hadamard_amplitudes())
    rng = np.random.default_rng(6)
    samples = np.asarray(encoders.sample(jax.random.PRNGKey(0), 32))
    masks = rng.integers(0, 2, size=(8, N_QUBITS)).astype(np.float64)
    ratios = np.asarray(encoders.amplitude_ratios(samples, masks))

    for condition in range(N_CONDITIONS):
        for index in range(samples.shape[1]):
            z = samples[condition, index].astype(np.int64)
            base = int((z * 2 ** np.arange(N_QUBITS)).sum())
            if abs(state[condition, base]) == 0.0:
                continue
            for mask_index, mask in enumerate(masks.astype(np.int64)):
                flipped = int(((z ^ mask) * 2 ** np.arange(N_QUBITS)).sum())
                expected = state[condition, flipped] / state[condition, base]
                assert abs(ratios[condition, index, mask_index] - expected) < 1e-12


@pytest.mark.parametrize("family", ["product", "spin_flip"])
def test_pauli_moments_match_pennylane(family: str) -> None:
    """Estimator moments against exact PennyLane expectation values.

    The model carries one ancilla, so the comparison also exercises the
    visible-mask padding and the ancilla marginalization: PennyLane measures Z
    on the same visible wires and identity elsewhere.
    """
    angles = angles_for(family, seed=7)
    ansatz = iqc.complete_ansatz(N_QUBITS)
    rng = np.random.default_rng(8)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    masks = rng.integers(0, 2, size=(6, N_VISIBLE)).astype(np.float64)
    masks[0] = [1.0, 0.0, 0.0]
    encoders = iqc.encoder(family, angles)

    estimate = iqc.estimate_pauli_z(
        params,
        ansatz.generators,
        masks,
        encoders,
        jax.random.PRNGKey(9),
        200_000,
    )
    values = np.asarray(estimate.values)
    errors = np.asarray(estimate.standard_errors)

    for condition in range(N_CONDITIONS):
        for mask_index, mask in enumerate(masks):
            expected = pl_expectation(
                family,
                angles[condition],
                params,
                np.asarray(ansatz.generators),
                mask,
                N_VISIBLE,
                N_QUBITS,
            )
            assert abs(values[condition, mask_index] - expected) < 5.0 * max(
                errors[condition, mask_index],
                1e-12,
            )


@pytest.mark.parametrize("family", ["product", "spin_flip"])
def test_reference_sampling_matches_pennylane(family: str) -> None:
    """Sampled bitstring frequencies against the reference state's Born law.

    ``Encoder.sample`` draws from the post-Hadamard state, not from the model
    output, so the reference is PennyLane's state after the encoder and the
    first Hadamard layer.
    """
    angles = angles_for(family, seed=11)
    encoders = iqc.encoder(family, angles)
    n_samples = 100_000

    samples = np.asarray(encoders.sample(jax.random.PRNGKey(13), n_samples))
    for condition in range(N_CONDITIONS):
        codes = (samples[condition] * 2 ** np.arange(N_QUBITS)).sum(axis=-1)
        empirical = np.bincount(codes.astype(int), minlength=2**N_QUBITS) / n_samples
        state = pl_state_after_first_hadamard(family, angles[condition], N_QUBITS)
        expected = np.abs(state) ** 2
        expected = expected[reverse_bits(np.arange(2**N_QUBITS), N_QUBITS)]
        sigma = 0.5 * np.sqrt(np.sum(expected * (1.0 - expected)) / n_samples)
        assert 0.5 * np.sum(np.abs(empirical - expected)) < 5.0 * sigma + 1e-4


def test_sampled_mmd_expectation_matches_exact_pennylane_value() -> None:
    """The stochastic loss is unbiased for the exact PennyLane MMD.

    The target is an analytic product law, so its character moments are exact;
    the model moments come from PennyLane; the mask expectation is the exact
    Hamming-RBF character sum.  The estimator's mean over independent keys must
    land on that value within Monte Carlo error.
    """
    sigma = 2.0
    n_ops, n_samples, n_keys = 32, 64, 240
    family = "spin_flip"
    angles = angles_for(family, seed=17)
    ansatz = iqc.complete_ansatz(N_QUBITS)
    rng = np.random.default_rng(18)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    generators = np.asarray(ansatz.generators)

    target_probabilities = rng.uniform(0.2, 0.8, size=(N_CONDITIONS, N_VISIBLE))
    targets = np.stack(
        [
            (rng.uniform(size=(512, N_VISIBLE)) < target_probabilities[c]).astype(float)
            for c in range(N_CONDITIONS)
        ],
    )

    masks = ((np.arange(2**N_VISIBLE)[:, None] >> np.arange(N_VISIBLE)) & 1).astype(
        float,
    )
    p_mask = float(iqc.mmd_operator_probability(sigma))
    weights = np.prod(np.where(masks == 1.0, p_mask, 1.0 - p_mask), axis=1)

    exact = np.zeros(N_CONDITIONS)
    for condition in range(N_CONDITIONS):
        target_mean = np.asarray(targets[condition]).mean(axis=0)
        mu_p = np.prod(
            np.where(masks == 1.0, 1.0 - 2.0 * target_mean, 1.0),
            axis=1,
        )
        mu_q = np.array(
            [
                pl_expectation(
                    family,
                    angles[condition],
                    params,
                    generators,
                    mask,
                    N_VISIBLE,
                    N_QUBITS,
                )
                for mask in masks
            ],
        )
        exact[condition] = np.sum(weights * (mu_q - mu_p) ** 2)
    exact_loss = float(exact.mean())

    encoders = iqc.encoder(family, angles)

    def one_loss(key: jax.Array) -> jax.Array:
        return iqc.sampled_mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            key,
            sigmas=(sigma,),
            n_ops=n_ops,
            n_samples=n_samples,
        ).loss

    losses = np.asarray(
        [float(jax.jit(one_loss)(jax.random.PRNGKey(k))) for k in range(n_keys)]
    )
    standard_error = losses.std(ddof=1) / np.sqrt(n_keys)
    assert abs(losses.mean() - exact_loss) < 5.0 * standard_error + 1e-9
