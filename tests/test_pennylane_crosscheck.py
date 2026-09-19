"""Independent cross-check of the conditional IQP model against PennyLane.

PennyLane is a package dependency, so these tests always run.  They rebuild the
encoder state and the full conditional circuit from scratch with
``default.qubit`` and compare the exact statevector expectations against the
library's classically computable estimator.  The reference comes from PennyLane
rather than from the library's own exact path, so a shared convention bug cannot
pass unnoticed.

Wire order differs.  PennyLane returns states with wire 0 as the most significant
bit, and the library is little-endian.  State comparisons bit-reverse the index;
expectation comparisons build PennyLane observables on the same wires and do not
depend on the order.
"""

from __future__ import annotations

import jax
import numpy as np
import pennylane as qml

import iqp_cond as ciqp
from iqp_cond.operators import mmd_operator_probability


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit: the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.angle_encoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


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


def angles_for(seed: int = 3) -> np.ndarray:
    """Well-spread product-state angles, one per condition and qubit."""
    return np.random.default_rng(seed).uniform(0.4, 2.2, size=(N_CONDITIONS, N_QUBITS))


def prepare_ry_layer(angles: np.ndarray, n_qubits: int) -> None:
    """State preparation for one condition, in PennyLane's wire ordering."""
    for wire, angle in enumerate(angles):
        qml.RY(float(angle), wires=wire)


def prepare_angle_encoder(axes: tuple[str, ...], angles: np.ndarray) -> None:
    """Layered single-qubit rotations; layer 0 is applied first."""
    for layer, labels in enumerate(axes):
        for wire, label in enumerate(labels):
            getattr(qml, f"R{label}")(float(angles[layer, wire]), wires=wire)


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


def pl_state_after_first_hadamard(angles: np.ndarray, n_qubits: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        prepare_ry_layer(angles, n_qubits)
        for wire in range(n_qubits):
            qml.Hadamard(wire)
        return qml.state()

    return np.asarray(circuit())


def pl_layered_state(
    axes: tuple[str, ...],
    angles: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        prepare_angle_encoder(axes, angles)
        for wire in range(n_qubits):
            qml.Hadamard(wire)
        return qml.state()

    return np.asarray(circuit())


def pl_expectation(
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
        prepare_ry_layer(angles, n_qubits)
        apply_iqp(params, generators, n_qubits)
        return qml.expval(observable)

    return float(circuit())


def test_encoder_state_matches_pennylane() -> None:
    angles = angles_for()
    encoders = y_layer(angles)
    state = np.asarray(encoders.post_hadamard_amplitudes())

    for condition in range(N_CONDITIONS):
        expected = pl_state_after_first_hadamard(angles[condition], N_QUBITS)
        reordered = state[condition][reverse_bits(np.arange(2**N_QUBITS), N_QUBITS)]
        assert np.max(np.abs(reordered - expected)) < 1e-12


def test_amplitude_ratios_match_pennylane() -> None:
    angles = angles_for(seed=5)
    encoders = y_layer(angles)
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


def test_pauli_moments_match_pennylane() -> None:
    """Estimator moments against exact PennyLane expectation values.

    The model carries one ancilla, so the comparison also exercises the
    visible-mask padding and the ancilla marginalization: PennyLane measures Z
    on the same visible wires and identity elsewhere.
    """
    angles = angles_for(seed=7)
    ansatz = ciqp.complete_ansatz(N_QUBITS)
    rng = np.random.default_rng(8)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    masks = rng.integers(0, 2, size=(6, N_VISIBLE)).astype(np.float64)
    masks[0] = [1.0, 0.0, 0.0]
    encoders = y_layer(angles)

    estimate = ciqp.estimate_pauli_z(
        params,
        ansatz.generators,
        encoders,
        masks,
        key=jax.random.PRNGKey(9),
        n_samples=200_000,
        return_samples=True,
    )
    values = np.asarray(estimate.values)
    errors = np.asarray(estimate.standard_errors)

    for condition in range(N_CONDITIONS):
        for mask_index, mask in enumerate(masks):
            expected = pl_expectation(
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


def test_estimator_gradient_matches_independent_pennylane_gradient() -> None:
    """Check the complex ratio-gradient estimator against a separate QNode.

    The X/Z encoder layers make the reference amplitudes complex, while the
    three-bit observable mask is padded across the fourth, ancilla qubit.
    """
    axes = ("XYXX", "ZXYZ")
    angles = np.asarray(
        np.random.default_rng(41).uniform(0.3, 2.1, size=(1, len(axes), N_QUBITS))
    )
    encoders = ciqp.angle_encoder(angles, axes)
    ansatz = ciqp.complete_ansatz(N_QUBITS)
    generators = np.asarray(ansatz.generators)
    params = jax.numpy.asarray(
        np.random.default_rng(42).normal(0.0, 0.25, ansatz.n_parameters),
    )
    visible_mask = jax.numpy.asarray([[1.0, 0.0, 1.0]])
    samples = encoders.sample(jax.random.PRNGKey(43), 200_000)

    def sample_values(theta: jax.Array) -> jax.Array:
        estimate = ciqp.estimate_pauli_z(
            theta,
            ansatz.generators,
            encoders,
            visible_mask,
            samples=samples,
            return_samples=True,
        )
        assert estimate.sample_values is not None
        return estimate.sample_values[0, :, 0]

    per_sample_grad = np.asarray(jax.jacfwd(sample_values)(params))
    gradient = per_sample_grad.mean(axis=0)
    standard_error = per_sample_grad.std(axis=0, ddof=1) / np.sqrt(samples.shape[1])

    dev = qml.device("default.qubit", wires=N_QUBITS)
    jax_angles = jax.numpy.asarray(angles[0])

    @qml.qnode(dev, interface="jax")
    def circuit(theta: jax.Array) -> jax.Array:
        for layer, labels in enumerate(axes):
            for wire, label in enumerate(labels):
                getattr(qml, f"R{label}")(jax_angles[layer, wire], wires=wire)
        for wire in range(N_QUBITS):
            qml.Hadamard(wire)
        for index, generator in enumerate(generators):
            wires = [int(i) for i, bit in enumerate(generator) if bit]
            qml.MultiRZ(-2.0 * theta[index], wires=wires)
        for wire in range(N_QUBITS):
            qml.Hadamard(wire)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    exact_gradient = np.asarray(jax.grad(circuit)(params))
    assert np.all(np.abs(gradient - exact_gradient) <= 5.0 * standard_error + 2e-5)


def test_reference_sampling_matches_pennylane() -> None:
    """Sampled bitstring frequencies against the reference state's Born law.

    ``Encoder.sample`` draws from the post-Hadamard state, not from the model
    output, so the reference is PennyLane's state after the encoder and the
    first Hadamard layer.
    """
    angles = angles_for(seed=11)
    encoders = y_layer(angles)
    n_samples = 100_000

    samples = np.asarray(encoders.sample(jax.random.PRNGKey(13), n_samples))
    for condition in range(N_CONDITIONS):
        codes = (samples[condition] * 2 ** np.arange(N_QUBITS)).sum(axis=-1)
        empirical = np.bincount(codes.astype(int), minlength=2**N_QUBITS) / n_samples
        state = pl_state_after_first_hadamard(angles[condition], N_QUBITS)
        expected = np.abs(state) ** 2
        expected = expected[reverse_bits(np.arange(2**N_QUBITS), N_QUBITS)]
        sigma = 0.5 * np.sqrt(np.sum(expected * (1.0 - expected)) / n_samples)
        assert 0.5 * np.sum(np.abs(empirical - expected)) < 5.0 * sigma + 1e-4


def test_layered_angle_encoder_state_matches_pennylane() -> None:
    """The multi-layer family composes rotations in the documented order."""
    axes = ("YXXY", "ZXYZ", "YYZZ")
    angles = np.random.default_rng(23).uniform(
        0.4,
        2.2,
        size=(N_CONDITIONS, len(axes), N_QUBITS),
    )
    encoders = ciqp.angle_encoder(angles, axes)
    state = np.asarray(encoders.post_hadamard_amplitudes())

    for condition in range(N_CONDITIONS):
        expected = pl_layered_state(axes, angles[condition], N_QUBITS)
        reordered = state[condition][reverse_bits(np.arange(2**N_QUBITS), N_QUBITS)]
        assert np.max(np.abs(reordered - expected)) < 1e-12


def test_mmd_expectation_matches_exact_pennylane_value() -> None:
    """The stochastic loss is unbiased for the exact PennyLane MMD.

    The target is an analytic product law, so its character moments are exact;
    the model moments come from PennyLane; the mask expectation is the exact
    Hamming-RBF character sum.  The estimator's mean over independent keys must
    land on that value within Monte Carlo error.
    """
    sigma = 2.0
    n_ops, n_samples, n_keys = 32, 64, 240
    angles = angles_for(seed=17)
    ansatz = ciqp.complete_ansatz(N_QUBITS)
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
    p_mask = float(mmd_operator_probability(sigma))
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

    encoders = y_layer(angles)

    def one_loss(key: jax.Array) -> jax.Array:
        return ciqp.mmd(
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


def pl_visible_probabilities(
    angles: np.ndarray,
    params: np.ndarray,
    generators: np.ndarray,
    n_visible: int,
    n_qubits: int,
) -> np.ndarray:
    """PennyLane distribution over the visible wires after the full circuit."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        prepare_ry_layer(angles, n_qubits)
        apply_iqp(params, generators, n_qubits)
        return qml.probs(wires=range(n_visible))

    return np.asarray(circuit())


def test_simulation_probabilities_match_pennylane() -> None:
    """The bridge distribution, including ancilla marginalization."""
    angles = angles_for(seed=29)
    ansatz = ciqp.complete_ansatz(N_QUBITS)
    rng = np.random.default_rng(30)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    model = ciqp.ConditionalModel(ansatz, y_layer(angles), n_visible=N_VISIBLE)
    probabilities = ciqp.simulation.probabilities(model, params)

    for condition in range(N_CONDITIONS):
        expected = pl_visible_probabilities(
            angles[condition],
            params,
            np.asarray(ansatz.generators),
            N_VISIBLE,
            N_QUBITS,
        )
        # PennyLane returns wire 0 as the most significant bit.
        reordered = probabilities[condition][
            reverse_bits(np.arange(2**N_VISIBLE), N_VISIBLE)
        ]
        assert np.allclose(reordered, expected, atol=1e-12)
