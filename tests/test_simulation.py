"""The PennyLane bridge: circuit preparation, probabilities, and sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import pytest

import iqp_cond as ciqp


def _apply_hadamard(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Explicit ``<y|H^n|x> = (-1)^{x.y} / 2^{n/2}`` on little-endian indices."""
    out = np.zeros_like(state)
    for y in range(2**n_qubits):
        for x in range(2**n_qubits):
            out[y] += (-1.0) ** bin(x & y).count("1") * state[x] / 2 ** (n_qubits / 2)
    return out


def _exact_joint(model: ciqp.ConditionalModel, params: np.ndarray) -> np.ndarray:
    """Independent statevector oracle for one condition."""
    weights = 1 << np.arange(model.n_qubits, dtype=np.int64)
    generator_codes = np.asarray(model.generators, dtype=np.int64) @ weights
    psi = np.asarray(model.encoders.post_hadamard_amplitudes())[0]
    characters = np.array(
        [
            [1 - 2 * (bin(code & int(g)).count("1") % 2) for g in generator_codes]
            for code in range(2**model.n_qubits)
        ],
    )
    phases = characters @ np.asarray(params, dtype=np.float64)
    return _apply_hadamard(psi * np.exp(1j * phases), model.n_qubits)


def _model(
    axes: tuple[str, ...], n_visible: int | None = None
) -> ciqp.ConditionalModel:
    rng = np.random.default_rng(4)
    angles = rng.uniform(0.3, 2.4, size=(1, len(axes), len(axes[0])))
    n_qubits = len(axes[0])
    encoder = ciqp.angle_encoder(angles, axes=axes)
    if n_visible is None:
        return ciqp.ConditionalModel(ciqp.complete_ansatz(n_qubits), encoder)
    return ciqp.ConditionalModel(
        ciqp.complete_ansatz(n_qubits),
        encoder,
        n_visible=n_visible,
    )


def _params(model: ciqp.ConditionalModel, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        0.0,
        0.4,
        size=model.ansatz.n_parameters,
    )


def test_probabilities_match_the_statevector_oracle() -> None:
    model = _model(("YY",))
    params = _params(model)
    expected = np.abs(_exact_joint(model, params)) ** 2
    probabilities = ciqp.simulation.probabilities(model, params)
    assert probabilities.shape == (1, 4)
    assert np.allclose(probabilities[0], expected)


def test_probabilities_marginalize_ancillas() -> None:
    model = _model(("YY",), n_visible=1)
    params = _params(model)
    joint = np.abs(_exact_joint(model, params)) ** 2
    expected = joint.reshape(2, 2).sum(axis=0)
    probabilities = ciqp.simulation.probabilities(model, params)
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities[0], expected)
    assert np.isclose(probabilities.sum(), 1.0)


def test_probabilities_match_for_layered_axes() -> None:
    model = _model(("XY", "ZY", "YZ"))
    params = _params(model, seed=6)
    expected = np.abs(_exact_joint(model, params)) ** 2
    probabilities = ciqp.simulation.probabilities(model, params)
    assert np.allclose(probabilities[0], expected)


def test_probabilities_accept_a_device_instance() -> None:
    model = _model(("Y",))
    params = _params(model)
    default = ciqp.simulation.probabilities(model, params)
    device = qml.device("default.qubit", wires=1)
    bridged = ciqp.simulation.probabilities(model, params, device=device)
    assert np.allclose(default, bridged)


def test_sample_model_follows_the_exact_law() -> None:
    model = _model(("XY",), n_visible=1)
    params = _params(model, seed=7)
    probabilities = ciqp.simulation.probabilities(model, params)[0]

    n_samples = 40_000
    draws = ciqp.simulation.sample_model(model, params, n_samples, seed=8)
    assert draws.shape == (1, n_samples, 1)
    assert draws.dtype == np.float64
    assert np.isin(draws, (0.0, 1.0)).all()

    empirical = draws[0, :, 0].mean()
    sigma = np.sqrt(probabilities[1] * (1.0 - probabilities[1]) / n_samples)
    assert abs(empirical - probabilities[1]) < 5.0 * sigma + 1e-9


def test_sample_model_is_reproducible_in_the_seed() -> None:
    model = _model(("Y",))
    params = _params(model)
    first = ciqp.simulation.sample_model(model, params, 256, seed=11)
    repeat = ciqp.simulation.sample_model(model, params, 256, seed=11)
    other = ciqp.simulation.sample_model(model, params, 256, seed=12)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_sample_model_rejects_a_seeded_device_instance() -> None:
    model = _model(("Y",))
    params = _params(model)
    device = qml.device("default.qubit", wires=1)
    with pytest.raises(ValueError, match="seed applies"):
        ciqp.simulation.sample_model(model, params, 8, seed=1, device=device)


def test_build_qnode_samples_the_visible_wires() -> None:
    model = _model(("YY",), n_visible=1)
    params = _params(model)
    qnode = ciqp.simulation.build_qnode(model, params, 0, shots=16)
    samples = np.asarray(qnode())
    assert samples.shape == (16, 1)


@dataclass(frozen=True)
class _BadEncoder:
    name: str = "bad"

    @property
    def n_conditions(self) -> int:
        return 1

    @property
    def n_qubits(self) -> int:
        return 1

    def sample(self, key, n_samples):  # pragma: no cover - interface only
        raise NotImplementedError

    def amplitude_ratios(self, samples, masks):  # pragma: no cover
        raise NotImplementedError

    def preparation_ops(self, condition: int) -> tuple[ciqp.Operation, ...]:
        return (ciqp.Operation("H", (0,), ()),)


def test_simulation_validates_the_setup() -> None:
    model = _model(("Y",))
    params = _params(model)
    with pytest.raises(ValueError, match="params must have shape"):
        ciqp.simulation.probabilities(model, np.zeros(3))
    bad = params.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ciqp.simulation.probabilities(model, bad)
    with pytest.raises(IndexError, match="condition"):
        ciqp.simulation.build_qnode(model, params, 3, shots=4)

    unsupported = ciqp.ConditionalModel(ciqp.complete_ansatz(1), _BadEncoder())
    with pytest.raises(ValueError, match="unsupported preparation operation"):
        ciqp.simulation.probabilities(unsupported, np.zeros(1))

    class Minimal:
        n_conditions = 1
        n_qubits = 1

        def sample(self, key, n_samples):  # pragma: no cover - interface only
            raise NotImplementedError

        def amplitude_ratios(self, samples, masks):  # pragma: no cover
            raise NotImplementedError

    plain = ciqp.ConditionalModel(ciqp.complete_ansatz(1), Minimal())
    with pytest.raises(TypeError, match="preparation_ops"):
        ciqp.simulation.probabilities(plain, np.zeros(1))


@dataclass(frozen=True)
class _PreparedEncoder:
    """A test encoder whose preparation and exact reference state are given."""

    operations: tuple[ciqp.Operation, ...]
    amplitudes: np.ndarray

    @property
    def n_conditions(self) -> int:
        return 1

    @property
    def n_qubits(self) -> int:
        return int(np.log2(self.amplitudes.shape[-1]))

    def sample(self, key, n_samples):  # pragma: no cover - interface only
        raise NotImplementedError

    def amplitude_ratios(self, samples, masks):  # pragma: no cover
        raise NotImplementedError

    def preparation_ops(self, condition: int) -> tuple[ciqp.Operation, ...]:
        return self.operations

    def post_hadamard_amplitudes(self) -> np.ndarray:
        return self.amplitudes


def test_cnot_preparation_is_supported() -> None:
    """A GHZ-type preparation compiles and matches the reference state."""
    bell = np.zeros((1, 4), dtype=np.complex128)
    bell[0, 0] = 1.0 / np.sqrt(2.0)
    bell[0, 3] = 1.0 / np.sqrt(2.0)
    encoder = _PreparedEncoder(
        operations=(
            ciqp.Operation("RY", (0,), (np.pi / 2.0,)),
            ciqp.Operation("CNOT", (0, 1)),
        ),
        amplitudes=bell,
    )
    model = ciqp.ConditionalModel(ciqp.complete_ansatz(2), encoder)
    params = _params(model, seed=8)
    expected = np.abs(_exact_joint(model, params)) ** 2
    probabilities = ciqp.simulation.probabilities(model, params)
    assert probabilities.shape == (1, 4)
    assert np.allclose(probabilities[0], expected)


def test_preparation_gate_arity_is_validated() -> None:
    cases = (
        ((ciqp.Operation("CNOT", (0,)),), "CNOT preparation needs 2 wire"),
        (
            (ciqp.Operation("CNOT", (0, 1), (0.3,)),),
            "0 parameter",
        ),
        ((ciqp.Operation("RY", (0, 1), (0.3,)),), "RY preparation needs 1 wire"),
    )
    for operations, message in cases:
        encoder = _PreparedEncoder(
            operations=operations,
            amplitudes=np.zeros((1, 4), dtype=np.complex128),
        )
        model = ciqp.ConditionalModel(ciqp.complete_ansatz(2), encoder)
        with pytest.raises(ValueError, match=message):
            ciqp.simulation.probabilities(model, np.zeros(model.ansatz.n_parameters))
