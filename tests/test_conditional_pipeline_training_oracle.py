import numpy as np
import pytest

from benchmarks.conditional_pipeline.training_workload import exact_product_state_oracle

oracle = exact_product_state_oracle


def _reference(params, angles):
    n = len(angles)
    single = np.array(
        [[np.exp(-0.5j * a), np.exp(0.5j * a)] for a in angles]
    ) / np.sqrt(2)
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    state = np.prod(single[np.arange(n)[None, :], bits], axis=1)
    full_h = np.empty((1 << n, 1 << n), dtype=complex)
    for out in range(1 << n):
        for inp in range(1 << n):
            full_h[out, inp] = (-1) ** (out & inp).bit_count() / np.sqrt(1 << n)
    rows = list(range(n))
    cols = list(range(n))
    pair_rows, pair_cols = np.triu_indices(n, k=1)
    rows.extend(pair_rows)
    cols.extend(pair_cols)
    phase = np.zeros(1 << n)
    for theta, row, col in zip(params, rows, cols, strict=True):
        parity = bits[:, row] if row == col else (bits[:, row] + bits[:, col]) % 2
        phase += theta * (1 - 2 * parity)
    return np.abs(full_h @ (np.exp(1j * phase) * (full_h @ state))) ** 2


def test_oracle_matches_independent_three_qubit_reference():
    angles = np.array([0.2, -0.4, 0.7])
    params = np.arange(6, dtype=float) / 7
    actual = oracle(params, angles)
    np.testing.assert_allclose(
        actual, _reference(params, angles), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(actual.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_oracle_parameters_change_probabilities():
    angles = np.array([0.2, -0.4, 0.7])
    first = oracle(np.zeros(6), angles)
    second = oracle(np.ones(6) * 0.3, angles)
    assert not np.allclose(first, second)


def test_oracle_rejects_unsupported_shapes():
    with pytest.raises(ValueError):
        oracle(np.zeros(5), np.zeros(3))
    with pytest.raises(ValueError):
        oracle(np.zeros(3), np.zeros(17))


def test_oracle_matches_pennylane_circuit():
    import pennylane as qml

    from iqp_cond import Ansatz

    n = 3
    angles = np.array([0.2, -0.4, 0.7])
    params = np.arange(6) / 7
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        for q in range(n):
            qml.RY(np.pi / 2, wires=q)
            qml.RZ(angles[q], wires=q)
            qml.Hadamard(wires=q)
        for theta, support in zip(
            params, np.asarray(Ansatz.complete(n).generators), strict=True
        ):
            qml.MultiRZ(-2 * theta, wires=np.flatnonzero(support).tolist())
        for q in range(n):
            qml.Hadamard(wires=q)
        return qml.state()

    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    indices = bits @ (1 << np.arange(n - 1, -1, -1))
    np.testing.assert_allclose(
        oracle(params, angles), abs(np.asarray(circuit())[indices]) ** 2, atol=1e-12
    )
