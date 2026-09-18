"""Generator banks, the conditional model, and exact model evaluation."""

from __future__ import annotations

import numpy as np
import pytest

import iqp_cond as ciqp


def test_complete_ansatz_has_one_and_two_body_rows() -> None:
    ansatz = ciqp.complete_ansatz(4)
    rows = np.asarray(ansatz.generators)
    assert ansatz.n_qubits == 4
    assert ansatz.n_parameters == 4 + 6
    assert set(np.unique(rows)) == {0.0, 1.0}
    assert np.array_equal(rows[:4], np.eye(4))
    # One row per unordered pair, in lexicographic order.
    pairs = [np.flatnonzero(row) for row in rows[4:]]
    assert all(len(pair) == 2 for pair in pairs)
    assert [tuple(pair) for pair in pairs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_ansatz_from_generators_accepts_arbitrary_banks() -> None:
    rows = np.array([[1.0, 1.0], [0.0, 1.0]])
    ansatz = ciqp.ansatz_from_generators(rows, name="pair")
    assert ansatz.name == "pair"
    assert ansatz.n_qubits == 2
    assert ansatz.n_parameters == 2
    assert np.array_equal(np.asarray(ansatz.generators), rows)


def test_ansatz_from_generators_rejects_non_binary_rows() -> None:
    with pytest.raises(ValueError, match="only zeros and ones"):
        ciqp.ansatz_from_generators(np.array([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="shape"):
        ciqp.ansatz_from_generators(np.array([1.0, 0.0]))


def test_ansatz_constructor_validates_binary_rows() -> None:
    """Direct construction cannot bypass the bitmask contract."""
    with pytest.raises(ValueError, match="only zeros and ones"):
        ciqp.Ansatz(generators=np.array([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="only zeros and ones"):
        ciqp.Ansatz(generators=np.array([[2.0, 0.0]]))
    with pytest.raises(ValueError, match="only zeros and ones"):
        ciqp.Ansatz(generators=np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="shape"):
        ciqp.Ansatz(generators=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        ciqp.Ansatz(generators=np.zeros((0, 3)))


def test_ansatz_constructor_coerces_dtype() -> None:
    ansatz = ciqp.Ansatz(generators=np.array([[1, 0], [0, 1]], dtype=np.int32))
    assert ansatz.generators.dtype == np.float64
    assert ansatz.n_qubits == 2
    assert ansatz.n_parameters == 2


def test_complete_ansatz_rejects_empty_registers() -> None:
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        ciqp.complete_ansatz(0)


def test_ancilla_support_marks_rows_touching_ancillas() -> None:
    generators = np.asarray(ciqp.complete_ansatz(4).generators)
    support = ciqp.ancilla_support(generators, n_visible=2)
    assert support.shape == (generators.shape[0],)
    expected = np.any(generators[:, 2:] != 0, axis=1)
    assert np.array_equal(support, expected)
    with pytest.raises(ValueError, match="shape"):
        ciqp.ancilla_support(np.zeros(3), 1)
    with pytest.raises(ValueError, match="n_visible"):
        ciqp.ancilla_support(generators, 5)


def test_pad_visible_masks_pads_and_validates() -> None:
    masks = np.array([[1.0, 0.0]])
    padded = np.asarray(ciqp.pad_visible_masks(masks, 2))
    assert np.array_equal(padded, [[1.0, 0.0, 0.0, 0.0]])
    assert np.array_equal(
        np.asarray(ciqp.pad_visible_masks(np.array([1.0, 1.0]), 0)),
        [[1.0, 1.0]],
    )
    with pytest.raises(ValueError, match="nonnegative"):
        ciqp.pad_visible_masks(masks, -1)
    with pytest.raises(ValueError, match="shape"):
        ciqp.pad_visible_masks(np.zeros((2, 2, 2)), 1)


def test_model_exposes_the_visible_ancilla_split() -> None:
    encoders = ciqp.angle_encoder(
        np.zeros((2, 1, 3)),
        axes=("YYY",),
    )
    model = ciqp.ConditionalModel(ciqp.complete_ansatz(3), encoders, n_visible=2)
    assert model.n_qubits == 3
    assert model.visible == 2
    assert model.n_ancillas == 1
    full = ciqp.ConditionalModel(ciqp.complete_ansatz(3), encoders)
    assert full.visible == 3
    assert full.n_ancillas == 0


def _apply_hadamard(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Explicit ``<y|H^n|x> = (-1)^{x.y} / 2^{n/2}`` on little-endian indices."""
    out = np.zeros_like(state)
    for y in range(2**n_qubits):
        for x in range(2**n_qubits):
            out[y] += (-1.0) ** bin(x & y).count("1") * state[x] / 2 ** (n_qubits / 2)
    return out


def _exact_joint(model: ciqp.ConditionalModel, params: np.ndarray) -> np.ndarray:
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


def test_exact_probabilities_match_a_direct_construction() -> None:
    rng = np.random.default_rng(0)
    angles = rng.uniform(0.3, 2.4, size=(1, 2))
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(2),
        ciqp.angle_encoder(angles[:, None, :], axes=("YY",)),
    )
    params = rng.normal(0.0, 0.4, size=model.ansatz.n_parameters)
    expected = np.abs(_exact_joint(model, params)) ** 2
    probabilities = ciqp.exact_probabilities(model, params)
    assert probabilities.shape == (1, 4)
    assert np.allclose(probabilities[0], expected)


def test_exact_probabilities_marginalize_ancillas() -> None:
    rng = np.random.default_rng(1)
    angles = rng.uniform(0.3, 2.4, size=(1, 2))
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(2),
        ciqp.angle_encoder(angles[:, None, :], axes=("YY",)),
        n_visible=1,
    )
    params = rng.normal(0.0, 0.4, size=model.ansatz.n_parameters)
    # Ancillas are traced out: sum the joint probabilities.
    # Little-endian: index = visible + 2 * ancilla.
    joint = np.abs(_exact_joint(model, params)) ** 2
    expected = joint.reshape(2, 2).sum(axis=0)
    probabilities = ciqp.exact_probabilities(model, params)
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities[0], expected)
    assert np.isclose(probabilities.sum(), 1.0)


def test_sample_exact_follows_the_exact_law() -> None:
    rng = np.random.default_rng(2)
    angles = rng.uniform(0.3, 2.4, size=(1, 2))
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(2),
        ciqp.angle_encoder(angles[:, None, :], axes=("XY",)),
    )
    params = rng.normal(0.0, 0.4, size=model.ansatz.n_parameters)
    probabilities = ciqp.exact_probabilities(model, params)[0]

    n_samples = 40_000
    samples = ciqp.sample_exact(model, params, n_samples, seed=3)
    assert samples.shape == (1, n_samples, 2)
    assert np.isin(samples, (0.0, 1.0)).all()
    codes = (samples[0] * 2 ** np.arange(2)).sum(axis=-1).astype(int)
    empirical = np.bincount(codes, minlength=4) / n_samples
    tolerance = 5.0 * np.sqrt(probabilities * (1.0 - probabilities) / n_samples)
    assert np.all(np.abs(empirical - probabilities) < tolerance + 1e-9)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        ciqp.sample_exact(model, params, 0)


def test_exact_probabilities_validate_the_setup() -> None:
    class Minimal:
        n_conditions = 1
        n_qubits = 2

        def sample(self, key, n_samples):  # pragma: no cover - interface only
            raise NotImplementedError

        def amplitude_ratios(self, samples, masks):  # pragma: no cover
            raise NotImplementedError

    model = ciqp.ConditionalModel(ciqp.complete_ansatz(2), Minimal())
    with pytest.raises(TypeError, match="post_hadamard_amplitudes"):
        ciqp.exact_probabilities(model, np.zeros(model.ansatz.n_parameters))

    exact = ciqp.ConditionalModel(
        ciqp.complete_ansatz(2),
        ciqp.angle_encoder(np.zeros((1, 1, 2)), axes=("YY",)),
    )
    with pytest.raises(ValueError, match="params must have shape"):
        ciqp.exact_probabilities(exact, np.zeros(2))
    bad = np.zeros(exact.ansatz.n_parameters)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ciqp.exact_probabilities(exact, bad)
