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
