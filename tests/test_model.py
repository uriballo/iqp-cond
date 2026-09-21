"""Generator banks, the conditional model, and exact model evaluation."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.estimator import estimate_pauli_z
from iqp_cond.loss import mmd
from iqp_cond.model import ancilla_support, pad_visible_masks
from iqp_cond.verification import EncoderCheck, check_encoder


def test_package_root_is_the_compact_workflow_api() -> None:
    assert ciqp.__all__ == [
        "AngleEncoder",
        "Ansatz",
        "ConditionalModel",
        "Encoder",
        "EncoderCheck",
        "TrainingConfig",
        "check_encoder",
        "train",
    ]
    # The verification helpers resolve lazily, without importing PennyLane.
    assert ciqp.check_encoder is check_encoder
    assert ciqp.EncoderCheck is EncoderCheck
    for removed in (
        "CircuitEncoder",
        "Operation",
        "angle_encoder",
        "complete_ansatz",
        "encoder",
        "register_encoder",
        "STEP",
        "CACHE",
    ):
        assert not hasattr(ciqp, removed)


def test_complete_ansatz_has_one_and_two_body_rows() -> None:
    ansatz = ciqp.Ansatz.complete(4)
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
    ansatz = ciqp.Ansatz(rows, name="pair")
    assert ansatz.name == "pair"
    assert ansatz.n_qubits == 2
    assert ansatz.n_parameters == 2
    assert np.array_equal(np.asarray(ansatz.generators), rows)


def test_ansatz_from_generators_rejects_non_binary_rows() -> None:
    with pytest.raises(ValueError, match="only zeros and ones"):
        ciqp.Ansatz(np.array([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="shape"):
        ciqp.Ansatz(np.array([1.0, 0.0]))


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
        ciqp.Ansatz.complete(0)


def test_ancilla_support_marks_rows_touching_ancillas() -> None:
    generators = np.asarray(ciqp.Ansatz.complete(4).generators)
    support = ancilla_support(generators, n_visible=2)
    assert support.shape == (generators.shape[0],)
    expected = np.any(generators[:, 2:] != 0, axis=1)
    assert np.array_equal(support, expected)
    with pytest.raises(ValueError, match="shape"):
        ancilla_support(np.zeros(3), 1)
    with pytest.raises(ValueError, match="n_visible"):
        ancilla_support(generators, 5)


def test_pad_visible_masks_pads_and_validates() -> None:
    masks = np.array([[1.0, 0.0]])
    padded = np.asarray(pad_visible_masks(masks, 2))
    assert np.array_equal(padded, [[1.0, 0.0, 0.0, 0.0]])
    assert np.array_equal(
        np.asarray(pad_visible_masks(np.array([1.0, 1.0]), 0)),
        [[1.0, 1.0]],
    )
    with pytest.raises(ValueError, match="nonnegative"):
        pad_visible_masks(masks, -1)
    with pytest.raises(ValueError, match="shape"):
        pad_visible_masks(np.zeros((2, 2, 2)), 1)


def test_ancilla_support_without_ancillas_is_empty() -> None:
    ansatz = ciqp.Ansatz.complete(3)
    support = ancilla_support(ansatz.generators, n_visible=3)
    assert not support.any()


def test_model_exposes_the_visible_ancilla_split() -> None:
    encoders = ciqp.AngleEncoder(
        np.zeros((2, 1, 3)),
        axes=("YYY",),
    )
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(3), encoders, n_visible=2)
    assert model.n_qubits == 3
    assert model.visible == 2
    assert model.n_ancillas == 1
    full = ciqp.ConditionalModel(ciqp.Ansatz.complete(3), encoders)
    assert full.visible == 3
    assert full.n_ancillas == 0


def test_model_estimate_pauli_z_delegates_to_the_estimator() -> None:
    angles = np.random.default_rng(5).uniform(0.4, 2.2, size=(2, 3))
    encoder = ciqp.AngleEncoder(angles[:, None, :], axes=("YYY",))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(3), encoder, n_visible=2)
    params = np.zeros(model.ansatz.n_parameters)
    masks = np.array([[1.0, 0.0], [0.0, 1.0]])
    key = jax.random.PRNGKey(6)

    direct = estimate_pauli_z(
        params,
        model.generators,
        encoder,
        masks,
        key=key,
        n_samples=64,
    )
    via_model = model.estimate_pauli_z(params, masks, key=key, n_samples=64)
    assert np.array_equal(np.asarray(via_model.values), np.asarray(direct.values))


def test_model_mmd_delegates_and_validates_the_target_width() -> None:
    angles = np.random.default_rng(7).uniform(0.4, 2.2, size=(2, 3))
    encoder = ciqp.AngleEncoder(angles[:, None, :], axes=("YYY",))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(3), encoder, n_visible=2)
    params = np.zeros(model.ansatz.n_parameters)
    targets = np.random.default_rng(8).integers(0, 2, size=(2, 16, 2)).astype(float)
    key = jax.random.PRNGKey(9)

    direct = mmd(
        params,
        model.generators,
        encoder,
        targets,
        key,
        sigmas=(1.0,),
        n_ops=8,
        n_samples=16,
    )
    via_model = model.mmd(params, targets, key, sigmas=(1.0,), n_ops=8, n_samples=16)
    assert float(via_model.loss) == float(direct.loss)

    with pytest.raises(ValueError, match="targets have 3 visible qubits"):
        model.mmd(
            params,
            np.zeros((2, 16, 3)),
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )
