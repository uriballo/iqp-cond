"""Ancilla conventions: mask padding and ancilla-support parameters."""

from __future__ import annotations

import numpy as np

import iqp_cond as iqc


def test_pad_visible_masks_appends_identity() -> None:
    masks = np.array([[1.0, 0.0], [0.0, 1.0]])
    padded = np.asarray(iqc.pad_visible_masks(masks, 3))
    assert padded.shape == (2, 5)
    assert np.array_equal(padded[:, :2], masks)
    assert np.all(padded[:, 2:] == 0.0)


def test_pad_visible_masks_without_ancillas_is_identity() -> None:
    masks = np.array([[1.0, 1.0]])
    assert np.array_equal(np.asarray(iqc.pad_visible_masks(masks, 0)), masks)


def test_pad_visible_masks_accepts_a_flat_mask() -> None:
    padded = np.asarray(iqc.pad_visible_masks(np.array([1.0, 0.0]), 2))
    assert padded.shape == (1, 4)
    assert np.array_equal(padded[0], [1.0, 0.0, 0.0, 0.0])


def test_ancilla_support_marks_generators_touching_ancillas() -> None:
    ansatz = iqc.complete_ansatz(4)
    support = iqc.ancilla_support(ansatz.generators, n_visible=3)
    assert support.shape == (ansatz.n_parameters,)
    assert support.sum() > 0
    # One-body rows on visible qubits, and two-body rows within the visible
    # block, are the only parameters initialized from data.
    generators = np.asarray(ansatz.generators)
    expected = np.any(generators[:, 3:] != 0, axis=1)
    assert (support == expected).all()


def test_ancilla_support_without_ancillas_is_empty() -> None:
    ansatz = iqc.complete_ansatz(3)
    support = iqc.ancilla_support(ansatz.generators, n_visible=3)
    assert not support.any()
