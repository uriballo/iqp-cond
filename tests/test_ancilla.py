"""Ancilla conventions: mask padding and visible marginalization."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_marginalize_ancillas_recovers_the_visible_law() -> None:
    visible = np.array([0.1, 0.2, 0.3, 0.4])
    ancilla = np.array([0.25, 0.75])
    joint = np.outer(ancilla, visible).ravel()
    marginal = iqc.marginalize_ancillas(joint, n_data_qubits=2, n_ancillas=1)
    assert np.allclose(marginal, visible)


def test_marginalize_ancillas_handles_batches() -> None:
    visible = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    ancilla = np.array([0.25, 0.75])
    joint = np.stack(
        [np.kron(np.array([a]), row) for row in visible for a in ancilla],
    ).reshape(2, 8)
    marginal = iqc.marginalize_ancillas(joint, n_data_qubits=2, n_ancillas=1)
    assert np.allclose(marginal, visible)


def test_marginalize_ancillas_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        iqc.marginalize_ancillas(np.zeros((2, 2, 2)), n_data_qubits=1, n_ancillas=1)


def test_ancilla_only_angles_zero_the_data_qubits() -> None:
    angles = np.array([[0.5, 1.5], [0.7, 1.1]])
    placed = np.asarray(iqc.ancilla_only_condition_angles(angles, n_data_qubits=3))
    assert placed.shape == (2, 5)
    assert np.all(placed[:, :3] == 0.0)
    assert np.array_equal(placed[:, 3:], angles)
