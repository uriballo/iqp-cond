"""Ancilla conventions: visible masks are padded with identity on ancillas."""

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
