"""Parameter initialization from observed data statistics.

The one-body phases come from the observed marginals and the two-body phases
from the observed covariance; both are classical statistics of the training
sample.  Unsupported rows are set to zero.  Ancilla-supported parameters have
no data to learn from, so they are initialized from small seeded noise instead.
"""

from __future__ import annotations

import numpy as np

from .model import ancilla_support

__all__ = [
    "parameters_from_data",
    "random_parameters",
]


def random_parameters(n_parameters: int, seed: int) -> np.ndarray:
    """Draw generator-count normalized Gaussian parameters."""
    if n_parameters < 1:
        raise ValueError("n_parameters must be positive")
    rng = np.random.default_rng(seed + 10_000)
    scale = 0.1 / np.sqrt(n_parameters)
    return np.asarray(rng.normal(0.0, scale, n_parameters), dtype=np.float64)


def _data_parameters(
    data: np.ndarray,
    rows: np.ndarray,
    scale: float | None,
) -> np.ndarray:
    """One- and two-body phases from binary observations (zero elsewhere).

    The prescription normalizes by the largest observed pair covariance.  If
    every visible pair is exactly uncorrelated, no pair covariance is nonzero
    to normalize by, so the two-body phases stay zero.
    """
    n_features = data.shape[1]
    signed = 2.0 * data - 1.0
    covariance = np.cov(signed, rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance)
    off_diagonal = covariance.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    result = np.zeros(rows.shape[0], dtype=np.float64)
    means = np.mean(data, axis=0)
    has_two_body = bool(np.any(np.sum(rows, axis=1) == 2))
    max_abs_cov = float(np.max(np.abs(off_diagonal))) if has_two_body else 0.0
    if has_two_body and max_abs_cov > 0.0:
        if scale is None:
            cov_sq_sum = float(np.sum(off_diagonal**2))
            scale = np.sqrt(2.0 * n_features / cov_sq_sum) * max_abs_cov / np.pi
    else:
        scale = None
    for index, row in enumerate(rows):
        weight = int(np.sum(row))
        if weight == 1:
            result[index] = np.arcsin(np.sqrt(means[np.flatnonzero(row)[0]]))
        elif weight == 2 and scale is not None:
            left, right = np.flatnonzero(row)
            result[index] = covariance[left, right] * scale * np.pi / max_abs_cov
    return result


def parameters_from_data(
    data: np.ndarray,
    generators: np.ndarray,
    *,
    n_visible: int | None = None,
    scale: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Initialize one- and two-body IQP parameters from binary observations.

    ``data`` carries only visible-qubit observations, one column per visible
    qubit; ``n_visible`` says how many of the generator columns those are, and
    defaults to the data width.  When the generator bank is wider than the
    data, the remaining columns are ancillas, and the function draws their
    parameters from a seeded Gaussian of width ``0.01/sqrt(P)`` so ancilla
    blocks start near zero without being exactly symmetric.

    One-body rows use ``arcsin(sqrt(mean))``.  For two-body rows, the function
    scales the population covariance, with ``bias=True``, of the ``{-1,+1}``
    representation by ``pi`` and by ``scale / max_abs_cov``.  With
    ``scale=None`` it uses the default heuristic
    ``sqrt(2*n_features/sum(cov**2))*max_abs_cov/pi``.  Rows of other weights
    are deterministically initialized to zero, and so are two-body rows when
    every visible pair covariance is exactly zero, because the prescription has
    no observed covariance to scale by.
    """
    values = np.asarray(data)
    rows = np.asarray(generators)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty rank-two array")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("data must contain only finite binary values 0 and 1")
    if rows.ndim != 2 or rows.shape[1] < values.shape[1]:
        raise ValueError("generators must cover all visible data qubits")
    if n_visible is None:
        n_visible = values.shape[1]
    if not 1 <= n_visible <= rows.shape[1]:
        raise ValueError("n_visible must be between 1 and the generator width")
    if values.shape[1] != n_visible:
        raise ValueError("data width must equal n_visible")
    if not np.isfinite(rows).all() or not np.isin(rows, (0, 1)).all():
        raise ValueError("generators must contain only finite binary values 0 and 1")
    if scale is not None and (not np.isfinite(scale) or scale < 0):
        raise ValueError("scale must be finite and nonnegative")

    params = _data_parameters(values, rows[:, :n_visible], scale)
    if rows.shape[1] > n_visible:
        supports = ancilla_support(rows, n_visible)
        rng = np.random.default_rng(seed)
        params = params.copy()
        params[supports] = rng.normal(0.0, 0.01 / np.sqrt(params.size), supports.sum())
    return params
