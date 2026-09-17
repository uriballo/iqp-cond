"""Data-based parameter initialization in the IQPopt convention.

One-body phases come from the observed marginals, two-body phases from the
observed covariance; both are classical statistics of the training sample, so
the model starts from the target's single- and two-body structure instead of
noise.  Unsupported rows start at zero.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "initialize_parameters_from_data",
    "initialize_parameters_with_ancillas",
]


def initialize_parameters_from_data(
    data: np.ndarray,
    generators: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Initialize one- and two-body IQP parameters from binary observations.

    One-body rows use ``arcsin(sqrt(mean))``.  For two-body rows, the
    covariance of the ``{-1,+1}`` representation is scaled by ``pi`` and by
    ``scale / max_abs_cov``.  With ``scale=None`` the IQPopt heuristic
    ``sqrt(2*n_features/sum(cov**2))*max_abs_cov/pi`` is used.  Rows of other
    weights are deterministically initialized to zero.
    """
    values = np.asarray(data)
    rows = np.asarray(generators)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty rank-two array")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("data must contain only finite binary values 0 and 1")
    if rows.ndim != 2 or rows.shape[1] != values.shape[1]:
        raise ValueError("generators must have shape (n_generators, n_features)")
    if not np.isfinite(rows).all() or not np.isin(rows, (0, 1)).all():
        raise ValueError("generators must contain only finite binary values 0 and 1")
    if scale is not None and (not np.isfinite(scale) or scale < 0):
        raise ValueError("scale must be finite and nonnegative")

    n_features = values.shape[1]
    signed = 2.0 * values - 1.0
    covariance = np.cov(signed, rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance)
    off_diagonal = covariance.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    result = np.zeros(rows.shape[0], dtype=np.float64)
    means = np.mean(values, axis=0)
    has_two_body = bool(np.any(np.sum(rows, axis=1) == 2))
    if has_two_body:
        max_abs_cov = float(np.max(np.abs(off_diagonal)))
        if max_abs_cov == 0.0:
            raise ValueError(
                "covariance initialization requires nonzero off-diagonal covariance",
            )
        cov_sq_sum = float(np.sum(off_diagonal**2))
        if scale is None:
            scale = np.sqrt(2.0 * n_features / cov_sq_sum) * max_abs_cov / np.pi
    for index, row in enumerate(rows):
        weight = int(np.sum(row))
        if weight == 1:
            result[index] = np.arcsin(np.sqrt(means[np.flatnonzero(row)[0]]))
        elif weight == 2:
            left, right = np.flatnonzero(row)
            assert scale is not None
            result[index] = covariance[left, right] * scale * np.pi / max_abs_cov
    return result


def initialize_parameters_with_ancillas(
    data: np.ndarray,
    generators: np.ndarray,
    ancilla_support: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Data initialization on the visible qubits plus noise on ancilla supports.

    ``data`` carries only visible-qubit observations.  Visible parameters are
    initialized from those statistics; parameters whose generator touches an
    ancilla are drawn from a seeded Gaussian of width ``0.01/sqrt(P)`` on the
    supplied support, so ancilla blocks start near zero without being exactly
    symmetric.
    """
    rows = np.asarray(generators)
    if rows.ndim != 2 or rows.shape[1] < data.shape[1]:
        raise ValueError("generators must cover all visible data qubits")
    params = initialize_parameters_from_data(data, rows[:, : data.shape[1]])
    supports = np.asarray(ancilla_support, dtype=bool)
    if supports.shape != params.shape:
        raise ValueError("ancilla_support must match parameter vector")
    rng = np.random.default_rng(seed)
    params = params.copy()
    params[supports] = rng.normal(0.0, 0.01 / np.sqrt(params.size), supports.sum())
    return params
