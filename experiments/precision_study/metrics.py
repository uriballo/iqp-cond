"""Metrics of the precision study: trajectory deviation from fp64 and the
rounding error of a quantity in units of its Monte Carlo noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parameter_deviation(params: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return ``d_p(t)``, the distance per checkpoint modulo ``pi`` per parameter.

    Shifting a parameter by ``pi`` changes only a global phase, so each
    difference is wrapped into ``(-pi/2, pi/2]`` before taking the norm.
    Both arrays are ``(checkpoints, parameters)``.
    """
    difference = np.asarray(params) - np.asarray(reference)
    wrapped = np.pi / 2 - np.mod(np.pi / 2 - difference, np.pi)
    return np.linalg.norm(wrapped, axis=-1)


def iteration_lag(
    loss: np.ndarray, reference: np.ndarray, steps: np.ndarray, *, window: int
) -> np.ndarray:
    """Return ``tau_p(t)`` in steps, NaN where the reference stops falling.

    Both losses are smoothed by a trailing moving average over ``window``
    checkpoints, taken at the training ``steps``.  The lag at ``t`` is the step
    at which the smoothed loss first reaches the smoothed reference value at
    ``t``, minus the step of ``t``: negative for a run ahead of fp64, NaN if it
    never gets there.
    """

    def smooth(values: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(np.asarray(values, float))
        out = cumulative.copy()
        out[window:] = cumulative[window:] - cumulative[:-window]
        counts = np.minimum(np.arange(1, len(out) + 1), window)
        return out / counts

    loss, reference = smooth(loss), smooth(reference)
    steps = np.asarray(steps)
    lag = np.full(len(reference), np.nan)
    for t in range(len(reference) - 1):
        if not reference[t + 1] < reference[t]:
            continue
        reached = np.flatnonzero(loss <= reference[t])
        if reached.size:
            lag[t] = steps[reached[0]] - steps[t]
    return lag


def classify(
    deviation: np.ndarray,
    baseline: np.ndarray,
    distribution_differs: bool,
) -> str:
    """Classify a run as ``"tracks"``, ``"delayed"`` or ``"diverges"``.

    ``baseline`` holds the chaos-baseline distances, ``(runs, checkpoints)``.
    A run tracks fp64 if ``d_p(t)`` never exceeds the baseline's 90th
    percentile at the same checkpoint, and diverges if its endpoint lies
    beyond it and its output distribution differs too; otherwise it is delayed.
    """
    band = np.percentile(baseline, 90, axis=0)
    if np.all(deviation <= band):
        return "tracks"
    if deviation[-1] > band[-1] and distribution_differs:
        return "diverges"
    return "delayed"


@dataclass(frozen=True)
class NoiseRatio:
    """Rounding error in units of the fp64 Monte Carlo noise ``sigma``."""

    bias: float
    spread: float
    rho: float


def noise_ratio(values: np.ndarray, reference: np.ndarray) -> NoiseRatio:
    """Return ``rho_p`` and its bias and spread parts over independent draws.

    ``values`` and ``reference`` hold one quantity in format ``p`` and in fp64
    at the same draws.  ``sigma``, the Monte Carlo noise, is the unbiased
    (``ddof=1``) standard deviation of the fp64 values across draws; the bias,
    spread and RMS describe the observed errors themselves (``ddof=0``), so
    ``rho^2 = bias^2 + spread^2`` exactly.
    """
    error = np.asarray(values) - np.asarray(reference)
    sigma = np.std(reference, ddof=1)
    if not sigma > 0:
        raise ValueError("the fp64 reference shows no noise across draws")
    bias = float(np.mean(error) / sigma)
    spread = float(np.std(error) / sigma)
    rho = float(np.sqrt(np.mean(error**2)) / sigma)
    return NoiseRatio(bias, spread, rho)
