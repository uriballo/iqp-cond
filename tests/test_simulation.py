"""The PennyLane bridge, with device handling, seeds, shapes, and validation.

``test_pennylane_crosscheck.py`` checks circuit fidelity and the sampled law
against PennyLane; these tests cover the bridge API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import pytest

import iqp_cond as ciqp
from iqp_cond.simulation import build_qnode


def _model(
    axes: tuple[str, ...], n_visible: int | None = None
) -> ciqp.ConditionalModel:
    rng = np.random.default_rng(4)
    angles = rng.uniform(0.3, 2.4, size=(1, len(axes), len(axes[0])))
    n_qubits = len(axes[0])
    encoder = ciqp.AngleEncoder(angles, axes=axes)
    if n_visible is None:
        return ciqp.ConditionalModel(ciqp.Ansatz.complete(n_qubits), encoder)
    return ciqp.ConditionalModel(
        ciqp.Ansatz.complete(n_qubits),
        encoder,
        n_visible=n_visible,
    )


def _params(model: ciqp.ConditionalModel, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        0.0,
        0.4,
        size=model.ansatz.n_parameters,
    )


def test_probabilities_are_normalized_over_every_condition() -> None:
    model = _model(("YY",), n_visible=1)
    probabilities = model.probabilities(_params(model))
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_probabilities_accept_a_device_instance() -> None:
    model = _model(("Y",))
    params = _params(model)
    default = model.probabilities(params)
    device = qml.device("default.qubit", wires=1)
    bridged = model.probabilities(params, device=device)
    assert np.allclose(default, bridged)


def test_sample_model_returns_bits_with_the_documented_shape() -> None:
    model = _model(("XY",), n_visible=1)
    params = _params(model, seed=7)
    draws = model.sample(params, 64, seed=8)
    assert draws.shape == (1, 64, 1)
    assert draws.dtype == np.float64
    assert np.isin(draws, (0.0, 1.0)).all()


def test_sample_model_is_reproducible_in_the_seed() -> None:
    model = _model(("Y",))
    params = _params(model)
    first = model.sample(params, 256, seed=11)
    repeat = model.sample(params, 256, seed=11)
    other = model.sample(params, 256, seed=12)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_simulation_rejects_bad_devices_draw_counts_and_operations() -> None:
    model = _model(("Y",))
    params = _params(model)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        model.sample(params, 0, seed=1)
    with pytest.raises(ValueError, match="device must act on 1 wires"):
        model.probabilities(params, device=qml.device("default.qubit", wires=2))

    unsupported = ciqp.ConditionalModel(ciqp.Ansatz.complete(1), _BadEncoder())
    with pytest.raises(TypeError, match="must return a callable"):
        unsupported.probabilities(np.zeros(unsupported.ansatz.n_parameters))


def test_sample_model_rejects_a_seeded_device_instance() -> None:
    model = _model(("Y",))
    params = _params(model)
    device = qml.device("default.qubit", wires=1)
    with pytest.raises(ValueError, match="seed applies"):
        model.sample(params, 8, seed=1, device=device)


def test_build_qnode_samples_the_visible_wires() -> None:
    model = _model(("YY",), n_visible=1)
    params = _params(model)
    qnode = build_qnode(model, params, 0, shots=16)
    samples = np.asarray(qnode())
    assert samples.shape == (16, 1)


@dataclass(frozen=True)
class _BadEncoder:
    name: str = "bad"

    @property
    def n_conditions(self) -> int:
        return 1

    @property
    def n_qubits(self) -> int:
        return 1

    def sample(self, key, n_samples):  # pragma: no cover - interface only
        raise NotImplementedError

    def amplitude_ratios(self, samples, masks):  # pragma: no cover
        raise NotImplementedError

    def preparation(self, condition: int) -> object:
        # Returns operations instead of the required callable.
        return (qml.RY(0.3, wires=0),)


def test_simulation_validates_the_setup() -> None:
    model = _model(("Y",))
    params = _params(model)
    with pytest.raises(ValueError, match="params must have shape"):
        model.probabilities(np.zeros(3))
    bad = params.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.probabilities(bad)
    with pytest.raises(IndexError, match="condition"):
        build_qnode(model, params, 3, shots=4)

    class Minimal:
        n_conditions = 1
        n_qubits = 1

        def sample(self, key, n_samples):  # pragma: no cover - interface only
            raise NotImplementedError

        def amplitude_ratios(self, samples, masks):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(TypeError, match="Encoder protocol"):
        ciqp.ConditionalModel(ciqp.Ansatz.complete(1), Minimal())
