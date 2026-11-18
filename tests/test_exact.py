"""Exact output distributions by the Walsh-Hadamard transform."""

from __future__ import annotations

import numpy as np
import pytest
from fp64_golden import inputs

import iqp_cond as ciqp
from iqp_cond.exact import probabilities


@pytest.mark.parametrize("n_visible", [None, 3], ids=["all visible", "two ancillas"])
def test_probabilities_match_pennylane(n_visible):
    data = inputs()
    model = ciqp.ConditionalModel(
        data["model"].ansatz, data["model"].encoder, n_visible=n_visible
    )
    # Both sides are exact fp64 computations over 2^5 amplitudes; they differ
    # only by roundoff, of order 1e-15 per amplitude.
    np.testing.assert_allclose(
        probabilities(model, data["params"]),
        model.probabilities(data["params"]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "encoder",
    [
        # One condition with zero angles: the input |+>^n of a standard IQP
        # circuit.
        ciqp.AngleEncoder(np.zeros((1, 1, 5)), axes=("YYYYY",)),
        ciqp.TwistedParityEncoder([0.4, 1.1], [0.3, -0.7], n_qubits=5),
    ],
    ids=["standard IQP", "twisted parity"],
)
def test_probabilities_match_pennylane_for_other_inputs(encoder):
    ansatz = ciqp.Ansatz.complete(5)
    params = np.random.default_rng(12).normal(0.0, 0.6, ansatz.n_parameters)
    model = ciqp.ConditionalModel(ansatz, encoder)
    np.testing.assert_allclose(
        probabilities(model, params), model.probabilities(params), atol=1e-12
    )


def test_single_precision_probabilities_match_to_float32_roundoff():
    data = inputs()
    model = data["model"]
    # complex64 halves the memory of large registers; each probability is a
    # sum of squares of transformed amplitudes, accurate to about 1e-6.
    np.testing.assert_allclose(
        probabilities(model, data["params"], dtype=np.complex64),
        model.probabilities(data["params"]),
        atol=1e-6,
    )
