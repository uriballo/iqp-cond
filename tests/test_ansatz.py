"""Generator banks: construction from graphs, generators, and complete graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import iqp_cond as iqc


def test_complete_ansatz_has_one_and_two_body_rows() -> None:
    ansatz = iqc.complete_ansatz(4)
    rows = np.asarray(ansatz.generators)
    assert ansatz.n_qubits == 4
    assert ansatz.n_parameters == 4 + 6
    assert len(set(map(tuple, rows))) == ansatz.n_parameters
    assert ansatz.graph is not None
    assert ansatz.graph.number_of_edges() == 6


def test_generators_from_graph_is_binary_and_ordered() -> None:
    graph = nx.path_graph(3)
    rows = np.asarray(iqc.generators_from_graph(graph))
    assert rows.shape == (3 + 2, 3)
    assert set(np.unique(rows)) == {0.0, 1.0}
    # One-body rows first, then one two-body row per edge.
    assert np.array_equal(rows[0], [1.0, 0.0, 0.0])
    assert np.array_equal(rows[3], [1.0, 1.0, 0.0])
    assert np.array_equal(rows[4], [0.0, 1.0, 1.0])


def test_ansatz_from_generators_accepts_arbitrary_banks() -> None:
    rows = np.array([[1.0, 1.0], [0.0, 1.0]])
    ansatz = iqc.ansatz_from_generators(rows, name="pair")
    assert ansatz.name == "pair"
    assert ansatz.n_qubits == 2
    assert ansatz.n_parameters == 2
    assert ansatz.graph is None
    assert np.array_equal(np.asarray(ansatz.generators), rows)


def test_ansatz_from_generators_rejects_non_binary_rows() -> None:
    with pytest.raises(ValueError, match="only zeros and ones"):
        iqc.ansatz_from_generators(np.array([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="shape"):
        iqc.ansatz_from_generators(np.array([1.0, 0.0]))


def test_complete_ansatz_rejects_empty_registers() -> None:
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        iqc.complete_ansatz(0)
