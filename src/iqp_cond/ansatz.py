"""IQP generator banks induced by an interaction graph.

A generator row is a binary mask over the register; the diagonal unitary is
``exp(i sum_j theta_j (-1)^{g_j . z})``.  The paper's production bank is the
complete two-local one built by :func:`complete_ansatz`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import networkx as nx
import numpy as np

__all__ = ["Ansatz", "complete_ansatz", "generators_from_graph"]


@dataclass(frozen=True)
class Ansatz:
    """One- and two-body IQP generators induced by an interaction graph."""

    name: str
    graph: nx.Graph
    generators: jnp.ndarray

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self.graph.number_of_nodes()

    @property
    def n_parameters(self) -> int:
        """Number of independently parameterized generators."""
        return int(self.generators.shape[0])


def generators_from_graph(graph: nx.Graph) -> jnp.ndarray:
    """Return all one-body rows followed by one two-body row per graph edge."""
    nodes = sorted(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("graph nodes must be contiguous integers starting at zero")
    n_qubits = len(nodes)
    rows = np.eye(n_qubits, dtype=np.float64).tolist()
    for left, right in sorted(tuple(sorted(edge)) for edge in graph.edges()):
        row = np.zeros(n_qubits, dtype=np.float64)
        row[left] = 1.0
        row[right] = 1.0
        rows.append(row.tolist())
    return jnp.asarray(rows, dtype=jnp.float64)


def _validate(name: str, graph: nx.Graph) -> Ansatz:
    if not nx.is_connected(graph):
        raise ValueError(f"{name} graph must be connected")
    return Ansatz(name=name, graph=graph, generators=generators_from_graph(graph))


def complete_ansatz(n_qubits: int) -> Ansatz:
    """Return the complete two-local ansatz on ``n_qubits`` qubits.

    The bank contains every one-body and distinct two-body diagonal generator,
    so it has ``n + n(n-1)/2`` parameters.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return _validate("complete", nx.complete_graph(n_qubits))
