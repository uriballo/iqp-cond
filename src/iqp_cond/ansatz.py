"""IQP generator banks induced by an interaction graph.

A generator row is a binary mask over the register; the diagonal unitary is
``exp(i sum_j theta_j (-1)^{g_j . z})``.  The generator bank is the model's data:
the graph is optional metadata kept only for the built-in constructors.  The
paper's production bank is the complete two-local one built by
:func:`complete_ansatz`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import networkx as nx
import numpy as np

__all__ = [
    "Ansatz",
    "ansatz_from_generators",
    "complete_ansatz",
    "generators_from_graph",
]


@dataclass(frozen=True)
class Ansatz:
    """A bank of diagonal IQP generator masks.

    ``generators`` is ``(n_parameters, n_qubits)`` and binary; ``graph`` is
    optional provenance for banks built from an interaction graph.
    """

    generators: jnp.ndarray
    name: str = "custom"
    graph: nx.Graph | None = None

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return int(self.generators.shape[1])

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


def ansatz_from_generators(
    generators: jnp.ndarray | np.ndarray,
    name: str = "custom",
) -> Ansatz:
    """Build an ansatz directly from a binary generator bank."""
    rows = np.asarray(generators)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("generators must have shape (n_parameters, n_qubits)")
    if not np.isfinite(rows).all() or not np.isin(rows, (0, 1)).all():
        raise ValueError("generators must contain only zeros and ones")
    return Ansatz(generators=jnp.asarray(rows, dtype=jnp.float64), name=name)


def complete_ansatz(n_qubits: int) -> Ansatz:
    """Return the complete two-local ansatz on ``n_qubits`` qubits.

    The bank contains every one-body and distinct two-body diagonal generator,
    so it has ``n + n(n-1)/2`` parameters.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    graph = nx.complete_graph(n_qubits)
    return Ansatz(
        generators=generators_from_graph(graph),
        name="complete",
        graph=graph,
    )
