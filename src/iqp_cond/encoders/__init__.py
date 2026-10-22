"""Encoders: the interface and the built-in encoder families.

A conditional IQP model applies one shared diagonal unitary ``U_theta`` to a
condition-dependent input state ``|phi(c)>``.  An encoder provides four
capabilities:

``sample``
    Draw bitstrings from ``|psi_c(z)|^2`` in polynomial time, where
    ``psi_c(z) = <z|H^n|phi(c)>`` is the post-Hadamard reference amplitude.
``amplitude_ratios``
    Evaluate ``psi_c(z xor a) / psi_c(z)`` for sampled ``z`` and Fourier masks ``a``.
``preparation``
    Return a callable that queues ``|phi(c)>`` in the active PennyLane queuing
    context, so the model can be compiled to a circuit and sampled on a
    simulator or backend.
``select``
    Restrict the condition batch to one condition for memory-bounded loss
    evaluation.

The first two operations make training classically tractable and keep the
trainer independent of any circuit simulator.  Only :mod:`iqp_cond.simulation`
uses the third.  Encoder families are ordinary Python objects, so a custom
implementation needs no registration and is used directly::

    import iqp_cond as ciqp

    states = MyEncoder(angles)
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(states.n_qubits), states)

The package is organised by shared machinery:

:mod:`~iqp_cond.encoders.base`
    The :class:`Encoder` and :class:`ExactEncoder` protocols and shared helpers.
:mod:`~iqp_cond.encoders.product`
    :class:`AngleEncoder`, a product state of fixed-axis rotation layers.
:mod:`~iqp_cond.encoders.symmetric`
    :class:`ParityEncoder`, :class:`TwistedParityEncoder` and
    :class:`CollectivePhaseEncoder`, entangled states whose amplitudes depend
    only on Hamming weights.

Implementations are batched over conditions: every array has a leading
condition axis.  Exact small-system work needs a further capability,
``post_hadamard_amplitudes``, declared separately by :class:`ExactEncoder`.
A new family goes in its own module and is re-exported here.
"""

from .base import Encoder, ExactEncoder
from .product import AngleEncoder, layered_amplitudes
from .symmetric import CollectivePhaseEncoder, ParityEncoder, TwistedParityEncoder

__all__ = [
    "AngleEncoder",
    "CollectivePhaseEncoder",
    "Encoder",
    "ExactEncoder",
    "ParityEncoder",
    "TwistedParityEncoder",
    "layered_amplitudes",
]
