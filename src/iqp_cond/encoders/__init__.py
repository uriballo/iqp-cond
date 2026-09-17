"""Encoders: the classically tractable interface behind conditional IQP.

A conditional IQP model fixes one diagonal unitary and varies only the input
state, so the whole modelling choice lives here.  An encoder family is trainable
exactly when it can sample from its own Born distribution and evaluate amplitude
ratios in polynomial time; see :mod:`iqp_cond.encoders.protocol`.

Built-in families are the plain product state and the spin-flip-symmetric
two-branch state used in the manuscript.  Radial families share the kernels in
:mod:`iqp_cond.encoders.kernels`, so a new permutation-symmetric family only
has to supply its shell table, and any other family just implements the
:class:`Encoder` protocol.
"""

from __future__ import annotations

from .kernels import radial_shell_ratios, radial_shell_samples, radial_shell_state
from .product import PRODUCT, ProductEncoder, local_amplitudes, product_encoder
from .protocol import (
    ENCODERS,
    Encoder,
    ExactEncoder,
    as_condition_batch,
    available_encoders,
    encoder,
    register_encoder,
    select_encoder,
)
from .spin_flip import SPIN_FLIP, SpinFlipEncoder, spin_flip_encoder

__all__ = [
    "ENCODERS",
    "PRODUCT",
    "SPIN_FLIP",
    "Encoder",
    "ExactEncoder",
    "ProductEncoder",
    "SpinFlipEncoder",
    "as_condition_batch",
    "available_encoders",
    "encoder",
    "local_amplitudes",
    "product_encoder",
    "radial_shell_ratios",
    "radial_shell_samples",
    "radial_shell_state",
    "register_encoder",
    "select_encoder",
    "spin_flip_encoder",
]

register_encoder(PRODUCT, product_encoder)
register_encoder(SPIN_FLIP, spin_flip_encoder)
