"""Encoders: the classically tractable interface behind conditional IQP.

A conditional IQP model fixes one diagonal unitary and varies only the input
state, so the whole modelling choice lives here.  An encoder family is trainable
exactly when it can sample from its own Born distribution and evaluate amplitude
ratios in polynomial time; see :mod:`iqp_cond.encoders.protocol`.

Built-in families are the plain product state and its layered angle
generalization.  Any other family just implements the :class:`Encoder` protocol
and registers a factory.
"""

from __future__ import annotations

from .product import (
    ANGLE,
    PRODUCT,
    AngleEncoder,
    ProductEncoder,
    angle_encoder,
    layered_amplitudes,
    local_amplitudes,
    product_encoder,
)
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

__all__ = [
    "ANGLE",
    "ENCODERS",
    "PRODUCT",
    "AngleEncoder",
    "Encoder",
    "ExactEncoder",
    "ProductEncoder",
    "angle_encoder",
    "as_condition_batch",
    "available_encoders",
    "encoder",
    "layered_amplitudes",
    "local_amplitudes",
    "product_encoder",
    "register_encoder",
    "select_encoder",
]

register_encoder(PRODUCT, product_encoder)
register_encoder(ANGLE, angle_encoder)
