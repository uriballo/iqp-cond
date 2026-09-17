"""Reference states: the classically tractable interface behind conditional IQP.

A conditional IQP model fixes one diagonal unitary and varies only the input
state, so the whole modelling choice lives here.  A family is trainable exactly
when it can sample from its own Born distribution and evaluate amplitude ratios
in polynomial time; see :mod:`iqp_cond.references.protocol`.

Built-in families are the plain product state and the spin-flip-symmetric
two-branch state used in the manuscript, plus reupload variants of the product
state.  Radial families share the kernels in :mod:`iqp_cond.references.kernels`,
so a new permutation-symmetric family only has to supply its shell table.
"""

from __future__ import annotations

from .kernels import radial_shell_ratios, radial_shell_samples, radial_shell_state
from .product import (
    CALIBRATED_REUPLOAD,
    PRODUCT,
    REUPLOAD_L2,
    REUPLOAD_L3,
    SYMMETRIC_REUPLOAD,
    ProductStates,
    product_states,
)
from .protocol import (
    FAMILIES,
    ReferenceStates,
    as_condition_batch,
    available_families,
    reference,
    register_reference,
)
from .spin_flip import SPIN_FLIP, SpinFlipStates, spin_flip_states

__all__ = [
    "CALIBRATED_REUPLOAD",
    "FAMILIES",
    "PRODUCT",
    "REUPLOAD_L2",
    "REUPLOAD_L3",
    "SPIN_FLIP",
    "SYMMETRIC_REUPLOAD",
    "ProductStates",
    "ReferenceStates",
    "SpinFlipStates",
    "as_condition_batch",
    "available_families",
    "product_states",
    "radial_shell_ratios",
    "radial_shell_samples",
    "radial_shell_state",
    "reference",
    "register_reference",
    "spin_flip_states",
]

register_reference(PRODUCT, product_states)
register_reference(SYMMETRIC_REUPLOAD, product_states)
register_reference(CALIBRATED_REUPLOAD, product_states)
register_reference(REUPLOAD_L2, product_states)
register_reference(REUPLOAD_L3, product_states)
register_reference(SPIN_FLIP, spin_flip_states)
