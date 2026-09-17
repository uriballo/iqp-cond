"""Conditional IQP generation with classically tractable reference states.

A fixed diagonal IQP unitary receives a condition-dependent input state, so
conditioning never touches the learned circuit.  Training uses sampled
Pauli-character estimates of a multiscale Hamming-RBF MMD, which is computable
classically whenever the input state supports basis sampling and amplitude-ratio
queries; that requirement is the :mod:`iqp_cond.references` interface.

Note: importing this package enables JAX float64
(``jax.config.update("jax_enable_x64", True)``).  The estimator, the loss, and
the trainer assume float64 throughout, so this must be set before JAX arrays are
created.  Disable it only if you know the precision change is acceptable.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .ancilla import (
    ancilla_only_condition_angles,
    marginalize_ancillas,
    pad_visible_masks,
)
from .ansatz import Ansatz, complete_ansatz, generators_from_graph
from .estimator import ExpectationEstimate, estimate_pauli_z
from .initialization import (
    initialize_parameters_from_data,
    initialize_parameters_with_ancillas,
)
from .loss import (
    COMPRESSED,
    PLAIN,
    ConditionalMMDResult,
    OperatorBank,
    bank_from_ragged,
    conditional_mmd,
    hamming_rbf,
    mmd_operator_probability,
    operator_bank,
    sample_mmd_operators,
    sample_nonidentity_masks,
    sampled_mmd,
    target_moments,
)
from .references import (
    FAMILIES,
    PRODUCT,
    SPIN_FLIP,
    ProductStates,
    ReferenceStates,
    SpinFlipStates,
    available_families,
    product_states,
    reference,
    register_reference,
    spin_flip_states,
)
from .trainer import (
    EPOCH,
    FIT,
    TrainingConfig,
    TrainingResult,
    initialize_parameters,
    train,
)

__all__ = [
    "COMPRESSED",
    "EPOCH",
    "FAMILIES",
    "FIT",
    "PLAIN",
    "PRODUCT",
    "SPIN_FLIP",
    "Ansatz",
    "ConditionalMMDResult",
    "ExpectationEstimate",
    "OperatorBank",
    "ProductStates",
    "ReferenceStates",
    "SpinFlipStates",
    "TrainingConfig",
    "TrainingResult",
    "ancilla_only_condition_angles",
    "available_families",
    "bank_from_ragged",
    "complete_ansatz",
    "conditional_mmd",
    "estimate_pauli_z",
    "generators_from_graph",
    "hamming_rbf",
    "initialize_parameters",
    "initialize_parameters_from_data",
    "initialize_parameters_with_ancillas",
    "marginalize_ancillas",
    "mmd_operator_probability",
    "operator_bank",
    "pad_visible_masks",
    "product_states",
    "reference",
    "register_reference",
    "sample_mmd_operators",
    "sample_nonidentity_masks",
    "sampled_mmd",
    "spin_flip_states",
    "target_moments",
    "train",
]
