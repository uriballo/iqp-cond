"""Conditional IQP generation with classically tractable encoders.

A fixed diagonal IQP unitary receives a condition-dependent input state, so
conditioning never touches the learned circuit.  Training uses sampled
Pauli-character estimates of a multiscale Hamming-RBF MMD, which is computable
classically whenever the encoder supports basis sampling and amplitude-ratio
queries; that requirement is the :mod:`iqp_cond.encoders` interface.

Note: importing this package enables JAX float64
(``jax.config.update("jax_enable_x64", True)``).  The estimator, the loss, and
the trainer assume float64 throughout, so this must be set before JAX arrays are
created.  Disable it only if you know the precision change is acceptable.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .ancilla import pad_visible_masks
from .ansatz import Ansatz, complete_ansatz, generators_from_graph
from .encoders import (
    ENCODERS,
    PRODUCT,
    SPIN_FLIP,
    Encoder,
    ExactEncoder,
    ProductEncoder,
    SpinFlipEncoder,
    available_encoders,
    encoder,
    product_encoder,
    register_encoder,
    spin_flip_encoder,
)
from .estimator import (
    ExpectationEstimate,
    estimate_pauli_z,
    estimate_pauli_z_from_samples,
    estimate_pauli_z_unchecked,
)
from .initialization import (
    initialize_parameters_from_data,
    initialize_parameters_with_ancillas,
)
from .loss import (
    ConditionalMMDResult,
    OperatorBank,
    bank_from_ragged,
    banked_mmd,
    hamming_rbf,
    mmd_operator_probability,
    operator_bank,
    sample_nonidentity_masks,
    sample_nonidentity_masks_jax,
    sampled_mmd,
    target_moments,
)
from .trainer import (
    CACHE,
    STEP,
    TrainingConfig,
    TrainingResult,
    initialize_parameters,
    train,
)

__all__ = [
    "CACHE",
    "ENCODERS",
    "PRODUCT",
    "SPIN_FLIP",
    "STEP",
    "Ansatz",
    "ConditionalMMDResult",
    "Encoder",
    "ExactEncoder",
    "ExpectationEstimate",
    "OperatorBank",
    "ProductEncoder",
    "SpinFlipEncoder",
    "TrainingConfig",
    "TrainingResult",
    "available_encoders",
    "bank_from_ragged",
    "banked_mmd",
    "complete_ansatz",
    "encoder",
    "estimate_pauli_z",
    "estimate_pauli_z_from_samples",
    "estimate_pauli_z_unchecked",
    "generators_from_graph",
    "hamming_rbf",
    "initialize_parameters",
    "initialize_parameters_from_data",
    "initialize_parameters_with_ancillas",
    "mmd_operator_probability",
    "operator_bank",
    "pad_visible_masks",
    "product_encoder",
    "register_encoder",
    "sample_nonidentity_masks",
    "sample_nonidentity_masks_jax",
    "sampled_mmd",
    "spin_flip_encoder",
    "target_moments",
    "train",
]
