"""Conditional IQP generation with classically tractable encoders.

A conditional IQP model applies one shared diagonal unitary to a
condition-dependent input state, so conditioning never touches the learned
circuit.  Training uses sampled Pauli-character estimates of a multiscale
Hamming-RBF MMD, which is computable classically whenever the encoder supports
basis sampling and amplitude-ratio queries; that requirement is the
:mod:`iqp_cond.encoders` interface.

The facade is the user journey: build a :class:`ConditionalModel` from an
:class:`Ansatz` and an :class:`Encoder`, estimate moments or the conditional
MMD, and train.  For registers small enough to hold the full state,
:func:`exact_probabilities` and :func:`sample_exact` evaluate the model
distribution without Monte Carlo.  The numerical building blocks — operator
banks, mask samplers, and the raw conditional estimator — live in
:mod:`iqp_cond.operators` and :mod:`iqp_cond.estimator` for advanced use.

Note: importing this package enables JAX float64
(``jax.config.update("jax_enable_x64", True)``).  The estimator, the loss, and
the trainer assume float64 throughout, so this must be set before JAX arrays are
created.  Disable it only if you know the precision change is acceptable.
"""

import jax

jax.config.update("jax_enable_x64", True)

from . import simulation
from .core import (
    Ansatz,
    ConditionalModel,
    ancilla_support,
    ansatz_from_generators,
    complete_ansatz,
    pad_visible_masks,
)
from .encoders import (
    ANGLE,
    AngleEncoder,
    CircuitEncoder,
    Encoder,
    ExactEncoder,
    Operation,
    angle_encoder,
    available_encoders,
    encoder,
    register_encoder,
)
from .estimator import (
    ExpectationEstimate,
    estimate_pauli_z,
    estimate_pauli_z_unchecked,
)
from .initialization import parameters_from_data, random_parameters
from .loss import ConditionalMMDResult, mmd
from .operators import expected_bodyness_sigmas
from .trainer import CACHE, STEP, TrainingConfig, TrainingResult, train

__all__ = [
    "ANGLE",
    "CACHE",
    "STEP",
    "AngleEncoder",
    "Ansatz",
    "CircuitEncoder",
    "ConditionalMMDResult",
    "ConditionalModel",
    "Encoder",
    "ExactEncoder",
    "ExpectationEstimate",
    "Operation",
    "TrainingConfig",
    "TrainingResult",
    "ancilla_support",
    "angle_encoder",
    "ansatz_from_generators",
    "available_encoders",
    "complete_ansatz",
    "encoder",
    "estimate_pauli_z",
    "estimate_pauli_z_unchecked",
    "expected_bodyness_sigmas",
    "mmd",
    "pad_visible_masks",
    "parameters_from_data",
    "random_parameters",
    "register_encoder",
    "simulation",
    "train",
]
