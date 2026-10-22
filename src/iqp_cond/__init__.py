"""Conditional IQP generation with classically tractable encoders.

A conditional IQP model applies one shared diagonal unitary to a
condition-dependent input state, so the condition never changes the learned
circuit.  Training uses sampled Pauli-character estimates of a multiscale
Hamming-RBF MMD, which is computable classically whenever the encoder supports
basis sampling and amplitude-ratio queries; that requirement is the
:mod:`iqp_cond.encoders` interface.

The compact public API covers the full workflow: build a
:class:`ConditionalModel` from an :class:`Ansatz` and an :class:`Encoder`,
estimate moments, train, and sample through model methods.  Numerical building
blocks remain available from their expert submodules.

Importing this package turns on JAX float64
(``jax.config.update("jax_enable_x64", True)``).  The estimator, the loss, and
the trainer assume float64 throughout, so enable it before you create JAX
arrays.  Disable it only if you know the precision change is acceptable.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .encoders import AngleEncoder, Encoder
from .model import Ansatz, ConditionalModel
from .symmetric_encoders import (
    CollectivePhaseEncoder,
    ParityEncoder,
    TwistedParityEncoder,
)
from .training import TrainingConfig, train

__all__ = [
    "AngleEncoder",
    "Ansatz",
    "CollectivePhaseEncoder",
    "ConditionalModel",
    "Encoder",
    "EncoderCheck",
    "ParityEncoder",
    "TrainingConfig",
    "TwistedParityEncoder",
    "check_encoder",
    "train",
]


def __getattr__(name: str) -> object:
    """Resolve the verification helpers on demand.

    They need PennyLane, which importing this package must not pull in, so the
    module is imported on first attribute access.
    """
    if name in {"EncoderCheck", "check_encoder"}:
        from .verification import EncoderCheck, check_encoder

        return {"EncoderCheck": EncoderCheck, "check_encoder": check_encoder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
