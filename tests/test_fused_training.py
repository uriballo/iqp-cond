"""Fused updates preserve reference trajectories from the unfused trainer.

The fixture was recorded from library f98225b on Strix Halo, before changing
its JIT boundary. It covers scheduling, optimizer state, both mask protocols,
complex amplitudes, ancillas, a partial sample chunk, and a custom encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.training import CACHE, STEP

CASES = (
    "real_step",
    "real_cache",
    "complex_ancilla",
    "mixed_precision_chunked",
    "custom",
)
REFERENCE = Path(__file__).parent / "fixtures" / "unfused_training.npz"


@dataclass(frozen=True)
class UniformEncoder:
    """A non-pytree encoder exposing only the general training operations."""

    n_conditions: int
    n_qubits: int

    def sample(self, key, n_samples):
        return jax.random.bernoulli(
            key, p=0.5, shape=(self.n_conditions, n_samples, self.n_qubits)
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples, masks):
        return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    def preparation_ops(self, condition):
        return ()

    def select(self, condition):
        return UniformEncoder(1, self.n_qubits)


def training_case(name):
    """Inputs shared by the baseline recorder and the regression check."""
    rng = np.random.default_rng(613)
    visible, conditions = 3, 2
    n = 5 if name == "complex_ancilla" else visible
    if name == "custom":
        encoder = UniformEncoder(conditions, n)
    elif name == "complex_ancilla":
        angles = rng.uniform(0.3, 1.2, (conditions, 2, n))
        encoder = ciqp.AngleEncoder(angles, axes=("XYXYX", "ZYZYZ"))
    else:
        angles = rng.uniform(0.3, 1.2, (conditions, 1, n))
        encoder = ciqp.AngleEncoder(angles, axes=("Y" * n,))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(n), encoder, n_visible=visible)
    train = rng.integers(0, 2, (conditions, 24, visible)).astype(float)
    valid = rng.integers(0, 2, (conditions, 12, visible)).astype(float)
    initial = rng.normal(0, 0.2, model.ansatz.n_parameters)
    config = ciqp.TrainingConfig(
        steps=6,
        batch_size=8,
        learning_rate=0.015,
        final_learning_rate=0.002,
        validation_interval=2,
        sigmas=(0.8, 1.3),
        n_ops=8,
        mc_samples=17,
        validation_n_ops=8,
        validation_mc_samples=19,
        mask_refresh=CACHE if name == "real_cache" else STEP,
        gemm_dtype=jnp.float32 if name == "mixed_precision_chunked" else None,
        sample_chunk=8 if name == "mixed_precision_chunked" else None,
    )
    return model, train, valid, config, initial


@pytest.mark.parametrize("case", CASES)
def test_fused_training_matches_unfused_reference(case):
    model, train, valid, config, initial = training_case(case)
    result = ciqp.train(
        model, train, valid, seed=31, config=config, initial_params=initial
    )
    with np.load(REFERENCE) as reference:
        reference_case = "mixed_chunk" if case == "mixed_precision_chunked" else case
        for field in (
            "train_losses",
            "validation_losses",
            "best_params",
            "final_params",
            "best_validation_loss",
        ):
            np.testing.assert_allclose(
                getattr(result, field),
                reference[f"{reference_case}_{field}"],
                rtol=1e-7 if case == "mixed_precision_chunked" else 1e-10,
                atol=1e-10 if case == "mixed_precision_chunked" else 1e-12,
                err_msg=f"{case}: {field}",
            )
        for field in ("train_steps", "validation_steps", "best_step"):
            np.testing.assert_array_equal(
                getattr(result, field), reference[f"{reference_case}_{field}"]
            )
