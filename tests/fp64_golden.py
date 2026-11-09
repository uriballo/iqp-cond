"""Reference fp64 outputs of the estimator, the loss and the trainer.

``python tests/fp64_golden.py OUT.npz`` records the outputs of the installed
``iqp_cond``; the stored file was recorded from 0.8.1 (commit d0d30e0), and the
regression test requires the fp64 path to reproduce it bit for bit.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

import iqp_cond as ciqp
from iqp_cond.estimator import estimate_pauli_z
from iqp_cond.loss import mmd

N_QUBITS = 5
N_CONDITIONS = 2


def inputs() -> dict:
    """Fixed inputs: a complex Y+Z angle encoder, so the log-polar path runs."""
    rng = np.random.default_rng(2026)
    angles = rng.uniform(0.3, 2.0, size=(N_CONDITIONS, 2, N_QUBITS))
    encoder = ciqp.AngleEncoder(angles, axes=("Y" * N_QUBITS, "Z" * N_QUBITS))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(N_QUBITS), encoder)
    params = rng.normal(0.0, 0.5, size=model.ansatz.n_parameters)
    masks = rng.integers(0, 2, size=(12, N_QUBITS)).astype(float)
    masks[masks.sum(axis=1) == 0, 0] = 1.0
    samples = encoder.sample(jax.random.key(7), 40)
    targets = rng.integers(0, 2, size=(N_CONDITIONS, 64, N_QUBITS)).astype(float)
    return {
        "model": model,
        "params": params,
        "masks": masks,
        "samples": samples,
        "targets": targets,
    }


def outputs() -> dict[str, np.ndarray]:
    data = inputs()
    model, params = data["model"], jnp.asarray(data["params"])
    generators = model.ansatz.generators
    estimate = estimate_pauli_z(
        params, generators, model.encoder, data["masks"], samples=data["samples"]
    )

    def loss(current):
        return mmd(
            current,
            generators,
            model.encoder,
            data["targets"],
            jax.random.key(3),
            sigmas=(1.0, 2.0),
            n_ops=16,
            n_samples=64,
        ).loss

    value, gradient = jax.value_and_grad(loss)(params)
    out = {
        "estimate": np.asarray(estimate.values),
        "loss": np.asarray(value),
        "gradient": np.asarray(gradient),
    }
    for refresh in ("step", "cache"):
        result = ciqp.train(
            model,
            data["targets"],
            data["targets"],
            seed=11,
            initial_params=data["params"],
            config=ciqp.TrainingConfig(
                steps=20,
                batch_size=8,
                validation_interval=10,
                n_ops=16,
                mc_samples=32,
                validation_n_ops=16,
                validation_mc_samples=32,
                mask_refresh=refresh,
            ),
        )
        out[f"train_{refresh}_final_params"] = result.final_params
        out[f"train_{refresh}_losses"] = result.train_losses
        out[f"train_{refresh}_validation_losses"] = result.validation_losses
    return out


if __name__ == "__main__":
    np.savez(sys.argv[1], **outputs())
