"""End-to-end conditional training and sampling on a synthetic Gaussian family.

A translated discrete Gaussian over Hamming weight, a broadcast product-state
encoder, and a short training run; the trained model is then sampled exactly
(the register is small enough) and compared with the targets.  Sample counts
are small so the script runs in seconds.

    uv run python examples/train_gaussian.py
"""

from __future__ import annotations

import numpy as np

import iqp_cond as ciqp

N_QUBITS = 12
TRAINING_CONDITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_TRAIN, N_VALIDATION = 500, 200


def gaussian_samples(
    rng: np.random.Generator,
    condition: float,
    n_samples: int,
) -> np.ndarray:
    """Bitstrings whose Hamming weight follows the Gaussian law."""
    mean = N_QUBITS * (0.15 + 0.70 * condition)
    width = 0.5 * np.sqrt(N_QUBITS)
    weights = np.arange(N_QUBITS + 1)
    shell_probabilities = np.exp(-0.5 * ((weights - mean) / width) ** 2)
    shell_probabilities /= shell_probabilities.sum()
    drawn_weights = rng.choice(N_QUBITS + 1, size=n_samples, p=shell_probabilities)
    samples = np.zeros((n_samples, N_QUBITS))
    for row, weight in enumerate(drawn_weights):
        if weight:
            samples[row, rng.choice(N_QUBITS, size=int(weight), replace=False)] = 1.0
    return samples


def hamming_profile(bits: np.ndarray) -> np.ndarray:
    """Probability of each Hamming weight under a bitstring sample."""
    weights = np.asarray(bits).sum(axis=-1).astype(int)
    counts = np.bincount(weights, minlength=N_QUBITS + 1)
    return counts / weights.size


def main() -> None:
    rng = np.random.default_rng(20260917)
    conditions = np.asarray(TRAINING_CONDITIONS)
    training = np.stack([gaussian_samples(rng, c, N_TRAIN) for c in conditions])
    validation = np.stack([gaussian_samples(rng, c, N_VALIDATION) for c in conditions])

    # The encoder owns "condition -> angle": one angle per condition,
    # alpha(c) = pi (0.35 - 0.70 c), broadcast to every qubit.
    angles = np.repeat((np.pi * (0.35 - 0.70 * conditions))[:, None], N_QUBITS, axis=1)
    model = ciqp.ConditionalModel(
        ciqp.complete_ansatz(N_QUBITS),
        ciqp.angle_encoder(angles[:, None, :], axes=("Y" * N_QUBITS,)),
    )

    result = ciqp.train(
        model,
        training,
        validation,
        seed=0,
        config=ciqp.TrainingConfig(
            steps=500,
            batch_size=100,
            n_ops=128,
            mc_samples=2048,
        ),
    )
    print(f"best step {result.best_step}")
    print(f"validation MMD {result.best_validation_loss:.6f}")

    # The simulation bridge compiles the trained model into a PennyLane
    # circuit: exact probabilities when the device supports analytic
    # execution, and shots otherwise (including hardware backends).
    draws = ciqp.simulation.sample_model(
        model,
        result.best_params,
        2000,
        seed=1,
    )
    probabilities = ciqp.simulation.probabilities(model, result.best_params)
    state_weights = np.bitwise_count(np.arange(2**N_QUBITS)).astype(int)
    profile_tvds = [
        0.5
        * float(
            np.abs(
                np.bincount(
                    state_weights,
                    weights=probabilities[index],
                    minlength=N_QUBITS + 1,
                )
                - hamming_profile(training[index]),
            ).sum(),
        )
        for index in range(conditions.size)
    ]
    print(f"exact samples {draws.shape}, mean profile TVD {np.mean(profile_tvds):.3f}")
    print("raise steps, n_ops, and mc_samples to tighten the fit")


if __name__ == "__main__":
    main()
