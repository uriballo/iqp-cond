"""End-to-end conditional training on a synthetic Gaussian family.

Reproduces the manuscript's simplest benchmark shape without any external data:
a translated discrete Gaussian over Hamming weight, a broadcast product-state
encoder, and a short training run.  Sample counts are small so the script runs
in seconds; the shipped `TrainingConfig` defaults are the production settings.

    PYTHONPATH=src python examples/train_gaussian.py
"""

from __future__ import annotations

import numpy as np

import iqp_cond as iqc

N_QUBITS = 12
TRAINING_CONDITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_TRAIN, N_VALIDATION = 500, 200


def gaussian_samples(
    rng: np.random.Generator,
    condition: float,
    n_samples: int,
) -> np.ndarray:
    """Bitstrings whose Hamming weight follows the campaign's Gaussian law."""
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


def main() -> None:
    rng = np.random.default_rng(20260917)
    conditions = np.asarray(TRAINING_CONDITIONS)
    training = np.stack([gaussian_samples(rng, c, N_TRAIN) for c in conditions])
    validation = np.stack([gaussian_samples(rng, c, N_VALIDATION) for c in conditions])

    # The encoder owns "condition -> angle"; the campaign broadcasts one angle
    # to every qubit, alpha(c) = pi (0.35 - 0.70 c).
    angles = np.repeat((np.pi * (0.35 - 0.70 * conditions))[:, None], N_QUBITS, axis=1)
    encoders = iqc.encoder(iqc.PRODUCT, angles)
    ansatz = iqc.complete_ansatz(N_QUBITS)

    result = iqc.train(
        ansatz,
        encoders,
        training,
        validation,
        seed=0,
        config=iqc.TrainingConfig(
            steps=200,
            batch_size=100,
            n_ops=64,
            n_iqp_samples=1024,
        ),
    )
    print(f"best step {result.best_step}")
    print(f"validation MMD {result.best_validation_loss:.6f}")
    print(f"parameters {result.best_params.shape}")


if __name__ == "__main__":
    main()
