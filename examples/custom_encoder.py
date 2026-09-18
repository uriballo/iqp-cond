"""Register a custom encoder and train through the public interface.

The encoder here is the simplest useful one: every condition is ``|+>^n``, so
sampling is uniform bits and every amplitude ratio is one.  It needs the two
interface operations plus ``select`` (it is not a JAX pytree), and the
estimator, loss, and trainer consume it without any change.

    uv run python examples/custom_encoder.py
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import iqp_cond as iqc


@dataclass(frozen=True)
class HadamardEncoder:
    """Every condition is ``|+>^n``: uniform bits, unit amplitude ratios."""

    angles: jnp.ndarray
    name: str = "hadamard"

    @property
    def n_conditions(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.angles.shape[1])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        return jax.random.bernoulli(
            key,
            p=0.5,
            shape=(self.n_conditions, n_samples, self.n_qubits),
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    def select(self, condition: int) -> HadamardEncoder:
        return HadamardEncoder(self.angles[condition : condition + 1])


def main() -> None:
    iqc.register_encoder("hadamard", lambda angles, **_: HadamardEncoder(angles))

    n_qubits, n_conditions = 6, 3
    encoders = iqc.encoder("hadamard", np.zeros((n_conditions, n_qubits)))
    model = iqc.ConditionalModel(iqc.complete_ansatz(n_qubits), encoders)

    # The trained circuit has to turn the uniform reference into a biased
    # product law whose bias moves with the condition.
    rng = np.random.default_rng(0)
    probabilities = (0.2 + 0.3 * np.arange(n_conditions))[:, None, None]
    targets = (rng.uniform(size=(n_conditions, 1000, n_qubits)) < probabilities).astype(
        float,
    )
    validation = targets[:, :200]

    result = iqc.train(
        model,
        targets,
        validation,
        seed=0,
        config=iqc.TrainingConfig(
            steps=100,
            batch_size=64,
            n_ops=32,
            mc_samples=1024,
        ),
    )
    print(f"best step {result.best_step}, validation {result.best_validation_loss:.5f}")

    estimate = model.estimate_pauli_z(
        result.best_params,
        np.array([[1.0] * n_qubits]),
        key=jax.random.PRNGKey(0),
        n_samples=8192,
    )
    print(f"<Z^{n_qubits}> per condition: {np.asarray(estimate.values)[:, 0]}")


if __name__ == "__main__":
    main()
