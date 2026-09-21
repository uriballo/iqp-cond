"""Instantiate a custom encoder and train through the public interface.

The encoder here is the simplest useful one.  Its post-Hadamard state is
``|+>^n``, so sampling is uniform bits and every amplitude ratio is one.  It
needs the three interface operations: sampling, amplitude ratios, and the
circuit representation.  It also needs ``select`` to expose one condition at
a time.  Registering the dataclass as a JAX pytree is optional, but lets the
trainer reuse one compiled optimizer step when every selected encoder has the
same pytree structure and array shapes.  The array state is dynamic and the
``name`` is static.  Encoders that are not pytrees still work
through the trainer's condition-specific closure fallback.  The estimator,
loss, trainer, and simulation bridge use either form unchanged.

    uv run python examples/custom_encoder.py
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import iqp_cond as ciqp


@dataclass(frozen=True)
class HadamardEncoder:
    """Every condition is ``|+>^n``; bits are uniform and ratios are one."""

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

    def preparation_ops(self, condition: int) -> tuple[ciqp.Operation, ...]:
        # |+>^n is the image of |0>^n under the first Hadamard layer.
        return ()

    def select(self, condition: int) -> HadamardEncoder:
        return HadamardEncoder(self.angles[condition : condition + 1], name=self.name)


# ``angles`` is dynamic JAX state; ``name`` is static metadata.  This is an
# optimization for the trainer, not a requirement of the public Encoder
# protocol. Opaque custom encoders keep using the closure fallback.
jax.tree_util.register_dataclass(
    HadamardEncoder,
    data_fields=["angles"],
    meta_fields=["name"],
)


def main() -> None:
    n_qubits, n_conditions = 6, 3
    encoder = HadamardEncoder(jnp.zeros((n_conditions, n_qubits)))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(n_qubits), encoder)

    # Training must map the uniform reference to a biased product law whose
    # bias depends on the condition.
    rng = np.random.default_rng(0)
    probabilities = (0.2 + 0.3 * np.arange(n_conditions))[:, None, None]
    targets = (rng.uniform(size=(n_conditions, 1000, n_qubits)) < probabilities).astype(
        float,
    )
    validation = targets[:, :200]

    result = ciqp.train(
        model,
        targets,
        validation,
        seed=0,
        config=ciqp.TrainingConfig(
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
