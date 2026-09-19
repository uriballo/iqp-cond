"""Train an unconditional model with a fixed, classically tractable encoder.

The current API represents an unconditional model with one condition.  This
example uses a fixed reference whose samples are uniform over even-parity
bitstrings.  Its amplitude ratio is one when a mask preserves parity and zero
otherwise.  The circuit description is a GHZ-style
``RY(pi/2)`` followed by a CNOT ladder, so the same fixed encoder can also be
used by the simulation bridge.

Even-parity reference support makes the final model distribution invariant
under complementing every output bit, for every parameter vector.  We train
against a target with that symmetry and report exact model probabilities for
the final checks.

    uv run python examples/unconditional_fixed_encoder.py
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import iqp_cond as ciqp


@dataclass(frozen=True)
class FixedParityEncoder:
    """One-condition parity reference with no trainable encoder state."""

    n_qubits: int

    @property
    def n_conditions(self) -> int:
        return 1

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        bits = jax.random.bernoulli(
            key, 0.5, shape=(n_samples, self.n_qubits - 1)
        ).astype(jnp.float64)
        parity = jnp.mod(jnp.sum(bits, axis=-1), 2.0)
        return jnp.concatenate((bits, parity[:, None]), axis=-1)[None, ...]

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        even_masks = jnp.mod(jnp.sum(masks, axis=-1), 2.0) == 0
        return jnp.broadcast_to(
            even_masks[None, None, :],
            (1, samples.shape[1], masks.shape[0]),
        ).astype(jnp.float64)

    def preparation_ops(self, condition: int) -> tuple[ciqp.Operation, ...]:
        if condition != 0:
            raise IndexError("the unconditional encoder has only condition 0")
        return (
            ciqp.Operation("RY", (0,), (np.pi / 2.0,)),
            *(
                ciqp.Operation("CNOT", (wire - 1, wire))
                for wire in range(1, self.n_qubits)
            ),
        )

    def select(self, condition: int) -> FixedParityEncoder:
        if condition != 0:
            raise IndexError("the unconditional encoder has only condition 0")
        return self


def complement_symmetric_target(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Draw an equal mixture of Bernoulli(p) and Bernoulli(1-p)."""
    component = rng.integers(0, 2, size=n_samples)
    p = np.where(component[:, None] == 0, probabilities, 1.0 - probabilities)
    return (rng.random((n_samples, probabilities.size)) < p).astype(float)


def main() -> None:
    n_qubits = 8
    states = np.arange(2**n_qubits)
    bits = ((states[:, None] >> np.arange(n_qubits)) & 1).astype(float)
    component = np.full(n_qubits, 0.12)
    target_probabilities = 0.5 * (
        np.prod(np.where(bits, component, 1.0 - component), axis=1)
        + np.prod(np.where(bits, 1.0 - component, component), axis=1)
    )

    rng = np.random.default_rng(12)
    target = complement_symmetric_target(rng, component, 2000)
    validation = complement_symmetric_target(rng, component, 1000)
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(n_qubits), FixedParityEncoder(n_qubits)
    )
    config = ciqp.TrainingConfig(
        steps=150,
        batch_size=128,
        sigmas=(0.6, 1.2),
        n_ops=128,
        mc_samples=512,
        validation_n_ops=256,
        validation_mc_samples=1000,
    )
    result = ciqp.train(
        model,
        target[None, ...],
        validation[None, ...],
        seed=9,
        config=config,
    )
    probabilities = np.asarray(model.probabilities(result.best_params))[0]
    target_tvd = 0.5 * np.abs(probabilities - target_probabilities).sum()
    complements = states ^ (2**n_qubits - 1)
    complement_error = np.max(np.abs(probabilities - probabilities[complements]))
    print(f"initial validation MMD {result.validation_losses[0]:.6f}")
    print(f"best validation MMD {result.best_validation_loss:.6f}")
    print(f"exact target TVD {target_tvd:.6f}")
    print(f"maximum complement-symmetry error {complement_error:.3e}")


if __name__ == "__main__":
    main()
