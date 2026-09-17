# iqp-cond

Conditional generation with instantaneous quantum polynomial-time circuits.

A conditional IQP model applies **one shared diagonal unitary** to a
**condition-dependent input state**: the learned circuit never changes with the
condition, only the state that feeds it. Training uses sampled Pauli-character
estimates of a multiscale Hamming-RBF maximum mean discrepancy, which is
classically computable whenever the input state supports two operations:

- draw bitstrings from its own Born distribution, and
- evaluate amplitude ratios `phi(z xor a) / phi(z)` for Fourier masks `a`.

That pair is the whole modelling interface, and it is also what keeps training
classical — the model itself is still hard to sample from in general.

## Install

```bash
pip install .        # from a checkout
pip install -e .     # editable
```

Python 3.11+ with JAX. Dependencies: `jax`, `jaxlib`, `numpy`, `optax`,
`networkx`.

## Quickstart

```python
import numpy as np
import iqp_cond as iqc

n_qubits, n_conditions = 8, 3

# One shared generator bank for every condition.
ansatz = iqc.complete_ansatz(n_qubits)

# One angle per model qubit per condition, plus the condition states it implies.
angles = np.linspace(0.6, 2.0, n_conditions * n_qubits).reshape(n_conditions, n_qubits)
references = iqc.reference(iqc.SPIN_FLIP, angles)

# Bitstring samples from the target family: (conditions, samples, visible qubits).
rng = np.random.default_rng(0)
train_targets = rng.integers(0, 2, size=(n_conditions, 1000, n_qubits)).astype(float)
valid_targets = rng.integers(0, 2, size=(n_conditions, 200, n_qubits)).astype(float)

result = iqc.train(
    ansatz,
    references,
    train_targets,
    valid_targets,
    seed=0,
    # Defaults are the published production settings (2000 updates, 8192 Monte
    # Carlo samples); this is a smaller run for a quick check.
    config=iqc.TrainingConfig(steps=50, batch_size=64, n_ops=16, n_iqp_samples=1024),
)

params = result.best_params          # selected on the validation split
```

## Querying expectations

At inference only the reference state changes with the condition; the learned
diagonal stays fixed.

```python
import jax

masks = np.array([[1.0] * n_qubits, [1.0, 0.0] * (n_qubits // 2)])
estimate = iqc.estimate_pauli_z(
    params,
    ansatz.generators,
    masks,
    iqc.reference(iqc.PRODUCT, angles),
    jax.random.PRNGKey(0),
    n_samples=8192,
)
estimate.values           # (conditions, masks)
estimate.standard_errors  # Monte Carlo uncertainty
```

## Reference states

`iqc.reference(family, angles)` takes angles of shape
`(n_conditions, n_qubits)` and returns an object exposing `sample`,
`amplitude_ratios`, `select`, and `post_hadamard_amplitudes`. `select(i)` returns
the same family restricted to condition `i`; the fixed-operator estimator
evaluates one condition at a time and needs it.

| Name | State | Used for |
|---|---|---|
| `product` | `RY(a_i)` on each qubit | translated Gaussian target |
| `spin_flip` | `\|p>^n + X^n \|p>^n`, a spin-flip-symmetric radial state | Curie–Weiss target |
| `symmetric_reupload`, `calibrated_reupload`, `reupload_l2`, `reupload_l3` | product states with noncommuting single-qubit layers | richer single-qubit maps |

### Adding your own

A family is any object with the three interface methods and the two size
properties:

```python
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import iqp_cond as iqc


@dataclass(frozen=True)
class HadamardFamily:
    """Every condition is |+>^n: uniform bits, unit amplitude ratios."""

    angles: jnp.ndarray

    @property
    def n_conditions(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.angles.shape[1])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        return jax.random.bernoulli(
            key, p=0.5, shape=(self.n_conditions, n_samples, self.n_qubits)
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    def select(self, condition: int) -> "HadamardFamily":
        return HadamardFamily(self.angles[condition : condition + 1])

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        return jnp.full(
            (self.n_conditions, 2**self.n_qubits),
            2.0 ** (-self.n_qubits / 2),
            dtype=jnp.complex128,
        )


iqc.register_reference("hadamard", lambda angles, **_: HadamardFamily(jnp.asarray(angles)))
states = iqc.reference("hadamard", angles)
```

Nothing else in the package changes: the estimator, the loss, and the trainer
consume the interface, not the family. `iqp_cond.references.ReferenceStates` is
a `Protocol`, so no inheritance is required.

## Ancillas

Extra qubits are allowed and are marginalized after measurement. Qubits are
ordered **visible first, ancillas last**:

- `angles` has one column per *model* qubit, including ancillas;
- targets keep the visible width only;
- masks over visible qubits are padded with identity on the ancilla positions
  automatically by `conditional_mmd` (or explicitly with
  `iqc.pad_visible_masks`);
- `iqc.marginalize_ancillas` sums ancilla outcomes out of joint distributions.

Ancilla parameters are not initialized from data. Use
`initialize_parameters_with_ancillas` to initialize visible parameters from
sample statistics and draw small seeded noise on ancilla-supported generators.

## API

| Module | Contents |
|---|---|
| `iqp_cond.ansatz` | `Ansatz`, `generators_from_graph`, `complete_ansatz` |
| `iqp_cond.references` | `ReferenceStates`, `reference`, `register_reference`, `available_families`, built-in families |
| `iqp_cond.estimator` | `estimate_pauli_z`, `ExpectationEstimate` |
| `iqp_cond.loss` | `sampled_mmd`, `operator_bank`, `OperatorBank`, `conditional_mmd`, `sample_nonidentity_masks`, `target_moments`, `hamming_rbf` |
| `iqp_cond.trainer` | `TrainingConfig`, `TrainingResult`, `train`, `initialize_parameters` |
| `iqp_cond.initialization` | `initialize_parameters_from_data`, `initialize_parameters_with_ancillas` |
| `iqp_cond.ancilla` | `pad_visible_masks`, `marginalize_ancillas`, `ancilla_only_condition_angles` |

## The two MMD estimators

Training minimizes a sampled estimate of the squared MMD, and the library offers
two estimators with the same target and different cost. `train` uses the first by
default; `estimator=iqc.PLAIN` selects the second.

**`sampled_mmd` (default)** builds compressed `OperatorBank` objects — `n_ops`
nonidentity draws per condition and bandwidth, with duplicates collapsed to
exact multiplicities and the sum rescaled by the nonidentity mass — and
evaluates one condition at a time. This is the estimator the manuscript's
Methods describe.

By default the training bank is **redrawn at every epoch** (`mask_refresh="epoch"`),
so each update's objective is unbiased for the true kernel MMD rather than
conditioned on one realized bank. Pass `mask_refresh="fit"` to draw once and
retain it, which is the protocol the published campaign used and saves the
redraw cost. Validation always uses a bank drawn once, so checkpoint selection
stays comparable across steps.

**`conditional_mmd` (reference)** draws every mask i.i.d. from the full Bernoulli
law, identity included, keeps duplicates, and evaluates all conditions together.
It is unbiased but spends a large share of its budget on a term that is
identically zero, and its peak allocation grows with the condition count. It is
kept for reproducing older runs and for the numerical equivalence tests.

At 12 visible qubits and 5 conditions, with the default `TrainingConfig`:

| | `sampled_mmd` | `conditional_mmd` |
|---|---|---|
| validation-pass spread (σ = 1 / 2 / 4) | 1.8e-3 / 1.7e-3 / 6.4e-4 | 6.6e-3 / 1.0e-2 / 7.1e-3 |
| added peak memory, one validation pass | ~0.4 GB | ~0.7 GB |
| time per validation pass | 108 ms | 217 ms |

Refresh cadence was compared directly at 8 visible qubits over 8 seeds: the two
cadences are statistically indistinguishable on an independent evaluation bank,
and per-epoch refresh costs wall clock — roughly 0.02 s per rebuilt bank, which
is ~80% overhead at 500 draws per entry and ~4% at paper-scale settings. It is
the default because it keeps the training objective unbiased, not because it
converges faster. Protocol and per-seed numbers: `experiments/mask_refresh/`.

Peak memory is still dominated by the `(samples, masks, qubits)` reduction inside
the per-condition evaluation, so it scales roughly as
`n_iqp_samples × distinct_masks × n_qubits × 16` bytes. Lower
`validation_n_iqp_samples` first; the bank is already compressed.

Not included in this package: exact-distribution evaluation (TVD, Walsh–Hadamard
transforms) and the benchmark target families. Exact evaluation needs the full
`2**n` output state, so it is useful only at small sizes, and the target
families carry experiment-specific constants; both live with the research code
that consumes this package.

## Precision

Importing `iqp_cond` enables JAX float64
(`jax.config.update("jax_enable_x64", True)`). The estimator, the loss, and the
trainer assume it: the Monte Carlo settings were chosen for float64, and
disabling it silently changes the numbers. Set it yourself if you import JAX
before this package.

## Status

Extracted from the research repository for the conditional-IQP manuscript.
Licence and citation metadata are pending author and institutional approval.
