# iqp-cond

Conditional generation with instantaneous quantum polynomial-time circuits.

A conditional IQP model applies **one shared diagonal unitary** to a
**condition-dependent input state**: the learned circuit never changes with the
condition, only the state that feeds it. Training uses sampled Pauli-character
estimates of a multiscale Hamming-RBF maximum mean discrepancy, which is
classically computable whenever the encoder supports two operations:

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

# spin_flip takes one angle per condition, broadcast over the model qubits.
angles = np.broadcast_to(
    np.linspace(0.6, 2.0, n_conditions)[:, None], (n_conditions, n_qubits)
)
encoders = iqc.encoder(iqc.SPIN_FLIP, angles)

# Bitstring samples from the target family: (conditions, samples, visible qubits).
rng = np.random.default_rng(0)
train_targets = rng.integers(0, 2, size=(n_conditions, 1000, n_qubits)).astype(float)
valid_targets = rng.integers(0, 2, size=(n_conditions, 200, n_qubits)).astype(float)

result = iqc.train(
    ansatz,
    encoders,
    train_targets,
    valid_targets,
    seed=0,
    # Defaults are the published production settings (2000 updates, 8192 Monte
    # Carlo samples); this is a smaller run for a quick check.
    config=iqc.TrainingConfig(steps=50, batch_size=64, n_ops=16, n_iqp_samples=1024),
)

params = result.best_params  # selected on the validation split
```

## Querying expectations

At inference only the encoder state changes with the condition; the learned
diagonal stays fixed.  Visible-width masks are padded with identity on the
ancillas automatically.

```python
import jax

masks = np.array([[1.0] * n_qubits, [1.0, 0.0] * (n_qubits // 2)])
estimate = iqc.estimate_pauli_z(
    params,
    ansatz.generators,
    masks,
    iqc.encoder(iqc.PRODUCT, angles),
    jax.random.PRNGKey(0),
    n_samples=8192,
)
estimate.values  # (conditions, masks)
estimate.standard_errors  # Monte Carlo uncertainty, computed on request
```

`estimate_pauli_z` validates its inputs.  Inside `jax.jit`, use
`estimate_pauli_z_unchecked`, which skips the Python-level checks and expects
model-width masks; `estimate_pauli_z_from_samples` evaluates a batch of masks
against pre-drawn reference samples, which is how several mask banks share one
draw. The loss uses both.

## Encoders

`iqc.encoder(family, angles)` takes angles of shape `(n_conditions, n_qubits)`
and returns an object with a `name` and the methods `sample`,
`amplitude_ratios`, and `select`; that is the whole training interface.
`select(i)` returns the same encoder restricted to condition `i`; the
fixed-operator estimator evaluates one condition at a time and needs it.

Both built-ins also implement `post_hadamard_amplitudes` (the optional
`ExactEncoder` capability) for exact small-system work. A custom family does not
need it.

| Name | State | Used for |
|---|---|---|
| `product` | `RY(a_i)` on each qubit | translated Gaussian target |
| `spin_flip` | `\|p>^n + X^n \|p>^n`, a spin-flip-symmetric radial state | Curie–Weiss target |

`spin_flip` rotates every qubit by a single angle, so its matrix must be a
broadcast row: every column equal to the first, or zero (the natural ancilla
column). A matrix with differing columns raises instead of silently reading only
column 0.

### Adding your own

An encoder is any object with the three interface methods and the two size
properties:

```python
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import iqp_cond as iqc


@dataclass(frozen=True)
class HadamardEncoder:
    """Every condition is |+>^n: uniform bits, unit amplitude ratios."""

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
            key, p=0.5, shape=(self.n_conditions, n_samples, self.n_qubits)
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    def select(self, condition: int) -> "HadamardEncoder":
        return HadamardEncoder(self.angles[condition : condition + 1])

    def post_hadamard_amplitudes(self) -> jnp.ndarray:
        return jnp.full(
            (self.n_conditions, 2**self.n_qubits),
            2.0 ** (-self.n_qubits / 2),
            dtype=jnp.complex128,
        )


iqc.register_encoder(
    "hadamard", lambda angles, **_: HadamardEncoder(jnp.asarray(angles))
)
states = iqc.encoder("hadamard", angles)
```

Nothing else in the package changes: the estimator, the loss, and the trainer
consume the interface, not the family. `iqp_cond.encoders.Encoder` is a
`Protocol`, so no inheritance is required, and registering a name that already
exists raises unless `replace=True`.

## Ancillas

Extra qubits are allowed and are marginalized after measurement. Qubits are
ordered **visible first, ancillas last**:

- `angles` has one column per *model* qubit, including ancillas;
- targets keep the visible width only;
- `estimate_pauli_z` and the loss pad visible masks with identity on the ancilla
  positions automatically (or explicitly with `iqc.pad_visible_masks`).

Ancilla parameters are not initialized from data. Use
`initialize_parameters_with_ancillas` to initialize visible parameters from
sample statistics and draw small seeded noise on ancilla-supported generators.

## API

| Module | Contents |
|---|---|
| `iqp_cond.ansatz` | `Ansatz`, `generators_from_graph`, `complete_ansatz` |
| `iqp_cond.encoders` | `Encoder`, `ExactEncoder`, `encoder`, `register_encoder`, `available_encoders`, built-in families |
| `iqp_cond.estimator` | `estimate_pauli_z`, `estimate_pauli_z_unchecked`, `estimate_pauli_z_from_samples`, `ExpectationEstimate` |
| `iqp_cond.operators` | `operator_bank`, `OperatorBank`, `bank_from_ragged`, `sample_nonidentity_masks`, `sample_nonidentity_masks_jax`, `mmd_operator_probability` |
| `iqp_cond.loss` | `sampled_mmd`, `banked_mmd`, `target_moments`, `hamming_rbf`, `ConditionalMMDResult` |
| `iqp_cond.trainer` | `TrainingConfig`, `TrainingResult`, `train`, `initialize_parameters` |
| `iqp_cond.initialization` | `initialize_parameters_from_data`, `initialize_parameters_with_ancillas` |
| `iqp_cond.ancilla` | `pad_visible_masks`, `ancilla_support` |

## Masks and the MMD estimator

Training minimizes the paper's stochastic operator estimator of the squared
multiscale Hamming–RBF MMD. Masks are **redrawn at every optimization step**
(`mask_refresh="step"`, the default), so each update's expectation equals the
true kernel MMD rather than a single realized mask draw; `mask_refresh="cache"`
draws one bank and retains it, which is the published campaign's protocol bit
for bit. Validation always uses one fixed bank, so checkpoint selection stays
comparable across steps.

Each training bank is compressed: masks are drawn from the nonidentity law,
duplicates are collapsed to exact multiplicities, and the sum is rescaled by the
nonidentity mass. Skipping the identity mask is exact — it contributes zero to
the kernel MMD, so spending draws on it only adds variance.

The library exposes both the compressed fixed-bank evaluator and a pure-JAX
sampler:

**`banked_mmd`** consumes a prebuilt `OperatorBank`. The trainer builds one per
step on the host and hands it to the compiled step as a pytree argument, which
keeps compilation to a single program. This is the fast path, because the bank
can collapse duplicate masks. With `shared_samples=True` each condition draws
one reference sample set and evaluates every bandwidth's masks on it, which is
the paper's estimator form and avoids resampling once per bandwidth; the
trainer's `"step"` path uses it, `"cache"` keeps the campaign's per-entry draw
layout.

**`sampled_mmd`** draws nonidentity masks inside `jax.jit` from the call's key
(`sample_nonidentity_masks_jax` is the exact, loop-free sampler behind it). It
needs no host loop, but it cannot collapse duplicates cheaply, so at equal draw
budgets it evaluates roughly 1.8× more mask columns and costs about 1.5–2× the
step time. Use it when the whole training loop must stay on device.

At 12 visible qubits and 5 conditions with the default `TrainingConfig`,
measured on an idle 32-core x86 host (JAX CPU, float64), one 2000-step fit takes
**59.8 s** with per-step refresh against **71.0 s** with the retained bank; the
pure-JAX path is ~1.7×. Per-step refresh is both the unbiased and the fastest
default. Protocol, timings, and reproduction commands:
`experiments/library_perf/`.

Peak memory is dominated by the `(samples, masks, qubits)` reduction inside the
per-condition evaluation, so it scales roughly as
`n_iqp_samples × distinct_masks × n_qubits × 16` bytes. Lower
`validation_n_iqp_samples` first; the bank is already compressed.

Not included in this package: exact-distribution evaluation (TVD,
Walsh–Hadamard transforms, visible marginalization) and the benchmark target
families. Exact evaluation needs the full `2**n` output state, so it is useful
only at small sizes, and the target families carry experiment-specific
constants; both live with the research code that consumes this package.

## Examples

`examples/train_gaussian.py` runs the manuscript's simplest benchmark shape
end to end on synthetic data, and `examples/custom_encoder.py` registers a
custom encoder and trains through the public interface. Both run from a
checkout with `PYTHONPATH=src python examples/<name>.py`.

## Tests

```bash
uv run pytest          # library suite
```

`tests/test_pennylane_crosscheck.py` rebuilds the encoder states and the full
conditional circuit in PennyLane and compares the estimator, the sampler, and
the amplitude ratios against exact statevector results. It skips unless
`pennylane` is installed (`pip install .[crosscheck]`).

## Precision

Importing `iqp_cond` enables JAX float64
(`jax.config.update("jax_enable_x64", True)`). The estimator, the loss, and the
trainer assume it: the Monte Carlo settings were chosen for float64, and
disabling it silently changes the numbers. Set it yourself if you import JAX
before this package.

## Status

Extracted from the research repository for the conditional-IQP manuscript.
Licence and citation metadata are pending author and institutional approval.
