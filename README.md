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

# A conditional model is a shared generator bank plus an encoder.
ansatz = iqc.complete_ansatz(n_qubits)

# One angle per condition and qubit; the angle family below stacks layers.
angles = np.broadcast_to(
    np.linspace(0.6, 2.0, n_conditions)[:, None], (n_conditions, n_qubits)
)
model = iqc.ConditionalModel(ansatz, iqc.encoder(iqc.PRODUCT, angles))

# Bitstring samples from the target family: (conditions, samples, visible qubits).
rng = np.random.default_rng(0)
train_targets = rng.integers(0, 2, size=(n_conditions, 1000, n_qubits)).astype(float)
valid_targets = rng.integers(0, 2, size=(n_conditions, 200, n_qubits)).astype(float)

result = iqc.train(
    model,
    train_targets,
    valid_targets,
    seed=0,
    # Shipped defaults: 2000 updates, batch 256, 512 masks per condition and
    # bandwidth, 1000 Monte Carlo samples, data-based initialization, and the
    # width-aware bodyness bandwidth schedule.  This quickstart shortens them.
    config=iqc.TrainingConfig(steps=50, batch_size=64, n_ops=16, mc_samples=1024),
)

params = result.best_params  # selected on the validation split
```

Parameters are initialized by default from the pooled training conditions with
the IQPopt mean-covariance prescription, so training starts from the target's
one- and two-body structure rather than noise.

## Querying expectations

At inference only the encoder state changes with the condition; the learned
diagonal stays fixed.  Visible-width masks are padded with identity on the
ancillas automatically.

```python
import jax

masks = np.array([[1.0] * n_qubits, [1.0, 0.0] * (n_qubits // 2)])
estimate = model.estimate_pauli_z(
    params,
    masks,
    key=jax.random.PRNGKey(0),
    n_samples=8192,
)
estimate.values  # (conditions, masks)
```

Pass `samples=` instead of `key=` to evaluate a pre-drawn reference draw, and
`return_samples=True` to materialise the per-sample values that
`estimate.standard_errors` needs.  Inside `jax.jit`, use the free function
`iqc.estimate_pauli_z_unchecked`, which skips validation; the loss uses it.

## Encoders

`iqc.encoder(family, angles, **params)` takes family-specific angles (all
built-ins but `angle` use `(n_conditions, n_qubits)`) and returns an object
implementing three things:

- `sample(key, n_samples)` — `(conditions, samples, qubits)` bitstrings from the
  encoder's Born law `|phi(c)|^2`;
- `amplitude_ratios(samples, masks)` — `(conditions, samples, masks)` ratios
  `phi(z xor a) / phi(z)`;
- `n_conditions` and `n_qubits`.

The library evaluates one condition at a time for memory, and does that with
`iqp_cond.encoders.select_encoder`: it slices every condition-batched leaf of a
JAX pytree, or falls back to an optional `select(condition)` method when the
encoder is not a pytree.  The built-ins are registered as pytrees, so a custom
family needs `select` only if it is plain Python.

All built-ins also implement `post_hadamard_amplitudes` (the optional
`ExactEncoder` capability) for exact small-system work.

| Name | State | Used for |
|---|---|---|
| `product` | `RY(a_i)` on each qubit | translated Gaussian target |
| `angle` | stack of fixed-axis rotations per qubit | general product conditioning |

The `angle` family generalizes `product` to several layers of non-commuting
single-qubit rotations: `iqc.encoder("angle", angles, axes=...)` takes angles of
shape `(conditions, layers, qubits)` and one axis label (`"X"`, `"Y"`, or
`"Z"`) per layer and qubit, with layer 0 applied first.  A layer can be any
sequence of labels, so `axes=("YXXY", "ZXYZ")` is valid for four qubits.  The
first layer may not contain `Z`, because `R_Z` on `|0>` is a global phase that
cancels from every probability and ratio, so accepting it would silently ignore
its angle.  Local amplitudes stay real for all-`Y` sequences and become complex
as soon as `X` or `Z` appears; sampling and ratio queries are unchanged either
way.

### Adding your own

An encoder is any object with the interface above; registering it makes
`iqc.encoder` return it.  See `examples/custom_encoder.py` for a runnable
version:

```python
@dataclass(frozen=True)
class HadamardEncoder:
    angles: jnp.ndarray

    @property
    def n_conditions(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.angles.shape[1])

    def sample(self, key, n_samples):
        return jax.random.bernoulli(
            key, p=0.5, shape=(self.n_conditions, n_samples, self.n_qubits)
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples, masks):
        return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    def select(self, condition):
        return HadamardEncoder(self.angles[condition : condition + 1])


iqc.register_encoder(
    "hadamard", lambda angles, **_: HadamardEncoder(jnp.asarray(angles))
)
```

Registering a name that already exists raises unless `replace=True`.

## Ancillas

Extra qubits are allowed and are marginalized after measurement. Qubits are
ordered **visible first, ancillas last**; the model declares the split:

```python
model = iqc.ConditionalModel(ansatz_on_5_qubits, encoder, n_visible=3)
model.n_ancillas  # 2
```

The encoder and the generator bank cover the full register, targets keep the
visible width only, and visible masks are padded with identity on the ancilla
positions automatically by the estimator and the loss (or explicitly with
`iqc.pad_visible_masks`).  `iqc.ancilla_support(generators, n_visible)` returns
the parameter mask whose generators touch an ancilla, which is what
`iqc.parameters_from_data` uses to leave ancilla parameters near zero.

## API

| Module | Contents |
|---|---|
| `iqp_cond.ansatz` | `Ansatz`, `ansatz_from_generators`, `generators_from_graph`, `complete_ansatz` |
| `iqp_cond.encoders` | `Encoder`, `ExactEncoder`, `encoder`, `register_encoder`, `available_encoders`, `select_encoder`, built-in families |
| `iqp_cond.estimator` | `estimate_pauli_z`, `estimate_pauli_z_unchecked`, `ExpectationEstimate` |
| `iqp_cond.loss` | `mmd`, `target_moments`, `hamming_rbf`, `ConditionalMMDResult` |
| `iqp_cond.model` | `ConditionalModel` |
| `iqp_cond.trainer` | `TrainingConfig`, `TrainingResult`, `train`, `STEP`, `CACHE` |
| `iqp_cond.initialization` | `random_parameters`, `parameters_from_data` |
| `iqp_cond.ancilla` | `pad_visible_masks`, `ancilla_support` |

Advanced building blocks live in `iqp_cond.operators` (mask law, `OperatorBank`,
`operator_bank`, `bank_from_ragged`, `sample_nonidentity_masks`,
`sample_nonidentity_masks_jax`, `mmd_operator_probability`); only
`expected_bodyness_sigmas` is re-exported, because `TrainingConfig` resolves
its default bandwidths through it.

## Masks and the MMD estimator

Training minimizes the paper's stochastic operator estimator of the squared
multiscale Hamming–RBF MMD through `iqc.mmd`. The reference-state randomness
always comes from the call's `key`; the mask source is exactly one of:

- `sigmas` + `n_ops`: draw nonidentity masks afresh inside the call, one draw
  per condition and bandwidth, the `iqpopt` cadence;
- `bank`: evaluate a prebuilt compressed `OperatorBank`, which is the
  evaluation and campaign-reproduction path.

A bandwidth `sigma` induces the mask law `a_i ~ Bernoulli(p_sigma)` with
`p_sigma = (1 - e^{-1/(2 sigma^2)})/2`, so `E[|a|] = n p_sigma` is the expected
bodyness. `iqc.expected_bodyness_sigmas(n, targets)` inverts a bodyness target
into its bandwidth; the library default targets `(2, n/4, 0.49n)`, capped at
`0.49n` because `p_sigma < 1/2` strictly. The trainer resolves
`TrainingConfig.sigmas=None` through that helper once it knows the model width;
the campaign instead targeted `1, (1+0.49n)/2, 0.49n`.

Skipping the identity mask is exact: it contributes zero to the kernel MMD, so
spending draws on it only adds variance.  Both mask sources accept
`shared_samples=True`, which draws one reference sample set per condition and
evaluates every bandwidth on it; that is the trainer's estimator form and the
paper's.  The default resamples per (bandwidth, condition) entry.  The trainer
redraws masks at every optimization step (`TrainingConfig.mask_refresh="step"`,
the default) so each update's expectation equals the true kernel MMD;
`mask_refresh="cache"` draws one bank and retains it, reproducing the published
campaign's protocol.  Validation always uses one fixed bank, so checkpoint
selection stays comparable across steps.

The shipped defaults are lighter than the campaign's: 512 masks and 1000 Monte
Carlo samples per condition, against 2000 masks and 8192 samples (the smallest
budget passing the campaign's preregistered estimator gates). Raise
`n_ops`/`mc_samples` for production fidelity. At 12 visible qubits and 5
conditions, one 2000-step fit at the campaign scale takes ~55–62 s on an idle
32-core host, with mixed precision
(`TrainingConfig(gemm_dtype=jnp.float32)`) another 6–16% faster. Peak memory is
dominated by the `(samples, masks, qubits)` reduction inside the per-condition
evaluation, so it scales roughly as
`mc_samples × distinct_masks × n_qubits × 16` bytes.

## Precision

Importing `iqp_cond` enables JAX float64
(`jax.config.update("jax_enable_x64", True)`). The estimator, the loss, and the
trainer assume it: the Monte Carlo settings were chosen for float64, and
disabling it silently changes the numbers. Set it yourself if you import JAX
before this package.

Two opt-in knobs cover large-budget regimes.

- `TrainingConfig(gemm_dtype=jnp.float32)` (or `gemm_dtype` on `mmd` /
  `estimate_pauli_z`) runs only the phase contraction in float32 and casts the
  result back to float64, so the loss and gradient stay float64-precise. It is
  the dominant lever on GPUs — the sibling `iqpopt-gpu` benchmark reports
  15–28× for the same estimator structure — and is worth 6–16% of a full fit on
  CPU with ~1e-7 gradient error.
- `TrainingConfig(sample_chunk=N)` streams the reference samples in slices and
  accumulates the per-mask moments, bounding the live ratio tensor. It pays off
  at large sample budgets (1.1–1.24× faster at 8192+ samples) and costs a little
  for small ones.

## Examples

`examples/train_gaussian.py` runs the manuscript's simplest benchmark shape end
to end on synthetic data, and `examples/custom_encoder.py` registers a custom
encoder and trains through the public interface. Both run from a checkout with
`PYTHONPATH=src python examples/<name>.py`.

## Tests

```bash
uv run pytest          # library suite
```

`tests/test_pennylane_crosscheck.py` rebuilds the encoder states and the full
conditional circuit in PennyLane and compares the estimator, the sampler, and
the amplitude ratios against exact statevector results. It skips unless
`pennylane` is installed (`pip install .[crosscheck]`).

## Status

Extracted from the research repository for the conditional-IQP manuscript.
Licence and citation metadata are pending author and institutional approval.
