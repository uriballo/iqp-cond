# iqp-cond

Conditional generation with instantaneous quantum polynomial-time circuits.

A conditional IQP model applies one shared diagonal unitary to a
condition-dependent input state. The condition changes only the state; the
learned circuit stays fixed. Training uses sampled Pauli-character estimates of
a multiscale Hamming-RBF maximum mean discrepancy, which is classically
computable whenever the encoder supports two operations:

- draw bitstrings from the post-Hadamard reference Born distribution, and
- evaluate reference-amplitude ratios `psi_c(z xor a) / psi_c(z)` for Fourier
  masks `a`.

Those operations keep training classical; the model itself is still hard to
sample from in general. Every encoder also describes its circuit preparation
and condition selection, so the same object trains classically and runs on a
PennyLane simulator or quantum backend.

## Install

```bash
uv sync   # library plus dev tools
```

The project is managed with [uv](https://docs.astral.sh/uv/); `uv sync` installs
the package in editable mode, and `uv run <command>` runs commands inside the
environment.  The package requires Python 3.11+ and depends on `jax`, `numpy`,
`optax`, and `pennylane` (which installs `pennylane-lightning`).

To install from Git:

```bash
uv add "iqp-cond @ git+https://github.com/uriballo/iqp-cond@v0.6.0"
```

### Migrating from 0.4

Version 0.5 removes the encoder registry and narrows the package root:

| 0.4 | 0.5 |
|---|---|
| `complete_ansatz(n)` | `Ansatz.complete(n)` |
| `ansatz_from_generators(rows)` | `Ansatz(rows)` |
| `angle_encoder(values, axes=...)` | `AngleEncoder(values, axes=...)` |
| `model.encoders` | `model.encoder` |
| `simulation.sample_model(model, ...)` | `model.sample(...)` |
| `simulation.probabilities(model, ...)` | `model.probabilities(...)` |
| `CircuitEncoder` | `Encoder` |
| `ciqp.STEP` / `ciqp.CACHE` | `"step"` / `"cache"` |

Custom encoder constructors now keep their own explicit signatures. They are
instantiated directly and must implement `preparation_ops` and `select` in
addition to sampling and amplitude ratios. Numerical helper functions remain
available from their named submodules.

### Version 0.6

Version 0.6 reduces training cost without changing the multiscale objective.
Each update samples one condition-bandwidth objective cell, while validation
still evaluates every cell with fixed randomness. Pytree-compatible encoders
reuse compiled optimizer steps across conditions; custom opaque encoders keep
the condition-specific closure path. The release also adds a one-condition
example in which a fixed, tractable GHZ reference imposes exact global
bit-flip symmetry on the model distribution.

## Quickstart

```python
import numpy as np
import iqp_cond as ciqp

n_qubits, n_conditions = 8, 3

# A conditional model is a shared generator bank plus an encoder.
ansatz = ciqp.Ansatz.complete(n_qubits)

# One RY layer per condition and qubit; see "Encoders" for stacked layers.
angles = np.broadcast_to(
    np.linspace(0.6, 2.0, n_conditions)[:, None], (n_conditions, n_qubits)
)
model = ciqp.ConditionalModel(
    ansatz,
    ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * n_qubits,)),
)

# Bitstring samples from the target family: (conditions, samples, visible qubits).
rng = np.random.default_rng(0)
train_targets = rng.integers(0, 2, size=(n_conditions, 1000, n_qubits)).astype(float)
valid_targets = rng.integers(0, 2, size=(n_conditions, 200, n_qubits)).astype(float)

result = ciqp.train(
    model,
    train_targets,
    valid_targets,
    seed=0,
    # Defaults: 2000 updates, batch 256, 512 masks for one uniformly selected
    # condition-bandwidth cell per update, 1000 Monte Carlo samples, pooled
    # data-based initialization, and the width-aware bandwidth schedule.
    config=ciqp.TrainingConfig(steps=50, batch_size=64, n_ops=16, mc_samples=1024),
)

params = result.best_params  # selected on the validation split
```

By default the trainer initializes parameters from the pooled training
conditions with the mean-covariance prescription, matching the target's one- and
two-body structure instead of noise.

## Querying expectations

At inference only the encoder state changes with the condition; the learned
diagonal stays fixed.  The estimator pads visible-width masks with identity on
the ancillas automatically.

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
`return_samples=True` to produce the per-sample values that
`estimate.standard_errors` needs. Expert estimator functions, including the
unchecked JIT kernel, live in `iqp_cond.estimator`.

## Sampling and backends

Training does not use PennyLane.  Sampling the trained model compiles it into a
PennyLane circuit that runs the encoder preparation, `H^n`, the shared diagonal
as `MultiRZ` per generator, and a second `H^n`, then evaluates it on the device
you choose:

```python
import pennylane as qml
from iqp_cond.simulation import build_qnode

probabilities = model.probabilities(params)
draws = model.sample(params, 1024, seed=0)

device = qml.device("default.qubit", wires=model.n_qubits)  # any plugin device
qnode = build_qnode(model, params, 0, device=device, shots=1000)
```

- `model.probabilities(params, *, device="lightning.qubit")` returns the exact
  visible distribution `(conditions, 2**n_visible)`; it needs an analytic device
  (a statevector simulator) and is a small-system path.
- `model.sample(params, n_samples, *, seed=None, device="lightning.qubit")`
  draws shot samples `(conditions, n_samples, n_visible)` and works on any
  PennyLane device, including hardware backends.  `seed` applies when the device
  is built from a name.
- `build_qnode(model, params, condition, device=None, *, shots=None, seed=None)`
  returns the QNode itself, so you can compile, batch, or submit it with any
  plugin.

Every exact sampler still represents the full `2**n_qubits` state, so simulator
sampling fits only registers that fit in memory; hardware runs use shots and the
same circuit.

## Encoders

`Encoder` is a structural protocol. A class implements it by providing
`n_conditions`, `n_qubits`, and four explicit operations:

- `sample(key, n_samples)`: `(conditions, samples, qubits)` bitstrings from the
  post-Hadamard reference Born law `|psi_c(z)|^2`, where
  `psi_c(z) = <z|H^n|phi(c)>`;
- `amplitude_ratios(samples, masks)`: `(conditions, samples, masks)` ratios
  `psi_c(z xor a) / psi_c(z)`;
- `preparation_ops(condition)`: `(gate, wires, parameters)` operations that
  prepare `|phi(c)>` before the first Hadamard, used only by
  `ciqp.simulation`.  The bridge maps `RX`, `RY`, and `RZ` (one wire, one
  angle) and `CNOT` (two wires, no parameter), which covers product states and
  entangling preparations.
- `select(condition)`: the same encoder restricted to a one-condition batch.

`ConditionalModel` validates the complete protocol when it is constructed.
Custom encoders require no base class, registration, family name, or shared
constructor signature.

The built-in family is `angle`.  It is a product state with a stack of
fixed-axis rotations per qubit. `ciqp.AngleEncoder(angles, axes=...)` takes
angles of shape `(conditions, layers, qubits)` and one axis label (`"X"`, `"Y"`,
or `"Z"`) per layer and qubit, with layer 0 applied first. A layer can be any
sequence of labels, so `axes=("YXXY", "ZXYZ")` is valid for four qubits; a
single all-`Y` layer is the plain `RY` product state.

The first layer may not contain `Z`, because `R_Z` on `|0>` is a global phase
that cancels from every probability and ratio, so accepting it would silently
ignore its angle.  Local amplitudes stay real for all-`Y` sequences and become
complex as soon as `X` or `Z` appears; sampling and ratio queries behave the
same either way.

To save memory, the loss evaluates one condition at a time through the required
`select(condition)` method. The built-in encoder is also a JAX pytree, but
custom encoders do not need JAX registration. `AngleEncoder` additionally
implements the optional `ExactEncoder` capability used by exact checks.

### Adding your own

An encoder is any object with the interface above. Instantiate it and pass it
directly to the model. See `examples/custom_encoder.py` for a runnable version:

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

    def preparation_ops(self, condition):
        # |+>^n is the image of |0>^n under the first Hadamard layer.
        return ()

    def select(self, condition):
        return HadamardEncoder(self.angles[condition : condition + 1])


encoder = HadamardEncoder(jnp.zeros((3, 6)))
model = ciqp.ConditionalModel(ciqp.Ansatz.complete(6), encoder)
```

## Ancillas

The model allows extra qubits and marginalizes them after measurement.  Qubits
are ordered visible first, ancillas last, and the model declares the split:

```python
model = ciqp.ConditionalModel(ansatz_on_5_qubits, encoder, n_visible=3)
model.n_ancillas  # 2
```

The encoder and the generator bank cover the full register, targets keep the
visible width only, and the estimator and the loss pad visible masks with
identity on the ancilla positions automatically. Advanced mask padding,
ancilla support, and data initialization are available from `iqp_cond.model`
and `iqp_cond.initialization`.

## API

| Module | Contents |
|---|---|
| `iqp_cond` | `Ansatz`, `ConditionalModel`, `Encoder`, `AngleEncoder`, `Operation`, `TrainingConfig`, `train` |
| `iqp_cond.model` | `Ansatz`, `ConditionalModel`, `pad_visible_masks`, `ancilla_support` |
| `iqp_cond.encoders` | `Encoder`, `ExactEncoder`, `Operation`, `AngleEncoder`, `layered_amplitudes` |
| `iqp_cond.simulation` | `build_qnode` |
| `iqp_cond.estimator` | `estimate_pauli_z`, `estimate_pauli_z_unchecked`, `ExpectationEstimate` |
| `iqp_cond.loss` | `mmd`, `target_moments`, `hamming_rbf`, `ConditionalMMDResult` |
| `iqp_cond.training` | `TrainingConfig`, `TrainingResult`, `train`, `STEP`, `CACHE` |
| `iqp_cond.initialization` | `random_parameters`, `parameters_from_data` |

The advanced building blocks are in `iqp_cond.operators` (mask law,
`OperatorBank`, `operator_bank`, `bank_from_ragged`,
`sample_nonidentity_masks`, `sample_nonidentity_masks_jax`,
`mmd_operator_probability`, `expected_bodyness_sigmas`).

## Masks and the MMD estimator

Training minimizes the stochastic operator estimator of the squared multiscale
Hamming–RBF MMD implemented by `iqp_cond.loss.mmd`. The reference-state randomness always
comes from the call's `key`; the mask source is exactly one of:

- `sigmas` + `n_ops`: draw nonidentity masks afresh inside the call, one draw
  per condition and bandwidth;
- `bank`: evaluate a prebuilt compressed `OperatorBank`, which is the path for
  fixed-bank evaluation and repeated scoring.

A bandwidth `sigma` induces the mask law `a_i ~ Bernoulli(p_sigma)` with
`p_sigma = (1 - e^{-1/(2 sigma^2)})/2`, so `E[|a|] = n p_sigma` is the expected
bodyness. `iqp_cond.operators.expected_bodyness_sigmas(n, targets)` inverts a bodyness
target into its bandwidth; the default targets are `(2, n/4, 0.49n)`, capped at
`0.49n` because `p_sigma < 1/2` strictly. `TrainingConfig.sigmas=None`
resolves through that helper once the model width is known.

Skipping the identity mask is exact.  It contributes zero to the kernel MMD, so
spending draws on it only adds variance.  Both mask sources accept
`shared_samples=True`, which draws one reference sample set per condition and
evaluates every selected bandwidth on it; the default resamples per (bandwidth,
condition) entry. At every optimization step, the trainer selects one condition
and one configured bandwidth uniformly. The selected-cell loss and gradient
are unbiased for the condition-balanced arithmetic-mean multiscale objective.
With `TrainingConfig.mask_refresh="step"`, masks are redrawn for the selected
cell; `mask_refresh="cache"` selects the cell from one retained full multiscale
bank. Validation always evaluates every condition and bandwidth with one fixed
bank, so checkpoint selection stays comparable across steps.

The shipped defaults are 512 masks and 1000 Monte Carlo samples per condition.
Peak memory comes from the per-condition sample/mask reduction, so larger
sample budgets and more distinct masks increase memory use.

## Estimator performance

Training compiles the loss gradient, Adam update and parameter update together.
On an idle Strix Halo CPU, this reduced measured 200-update fit times from
8.45 to 7.71 seconds at the default budget, and from 15.61 to 12.82 seconds
with 2000 masks, 1024 reference samples, float32 contraction and chunks of 256.
These are medians of three alternating runs on synthetic binary targets with
12 visible qubits and five conditions, including compilation and validation.

Across four runs on two Strix Halo CPU hosts, uniform bandwidth selection
reduced a 2000-update Gaussian fit from about 111 seconds to 50 seconds. It
changes the optimizer trajectory by adding unbiased bandwidth-sampling noise;
exhaustive fixed-bank validation and the estimated multiscale objective remain
unchanged. JAX may compile one training executable per distinct bandwidth
shape. Custom encoders that are not JAX pytrees also compile one closure per
condition, so very short cold-start runs can take longer despite the lower
steady-state cost.

The same selection applies to cached masks. Across two 1000-update runs on the
same hosts, selecting from a retained bank reduced the mean runtime from 49.7
seconds to 21.8 seconds, a 56% reduction. Cache mode still builds each
bandwidth once; it no longer evaluates every cached bandwidth on every update.

On the five-condition workload, selecting one condition further reduced the
mean runtime of a 1000-update fit from 27.2 seconds to 12.6 seconds with step
refresh and from 22.5 seconds to 11.4 seconds with cached masks. A full
2000-update frozen Gaussian comparison measured the combined bandwidth and
condition selection at 19.51 seconds, down from 50.85 seconds for bandwidth
selection alone. Condition selection changes gradient variance but preserves
the condition-balanced objective in expectation.

Pytree-compatible encoders pass their selected array state into one shared
compiled optimizer step. On two full frozen Gaussian runs this reduced mean
wall time from 20.30 to 16.78 seconds, with identical paired result summaries
and all 81 exhaustive validations. Custom encoders that are not registered as
JAX pytrees still use condition-specific compiled closures.

## Precision

Importing `iqp_cond` enables JAX float64
(`jax.config.update("jax_enable_x64", True)`). The estimator, the loss, and the
trainer assume float64, and the Monte Carlo settings depend on it; disabling it
silently changes the numbers. Set it yourself if you import JAX before this
package.

Two opt-in settings cover large-budget regimes.

- `TrainingConfig(gemm_dtype=jnp.float32)` (or `gemm_dtype` on the expert loss
  and estimator functions) runs only the phase contraction in float32 and casts the
  phase result back to float64 before subsequent operations. Outputs remain
  float64, with measured gradient error of about 1e-7. It gives the largest
  speedup on GPUs, and on CPU it saves 6–16% of a full fit.
- `TrainingConfig(sample_chunk=N)` streams the Monte Carlo samples in slices
  and accumulates the per-mask moments, bounding the live ratio tensor. It is
  faster at large sample budgets (1.1–1.24× at 8192+ samples) and slightly
  slower for small ones.

## Examples

`examples/train_gaussian.py` trains a conditional model on a synthetic Gaussian
family, then samples it through the PennyLane bridge and scores the Hamming
profiles; `examples/custom_encoder.py` instantiates a custom encoder and trains
through the public interface; `examples/unconditional_fixed_encoder.py` shows
an unconditional, fixed-reference encoder.  All run from a checkout with
`uv run python examples/<name>.py`.

## Tests

```bash
uv run pytest
```

`tests/test_simulation.py` checks the bridge against an independent NumPy
statevector oracle and shot statistics, and
`tests/test_pennylane_crosscheck.py` rebuilds the encoder states and the full
conditional circuit by hand in PennyLane, comparing the estimator, the sampler,
and the amplitude ratios against statevector results.

## Status

Version 0.6.0, a private development release. Licence and citation metadata are
pending author and institutional approval.
