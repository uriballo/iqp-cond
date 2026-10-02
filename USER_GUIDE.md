# User guide

This guide covers the public workflow for building, training, and querying conditional IQP models. The notebooks linked from [the README](README.md) give complete runnable examples.

## Model and shapes

An `Ansatz` is a binary generator bank with shape `(n_parameters, n_qubits)`. Each row defines a parity observable in the learned diagonal unitary. Use `Ansatz.complete(n_qubits)` for all one- and two-qubit generators, or pass a custom binary array to `Ansatz`.

A `ConditionalModel` pairs that shared ansatz with an encoder. The encoder has a leading condition axis; it changes the input state across conditions, while one parameter vector is shared by all conditions. Unless `n_visible` is set, all encoder qubits are measured. Any remaining qubits are ancillas, ordered last and marginalized. Target arrays always use visible qubits only.

The main shapes are:

| Value | Shape |
| --- | --- |
| Encoder samples | `(conditions, samples, model qubits)` |
| Training and validation targets | `(conditions, samples, visible qubits)` |
| Ansatz generators | `(parameters, model qubits)` |
| Angle encoder angles | `(conditions, layers, model qubits)` |
| Pauli-Z masks | `(masks, visible qubits)` |
| Sampled model outputs | `(conditions, samples, visible qubits)` |
| Exact model probabilities | `(conditions, 2**visible qubits)` |

Import `iqp_cond` before creating JAX arrays: package import enables JAX float64, which the estimator, loss, and trainer assume.

## Build an encoder and model

`AngleEncoder` is a product-state encoder built from fixed-axis rotations. Its angles have shape `(conditions, layers, qubits)` and `axes` gives one `X`, `Y`, or `Z` label for each layer and qubit. Layer zero acts first and cannot include `Z`, because a Z rotation on `|0>` changes only global phase. A single all-Y layer is the usual RY product state.

```python
import jax.numpy as jnp
import numpy as np
import iqp_cond as ciqp

n_conditions, n_qubits = 3, 6
angles = np.broadcast_to(
    np.linspace(0.3, 1.2, n_conditions)[:, None, None],
    (n_conditions, 1, n_qubits),
)
encoder = ciqp.AngleEncoder(angles, axes=("Y" * n_qubits,))
model = ciqp.ConditionalModel(ciqp.Ansatz.complete(n_qubits), encoder)
```

Custom ansatz generator arrays must be finite and contain only zeroes and ones. The ansatz and encoder must describe the same number of qubits.

## Prepare data and train

Targets are numeric bitstrings arranged as `(conditions, examples, visible_qubits)`. Training and validation targets must have the same number of conditions and visible width as the model. Keep the validation examples held out: `train` selects the best checkpoint using the validation objective.

```python
rng = np.random.default_rng(0)
training_targets = rng.integers(0, 2, (n_conditions, 256, n_qubits))
validation_targets = rng.integers(0, 2, (n_conditions, 64, n_qubits))
config = ciqp.TrainingConfig(steps=50, n_ops=32, mc_samples=256)
result = ciqp.train(
    model,
    training_targets,
    validation_targets,
    seed=0,
    config=config,
)
params = result.best_params
```

`TrainingConfig` controls optimizer and Monte Carlo budgets. Defaults are 2000 updates, batch size 256, learning rates from `1e-2` to `1e-3`, validation every 25 steps, 512 operator draws, and 1000 reference samples for both training and validation. The default bandwidth schedule is derived from visible width. `mask_refresh="step"` draws `n_ops` nonidentity masks directly for each selected objective cell; use `mask_refresh="cache"` to retain a compressed bank for the fit. Reduced precision and streamed samples (`sample_chunk`) are optional memory/performance controls. `gemm_dtype=jnp.float32` reduces only the final phase contraction. `estimator_dtype=jnp.float32` also computes parities and phases in float32 and evaluates the ratio-phase product in complex64 before returning to float64 for statistics and loss accumulation. The two dtype options are mutually exclusive.

If `initial_params` is omitted, initialization uses pooled training data and a mean-covariance prescription, with seeded noise for generator rows involving ancillas. Otherwise provide a finite vector of shape `(n_parameters,)`.

`TrainingResult` contains `best_params` and `final_params`, the `best_step` and `best_validation_loss`, plus `train_steps`, `train_losses`, `validation_steps`, and `validation_losses`. Use `best_params` for evaluation; validation selects that checkpoint.

## Estimate expectations and sample

`model.estimate_pauli_z(params, masks, key=..., n_samples=...)` estimates visible Pauli-Z moments and returns an `ExpectationEstimate` with values shaped `(conditions, masks)`. Supply either a random key and a sample count, or pre-drawn `samples=` instead. Set `return_samples=True` to retain per-sample values and access Monte Carlo standard errors through `standard_errors`. Ancilla identity padding is handled automatically for visible-width masks.

`model.sample(params, n_samples, seed=None, device="lightning.qubit")` returns bitstrings with shape `(conditions, samples, visible_qubits)`. It compiles the encoder preparation and learned diagonal into a PennyLane circuit. A device may be named or passed as an existing PennyLane device; a seed is accepted only when the library constructs the device from its name. Hardware or shot-based devices can sample without building the exact statevector.

`model.probabilities(params, device="lightning.qubit", seed=None)` returns the exact visible distribution. This requires an analytic device and enough memory for a statevector over all model qubits. For direct circuit control, `iqp_cond.simulation.build_qnode(model, params, condition, device=None, *, shots=None, seed=None)` returns the PennyLane QNode for one condition.

## Write and verify a custom encoder

`Encoder` is a structural protocol; there is no base class or registration step. An encoder provides `n_conditions`, `n_qubits`, and four operations:

- `sample(key, n_samples)` returns `(conditions, samples, qubits)` draws from the post-Hadamard reference distribution `|psi_c(z)|**2`, where `psi_c(z) = <z|H**n|phi(c)>`.
- `amplitude_ratios(samples, masks)` returns `(conditions, samples, masks)` values `psi_c(z xor a) / psi_c(z)`.
- `preparation(condition)` returns a callable that queues the input state `|phi(c)>` using PennyLane operations. The simulation bridge calls it before its first Hadamard layer; training does not call it.
- `select(condition)` returns an encoder containing just that condition. The loss evaluates conditions individually to bound memory.

All four operations must describe the same reference state. Run `ciqp.check_encoder(encoder)` to compare the sampling and ratio methods with the PennyLane preparation. It returns an `EncoderCheck`; inspect `.ok`, `.failures`, and `.summary`. The check builds a reference statevector and defaults to a maximum of 16 qubits, so use it on small systems.

## Ancillas, precision, and common limits

Set `n_visible` on `ConditionalModel` when the register includes ancillas. The visible qubits come first; ancillas come last. Targets and measured outputs contain only visible bits. Expectation masks are visible-width and are padded with identity on ancillas.

The library enables float64 at import because its numerical routines rely on that precision. Leave both dtype options unset for the default path. Use `TrainingConfig(gemm_dtype=jnp.float32)` for the conservative phase-contraction option, or `TrainingConfig(estimator_dtype=jnp.float32)` for the wider and faster mixed-precision estimator. Both modes retain float64 outputs and loss accumulation. The wider mode supports only float32 or an explicit float64 spelling; it rejects float16 and bfloat16.

Exact probabilities and `check_encoder` allocate statevector-scale memory, which grows exponentially with the total model qubit count. Use shot-based sampling for larger registers. Training also forms ratio intermediates over samples, masks, and qubits; lower `mc_samples` or set `sample_chunk` if memory is limited.

Common input errors include mismatched condition counts or visible widths, non-binary ansatz generators, invalid angle shapes or rotation axes, and initial-parameter vectors with the wrong length or non-finite values. Check the shape table above and the exception message for the failing input.
