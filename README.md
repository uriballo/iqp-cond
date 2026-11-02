# iqp-cond

`iqp-cond` trains conditional IQP generators with a shared learned diagonal circuit. Conditions change the input state, while the generator parameters are shared. Training estimates a multiscale Hamming-RBF MMD using classical samples and amplitude ratios from the encoder; drawing model samples requires a PennyLane simulator or quantum device. The encoder requirements are the main constraint on which reference states can be trained.

## Get started

Clone the repository and install its project and notebook dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/uriballo/iqp-cond.git
cd iqp-cond
uv sync
```

This installs the checkout in editable mode. Use `uv run` to run Python, notebooks, and development commands in the project environment.

```python
import jax.numpy as jnp
import numpy as np
import iqp_cond as ciqp

n_conditions, n_qubits = 3, 6
angles = np.broadcast_to(
    np.linspace(0.3, 1.2, n_conditions)[:, None, None],
    (n_conditions, 1, n_qubits),
)
model = ciqp.ConditionalModel(
    ciqp.Ansatz.complete(n_qubits),
    ciqp.AngleEncoder(angles, axes=("Y" * n_qubits,)),
)
rng = np.random.default_rng(0)
train = rng.integers(0, 2, (n_conditions, 256, n_qubits))
valid = rng.integers(0, 2, (n_conditions, 64, n_qubits))
result = ciqp.train(
    model,
    train,
    valid,
    seed=0,
    config=ciqp.TrainingConfig(steps=10, n_ops=16, mc_samples=128),
)
params = result.best_params
draws = model.sample(params, 100, seed=0)
```

## Learn the library

Read [`USER_GUIDE.md`](USER_GUIDE.md) for shapes, training, inference, and custom encoders. Then work through the notebooks:

1. [`examples/train_gaussian.ipynb`](examples/train_gaussian.ipynb) builds and trains a conditional model with the built-in angle encoder.
2. [`examples/unconditional_fixed_encoder.ipynb`](examples/unconditional_fixed_encoder.ipynb) uses a fixed reference to impose an exact output symmetry.
3. [`examples/custom_encoder.ipynb`](examples/custom_encoder.ipynb) implements and checks a custom encoder.

## API map

- `iqp_cond`: `Ansatz`, `ConditionalModel`, the encoders `AngleEncoder`, `ParityEncoder`, `TwistedParityEncoder` and `CollectivePhaseEncoder`, `TrainingConfig`, `train`, and `check_encoder`.
- `iqp_cond.encoders`: the `Encoder` and `ExactEncoder` protocols and every built-in encoder, one module per family (`product`, `symmetric`).
- `model`: moment estimation, MMD evaluation, exact probabilities, and sampling are methods on `ConditionalModel`.
- `iqp_cond.estimator`, `iqp_cond.initialization`, `iqp_cond.operators`, and `iqp_cond.simulation`: numerical routines and circuit construction for advanced workflows.

Importing `iqp_cond` enables JAX float64, which the estimator, loss, and trainer use internally. Create JAX arrays after importing the package.

Precision is one named policy, `TrainingConfig(precision=...)` or the `precision=` argument of the estimator and the loss, chosen from `ciqp.PRECISIONS`. The default `"fp64"` runs everything in float64. `"gemm_f32"` runs only the final phase contraction in float32, `"estimator_f32"` also builds parities and phases and evaluates the ratio-phase product in float32 before returning to float64 for the statistics, and `"fp32"` keeps the statistics, the loss, the parameters and the optimizer state in float32 too.

## Benchmark the library

The [conditional pipeline benchmark](benchmarks/conditional_pipeline/README.md) measures encoder operations, estimation, MMD, gradients, and one fixed 16-qubit training fit. Its [comparison workflow](benchmarks/conditional_pipeline/HANDOFF.md) explains how to compare a fork with shared inputs and saved numerical outputs.
