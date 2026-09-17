"""Training loop for shared conditional IQP models.

One parameter vector serves every condition; only the encoder state changes
with the condition.  Updates are condition-balanced (each condition contributes
the same number of samples per step) and checkpoints are selected on a held-out
validation split by the same MMD objective.

Masks refresh like ``iqpopt``: unless ``mask_refresh="cache"`` is requested, every
optimization step draws a fresh compressed, nonidentity mask bank and hands it to
the compiled step as an argument.  Building the bank on the host rather than in
JAX is deliberate: the in-JAX sampler in :mod:`iqp_cond.loss` cannot collapse
duplicates cheaply, so at equal draw budgets it evaluates ~1.8x more mask columns
and costs ~2x the step time (measured in ``experiments/library_perf``).  The
law is the paper's either way; compression is a pure compute optimization.
Validation always consumes one fixed bank, so checkpoint scores stay comparable
across steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import Ansatz
from .encoders import Encoder
from .loss import banked_mmd
from .operators import OperatorBank, operator_bank

__all__ = [
    "CACHE",
    "STEP",
    "TrainingConfig",
    "TrainingResult",
    "initialize_parameters",
    "train",
]

STEP = "step"
CACHE = "cache"

#: Headroom over the pilot distinct-mask count when fixing a refreshed bank's
#: width.  Distinct counts concentrate across banks, so this margin makes a
#: redraw (which conditions the entry on ``distinct <= width``) rare, while every
#: step still pays only the columns the law realistically reaches.  A tighter
#: multi-bank maximum was measured on an idle 32-core host and came out ~6%
#: slower per step, so the width is not worth tuning by column count alone.
_BANK_HEADROOM = 1.15
_BANK_MARGIN = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and Monte Carlo settings.

    The defaults reproduce the published conditional-IQP experiments; they are
    a starting point, not a tuned recommendation for other problems.

    ``n_ops`` and ``validation_n_ops`` are the number of nonidentity operator
    draws per condition and bandwidth that feed the compressed operator banks.
    Under the default per-step refresh a training bank is built before every
    update; the validation bank is drawn once.

    ``gemm_dtype`` runs the phase contraction in a reduced dtype (for example
    ``jnp.float32``) and casts the result back to float64, which is the mixed
    precision mode that pays off on GPUs; the reductions stay float64.  The
    default keeps every operation in float64.
    """

    steps: int = 2_000
    batch_size: int = 256
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    validation_interval: int = 25
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    n_ops: int = 64
    n_iqp_samples: int = 8_192
    validation_n_ops: int = 256
    validation_n_iqp_samples: int = 16_384
    gemm_dtype: jnp.dtype | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Best parameters and scalar training history."""

    seed: int
    generators: np.ndarray
    angles: np.ndarray
    best_params: np.ndarray
    final_params: np.ndarray
    best_step: int
    best_validation_loss: float
    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray
    family: str = ""
    n_banks: int = 0


def initialize_parameters(n_parameters: int, seed: int) -> jnp.ndarray:
    """Draw generator-count normalized Gaussian parameters."""
    if n_parameters < 1:
        raise ValueError("n_parameters must be positive")
    rng = np.random.default_rng(seed + 10_000)
    scale = 0.1 / np.sqrt(n_parameters)
    return jnp.asarray(rng.normal(0.0, scale, n_parameters), dtype=jnp.float64)


def _batch_schedule(
    n_conditions: int,
    n_examples: int,
    config: TrainingConfig,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        n_examples,
        size=(config.steps, n_conditions, config.batch_size),
    )


def _validate_training_inputs(
    ansatz: Ansatz,
    encoders: Encoder,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
) -> int:
    """Check the shared model/data conventions; return the visible width."""
    if training_targets.ndim != 3:
        raise ValueError("training_targets must be rank three")
    if encoders.n_conditions != training_targets.shape[0]:
        raise ValueError(
            f"encoder covers {encoders.n_conditions} conditions but targets "
            f"have {training_targets.shape[0]}",
        )
    if encoders.n_qubits != ansatz.n_qubits:
        raise ValueError(
            f"encoder acts on {encoders.n_qubits} qubits but the ansatz has "
            f"{ansatz.n_qubits}",
        )
    if validation_targets.shape[0] != training_targets.shape[0]:
        raise ValueError("training and validation conditions must align")
    if validation_targets.shape[2] > ansatz.n_qubits:
        raise ValueError(
            "validation target qubit count cannot exceed ansatz qubit count",
        )
    return int(training_targets.shape[2])


def train(
    ansatz: Ansatz,
    encoders: Encoder,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: TrainingConfig,
    *,
    initial_params: np.ndarray | None = None,
    mask_refresh: str = STEP,
) -> TrainingResult:
    """Train one shared conditional model with fixed validation randomness.

    ``mask_refresh`` selects how the training masks are drawn: ``"step"``
    (default) builds a fresh compressed bank before every update, so each
    update's expectation equals the true kernel MMD; ``"cache"`` draws one bank
    and retains it, which reproduces the published campaign's protocol exactly.
    Both arms hand the bank to the compiled step as an argument, so the shapes
    stay static and the program is compiled once.
    """
    training_targets = np.asarray(training_targets, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    if mask_refresh not in (STEP, CACHE):
        raise ValueError(
            f"unknown mask_refresh: {mask_refresh!r}; expected {STEP!r} or {CACHE!r}",
        )
    n_visible = _validate_training_inputs(
        ansatz,
        encoders,
        training_targets,
        validation_targets,
    )

    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    if initial_params is None:
        params = initialize_parameters(ansatz.n_parameters, seed)
    else:
        candidate = np.asarray(initial_params, dtype=np.float64)
        if candidate.shape != (ansatz.n_parameters,):
            raise ValueError(
                f"initial_params must have shape {(ansatz.n_parameters,)}",
            )
        if not np.isfinite(candidate).all():
            raise ValueError("initial_params must contain only finite values")
        params = jnp.asarray(candidate, dtype=jnp.float64).copy()
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    batch_indices = _batch_schedule(
        training_targets.shape[0],
        training_targets.shape[1],
        config,
        seed,
    )
    random_key = jax.random.PRNGKey(seed)

    validation_bank = operator_bank(
        config.sigmas,
        config.validation_n_ops,
        encoders.n_conditions,
        n_visible,
        seed + 1,
    )
    # One width per bandwidth for every refreshed bank, so the compiled step is
    # built once.  The pilot draw sizes each width; the margin keeps a redraw
    # (which conditions the entry on ``distinct <= width``) rare.
    pilot = operator_bank(
        config.sigmas,
        config.n_ops,
        encoders.n_conditions,
        n_visible,
        seed,
    )
    max_masks = tuple(
        int(width * _BANK_HEADROOM) + _BANK_MARGIN for width in pilot.widths
    )
    training_bank: OperatorBank | None = None
    n_banks = config.steps
    if mask_refresh == CACHE:
        # Retained for the whole run; the loop never rebuilds it.
        training_bank = operator_bank(
            config.sigmas,
            config.n_ops,
            encoders.n_conditions,
            n_visible,
            seed,
        )
        n_banks = 1

    def training_loss(
        current_params: jnp.ndarray,
        target_batch: jnp.ndarray,
        current_bank: OperatorBank,
        step_key: jax.Array,
    ) -> jnp.ndarray:
        return banked_mmd(
            current_params,
            generators,
            encoders,
            target_batch,
            current_bank,
            config.n_iqp_samples,
            step_key,
            # The refreshed path reuses one reference draw per condition across
            # bandwidths (the paper's form); the retained path keeps the
            # campaign's per-entry draw layout bit for bit.
            shared_samples=mask_refresh == STEP,
            gemm_dtype=config.gemm_dtype,
        ).loss

    value_and_gradient = jax.jit(jax.value_and_grad(training_loss))

    def validation_loss(current_params: jnp.ndarray) -> jnp.ndarray:
        return banked_mmd(
            current_params,
            generators,
            encoders,
            jnp.asarray(validation_targets),
            validation_bank,
            config.validation_n_iqp_samples,
            jax.random.PRNGKey(seed + 1_000_000),
            gemm_dtype=config.gemm_dtype,
        ).loss

    evaluate_validation = jax.jit(validation_loss)

    train_steps = []
    train_losses = []
    validation_steps = []
    validation_losses = []
    best_params = params
    best_step = 0
    best_validation = float(evaluate_validation(params))

    validation_steps.append(0)
    validation_losses.append(best_validation)
    for step in range(config.steps):
        if training_bank is None or mask_refresh == STEP:
            training_bank = operator_bank(
                config.sigmas,
                config.n_ops,
                encoders.n_conditions,
                n_visible,
                seed + 1_000_003 * (step + 1),
                max_masks=max_masks,
            )
        indices = batch_indices[step]
        batch = np.stack(
            [
                training_targets[condition, indices[condition]]
                for condition in range(training_targets.shape[0])
            ],
        )
        step_key = jax.random.fold_in(random_key, step)
        loss, gradient = value_and_gradient(
            params,
            jnp.asarray(batch),
            training_bank,
            step_key,
        )
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        train_steps.append(step + 1)
        # Defer the device sync: no ``float(loss)`` here, so next step's host
        # bank build overlaps with the running device program.
        train_losses.append(loss)

        completed_step = step + 1
        if completed_step % config.validation_interval == 0:
            current_validation = float(evaluate_validation(params))
            validation_steps.append(completed_step)
            validation_losses.append(current_validation)
            if current_validation < best_validation:
                best_validation = current_validation
                best_params = params
                best_step = completed_step

    return TrainingResult(
        seed=seed,
        generators=np.asarray(ansatz.generators),
        angles=np.asarray(encoders.angles),
        best_params=np.asarray(best_params),
        final_params=np.asarray(params),
        best_step=best_step,
        best_validation_loss=best_validation,
        train_steps=np.asarray(train_steps),
        train_losses=np.asarray(jnp.stack(train_losses)),
        validation_steps=np.asarray(validation_steps),
        validation_losses=np.asarray(validation_losses),
        family=encoders.name,
        n_banks=n_banks,
    )
