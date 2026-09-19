"""Training loop for shared conditional IQP models.

The model uses one parameter vector for every condition; only the encoder state
changes with the condition.  Updates are condition-balanced, each condition
contributing the same number of samples per step, and the trainer selects
checkpoints on a held-out validation split by the same MMD objective.

The trainer refreshes masks as in ``iqpopt``.  Unless
``mask_refresh="cache"`` is requested, every optimization step draws a fresh
compressed, nonidentity mask bank and passes it to the compiled step as an
argument.  The trainer builds the bank on the host rather than in JAX on
purpose, because the in-JAX sampler cannot collapse duplicates cheaply and so
evaluates more mask columns at equal draw budgets.  Validation always uses one
fixed bank, so checkpoint scores stay comparable across steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .initialization import parameters_from_data
from .model import ConditionalModel
from .operators import OperatorBank, expected_bodyness_sigmas, operator_bank

__all__ = [
    "CACHE",
    "STEP",
    "TrainingConfig",
    "TrainingResult",
    "train",
]

STEP = "step"
CACHE = "cache"

#: Extra width over the pilot distinct-mask count when fixing a refreshed bank's
#: width.  Distinct counts concentrate across banks, so this margin keeps a
#: redraw rare.  A redraw conditions the entry on ``distinct <= width``, and
#: every step still evaluates only the columns the law realistically reaches.
_BANK_HEADROOM = 1.15
_BANK_MARGIN = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and Monte Carlo settings.

    The defaults follow the paper's protocol where the library can: 2000
    updates, batches of 256 per condition, initialization from the pooled
    training data by the mean-covariance prescription, checkpoints every 25
    steps by validation MMD, and a width-aware bandwidth schedule, with
    ``sigmas=None`` resolving through :meth:`resolved_sigmas` to the bodyness
    targets ``(2, n/4, 0.49 n)``.  The mask and Monte Carlo budgets are
    lighter than the paper's so that default runs finish quickly:
    ``n_ops``/``validation_n_ops`` default to 512 where the paper used 2000
    masks, and ``mc_samples``/``validation_mc_samples`` to 1000 where the paper
    used 8192 reference draws, the smallest calibrated budget passing the
    estimator gates.  Raise them for production fidelity.  The paper also
    retained one mask bank per fit, the ``mask_refresh="cache"`` setting,
    instead of redrawing every step.

    ``n_ops`` and ``validation_n_ops`` give the number of nonidentity operator
    draws per condition and bandwidth that feed the compressed operator banks.
    Under the default per-step refresh the trainer builds a training bank
    before every update and draws the validation bank once.

    ``gemm_dtype`` runs the phase contraction in a reduced dtype such as
    ``jnp.float32`` and casts the result back to float64, which is the mixed
    precision mode that helps on GPUs; the reductions stay float64.

    ``sample_chunk`` streams the Monte Carlo samples in slices, bounding the live
    ``(samples, masks, qubits)`` ratio tensor.  It is faster at large sample
    budgets and slightly slower for small ones, so it is off by default.
    """

    steps: int = 2_000
    batch_size: int = 256
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    validation_interval: int = 25
    sigmas: tuple[float, ...] | None = None
    n_ops: int = 512
    mc_samples: int = 1_000
    validation_n_ops: int = 512
    validation_mc_samples: int = 1_000
    gemm_dtype: jnp.dtype | None = None
    sample_chunk: int | None = None
    mask_refresh: str = STEP

    def resolved_sigmas(self, n_visible: int) -> tuple[float, ...]:
        """Return the bandwidth schedule for ``n_visible`` visible qubits.

        Explicit ``sigmas`` win; otherwise the default bodyness targets
        ``(2, n/4, 0.49 n)`` are converted with
        :func:`iqp_cond.expected_bodyness_sigmas`.
        """
        if self.sigmas is not None:
            return self.sigmas
        return expected_bodyness_sigmas(n_visible)

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not (0.0 < self.learning_rate and 0.0 < self.final_learning_rate):
            raise ValueError("learning rates must be positive")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if self.sigmas is not None and (
            not self.sigmas
            or any(
                not math.isfinite(float(sigma)) or float(sigma) <= 0.0
                for sigma in self.sigmas
            )
        ):
            raise ValueError("sigmas must be positive and finite")
        if self.n_ops < 1 or self.validation_n_ops < 1:
            raise ValueError("n_ops and validation_n_ops must be positive")
        if self.mc_samples < 2 or self.validation_mc_samples < 2:
            raise ValueError(
                "mc_samples and validation_mc_samples must be at least two",
            )
        if self.sample_chunk is not None and self.sample_chunk < 1:
            raise ValueError("sample_chunk must be positive")
        if self.mask_refresh not in (STEP, CACHE):
            raise ValueError(
                f"unknown mask_refresh: {self.mask_refresh!r}; "
                f"expected {STEP!r} or {CACHE!r}",
            )


@dataclass(frozen=True)
class TrainingResult:
    """Best parameters and scalar training history."""

    seed: int
    best_params: np.ndarray
    final_params: np.ndarray
    best_step: int
    best_validation_loss: float
    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray


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
    model: ConditionalModel,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
) -> None:
    """Check the shared model/data conventions."""
    if training_targets.ndim != 3:
        raise ValueError("training_targets must be rank three")
    if validation_targets.ndim != 3:
        raise ValueError("validation_targets must be rank three")
    for name, values in (
        ("training_targets", training_targets),
        ("validation_targets", validation_targets),
    ):
        if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError(
                f"{name} must contain only finite binary values 0 and 1",
            )
        if values.shape[1] < 2:
            raise ValueError("each condition needs at least two target samples")
    if model.encoder.n_conditions != training_targets.shape[0]:
        raise ValueError(
            f"model covers {model.encoder.n_conditions} conditions but targets "
            f"have {training_targets.shape[0]}",
        )
    if training_targets.shape[2] != model.visible:
        raise ValueError(
            f"model declares {model.visible} visible qubits but training targets "
            f"have {training_targets.shape[2]}",
        )
    if validation_targets.shape[0] != training_targets.shape[0]:
        raise ValueError("training and validation conditions must align")
    if validation_targets.shape[2] != model.visible:
        raise ValueError(
            f"model declares {model.visible} visible qubits but validation "
            f"targets have {validation_targets.shape[2]}",
        )


def _refreshed_widths(
    sigmas: tuple[float, ...],
    config: TrainingConfig,
    model: ConditionalModel,
    seed: int,
) -> tuple[int, ...]:
    """Mask-axis widths a refreshed bank will almost never exceed."""
    pilot = operator_bank(
        sigmas,
        config.n_ops,
        model.encoder.n_conditions,
        model.visible,
        seed,
    )
    return tuple(int(width * _BANK_HEADROOM) + _BANK_MARGIN for width in pilot.widths)


def train(
    model: ConditionalModel,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    config: TrainingConfig,
    initial_params: np.ndarray | None = None,
) -> TrainingResult:
    """Train one shared conditional model with fixed validation randomness.

    ``config.mask_refresh`` selects how the training masks are drawn:
    ``"step"``, the default, draws a fresh bank inside every compiled step from
    a per-step key, so each update's expectation equals the true kernel MMD;
    ``"cache"`` draws one compressed NumPy bank and retains it, which
    reproduces the mask protocol of the paper.  Every update uses the
    paper's estimator form either way, with one reference draw per condition
    shared by all masks and bandwidths.

    Without ``initial_params`` the parameters start from the paper's
    mean-covariance prescription applied to the pooled training conditions,
    with seeded noise on ancilla-supported rows.  The bandwidth schedule comes
    from ``config.resolved_sigmas(model.visible)``.
    """
    training_targets = np.asarray(training_targets, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    _validate_training_inputs(model, training_targets, validation_targets)

    encoders = model.encoder
    sigmas = config.resolved_sigmas(model.visible)
    if initial_params is None:
        params = jnp.asarray(
            parameters_from_data(
                training_targets.reshape(-1, model.visible),
                model.ansatz.generators,
                n_visible=model.visible,
                seed=seed,
            ),
        )
    else:
        candidate = np.asarray(initial_params, dtype=np.float64)
        if candidate.shape != (model.ansatz.n_parameters,):
            raise ValueError(
                f"initial_params must have shape {(model.ansatz.n_parameters,)}",
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

    # One width per bandwidth for every refreshed bank, so the trainer compiles
    # the step once.  The pilot draw sizes each width.  The margin keeps a
    # redraw rare; a redraw conditions the entry on ``distinct <= width``.
    max_masks: tuple[int, ...] | None = None
    if config.mask_refresh == STEP:
        max_masks = _refreshed_widths(sigmas, config, model, seed)
    training_bank: OperatorBank | None = None
    if config.mask_refresh == CACHE:
        # Retained for the whole run; the loop never rebuilds it.
        training_bank = operator_bank(
            sigmas,
            config.n_ops,
            encoders.n_conditions,
            model.visible,
            seed,
        )
    validation_bank = operator_bank(
        sigmas,
        config.validation_n_ops,
        encoders.n_conditions,
        model.visible,
        seed + 1,
    )

    def training_loss(
        current_params: jnp.ndarray,
        target_batch: jnp.ndarray,
        current_bank: OperatorBank | None,
        step_key: jax.Array,
    ) -> jnp.ndarray:
        if current_bank is None:
            return model.mmd(
                current_params,
                target_batch,
                step_key,
                sigmas=sigmas,
                n_ops=config.n_ops,
                n_samples=config.mc_samples,
                shared_samples=True,
                gemm_dtype=config.gemm_dtype,
                sample_chunk=config.sample_chunk,
            ).loss
        return model.mmd(
            current_params,
            target_batch,
            step_key,
            bank=current_bank,
            n_samples=config.mc_samples,
            shared_samples=True,
            gemm_dtype=config.gemm_dtype,
            sample_chunk=config.sample_chunk,
        ).loss

    # Compile differentiation and the full optimizer update together to avoid
    # dispatching the optimizer operations separately from the estimator.
    @jax.jit
    def training_step(
        current_params: jnp.ndarray,
        current_optimizer_state: optax.OptState,
        target_batch: jnp.ndarray,
        current_bank: OperatorBank | None,
        step_key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, optax.OptState]:
        loss, gradient = jax.value_and_grad(training_loss)(
            current_params,
            target_batch,
            current_bank,
            step_key,
        )
        updates, new_optimizer_state = optimizer.update(
            gradient,
            current_optimizer_state,
            current_params,
        )
        new_params = optax.apply_updates(current_params, updates)
        return loss, new_params, new_optimizer_state

    def validation_loss(current_params: jnp.ndarray) -> jnp.ndarray:
        return model.mmd(
            current_params,
            jnp.asarray(validation_targets),
            jax.random.PRNGKey(seed + 1_000_000),
            bank=validation_bank,
            n_samples=config.validation_mc_samples,
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
        if training_bank is None or config.mask_refresh == STEP:
            training_bank = operator_bank(
                sigmas,
                config.n_ops,
                encoders.n_conditions,
                model.visible,
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
        loss, params, optimizer_state = training_step(
            params,
            optimizer_state,
            jnp.asarray(batch),
            training_bank,
            step_key,
        )
        train_steps.append(step + 1)
        # Defer the device sync. There is no ``float(loss)`` here, so the host
        # bank build for the next step overlaps with the running device program.
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
        best_params=np.asarray(best_params),
        final_params=np.asarray(params),
        best_step=best_step,
        best_validation_loss=best_validation,
        train_steps=np.asarray(train_steps),
        train_losses=np.asarray(jnp.stack(train_losses)),
        validation_steps=np.asarray(validation_steps),
        validation_losses=np.asarray(validation_losses),
    )
