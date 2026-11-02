"""Training loop for shared conditional IQP models.

The model uses one parameter vector for every condition; only the encoder state
changes with the condition.  Every update selects one condition and one
bandwidth uniformly, so the expected loss and gradient of that update equal the
condition-balanced multiscale objective.  The trainer selects checkpoints on a
held-out validation split by the full objective.

Step refresh draws a fixed number of masks directly in JAX for the selected
objective cell; cache refresh selects that cell from one full bank retained for
the fit. Validation uses fixed randomness so checkpoint scores stay comparable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._precision import resolve_estimator_dtype
from .initialization import parameters_from_data
from .loss import (
    _dynamic_encoder_support,
    _fixed_mmd,
    _fixed_mmd_plan,
    _FixedMMDPlan,
)
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
MaskRefresh = Literal["step", "cache"]
PrecisionMode = Literal["fp64", "gemm_f32", "estimator_f32", "full_fp32"]

_BANDWIDTH_SEED_OFFSET = 83_001
_CONDITION_SEED_OFFSET = 191_003


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and Monte Carlo settings.

    Defaults: 2000 updates, batches of 256 per condition, initialization from
    the pooled training data by the mean-covariance prescription, checkpoints
    every 25 steps by validation MMD, and a width-aware bandwidth schedule.
    ``sigmas=None`` resolves through :meth:`resolved_sigmas` to the bodyness
    targets ``(2, n/4, 0.49 n)``.  The mask and Monte Carlo budgets are small,
    so default runs stay short.  ``n_ops`` and ``validation_n_ops`` default to
    512 and ``mc_samples`` and ``validation_mc_samples`` to 1000, the smallest
    calibrated budget passing the estimator gates; raise them toward 2000 masks
    and 8192 reference draws for full-scale runs.  ``mask_refresh="cache"``
    retains one mask bank per fit instead of redrawing every step; both refresh
    modes select one condition-bandwidth objective cell per update.  See
    :func:`train`.

    ``n_ops`` and ``validation_n_ops`` give the number of nonidentity operator
    draws per condition and bandwidth. Step refresh uses direct fixed-width
    sampling; cache refresh uses compressed operator banks.
    The trainer selects one condition and one bandwidth uniformly before every
    update.  Under the default per-step refresh the trainer samples directly
    for that objective cell; cache refresh selects the matching slice from a
    retained bank. In expectation both estimate the same condition-balanced
    arithmetic-mean multiscale objective.

    ``gemm_dtype`` runs the phase contraction in a reduced dtype such as
    ``jnp.float32`` and casts the result back to float64.  This mixed precision
    mode helps on GPUs; the reductions stay float64.

    ``estimator_dtype=jnp.float32`` also runs parity and phase construction in
    single precision and evaluates the ratio-phase product there, in complex64
    only for complex ratios. It returns to float64 before statistics and MMD
    accumulation. It is mutually exclusive with ``gemm_dtype``. ``None``
    retains the default float64 estimator.

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
    estimator_dtype: jnp.dtype | None = None
    precision_mode: PrecisionMode = "full_fp32"
    sample_chunk: int | None = None
    mask_refresh: MaskRefresh = STEP

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
        if self.precision_mode not in ("fp64", "gemm_f32", "estimator_f32", "full_fp32"):
            raise ValueError(f"unknown precision_mode: {self.precision_mode!r}")
        if self.precision_mode == "full_fp32":
            # full_fp32 maps to estimator_f32 for backward compatibility
            # (runs estimator in fp32, returns to fp64 for MMD/optimizer)
            object.__setattr__(self, "estimator_dtype", jnp.float32)
            object.__setattr__(self, "gemm_dtype", None)
        elif self.precision_mode == "fp64":
            object.__setattr__(self, "estimator_dtype", None)
            object.__setattr__(self, "gemm_dtype", None)
        elif self.precision_mode == "gemm_f32":
            object.__setattr__(self, "gemm_dtype", jnp.float32)
            object.__setattr__(self, "estimator_dtype", None)
        elif self.precision_mode == "estimator_f32":
            object.__setattr__(self, "estimator_dtype", jnp.float32)
            object.__setattr__(self, "gemm_dtype", None)
        else:
            resolve_estimator_dtype(self.gemm_dtype, self.estimator_dtype)
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
    best_selection_score: float
    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray
    selection_scores: np.ndarray


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


def _bandwidth_schedule(n_bandwidths: int, steps: int, seed: int) -> np.ndarray:
    """Return independent uniform bandwidth choices for training updates."""
    rng = np.random.default_rng(seed + _BANDWIDTH_SEED_OFFSET)
    return rng.integers(0, n_bandwidths, size=steps)


def _condition_schedule(n_conditions: int, steps: int, seed: int) -> np.ndarray:
    """Return independent uniform condition choices for training updates."""
    rng = np.random.default_rng(seed + _CONDITION_SEED_OFFSET)
    return rng.integers(0, n_conditions, size=steps)


def _select_bandwidth(bank: OperatorBank, bandwidth: int) -> OperatorBank:
    """Return one exact bandwidth slice from a multiscale operator bank."""
    width = bank.width(bandwidth)
    return OperatorBank(
        sigmas=(bank.sigmas[bandwidth],),
        masks=bank.masks[bandwidth : bandwidth + 1, :, :width],
        weights=bank.weights[bandwidth : bandwidth + 1, :, :width],
        n_ops=bank.n_ops,
        n_visible=bank.n_visible,
        n_masks=width,
        widths=(width,),
    )


def _select_condition(bank: OperatorBank, condition: int) -> OperatorBank:
    """Return one exact condition slice from an operator bank."""
    return OperatorBank(
        sigmas=bank.sigmas,
        masks=bank.masks[:, condition : condition + 1],
        weights=bank.weights[:, condition : condition + 1],
        n_ops=bank.n_ops,
        n_visible=bank.n_visible,
        n_masks=bank.n_masks,
        widths=bank.widths,
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


def _initial_parameters(
    model: ConditionalModel,
    training_targets: np.ndarray,
    initial_params: np.ndarray | None,
    seed: int,
) -> jnp.ndarray:
    """Validate explicit initial parameters or derive them from the data."""
    if initial_params is None:
        return jnp.asarray(
            parameters_from_data(
                training_targets.reshape(-1, model.visible),
                model.ansatz.generators,
                n_visible=model.visible,
                seed=seed,
            ),
        )
    candidate = np.asarray(initial_params, dtype=np.float64)
    if candidate.shape != (model.ansatz.n_parameters,):
        raise ValueError(
            f"initial_params must have shape {(model.ansatz.n_parameters,)}",
        )
    if not np.isfinite(candidate).all():
        raise ValueError("initial_params must contain only finite values")
    return jnp.asarray(candidate, dtype=jnp.float64).copy()


def _cached_training_banks(
    sigmas: tuple[float, ...],
    config: TrainingConfig,
    n_conditions: int,
    n_visible: int,
    seed: int,
) -> tuple[tuple[OperatorBank, ...], ...]:
    """Draw the bank retained by cache refresh, sliced per (bandwidth, condition).

    The fit keeps it whole; the loop only selects exact slices.
    """
    cached_bank = operator_bank(sigmas, config.n_ops, n_conditions, n_visible, seed)
    return tuple(
        tuple(
            _select_condition(_select_bandwidth(cached_bank, bandwidth), condition)
            for condition in range(n_conditions)
        )
        for bandwidth in range(cached_bank.n_bandwidths)
    )


def _make_training_step(
    model: ConditionalModel,
    config: TrainingConfig,
    optimizer: optax.GradientTransformation,
    random_key: jax.Array,
    training_targets: jnp.ndarray,
    batch_indices: jnp.ndarray,
    closed_encoder: object | None,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, optax.OptState]]:
    """Compile one update for every bandwidth, sharing or closing the encoder.

    The bandwidth enters as a traced sigma under per-step refresh and through
    the selected bank under cache refresh, so one compiled program serves every
    bandwidth whose bank shape agrees.  The targets and the batch schedule stay
    on the device; the step slices its batch and derives its key from the step
    and condition scalars.  ``closed_encoder`` is ``None`` for pytree encoders,
    which arrive as an argument, and the fixed condition's encoder otherwise.
    """

    def training_loss(
        current_params: jnp.ndarray,
        target_batch: jnp.ndarray,
        current_bank: OperatorBank | None,
        step_key: jax.Array,
        selected_encoder: object,
        sigma: jax.Array | None,
    ) -> jnp.ndarray:
        active_encoder = selected_encoder if closed_encoder is None else closed_encoder
        selected_model = ConditionalModel(
            model.ansatz,
            active_encoder,
            n_visible=model.n_visible,
        )
        mask_source = (
            {"bank": current_bank}
            if current_bank is not None
            else {"sigmas": (sigma,), "n_ops": config.n_ops}
        )
        return selected_model.mmd(
            current_params,
            target_batch,
            step_key,
            **mask_source,
            n_samples=config.mc_samples,
            shared_samples=True,
            gemm_dtype=config.gemm_dtype,
            estimator_dtype=config.estimator_dtype,
            sample_chunk=config.sample_chunk,
        ).loss

    # Compile differentiation and the full optimizer update together to avoid
    # dispatching optimizer operations separately from the estimator.
    @jax.jit
    def training_step(
        current_params: jnp.ndarray,
        current_optimizer_state: optax.OptState,
        step: jax.Array,
        condition: jax.Array,
        current_bank: OperatorBank | None,
        selected_encoder: object,
        sigma: jax.Array | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, optax.OptState]:
        indices = batch_indices[step, condition]
        target_batch = training_targets[condition, indices][None]
        loss, gradient = jax.value_and_grad(training_loss)(
            current_params,
            target_batch,
            current_bank,
            jax.random.fold_in(random_key, step),
            selected_encoder,
            sigma,
        )
        updates, new_optimizer_state = optimizer.update(
            gradient,
            current_optimizer_state,
            current_params,
        )
        new_params = optax.apply_updates(current_params, updates)
        return loss, new_params, new_optimizer_state

    return training_step


def _make_validation(
    model: ConditionalModel,
    config: TrainingConfig,
    validation_targets: np.ndarray,
    sigmas: tuple[float, ...],
    seed: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the fixed-randomness validation MMD as a function of parameters.

    Every checkpoint would redraw the same masks and samples, so a plan keeps
    them, the amplitude ratios, and the target moments for the whole fit and
    only the phases are recomputed.  Without room for the plan, validation
    samples afresh with the same key.
    """
    key = jax.random.PRNGKey(seed + 1_000_000)
    targets = jnp.asarray(validation_targets)
    plan = _fixed_mmd_plan(
        model.encoder,
        targets,
        key,
        sigmas=sigmas,
        n_ops=config.validation_n_ops,
        n_samples=config.validation_mc_samples,
    )
    if plan is not None:

        @jax.jit
        def validation_loss(
            current_params: jnp.ndarray,
            plan: _FixedMMDPlan,
        ) -> jnp.ndarray:
            return _fixed_mmd(
                plan,
                current_params,
                model.generators,
                gemm_dtype=config.gemm_dtype,
                estimator_dtype=config.estimator_dtype,
            ).loss

        return lambda current_params: validation_loss(current_params, plan)

    @jax.jit
    def validation_loss(
        current_params: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> jnp.ndarray:
        return model.mmd(
            current_params,
            targets,
            key,
            sigmas=sigmas,
            n_ops=config.validation_n_ops,
            n_samples=config.validation_mc_samples,
            gemm_dtype=config.gemm_dtype,
            estimator_dtype=config.estimator_dtype,
        ).loss

    return lambda current_params: validation_loss(current_params, targets)


def train(
    model: ConditionalModel,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    config: TrainingConfig,
    initial_params: np.ndarray | None = None,
    selection_metric: Callable[[np.ndarray], float] | None = None,
) -> TrainingResult:
    """Train one shared conditional model with fixed validation randomness.

    ``config.mask_refresh`` selects how the training masks are drawn.  Both
    modes pick one condition and one bandwidth uniformly per update, so each
    update is unbiased for the condition-balanced arithmetic-mean multiscale
    MMD.  ``"step"``, the default, draws ``config.n_ops`` masks directly in JAX
    for the selected objective cell before every compiled step.  ``"cache"``
    draws one full compressed NumPy bank and selects the corresponding cached
    cell for each update.

    Without ``initial_params`` the parameters start from the mean-covariance
    prescription applied to the pooled training conditions, with seeded noise on
    ancilla-supported rows.  The bandwidth schedule comes from
    ``config.resolved_sigmas(model.visible)``. By default, sampled validation
    MMD selects checkpoints. ``selection_metric`` may supply a deterministic
    scalar diagnostic such as exact TVD instead. Validation MMD is still
    recorded at every checkpoint when an alternative selection metric is used.
    """
    training_targets = np.asarray(training_targets, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    _validate_training_inputs(model, training_targets, validation_targets)

    encoders = model.encoder
    n_conditions = encoders.n_conditions
    sigmas = config.resolved_sigmas(model.visible)
    params = _initial_parameters(model, training_targets, initial_params, seed)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    batch_indices = _batch_schedule(
        n_conditions,
        training_targets.shape[1],
        config,
        seed,
    )
    bandwidth_indices = _bandwidth_schedule(len(sigmas), config.steps, seed)
    condition_indices = _condition_schedule(n_conditions, config.steps, seed)
    cached_banks = (
        _cached_training_banks(sigmas, config, n_conditions, model.visible, seed)
        if config.mask_refresh == CACHE
        else None
    )
    sigma_values = tuple(jnp.asarray(sigma, dtype=jnp.float64) for sigma in sigmas)
    selected_encoders = tuple(
        encoders.select(condition) for condition in range(n_conditions)
    )

    device_targets = jnp.asarray(training_targets)
    device_batch_indices = jnp.asarray(batch_indices)

    def make_step(closed_encoder: object | None):
        return _make_training_step(
            model,
            config,
            optimizer,
            jax.random.PRNGKey(seed),
            device_targets,
            device_batch_indices,
            closed_encoder,
        )

    # Pytree encoders share one compiled step and arrive as an argument; any
    # other encoder is closed over, one compiled step per condition.
    if _dynamic_encoder_support(encoders):
        shared_step = make_step(None)
        training_steps = (shared_step,) * n_conditions
        step_encoders: tuple[object, ...] = selected_encoders
    else:
        training_steps = tuple(make_step(encoder) for encoder in selected_encoders)
        step_encoders = ((),) * n_conditions
    evaluate_validation = _make_validation(
        model,
        config,
        validation_targets,
        sigmas,
        seed,
    )

    def score(current_params: jnp.ndarray) -> tuple[float, float]:
        validation = float(evaluate_validation(current_params))
        if selection_metric is None:
            return validation, validation
        return validation, float(selection_metric(np.asarray(current_params)))

    train_losses = []
    best_params = params
    best_step = 0
    best_validation, best_selection = score(params)
    validation_steps = [0]
    validation_losses = [best_validation]
    selection_scores = [best_selection]
    for step in range(config.steps):
        bandwidth = int(bandwidth_indices[step])
        condition = int(condition_indices[step])
        if cached_banks is None:
            training_bank, sigma = None, sigma_values[bandwidth]
        else:
            training_bank, sigma = cached_banks[bandwidth][condition], None
        loss, params, optimizer_state = training_steps[condition](
            params,
            optimizer_state,
            step,
            condition,
            training_bank,
            step_encoders[condition],
            sigma,
        )
        # Defer the device sync so Python does not block on each optimizer update.
        train_losses.append(loss)

        completed_step = step + 1
        if completed_step % config.validation_interval == 0:
            current_validation, current_selection = score(params)
            validation_steps.append(completed_step)
            validation_losses.append(current_validation)
            selection_scores.append(current_selection)
            if current_selection < best_selection:
                best_validation = current_validation
                best_selection = current_selection
                best_params = params
                best_step = completed_step

    return TrainingResult(
        seed=seed,
        best_params=np.asarray(best_params),
        final_params=np.asarray(params),
        best_step=best_step,
        best_validation_loss=best_validation,
        best_selection_score=best_selection,
        train_steps=np.arange(1, config.steps + 1),
        train_losses=np.asarray(jnp.stack(train_losses)),
        validation_steps=np.asarray(validation_steps),
        validation_losses=np.asarray(validation_losses),
        selection_scores=np.asarray(selection_scores),
    )
