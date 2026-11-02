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

from ._precision import resolve_precision
from .encoders.base import _select_condition
from .initialization import parameters_from_data
from .loss import _Entries, _mmd_entries, _mmd_from_entries, mmd
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

    ``precision`` names a policy from :data:`iqp_cond.PRECISIONS`.  ``"fp64"``
    is the default.  ``"gemm_f32"`` runs only the final phase contraction in
    float32.  ``"estimator_f32"`` also builds parities and phases and evaluates
    the ratio-phase product in float32, returning to float64 for the statistics.
    ``"fp32"`` keeps the statistics, the loss, the parameters and the optimizer
    state in float32 as well.

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
    precision: str = "fp64"
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
        resolve_precision(self.precision)
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


def _cell_bank(
    bank: OperatorBank,
    bandwidth: int,
    condition: jax.Array,
) -> OperatorBank:
    """Return the cached bank's (bandwidth, condition) cell at its exact width."""
    width = bank.width(bandwidth)
    return OperatorBank(
        sigmas=(bank.sigmas[bandwidth],),
        masks=bank.masks[bandwidth, condition, :width][None, None],
        weights=bank.weights[bandwidth, condition, :width][None, None],
        n_ops=bank.n_ops,
        n_visible=bank.n_visible,
        n_masks=width,
        widths=(width,),
    )


def _make_training_steps(
    model: ConditionalModel,
    config: TrainingConfig,
    optimizer: optax.GradientTransformation,
    random_key: jax.Array,
    training_targets: jnp.ndarray,
    batch_indices: jnp.ndarray,
    sigmas: jnp.ndarray,
) -> Callable[..., tuple[jnp.ndarray, optax.OptState, jnp.ndarray]]:
    """Compile a scan of updates, one (condition, bandwidth) cell per update.

    The condition indexes the encoder leaves, the targets, and the batch
    schedule on the device, so one program serves every condition.  Under
    per-step refresh the bandwidth indexes ``sigmas``; under cache refresh it
    switches between the bank's bandwidths, whose evaluated widths differ.
    """

    def cell_loss(
        params: jnp.ndarray,
        encoders: object,
        bank: OperatorBank | None,
        step: jax.Array,
        condition: jax.Array,
        bandwidth: jax.Array,
    ) -> jnp.ndarray:
        encoder = _select_condition(encoders, condition)
        target_batch = training_targets[condition, batch_indices[step, condition]][None]

        def loss(**mask_source: object) -> jnp.ndarray:
            return mmd(
                params,
                model.generators,
                encoder,
                target_batch,
                jax.random.fold_in(random_key, step),
                **mask_source,
                n_samples=config.mc_samples,
                shared_samples=True,
                precision=config.precision,
                sample_chunk=config.sample_chunk,
            ).loss

        if bank is None:
            return loss(sigmas=(sigmas[bandwidth],), n_ops=config.n_ops)
        return jax.lax.switch(
            bandwidth,
            [
                lambda index=index: loss(bank=_cell_bank(bank, index, condition))
                for index in range(bank.n_bandwidths)
            ],
        )

    @jax.jit
    def training_steps(
        params: jnp.ndarray,
        optimizer_state: optax.OptState,
        encoders: object,
        bank: OperatorBank | None,
        cells: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
        def update(carry, cell):
            current_params, current_state = carry
            loss, gradient = jax.value_and_grad(cell_loss)(
                current_params,
                encoders,
                bank,
                *cell,
            )
            updates, new_state = optimizer.update(
                gradient,
                current_state,
                current_params,
            )
            return (optax.apply_updates(current_params, updates), new_state), loss

        (params, optimizer_state), losses = jax.lax.scan(
            update,
            (params, optimizer_state),
            cells,
        )
        return params, optimizer_state, losses

    return training_steps


def _make_validation(
    model: ConditionalModel,
    config: TrainingConfig,
    validation_targets: np.ndarray,
    sigmas: tuple[float, ...],
    seed: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the fixed-randomness validation MMD as a function of parameters.

    Every checkpoint would redraw the same masks and samples, so they are drawn
    once for the fit; the ratios and phases are recomputed at each checkpoint.
    """
    targets = jnp.asarray(validation_targets)
    entries = _mmd_entries(
        model.encoder,
        targets,
        jax.random.PRNGKey(seed + 1_000_000),
        bank=None,
        sigmas=sigmas,
        n_ops=config.validation_n_ops,
        n_samples=config.validation_mc_samples,
        shared_samples=False,
    )

    @jax.jit
    def validation_loss(
        current_params: jnp.ndarray,
        encoders: object,
        entries: tuple[_Entries, ...],
    ) -> jnp.ndarray:
        return _mmd_from_entries(
            current_params,
            model.generators,
            encoders,
            entries,
            precision=config.precision,
        ).loss

    return lambda current_params: validation_loss(
        current_params,
        model.encoder,
        entries,
    )


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
    # The parameters, and with them the optimizer state, live in the policy's
    # accumulation dtype.
    params = _initial_parameters(model, training_targets, initial_params, seed)
    params = params.astype(resolve_precision(config.precision).accumulate)
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
    bank = (
        operator_bank(sigmas, config.n_ops, n_conditions, model.visible, seed)
        if config.mask_refresh == CACHE
        else None
    )
    training_steps = _make_training_steps(
        model,
        config,
        optimizer,
        jax.random.PRNGKey(seed),
        jnp.asarray(training_targets),
        jnp.asarray(batch_indices),
        jnp.asarray(sigmas, dtype=jnp.float64),
    )
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
    for start in range(0, config.steps, config.validation_interval):
        stop = min(start + config.validation_interval, config.steps)
        params, optimizer_state, losses = training_steps(
            params,
            optimizer_state,
            encoders,
            bank,
            (
                jnp.arange(start, stop),
                jnp.asarray(condition_indices[start:stop]),
                jnp.asarray(bandwidth_indices[start:stop]),
            ),
        )
        # Defer the device sync so Python does not block on each interval.
        train_losses.append(losses)
        if stop % config.validation_interval == 0:
            current_validation, current_selection = score(params)
            validation_steps.append(stop)
            validation_losses.append(current_validation)
            selection_scores.append(current_selection)
            if current_selection < best_selection:
                best_validation = current_validation
                best_selection = current_selection
                best_params = params
                best_step = stop

    return TrainingResult(
        seed=seed,
        best_params=np.asarray(best_params),
        final_params=np.asarray(params),
        best_step=best_step,
        best_validation_loss=best_validation,
        best_selection_score=best_selection,
        train_steps=np.arange(1, config.steps + 1),
        train_losses=np.asarray(jnp.concatenate(train_losses)),
        validation_steps=np.asarray(validation_steps),
        validation_losses=np.asarray(validation_losses),
        selection_scores=np.asarray(selection_scores),
    )
