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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._precision import Precision, _rounding_key, resolve_precision
from .encoders import Encoder
from .encoders.base import _check_condition_pytree, _select_condition
from .initialization import parameters_from_data
from .loss import TargetMoments, _Entries, _mmd_entries, _mmd_from_entries, mmd
from .model import ConditionalModel
from .operators import OperatorBank, expected_bodyness_sigmas, operator_bank

if TYPE_CHECKING:
    from .guidelines import Measurement

__all__ = [
    "CACHE",
    "STEP",
    "Adaptation",
    "TrainingConfig",
    "TrainingResult",
    "train",
]

STEP = "step"
CACHE = "cache"
MaskRefresh = Literal["step", "cache"]

_BANDWIDTH_SEED_OFFSET = 83_001
_CONDITION_SEED_OFFSET = 191_003
_MEASUREMENT_SEED_OFFSET = 307_007


@dataclass(frozen=True)
class Adaptation:
    """Remeasure the sample and mask counts during training.

    Every ``every`` updates, from the first, :func:`train` measures the step
    noise at the current parameters with :func:`iqp_cond.measure`, on
    ``masks`` masks per table, and trains the following updates at its
    ``samples`` and ``masks`` in place of ``mc_samples`` and ``n_ops``;
    ``r_cap``, a power of two, caps the mask count.  An invalid estimate keeps
    the previous settings, which start from ``mc_samples`` and ``n_ops``.  The
    measurements draw from their own random stream, so the training draws are
    unchanged.
    """

    every: int = 100
    r_cap: int = 65_536
    masks: int = 1_024

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError("every must be positive")
        if self.r_cap < 1 or self.r_cap & (self.r_cap - 1):
            raise ValueError("r_cap must be a positive power of two")
        if self.masks < 2:
            raise ValueError("masks must be at least two")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and Monte Carlo settings.

    Defaults: 2000 updates, batches of 256 per condition, initialization from
    the pooled training data by the mean-covariance prescription, checkpoints
    every 25 steps by validation MMD, and a width-aware bandwidth schedule.
    ``sigmas=None`` resolves through :meth:`resolved_sigmas` to the bodyness
    targets ``(2, n/4, 0.49 n)``; explicit ``sigmas`` are scalars or, for the
    anisotropic kernel, vectors of one bandwidth per visible qubit, all of one
    shape.  The mask and Monte Carlo budgets are small, so default runs stay
    short: ``n_ops`` and ``validation_n_ops`` default to 512 and
    ``mc_samples`` and ``validation_mc_samples`` to 1000.
    :func:`iqp_cond.measure` and ``adapt`` set the training budgets from the
    step noise.  ``mask_refresh="cache"`` retains one mask bank per fit
    instead of redrawing every step; both refresh modes select one
    condition-bandwidth objective cell per update.  See :func:`train`.

    ``n_ops`` and ``validation_n_ops`` give the number of nonidentity operator
    draws per condition and bandwidth. Step refresh uses direct fixed-width
    sampling; cache refresh uses compressed operator banks.
    The trainer selects one condition and one bandwidth uniformly before every
    update.  Under the default per-step refresh the trainer samples directly
    for that objective cell; cache refresh selects the matching slice from a
    retained bank. In expectation both estimate the same condition-balanced
    arithmetic-mean multiscale objective.

    ``precision`` is a :class:`iqp_cond.Precision` or the name of a preset in
    :data:`iqp_cond.PRECISIONS`, ``"fp64"`` by default.  It sets the format of
    each operation of the loss; the parameters and the optimizer state stay
    float64 under every precision.  When ``precision.loss_scaled`` (fp16 in
    the uniform variant), the loss is scaled before differentiation, starting
    from ``initial_loss_scale``: a non-finite gradient skips the update and
    halves the scale, and the scale doubles after 2000 consecutive finite
    updates.  The scale itself is the first cotangent the fp16 rounders see,
    so it must start below fp16's largest value, 65504.

    ``sample_chunk`` streams the Monte Carlo samples in slices, bounding the live
    ``(samples, masks, qubits)`` ratio tensor.  It is faster at large sample
    budgets and slightly slower for small ones, so it is off by default.

    ``adapt``, an :class:`Adaptation`, sets ``mc_samples`` and ``n_ops`` from
    the measured step noise during training.  It needs step refresh and
    raw-sample targets, and ``adapt.every`` must be a multiple of
    ``validation_interval`` and divide ``steps``.
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
    precision: str | Precision = "fp64"
    initial_loss_scale: float = 2.0**15
    sample_chunk: int | None = None
    mask_refresh: MaskRefresh = STEP
    adapt: Adaptation | None = None

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
                np.ndim(sigma) > 1
                or not np.all(np.isfinite(np.asarray(sigma, dtype=np.float64)))
                or np.any(np.asarray(sigma, dtype=np.float64) <= 0.0)
                for sigma in self.sigmas
            )
        ):
            raise ValueError("sigmas must be positive and finite")
        if (
            self.sigmas is not None
            and len({np.shape(sigma) for sigma in self.sigmas}) > 1
        ):
            raise ValueError("sigmas must be all scalars or all vectors of one length")
        if self.n_ops < 1 or self.validation_n_ops < 1:
            raise ValueError("n_ops and validation_n_ops must be positive")
        if self.mc_samples < 2 or self.validation_mc_samples < 2:
            raise ValueError(
                "mc_samples and validation_mc_samples must be at least two",
            )
        resolve_precision(self.precision)
        if not self.initial_loss_scale > 0:
            raise ValueError("initial_loss_scale must be positive")
        if self.sample_chunk is not None and self.sample_chunk < 1:
            raise ValueError("sample_chunk must be positive")
        if self.mask_refresh not in (STEP, CACHE):
            raise ValueError(
                f"unknown mask_refresh: {self.mask_refresh!r}; "
                f"expected {STEP!r} or {CACHE!r}",
            )
        if self.adapt is not None:
            if self.mask_refresh != STEP:
                raise ValueError("adapt needs step mask refresh")
            if self.adapt.every % self.validation_interval:
                raise ValueError(
                    "adapt.every must be a multiple of validation_interval"
                )
            if self.steps % self.adapt.every:
                raise ValueError("adapt.every must divide steps")


@dataclass(frozen=True)
class TrainingResult:
    """Best parameters and scalar training history.

    ``loss_scales`` holds the loss scale used at each update (1 unless the
    precision is loss-scaled), and ``skipped_steps`` the 1-based updates that
    were skipped because their gradient overflowed.  ``checkpoint_params``
    holds the parameters at each of ``validation_steps`` and
    ``selection_scores`` the checkpoint scores, the validation MMD unless
    ``train`` was given a ``selection_metric``.  Under
    ``config.adapt``, ``measurements`` holds each
    :class:`iqp_cond.Measurement`, taken before the 0-based update in
    ``measurement_steps``; both are empty otherwise.
    """

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
    loss_scales: np.ndarray
    skipped_steps: np.ndarray
    checkpoint_params: np.ndarray
    measurement_steps: np.ndarray
    measurements: tuple[Measurement, ...]


_LOSS_SCALE_GROWTH_INTERVAL = 2000


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


def _validate_moment_targets(model: ConditionalModel, name: str, targets) -> None:
    if targets.bank.n_conditions != model.encoder.n_conditions:
        raise ValueError(
            f"model covers {model.encoder.n_conditions} conditions but {name} "
            f"have {targets.bank.n_conditions}",
        )
    if targets.bank.n_visible != model.visible:
        raise ValueError(
            f"model declares {model.visible} visible qubits but {name} "
            f"have {targets.bank.n_visible}",
        )


def _validate_training_inputs(
    model: ConditionalModel,
    training_targets: np.ndarray | TargetMoments,
    validation_targets: np.ndarray | TargetMoments,
) -> None:
    """Check the shared model/data conventions."""
    sources = (
        ("training_targets", training_targets),
        ("validation_targets", validation_targets),
    )
    if any(isinstance(targets, TargetMoments) for _, targets in sources):
        for name, targets in sources:
            if not isinstance(targets, TargetMoments):
                raise ValueError("training and validation targets must both be moments")
            _validate_moment_targets(model, name, targets)
        _check_condition_pytree(model.encoder)
        return
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
    _check_condition_pytree(model.encoder)


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
    sigmas: jnp.ndarray,
    n_samples: int,
    n_ops: int,
) -> Callable:
    """Compile a scan of updates, one (condition, bandwidth) cell per update.

    The condition indexes the encoder leaves, the targets, and the batch
    rows on the device, so one program serves every condition.  Under
    per-step refresh the bandwidth indexes ``sigmas``; under cache refresh it
    switches between the bank's bandwidths, whose evaluated widths differ.
    ``n_samples`` and ``n_ops`` fix the program's shapes, so each setting
    compiles once.
    """

    def cell_loss(
        params: jnp.ndarray,
        encoders: Encoder,
        bank: OperatorBank | TargetMoments | None,
        targets: jnp.ndarray | None,
        step: jax.Array,
        condition: jax.Array,
        bandwidth: jax.Array,
        batch: jax.Array | None,
    ) -> jnp.ndarray:
        encoder = _select_condition(encoders, condition)
        if isinstance(bank, TargetMoments):
            return _moment_cell_loss(params, encoder, bank, step, condition, bandwidth)
        target_batch = targets[condition, batch[condition]][None]

        def loss(**mask_source: object) -> jnp.ndarray:
            return mmd(
                params,
                model.generators,
                encoder,
                target_batch,
                jax.random.fold_in(random_key, step),
                **mask_source,
                n_samples=n_samples,
                # One cell has one condition and one bandwidth, so sharing only
                # fixes the key layout of the cached-bank cell.
                shared_samples=True,
                precision=config.precision,
                sample_chunk=config.sample_chunk,
            ).loss

        if bank is None:
            return loss(sigmas=(sigmas[bandwidth],), n_ops=n_ops)
        return jax.lax.switch(
            bandwidth,
            [
                lambda index=index: loss(bank=_cell_bank(bank, index, condition))
                for index in range(bank.n_bandwidths)
            ],
        )

    def _moment_cell_loss(params, encoder, targets, step, condition, bandwidth):
        """One (condition, bandwidth) cell of a moment target.

        Cache refresh evaluates the pool's cell; step refresh draws
        ``n_ops`` of its masks in proportion to their weights, each
        weighted ``q / n_ops`` with ``q`` the cell's nonidentity mass.
        """
        pool = targets.bank

        def cell(index: int) -> jnp.ndarray:
            width = pool.width(index)
            masks = pool.masks[index, condition, :width]
            weights = pool.weights[index, condition, :width]
            moments = targets.moments[index, condition, :width]
            if config.mask_refresh == STEP:
                mask_key = jax.random.fold_in(jax.random.fold_in(random_key, step), 1)
                mass = jnp.sum(weights)
                drawn = jax.random.choice(mask_key, width, (n_ops,), p=weights / mass)
                masks, moments = masks[drawn], moments[drawn]
                weights = jnp.full(n_ops, mass / n_ops)
                width = n_ops
            cell_bank = OperatorBank(
                sigmas=(pool.sigmas[index],),
                masks=masks[None, None],
                weights=weights[None, None],
                n_ops=pool.n_ops,
                n_visible=pool.n_visible,
                n_masks=width,
                widths=(width,),
            )
            return mmd(
                params,
                model.generators,
                encoder,
                TargetMoments(cell_bank, moments[None, None]),
                jax.random.fold_in(random_key, step),
                n_samples=n_samples,
                shared_samples=True,
                precision=config.precision,
                sample_chunk=config.sample_chunk,
            ).loss

        return jax.lax.switch(
            bandwidth,
            [lambda index=index: cell(index) for index in range(pool.n_bandwidths)],
        )

    loss_scaled = resolve_precision(config.precision).loss_scaled

    @jax.jit
    def training_steps(
        params: jnp.ndarray,
        optimizer_state: optax.OptState,
        scaling: tuple[jax.Array, jax.Array],
        encoders: Encoder,
        bank: OperatorBank | None,
        targets: jnp.ndarray | None,
        cells: tuple[jax.Array, jax.Array, jax.Array, jax.Array | None],
    ):
        def update(carry, cell):
            current_params, current_state, (scale, finite_run) = carry
            if not loss_scaled:
                loss, gradient = jax.value_and_grad(cell_loss)(
                    current_params,
                    encoders,
                    bank,
                    targets,
                    *cell,
                )
                updates, new_state = optimizer.update(
                    gradient,
                    current_state,
                    current_params,
                )
                new_params = optax.apply_updates(current_params, updates)
                return (new_params, new_state, (scale, finite_run)), (
                    loss,
                    scale,
                    jnp.array(True),
                )
            scaled_loss, scaled_gradient = jax.value_and_grad(
                lambda *arguments: cell_loss(*arguments) * scale
            )(current_params, encoders, bank, targets, *cell)
            gradient = scaled_gradient / scale
            finite = jnp.all(jnp.isfinite(gradient))
            updates, new_state = optimizer.update(
                jnp.where(finite, gradient, 0.0),
                current_state,
                current_params,
            )
            new_params = jnp.where(
                finite, optax.apply_updates(current_params, updates), current_params
            )
            new_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old), new_state, current_state
            )
            grown = finite_run + 1 >= _LOSS_SCALE_GROWTH_INTERVAL
            new_scale = jnp.where(
                finite, jnp.where(grown, scale * 2.0, scale), scale / 2.0
            )
            new_run = jnp.where(finite & ~grown, finite_run + 1, 0)
            return (new_params, new_state, (new_scale, new_run)), (
                scaled_loss / scale,
                scale,
                finite,
            )

        (params, optimizer_state, scaling), (losses, scales, finite) = jax.lax.scan(
            update,
            (params, optimizer_state, scaling),
            cells,
        )
        return params, optimizer_state, scaling, losses, scales, finite

    return training_steps


def _make_validation(
    model: ConditionalModel,
    config: TrainingConfig,
    validation_targets: np.ndarray | TargetMoments,
    sigmas: tuple[float, ...],
    seed: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the fixed-randomness validation MMD as a function of parameters.

    Every checkpoint would redraw the same masks and samples, so they are drawn
    once for the fit; the ratios and phases are recomputed at each checkpoint.
    Moment targets are evaluated on every mask of their own bank.
    """
    moments = isinstance(validation_targets, TargetMoments)
    key = jax.random.PRNGKey(seed + 1_000_000)
    entries = _mmd_entries(
        model.encoder,
        validation_targets if moments else jnp.asarray(validation_targets),
        key,
        bank=validation_targets.bank if moments else None,
        sigmas=None if moments else sigmas,
        n_ops=None if moments else config.validation_n_ops,
        n_samples=config.validation_mc_samples,
        shared_samples=False,
    )

    @jax.jit
    def validation_loss(
        current_params: jnp.ndarray,
        encoders: Encoder,
        entries: tuple[_Entries, ...],
    ) -> jnp.ndarray:
        return _mmd_from_entries(
            current_params,
            model.generators,
            encoders,
            entries,
            precision=config.precision,
            # Fixed bits, like the fixed entries, keep validation comparable.
            rounding_key=_rounding_key(key),
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

    Both target arguments may instead be :class:`iqp_cond.loss.TargetMoments`,
    which carry their own masks: cache refresh evaluates the training moments'
    cells, step refresh draws ``config.n_ops`` of a cell's masks per update in
    proportion to their weights, and validation evaluates every mask of the
    validation moments.  The bandwidths then come from the training bank, and
    ``initial_params`` is required.

    With ``config.adapt`` the sample and mask counts follow the measured step
    noise; see :class:`Adaptation`.
    """
    moments = isinstance(training_targets, TargetMoments)
    if not moments:
        training_targets = np.asarray(training_targets, dtype=np.float64)
        validation_targets = np.asarray(validation_targets, dtype=np.float64)
    _validate_training_inputs(model, training_targets, validation_targets)
    if moments and initial_params is None:
        raise ValueError("moment targets need initial_params")
    if moments and config.adapt is not None:
        raise ValueError("adapt needs raw-sample targets")

    encoders = model.encoder
    n_conditions = encoders.n_conditions
    sigmas = (
        training_targets.bank.sigmas
        if moments
        else config.resolved_sigmas(model.visible)
    )
    params = _initial_parameters(model, training_targets, initial_params, seed)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    bandwidth_indices = _bandwidth_schedule(len(sigmas), config.steps, seed)
    condition_indices = _condition_schedule(n_conditions, config.steps, seed)
    if moments:
        # The moment source carries its own masks and replaces the batches.
        bank, sample_targets, batch_indices = training_targets, None, None
    else:
        bank = (
            operator_bank(sigmas, config.n_ops, n_conditions, model.visible, seed)
            if config.mask_refresh == CACHE
            else None
        )
        sample_targets = jnp.asarray(training_targets)
        batch_indices = jnp.asarray(
            _batch_schedule(n_conditions, training_targets.shape[1], config, seed)
        )
    compiled_steps = {}

    def training_steps(n_samples: int, n_ops: int) -> Callable:
        if (n_samples, n_ops) not in compiled_steps:
            compiled_steps[n_samples, n_ops] = _make_training_steps(
                model,
                config,
                optimizer,
                jax.random.PRNGKey(seed),
                jnp.asarray(sigmas, dtype=jnp.float64),
                n_samples,
                n_ops,
            )
        return compiled_steps[n_samples, n_ops]

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

    train_losses, loss_scales, finite_steps = [], [], []
    initial_scale = (
        config.initial_loss_scale
        if resolve_precision(config.precision).loss_scaled
        else 1.0
    )
    scaling = (jnp.asarray(initial_scale, jnp.float64), jnp.asarray(0))
    best_params = params
    best_step = 0
    best_validation, best_selection = score(params)
    validation_steps = [0]
    checkpoint_params = [params]
    validation_losses = [best_validation]
    selection_scores = [best_selection]
    n_samples, n_ops = config.mc_samples, config.n_ops
    measurement_steps, measurements = [], []
    for start in range(0, config.steps, config.validation_interval):
        stop = min(start + config.validation_interval, config.steps)
        if config.adapt is not None and start % config.adapt.every == 0:
            # Imported here: the guidelines module builds on this one.
            from .guidelines import _measure

            measurement = _measure(
                model,
                params,
                training_targets,
                config,
                jax.random.fold_in(
                    jax.random.PRNGKey(seed + _MEASUREMENT_SEED_OFFSET), start
                ),
                config.adapt.masks,
                config.adapt.r_cap,
                (n_samples, n_ops),
            )
            n_samples, n_ops = measurement.samples, measurement.masks
            measurement_steps.append(start)
            measurements.append(measurement)
        params, optimizer_state, scaling, losses, scales, finite = training_steps(
            n_samples, n_ops
        )(
            params,
            optimizer_state,
            scaling,
            encoders,
            bank,
            sample_targets,
            (
                jnp.arange(start, stop),
                jnp.asarray(condition_indices[start:stop]),
                jnp.asarray(bandwidth_indices[start:stop]),
                None if batch_indices is None else batch_indices[start:stop],
            ),
        )
        train_losses.append(losses)
        loss_scales.append(scales)
        finite_steps.append(finite)
        if stop % config.validation_interval == 0:
            current_validation, current_selection = score(params)
            validation_steps.append(stop)
            checkpoint_params.append(params)
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
        loss_scales=np.asarray(jnp.concatenate(loss_scales)),
        checkpoint_params=np.asarray(jnp.stack(checkpoint_params)),
        skipped_steps=np.flatnonzero(~np.asarray(jnp.concatenate(finite_steps))) + 1,
        measurement_steps=np.asarray(measurement_steps, dtype=np.int64),
        measurements=tuple(measurements),
    )
