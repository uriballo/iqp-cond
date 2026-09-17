"""Training loop for shared conditional IQP models.

One parameter vector serves every condition; only the reference state changes
with the condition.  Updates are condition-balanced (each condition contributes
the same number of samples per step) and checkpoints are selected on a held-out
validation split by the same MMD objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import Ansatz
from .loss import (
    COMPRESSED,
    PLAIN,
    OperatorBank,
    conditional_mmd,
    operator_bank,
    sampled_mmd,
)
from .references import ReferenceStates

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "initialize_parameters",
    "train",
]

EPOCH = "epoch"
FIT = "fit"

#: Headroom over the pilot distinct-mask count when fixing the bank width.
#: Distinct counts concentrate across banks, so a small margin covers almost
#: every epoch; a bank that still overflows widens the cap once (see ``train``)
#: rather than failing the fit.  Kept small because every padded row is
#: estimated, so the margin is paid in compute on every step.
_BANK_HEADROOM = 1.10


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and Monte Carlo settings.

    The defaults reproduce the published conditional-IQP experiments; they are
    a starting point, not a tuned recommendation for other problems.

    With the default :data:`iqp_cond.loss.COMPRESSED` estimator, ``n_ops`` and
    ``validation_n_ops`` are the number of nonidentity operator *draws* per
    condition and bandwidth that feed the operator bank.  With
    ``estimator=PLAIN`` they are redrawn at every step and every condition is
    evaluated together, which costs more memory and, at the larger bandwidths,
    more variance for the same budget.

    ``mask_refresh="epoch"`` (the default) redraws the training bank at the start
    of every epoch, so each update's objective is unbiased for the true kernel
    MMD rather than conditioned on one realized bank.  ``mask_refresh="fit"``
    draws once and retains it, which is the protocol the published campaign used;
    it saves the redraw cost but trains against a fixed perturbation of the
    objective.  Validation always uses a bank drawn once, so checkpoint
    selection stays comparable across steps.
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


def train(
    ansatz: Ansatz,
    references: ReferenceStates,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: TrainingConfig,
    *,
    initial_params: np.ndarray | None = None,
    estimator: str = COMPRESSED,
    mask_refresh: str = EPOCH,
) -> TrainingResult:
    """Train one shared conditional model with fixed validation randomness.

    ``estimator`` selects the MMD estimator.  ``COMPRESSED`` (default) builds
    compressed operator banks, evaluates a single condition at a time, and skips
    the identity mask; ``PLAIN`` keeps the original estimator for reference and
    for reproducing older runs.

    ``mask_refresh`` selects how often the training bank is redrawn: ``"epoch"``
    (default) at the start of every epoch, ``"fit"`` once for the whole run.
    """
    training_targets = np.asarray(training_targets, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    if training_targets.ndim != 3:
        raise ValueError("training_targets must be rank three")
    if references.n_conditions != training_targets.shape[0]:
        raise ValueError(
            f"reference covers {references.n_conditions} conditions but targets "
            f"have {training_targets.shape[0]}",
        )
    if references.n_qubits != ansatz.n_qubits:
        raise ValueError(
            f"reference acts on {references.n_qubits} qubits but the ansatz has "
            f"{ansatz.n_qubits}",
        )
    if validation_targets.shape[0] != training_targets.shape[0]:
        raise ValueError("training and validation conditions must align")
    if validation_targets.shape[2] > ansatz.n_qubits:
        raise ValueError(
            "validation target qubit count cannot exceed ansatz qubit count",
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

    bank: OperatorBank | None = None
    validation_bank: OperatorBank | None = None
    bank_widths: tuple[int, ...] = ()
    n_banks = 0
    steps_per_epoch = 1
    if estimator == COMPRESSED:
        if mask_refresh not in (EPOCH, FIT):
            raise ValueError(
                f"unknown mask_refresh: {mask_refresh!r}; "
                f"expected {EPOCH!r} or {FIT!r}",
            )
        n_visible = training_targets.shape[2]
        steps_per_epoch = max(1, training_targets.shape[1] // config.batch_size)
        # One width *per bandwidth* for every training bank, so refreshed banks
        # keep every shape and the compiled step is built once, while a
        # bandwidth whose mask law cannot reach many masks does not pay for the
        # widest one.  The pilot draw sizes each width.
        pilot = operator_bank(
            config.sigmas,
            config.n_ops,
            references.n_conditions,
            n_visible,
            seed,
        )
        bank_widths = tuple(
            int(width * _BANK_HEADROOM) + 1 for width in pilot.widths
        )
        if mask_refresh == FIT:
            # Retained for the whole run; the loop never rebuilds it.
            bank = operator_bank(
                config.sigmas,
                config.n_ops,
                references.n_conditions,
                n_visible,
                seed,
                max_masks=bank_widths,
            )
            n_banks = 1
        validation_bank = operator_bank(
            config.sigmas,
            config.validation_n_ops,
            references.n_conditions,
            n_visible,
            seed + 1,
        )
    elif estimator != PLAIN:
        raise ValueError(
            f"unknown estimator: {estimator!r}; expected {COMPRESSED!r} or {PLAIN!r}",
        )

    def rebuild_bank(epoch: int, widths: tuple[int, ...]) -> OperatorBank:
        return operator_bank(
            config.sigmas,
            config.n_ops,
            references.n_conditions,
            training_targets.shape[2],
            seed + 1_000_003 * (epoch + 1),
            max_masks=widths,
        )

    if estimator == PLAIN:

        def training_loss(
            current_params: jnp.ndarray,
            target_batch: jnp.ndarray,
            step_key: jax.Array,
        ) -> jnp.ndarray:
            return conditional_mmd(
                current_params,
                generators,
                references,
                target_batch,
                config.sigmas,
                config.n_ops,
                config.n_iqp_samples,
                step_key,
            ).loss

        value_and_gradient = jax.jit(jax.value_and_grad(training_loss))

        def run_step(params, batch, step_key, current_bank):
            return value_and_gradient(params, batch, step_key)

    else:

        def banked_loss(
            current_params: jnp.ndarray,
            target_batch: jnp.ndarray,
            step_key: jax.Array,
            current_bank: OperatorBank,
        ) -> jnp.ndarray:
            return sampled_mmd(
                current_params,
                generators,
                references,
                target_batch,
                current_bank,
                config.n_iqp_samples,
                step_key,
            ).loss

        banked_value_and_gradient = jax.jit(jax.value_and_grad(banked_loss))

        def run_step(params, batch, step_key, current_bank):
            return banked_value_and_gradient(params, batch, step_key, current_bank)

    if validation_bank is None:

        def validation_loss(current_params: jnp.ndarray) -> jnp.ndarray:
            return conditional_mmd(
                current_params,
                generators,
                references,
                jnp.asarray(validation_targets),
                config.sigmas,
                config.validation_n_ops,
                config.validation_n_iqp_samples,
                jax.random.PRNGKey(seed + 1_000_000),
            ).loss

        evaluate_validation = jax.jit(validation_loss)
    else:

        def banked_validation(
            current_params: jnp.ndarray,
            current_bank: OperatorBank,
        ) -> jnp.ndarray:
            return sampled_mmd(
                current_params,
                generators,
                references,
                jnp.asarray(validation_targets),
                current_bank,
                config.validation_n_iqp_samples,
                jax.random.PRNGKey(seed + 1_000_000),
            ).loss

        evaluate_validation = jax.jit(banked_validation)

    train_steps = []
    train_losses = []
    validation_steps = []
    validation_losses = []
    best_params = params
    best_step = 0
    best_validation = float(
        evaluate_validation(params, validation_bank)
        if validation_bank is not None
        else evaluate_validation(params),
    )

    validation_steps.append(0)
    validation_losses.append(best_validation)
    for step in range(config.steps):
        if estimator == COMPRESSED and (
            bank is None or (mask_refresh == EPOCH and step % steps_per_epoch == 0)
        ):
            epoch = step // steps_per_epoch
            try:
                bank = rebuild_bank(epoch, bank_widths)
            except RuntimeError:
                # An epoch drew more distinct masks than the caps allow.  Widen
                # once and continue: the step recompiles for the new shapes, but
                # the fit survives instead of failing.
                bank_widths = tuple(width * 2 for width in bank_widths)
                bank = rebuild_bank(epoch, bank_widths)
            n_banks += 1
        indices = batch_indices[step]
        batch = np.stack(
            [
                training_targets[condition, indices[condition]]
                for condition in range(training_targets.shape[0])
            ],
        )
        step_key = jax.random.fold_in(random_key, step)
        loss, gradient = run_step(params, jnp.asarray(batch), step_key, bank)
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        train_steps.append(step + 1)
        train_losses.append(float(loss))

        completed_step = step + 1
        if completed_step % config.validation_interval == 0:
            current_validation = float(
                evaluate_validation(params, validation_bank)
                if validation_bank is not None
                else evaluate_validation(params),
            )
            validation_steps.append(completed_step)
            validation_losses.append(current_validation)
            if current_validation < best_validation:
                best_validation = current_validation
                best_params = params
                best_step = completed_step

    return TrainingResult(
        seed=seed,
        generators=np.asarray(ansatz.generators),
        angles=np.asarray(references.angles),
        best_params=np.asarray(best_params),
        final_params=np.asarray(params),
        best_step=best_step,
        best_validation_loss=best_validation,
        train_steps=np.asarray(train_steps),
        train_losses=np.asarray(train_losses),
        validation_steps=np.asarray(validation_steps),
        validation_losses=np.asarray(validation_losses),
        family=getattr(references, "name", ""),
        n_banks=n_banks,
    )
