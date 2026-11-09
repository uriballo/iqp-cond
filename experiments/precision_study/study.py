"""One precision-study run: build, train, evaluate in fp64, save and reload.

A run trains the complete 2-local student, initialized from N(0, 1/m),
against a sparse Erdos-Renyi teacher's moments on the bodyness-2 band.  The
training masks come from a pool drawn from that band; the fp64 evaluation uses
a separate fixed bank and a fixed reference-sample key, so evaluations of
different runs are comparable.  Everything needed to rebuild a run is in its
``RunConfig``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import jax
import numpy as np

import iqp_cond as ciqp
from iqp_cond.loss import TargetMoments, mmd
from iqp_cond.operators import expected_bodyness_sigmas, operator_bank

from .teacher import teacher, teacher_moments


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a run; ``precision`` holds ``Precision`` fields."""

    n_qubits: int
    teacher_seed: int
    seed: int
    precision: dict = field(default_factory=dict)
    mc_samples: int = 1000
    mask_refresh: str = "step"
    steps: int = 2000
    validation_interval: int = 25
    n_ops: int = 512
    pool_draws: int = 32768
    eval_masks: int = 4096
    eval_samples: int = 16384


@dataclass(frozen=True)
class _Setup:
    model: ciqp.ConditionalModel
    targets: TargetMoments
    evaluation: TargetMoments


def _moment_targets(n_qubits, generators, params, draws, seed) -> TargetMoments:
    sigmas = expected_bodyness_sigmas(n_qubits, (2,))
    bank = operator_bank(sigmas, draws, 1, n_qubits, seed)
    width = bank.width(0)
    moments = np.ones(bank.weights.shape)
    moments[0, 0, :width] = teacher_moments(
        generators,
        params,
        np.asarray(bank.masks[0, 0, :width]),
        key=jax.random.key(seed),
    )
    return TargetMoments(bank, moments)


@cache
def _setup(n_qubits: int, teacher_seed: int, pool_draws: int, eval_masks: int):
    generators, params = teacher(n_qubits, teacher_seed)
    encoder = ciqp.AngleEncoder(np.zeros((1, 1, n_qubits)), axes=("Y" * n_qubits,))
    return _Setup(
        model=ciqp.ConditionalModel(ciqp.Ansatz.complete(n_qubits), encoder),
        targets=_moment_targets(n_qubits, generators, params, pool_draws, teacher_seed),
        evaluation=_moment_targets(
            n_qubits, generators, params, eval_masks, teacher_seed + 1
        ),
    )


def build(config: RunConfig) -> _Setup:
    return _setup(
        config.n_qubits, config.teacher_seed, config.pool_draws, config.eval_masks
    )


def initial_params(config: RunConfig) -> np.ndarray:
    m = build(config).model.ansatz.n_parameters
    return np.random.default_rng(config.seed).normal(0.0, 1 / np.sqrt(m), m)


def run(config: RunConfig) -> ciqp.training.TrainingResult:
    setup = build(config)
    return ciqp.train(
        setup.model,
        setup.targets,
        setup.evaluation,
        seed=config.seed,
        initial_params=initial_params(config),
        config=ciqp.TrainingConfig(
            steps=config.steps,
            validation_interval=config.validation_interval,
            n_ops=config.n_ops,
            mc_samples=config.mc_samples,
            validation_mc_samples=config.eval_samples,
            precision=ciqp.Precision(**config.precision),
            mask_refresh=config.mask_refresh,
        ),
    )


def evaluate(config: RunConfig, params: np.ndarray) -> float:
    """Return the fp64 loss on the fixed evaluation bank and sample key."""
    setup = build(config)
    return float(
        mmd(
            params,
            setup.model.generators,
            setup.model.encoder,
            setup.evaluation,
            jax.random.key(0),
            n_samples=config.eval_samples,
        ).loss
    )


def save(folder: Path, config: RunConfig, result) -> None:
    """Save the parameters, traces, config and a short note of a run."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fp64_losses = np.array([evaluate(config, p) for p in result.checkpoint_params])
    np.savez(
        folder / "run.npz",
        checkpoint_params=result.checkpoint_params,
        validation_steps=result.validation_steps,
        final_params=result.final_params,
        train_losses=result.train_losses,
        loss_scales=result.loss_scales,
        skipped_steps=result.skipped_steps,
        fp64_losses=fp64_losses,
    )
    (folder / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (folder / "NOTE.md").write_text(
        f"""# Precision-study run

Complete 2-local student on {config.n_qubits} qubits trained against the
sparse Erdos-Renyi teacher (seed {config.teacher_seed}) with precision
{config.precision or "fp64"}, S = {config.mc_samples}, {config.mask_refresh}
masks, seed {config.seed}, {config.steps} steps.

Final fp64 loss: {fp64_losses[-1]:.6g}; skipped steps: {result.skipped_steps.size}.

Load with `experiments.precision_study.study.load(folder)`, which returns the
`RunConfig` and the arrays; `evaluate(config, params)` recomputes the fp64 loss.
"""
    )


def load(folder: Path) -> tuple[RunConfig, dict[str, np.ndarray]]:
    folder = Path(folder)
    config = RunConfig(**json.loads((folder / "config.json").read_text()))
    with np.load(folder / "run.npz") as arrays:
        return config, dict(arrays)
