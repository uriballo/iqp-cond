"""One precision-study run: build, train, evaluate in fp64, save and reload.

A run trains the complete 2-local student, initialized from
``N(0, init_scale^2 / m)``, against a sparse Erdos-Renyi teacher's moments on
the bodyness-2 band.  The cached arm trains on one fixed bank of ``n_ops``
draws; the refreshed arm draws ``n_ops`` masks per step from a pool of
``pool_draws`` draws.  Teacher moments are exact for supports up to
``teacher_exact_limit`` qubits and Monte Carlo otherwise, and are cached on disk
per teacher, since every run of a teacher shares them.

The fp64 evaluation uses a separate fixed bank and reference-sample key, so the
evaluations of different runs are comparable.  ``init_seed`` sets the initial
parameters and ``seed`` the sampling, so the chaos baseline varies one with the
other fixed, or nudges every initial parameter by one float32 ulp
(``theta0_ulp``).  Everything that defines a run is in its ``RunConfig``.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import jax
import numpy as np
import yaml

import iqp_cond as ciqp
from iqp_cond.loss import TargetMoments, mmd
from iqp_cond.operators import OperatorBank, expected_bodyness_sigmas, operator_bank

from .teacher import teacher, teacher_moments, teacher_samples

CACHE = Path(__file__).parent / "cache"
_TRAINING_BANK_SEED, _EVALUATION_BANK_SEED = 10_000, 20_000


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a run; ``precision`` holds ``Precision`` fields."""

    n_qubits: int
    teacher_seed: int
    init_seed: int
    seed: int
    precision: dict = field(default_factory=dict)
    theta0_ulp: bool = False
    mc_samples: int = 1000
    mask_refresh: str = "step"
    refresh_source: str = "auto"
    teacher_sample_count: int = 100_000
    steps: int = 2000
    validation_interval: int = 25
    n_ops: int = 512
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    initial_loss_scale: float = 2.0**15
    init_scale: float = 1.0
    teacher_c: float = 2.5
    teacher_angle: float = float(np.pi / 8)
    teacher_exact_limit: int = 20
    teacher_samples: int = 10**7
    pool_draws: int = 32768
    eval_masks: int = 4096
    eval_samples: int = 16384
    eval_every: int = 1

    def __hash__(self) -> int:
        # ``precision`` is a dict, kept for readable configs; hash its items.
        return hash(
            tuple(
                tuple(sorted(value.items())) if isinstance(value, dict) else value
                for value in asdict(self).values()
            )
        )


def _bank_moments(config: RunConfig, draws: int, seed: int) -> TargetMoments:
    """Draw a bodyness-2 bank and its teacher moments, cached on disk."""
    name = (
        f"n{config.n_qubits}-teacher{config.teacher_seed}-c{config.teacher_c}"
        f"-angle{config.teacher_angle:.6f}-draws{draws}-seed{seed}"
        f"-exact{config.teacher_exact_limit}-mc{config.teacher_samples}.npz"
    )
    path = CACHE / name
    sigmas = expected_bodyness_sigmas(config.n_qubits, (2,))
    if path.exists():
        with np.load(path) as stored:
            bank = OperatorBank(
                sigmas=sigmas,
                masks=stored["masks"],
                weights=stored["weights"],
                n_ops=draws,
                n_visible=config.n_qubits,
                n_masks=stored["masks"].shape[2],
                widths=tuple(stored["widths"]),
            )
            return TargetMoments(bank, stored["moments"])
    bank = operator_bank(sigmas, draws, 1, config.n_qubits, seed)
    generators, params = teacher(
        config.n_qubits,
        config.teacher_seed,
        c=config.teacher_c,
        angle=config.teacher_angle,
    )
    width = bank.width(0)
    moments = np.ones(bank.weights.shape)  # padding rows carry zero weight
    moments[0, 0, :width] = teacher_moments(
        generators,
        params,
        np.asarray(bank.masks[0, 0, :width]),
        exact_limit=config.teacher_exact_limit,
        n_samples=config.teacher_samples,
        key=jax.random.key(seed),
    )
    CACHE.mkdir(exist_ok=True)
    np.savez(
        path,
        masks=bank.masks,
        weights=bank.weights,
        widths=np.asarray(bank.widths or (bank.n_masks,)),
        moments=moments,
    )
    return TargetMoments(bank, moments)


def _source(config: RunConfig) -> str:
    """``"samples"``, ``"pool"`` or ``"bank"``: where the training targets come from.

    The refreshed arm trains on exact teacher samples up to 30 qubits and on
    the moment pool above, unless ``refresh_source`` names one; the cached arm
    always trains on a fixed moment bank.
    """
    if config.mask_refresh != "step":
        return "bank"
    if config.refresh_source != "auto":
        return config.refresh_source
    return "samples" if config.n_qubits <= 30 else "pool"


def _teacher_samples(config: RunConfig, count: int, seed: int) -> np.ndarray:
    """Exact teacher samples, ``(1, count, n)``, cached on disk."""
    path = CACHE / (
        f"n{config.n_qubits}-teacher{config.teacher_seed}-c{config.teacher_c}"
        f"-angle{config.teacher_angle:.6f}-samples{count}-seed{seed}.npy"
    )
    if path.exists():
        return np.load(path)
    generators, params = teacher(
        config.n_qubits,
        config.teacher_seed,
        c=config.teacher_c,
        angle=config.teacher_angle,
    )
    samples = teacher_samples(generators, params, count, key=jax.random.key(seed))
    CACHE.mkdir(exist_ok=True)
    np.save(path, samples[None])
    return samples[None]


@dataclass(frozen=True)
class _Setup:
    model: ciqp.ConditionalModel
    targets: TargetMoments | np.ndarray
    validation: TargetMoments | np.ndarray
    evaluation: TargetMoments


@cache
def _setup(config: RunConfig) -> _Setup:
    n = config.n_qubits
    encoder = ciqp.AngleEncoder(np.zeros((1, 1, n)), axes=("Y" * n,))
    evaluation = _bank_moments(
        config, config.eval_masks, config.teacher_seed + _EVALUATION_BANK_SEED
    )
    source = _source(config)
    seed = config.teacher_seed + _TRAINING_BANK_SEED
    if source == "samples":
        targets = _teacher_samples(config, config.teacher_sample_count, seed)
        # The trainer's own validation is unused; two samples are its minimum.
        validation = _teacher_samples(config, 2, seed + 1)
    else:
        draws = config.pool_draws if source == "pool" else config.n_ops
        targets = _bank_moments(config, draws, seed)
        validation = evaluation
    return _Setup(
        model=ciqp.ConditionalModel(ciqp.Ansatz.complete(n), encoder),
        targets=targets,
        validation=validation,
        evaluation=evaluation,
    )


def _shared(config: RunConfig) -> RunConfig:
    """The fields that shape the setup and the evaluation, the rest at defaults.

    Runs that differ only in precision, seeds or schedule share one setup.
    """
    return RunConfig(
        n_qubits=config.n_qubits,
        teacher_seed=config.teacher_seed,
        init_seed=0,
        seed=0,
        mask_refresh=config.mask_refresh,
        refresh_source=config.refresh_source,
        teacher_sample_count=config.teacher_sample_count,
        n_ops=config.n_ops,
        teacher_c=config.teacher_c,
        teacher_angle=config.teacher_angle,
        teacher_exact_limit=config.teacher_exact_limit,
        teacher_samples=config.teacher_samples,
        pool_draws=config.pool_draws,
        eval_masks=config.eval_masks,
        eval_samples=config.eval_samples,
    )


def build(config: RunConfig) -> _Setup:
    """Return the run's model, training moments and fp64 evaluation moments."""
    return _setup(_shared(config))


def clear_cache() -> None:
    """Forget the in-process setups, so the next build reads from disk."""
    _setup.cache_clear()
    _evaluator.cache_clear()


def initial_params(config: RunConfig) -> np.ndarray:
    m = config.n_qubits * (config.n_qubits + 1) // 2
    rng = np.random.default_rng(config.init_seed)
    params = rng.normal(0.0, config.init_scale / np.sqrt(m), m)
    if config.theta0_ulp:
        params = params + np.spacing(params.astype(np.float32)).astype(np.float64)
    return params


def run(config: RunConfig) -> ciqp.training.TrainingResult:
    setup = build(config)
    samples = _source(config) == "samples"
    return ciqp.train(
        setup.model,
        setup.targets,
        setup.validation,
        seed=config.seed,
        initial_params=initial_params(config),
        config=ciqp.TrainingConfig(
            # Samples refresh masks through the library's own path, on the
            # bodyness-2 band, with every teacher sample in each step's batch.
            sigmas=expected_bodyness_sigmas(config.n_qubits, (2,)) if samples else None,
            batch_size=config.teacher_sample_count if samples else 256,
            validation_n_ops=1,
            steps=config.steps,
            validation_interval=config.validation_interval,
            n_ops=config.n_ops,
            mc_samples=config.mc_samples,
            # Checkpoints are compared in fp64 after training, so the
            # trainer's own validation only needs its minimum budget.
            validation_mc_samples=2,
            learning_rate=config.learning_rate,
            final_learning_rate=config.final_learning_rate,
            initial_loss_scale=config.initial_loss_scale,
            precision=ciqp.Precision(**config.precision),
            mask_refresh=config.mask_refresh,
        ),
    )


@cache
def _evaluator(config: RunConfig):
    setup = build(config)
    model = setup.model

    @jax.jit
    def loss(params):
        return mmd(
            params,
            model.generators,
            model.encoder,
            setup.evaluation,
            jax.random.key(0),
            n_samples=config.eval_samples,
        ).loss

    return loss


def evaluate(config: RunConfig, params: np.ndarray) -> float:
    """Return the fp64 loss on the fixed evaluation bank and sample key."""
    return float(_evaluator(_shared(config))(np.asarray(params, dtype=np.float64)))


def _provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    from importlib.metadata import version

    return {"iqp_cond": version("iqp-cond"), "git": sha}


def save(folder: Path, config: RunConfig, result) -> None:
    """Save the parameters, traces, fp64 evaluations, config and a note."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    evaluated = np.arange(len(result.checkpoint_params))[:: config.eval_every]
    if evaluated[-1] != len(result.checkpoint_params) - 1:
        evaluated = np.append(evaluated, len(result.checkpoint_params) - 1)
    fp64_losses = np.array(
        [evaluate(config, result.checkpoint_params[i]) for i in evaluated]
    )
    np.savez(
        folder / "run.npz",
        checkpoint_params=result.checkpoint_params,
        validation_steps=result.validation_steps,
        validation_losses=result.validation_losses,
        final_params=result.final_params,
        train_losses=result.train_losses,
        loss_scales=result.loss_scales,
        skipped_steps=result.skipped_steps,
        fp64_checkpoints=evaluated,
        fp64_losses=fp64_losses,
    )
    record = {"config": asdict(config), "provenance": _provenance()}
    (folder / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    nudge = " nudged by one float32 ulp" if config.theta0_ulp else ""
    (folder / "NOTE.md").write_text(
        f"""# Precision-study run

Complete 2-local student on {config.n_qubits} qubits trained against the
sparse Erdos-Renyi teacher (seed {config.teacher_seed}, c {config.teacher_c},
angles within +-{config.teacher_angle:.4f}) with precision
{config.precision or "fp64"}, S = {config.mc_samples}, {config.mask_refresh}
masks, init seed {config.init_seed}{nudge}, sampling seed {config.seed},
{config.steps} steps.

Final fp64 loss: {fp64_losses[-1]:.6g}; skipped steps: {result.skipped_steps.size}.
fp64 evaluations at every {config.eval_every} checkpoint(s), on {config.eval_masks}
mask draws and {config.eval_samples} reference samples.

Load with `experiments.precision_study.study.load(folder)`, which returns the
`RunConfig` and the arrays; `evaluate(config, params)` recomputes the fp64 loss.
Provenance (library version, git commit) is in config.yaml.
"""
    )


def load(folder: Path) -> tuple[RunConfig, dict[str, np.ndarray]]:
    folder = Path(folder)
    record = yaml.safe_load((folder / "config.yaml").read_text())
    config = RunConfig(**record["config"])
    with np.load(folder / "run.npz") as arrays:
        return config, dict(arrays)
