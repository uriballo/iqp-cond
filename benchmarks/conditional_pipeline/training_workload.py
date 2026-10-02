"""End-to-end Curie-Weiss training workload used by the benchmark."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

import iqp_cond as ciqp
from iqp_cond.operators import expected_bodyness_sigmas

from .fixtures import fixture_hash

_DEFAULT = {
    "n_qubits": 16,
    "steps": 2000,
    "seed": 410,
    "batch_size": 450,
    "n_train": 10000,
    "n_validation": 2000,
    "n_test": 2000,
    "n_ops": 256,
    "mc_samples": 8192,
    "validation_n_ops": 256,
    "validation_mc_samples": 16384,
    "validation_interval": 25,
    "sample_chunk": 256,
    "gemm_dtype": "float32",
    "estimator_dtype": None,
    "profile": "paper",
    "precision_policy": "gemm_f32",
    "betas": (0.6, 0.8, 1.2, 1.4),
    "development_betas": (0.7, 1.3),
    "evaluation_betas": (0.4, 1.0, 1.6),
    "learning_rate": 0.01,
    "final_learning_rate": 0.001,
    "initialization": "normal",
    "initial_scale": 0.5,
    "initial_seed_offset": 101,
    "encoder_axes": ("Y", "Z"),
    "ansatz": "complete",
    "ancilla": False,
}


def resolve_training_config(
    config: dict[str, Any] | None = None, smoke: bool = False
) -> dict[str, Any]:
    out = dict(_DEFAULT)
    out.update(config or {})
    if smoke:
        out.update(
            {
                "profile": "smoke",
                "steps": 2,
                "n_train": 32,
                "n_validation": 32,
                "n_test": 32,
                "batch_size": 8,
                "n_ops": 4,
                "mc_samples": 16,
                "validation_n_ops": 4,
                "validation_mc_samples": 16,
                "validation_interval": 1,
                "sample_chunk": 8,
            }
        )
    if out.get("precision_policy") == "fp64":
        out.update(gemm_dtype=None, estimator_dtype=None)
    elif out.get("precision_policy") == "estimator_f32":
        out.update(gemm_dtype=None, estimator_dtype="float32")
    elif out.get("precision_policy") == "gemm_f32":
        out.update(gemm_dtype="float32", estimator_dtype=None)
    for k in (
        "n_qubits",
        "steps",
        "seed",
        "batch_size",
        "n_train",
        "n_validation",
        "n_test",
        "n_ops",
        "mc_samples",
        "validation_n_ops",
        "validation_mc_samples",
        "validation_interval",
    ):
        out[k] = int(out[k])
    if out["sample_chunk"] is not None:
        out["sample_chunk"] = int(out["sample_chunk"])
    out["betas"] = tuple(float(x) for x in out["betas"])
    if not 0 < out["n_qubits"] <= 16 or out["steps"] < 1 or out["batch_size"] < 2:
        raise ValueError("n_qubits must be in 1..16, steps positive, batch_size >= 2")
    if out["seed"] < 0 or any(not np.isfinite(x) for x in out["betas"]):
        raise ValueError("seed must be nonnegative and betas finite")
    for key in (
        "n_train",
        "n_validation",
        "n_test",
        "n_ops",
        "mc_samples",
        "validation_n_ops",
        "validation_mc_samples",
    ):
        if out[key] < 2:
            raise ValueError(f"{key} must be at least two")
    if out["sample_chunk"] is not None and out["sample_chunk"] < 1:
        raise ValueError("sample_chunk must be positive or None")
    for key in ("learning_rate", "final_learning_rate", "initial_scale"):
        if not np.isfinite(out[key]) or out[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("betas", "development_betas", "evaluation_betas"):
        out[key] = tuple(float(x) for x in out[key])
        if not out[key] or not np.isfinite(out[key]).all():
            raise ValueError(f"{key} must be nonempty and finite")
    if out["validation_interval"] < 1:
        raise ValueError("validation_interval must be positive")
    if out["precision_policy"] not in {"fp64", "gemm_f32", "estimator_f32"}:
        raise ValueError("unknown precision policy")
    return out


def _shell_distribution(n: int, beta: float) -> np.ndarray:
    k = np.arange(n + 1)
    logw = np.array([math.log(math.comb(n, int(i))) for i in k])
    logw += beta * (2 * k - n) ** 2 / (2 * n)
    mass = np.exp(logw - logw.max())
    return mass / mass.sum()


def exact_product_state_oracle(params: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Independent Walsh-Hadamard oracle for a small product-state IQP model."""
    params = np.asarray(params)
    angles = np.asarray(angles)
    if angles.ndim != 1:
        raise ValueError("angles must be a one-dimensional vector")
    n = int(angles.size)
    if n < 1 or n > 16:
        raise ValueError("exact oracle is limited to 16 qubits")
    expected = n * (n + 1) // 2
    if params.ndim != 1 or params.size != expected:
        raise ValueError(f"params must have shape ({expected},)")
    size = 1 << n
    bits = ((np.arange(size)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    state = np.ones(size, dtype=np.complex128)
    for q in range(n):
        local = np.array(
            [np.exp(-0.5j * angles[q]), np.exp(0.5j * angles[q])]
        ) / np.sqrt(2.0)
        state *= local[bits[:, q]]

    def hadamard(vector):
        result = vector.copy()
        for q in range(n):
            stride = 1 << q
            blocks = result.reshape(-1, 2, stride)
            left = blocks[:, 0, :].copy()
            right = blocks[:, 1, :].copy()
            blocks[:, 0, :] = (left + right) / np.sqrt(2.0)
            blocks[:, 1, :] = (left - right) / np.sqrt(2.0)
        return result

    state = hadamard(state)
    pair_rows, pair_cols = np.triu_indices(n, k=1)
    generators = np.concatenate(
        (
            np.eye(n, dtype=np.int8),
            np.eye(n, dtype=np.int8)[pair_rows] + np.eye(n, dtype=np.int8)[pair_cols],
        )
    )
    parity = (bits @ generators.T) % 2
    state *= np.exp(1j * (1 - 2 * parity) @ params)
    return np.abs(hadamard(state)) ** 2


def generate_training_fixture(config: dict[str, Any]) -> dict[str, np.ndarray]:
    c = resolve_training_config(config)
    arrays: dict[str, np.ndarray] = {}
    shared_rng = np.random.default_rng(c["seed"])
    for name, count, betas, local in (
        ("train_targets", c["n_train"], c["betas"], shared_rng),
        ("validation_targets", c["n_validation"], c["betas"], shared_rng),
        (
            "test_targets",
            c["n_test"],
            c["evaluation_betas"],
            np.random.default_rng(c["seed"] + 2_000_001),
        ),
    ):
        rows = []
        for beta in betas:
            shell = local.choice(
                c["n_qubits"] + 1,
                size=count,
                p=_shell_distribution(c["n_qubits"], beta),
            )
            samples = np.zeros((count, c["n_qubits"]))
            for i, weight in enumerate(shell):
                if weight:
                    samples[
                        i, local.choice(c["n_qubits"], size=int(weight), replace=False)
                    ] = 1
            rows.append(samples)
        arrays[name] = np.stack(rows)
    arrays["initial_params"] = np.random.default_rng(
        c["seed"] + c["initial_seed_offset"]
    ).normal(0, c["initial_scale"], size=c["n_qubits"] * (c["n_qubits"] + 1) // 2)
    arrays["betas"] = np.asarray(c["betas"], dtype=np.float64)
    return arrays


class _TrainingObserver:
    """Observe JIT timing without synchronizing every optimizer update.

    Loop timing may be unavailable if a fork changes the trainer's JIT calls.
    """

    def __init__(self):
        self.loop_start = None
        self.validation_seconds = 0.0
        self.selection_seconds = 0.0
        self.loop_compile_seconds = 0.0
        self.compilation_seconds = 0.0
        self.lowering_seconds = 0.0
        self.compilation_count = 0
        self.initial_validation_seconds = 0.0
        self.loop_selection_seconds = 0.0

    def __getattr__(self, name):
        return getattr(jax, name)

    def jit(self, fn, *args, **kwargs):
        wrapped = jax.jit(fn, *args, **kwargs)
        if fn.__name__ not in {"training_step", "validation_loss"}:
            return wrapped
        executable = None

        def execute(*values):
            nonlocal executable
            start = time.perf_counter()
            if executable is None:
                lowered = wrapped.lower(*values)
                middle = time.perf_counter()
                executable = lowered.compile()
                end = time.perf_counter()
                self.lowering_seconds += middle - start
                self.compilation_seconds += end - middle
                self.compilation_count += 1
                if fn.__name__ == "training_step" and self.loop_start is not None:
                    self.loop_compile_seconds += end - start
            if fn.__name__ == "training_step":
                if self.loop_start is None:
                    self.loop_start = time.perf_counter()
                return executable(*values)
            result = jax.block_until_ready(executable(*values))
            elapsed = time.perf_counter() - start
            self.validation_seconds += elapsed
            if self.loop_start is None:
                self.initial_validation_seconds += elapsed
            return result

        return execute


def _file_hash(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profile(probabilities, n):
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).sum(axis=1)
    return np.bincount(bits, weights=probabilities, minlength=n + 1)


def _diagnostics(params, c, betas):
    n = c["n_qubits"]
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).sum(axis=1)
    bit_tvd, profile_tvd = [], []
    for beta in betas:
        got = exact_product_state_oracle(params, np.full(n, 0.75 * beta))
        target_profile = _shell_distribution(n, beta)
        counts = np.array([math.comb(n, int(k)) for k in bits], dtype=float)
        target = target_profile[bits] / counts
        bit_tvd.append(float(0.5 * abs(got - target).sum()))
        profile_tvd.append(float(0.5 * abs(_profile(got, n) - target_profile).sum()))
    return {
        "betas": list(betas),
        "raw_probability_tvd": bit_tvd,
        "profile_tvd": profile_tvd,
    }


def run_training_cell(
    config, *, backend="cpu", artifacts=None, fixture=None, fixtures_dir=None
):
    import jax.numpy as jnp

    from iqp_cond import training as training_module

    c = resolve_training_config(config)
    path = Path(artifacts or "benchmarks/conditional_pipeline/results/training")
    path.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 3,
        "benchmark_kind": "training",
        "config": c,
        "profile": c["profile"],
        "backend": backend,
        "status": "failed",
        "completed_updates": 0,
        "timings_seconds": {},
        "validity": {
            "status": "unverified",
            "reason": "training outcomes require review",
        },
    }
    process_start = time.perf_counter()
    try:
        jax.config.update("jax_enable_compilation_cache", False)
        device = jax.devices(backend)[0]
        if device.platform != backend:
            raise RuntimeError(f"requested {backend}, got {device.platform}")
        row["device"] = {
            "kind": device.device_kind,
            "id": device.id,
            "platform": device.platform,
        }
        with jax.default_device(device):
            jax.block_until_ready(jnp.zeros(()))
            started = time.perf_counter()
            import hashlib

            fixture_config = {
                k: c[k]
                for k in (
                    "n_qubits",
                    "seed",
                    "n_train",
                    "n_validation",
                    "n_test",
                    "betas",
                    "evaluation_betas",
                    "initial_scale",
                    "initial_seed_offset",
                )
            }
            fixture_id = hashlib.sha256(
                json.dumps(fixture_config, sort_keys=True).encode()
            ).hexdigest()[:24]
            fixture_path = Path(fixtures_dir or path) / f"training-{fixture_id}.npz"
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            checksum = fixture_path.with_suffix(".sha256")
            if fixture is None:
                if fixture_path.exists():
                    if not checksum.exists() or checksum.read_text() != _file_hash(
                        fixture_path
                    ):
                        raise ValueError("training fixture checksum failed")
                    with np.load(fixture_path, allow_pickle=False) as data:
                        fixture = dict(data)
                else:
                    fixture = generate_training_fixture(c)
            if not fixture_path.exists():
                np.savez(fixture_path, **fixture)
                checksum.write_text(_file_hash(fixture_path))
            row.update(
                fixture_hash=fixture_hash(fixture),
                fixture_path=str(fixture_path),
                fixture_file_hash=_file_hash(fixture_path),
            )
            angles = np.stack(
                [
                    np.tile([np.pi / 2, 0.75 * b], (c["n_qubits"], 1)).T
                    for b in c["betas"]
                ]
            )
            encoder = ciqp.AngleEncoder(
                angles, axes=("Y" * c["n_qubits"], "Z" * c["n_qubits"])
            )
            model = ciqp.ConditionalModel(ciqp.Ansatz.complete(c["n_qubits"]), encoder)
            sigmas = expected_bodyness_sigmas(c["n_qubits"])[:2]
            c["sigmas"] = list(sigmas)
            tc = ciqp.TrainingConfig(
                steps=c["steps"],
                batch_size=c["batch_size"],
                learning_rate=c["learning_rate"],
                final_learning_rate=c["final_learning_rate"],
                validation_interval=c["validation_interval"],
                sigmas=sigmas,
                n_ops=c["n_ops"],
                mc_samples=c["mc_samples"],
                validation_n_ops=c["validation_n_ops"],
                validation_mc_samples=c["validation_mc_samples"],
                gemm_dtype=jnp.float32 if c["gemm_dtype"] else None,
                estimator_dtype=jnp.float32 if c["estimator_dtype"] else None,
                sample_chunk=c["sample_chunk"],
                mask_refresh="step",
            )
            jax.block_until_ready((model.encoder, model.ansatz.generators))
            observer = _TrainingObserver()
            checkpoints = []

            def selection(params):
                begin = time.perf_counter()
                checkpoints.append(np.asarray(params).copy())
                score = float(
                    np.mean(
                        _diagnostics(params, c, c["development_betas"])["profile_tvd"]
                    )
                )
                elapsed = time.perf_counter() - begin
                observer.selection_seconds += elapsed
                if observer.loop_start is not None:
                    observer.loop_selection_seconds += elapsed
                return score

            call_start = time.perf_counter()
            original_jax = training_module.jax
            try:
                training_module.jax = observer
                result = ciqp.train(
                    model,
                    fixture["train_targets"],
                    fixture["validation_targets"],
                    seed=c["seed"],
                    config=tc,
                    initial_params=fixture["initial_params"],
                    selection_metric=selection,
                )
                jax.block_until_ready((result.final_params, result.train_losses))
            finally:
                training_module.jax = original_jax
            call_end = time.perf_counter()
            outputs = {
                name: np.asarray(getattr(result, name))
                for name in (
                    "best_params",
                    "final_params",
                    "train_steps",
                    "train_losses",
                    "validation_steps",
                    "validation_losses",
                    "selection_scores",
                )
            }
            outputs["checkpoint_params"] = np.asarray(checkpoints)
            output_path = path / "training_history.npz"
            np.savez(output_path, **outputs)
            total_end = time.perf_counter()
            loop = None
            if observer.loop_start is not None:
                # Initial validation precedes the loop.
                initial_validation = observer.initial_validation_seconds
                loop = max(
                    0.0,
                    call_end
                    - observer.loop_start
                    - observer.loop_compile_seconds
                    - (observer.validation_seconds - initial_validation)
                    - observer.loop_selection_seconds,
                )
            row.update(
                status="ok",
                completed_updates=len(result.train_steps),
                best_step=int(result.best_step),
                output_path=str(output_path),
                output_file_hash=_file_hash(output_path),
            )
            if not np.array_equal(result.train_steps, np.arange(1, c["steps"] + 1)):
                raise ValueError(
                    "trainer did not complete the requested update sequence"
                )
            if not all(np.isfinite(x).all() for x in outputs.values()):
                raise FloatingPointError("nonfinite training result")
            row["timings_seconds"] = {
                "total_training": total_end - started,
                "preparation": call_start - started,
                "training_call": call_end - call_start,
                "lowering": observer.lowering_seconds,
                "compile": observer.compilation_seconds,
                "training_loop": loop,
                "validation": observer.validation_seconds,
                "checkpoint_selection": observer.selection_seconds,
                "artifact_write": total_end - call_end,
            }
            row["compilation_count"] = observer.compilation_count
            row["measurement_policy"] = {
                "single_fit": True,
                "persistent_cache": "disabled",
                "selection": "exact_development_profile_tvd",
                "total_scope": (
                    "fixture preparation, transfer, compile, updates, validation, "
                    "selection, checkpoint artifacts"
                ),
                "loop_scope": (
                    "first update to completed fit excluding compilation, validation "
                    "and selection; async updates are not individually synchronized"
                ),
            }
            if loop is None:
                row["training_loop_time_reason"] = (
                    "fork trainer internals do not support the benchmark observer"
                )
            diagnostic_start = time.perf_counter()
            row["outcomes"] = {
                name: {
                    split: _diagnostics(params, c, c[key])
                    for split, key in (
                        ("development", "development_betas"),
                        ("evaluation", "evaluation_betas"),
                    )
                }
                for name, params in (
                    ("best", result.best_params),
                    ("final", result.final_params),
                )
            }
            row["timings_seconds"]["final_diagnostics"] = (
                time.perf_counter() - diagnostic_start
            )
            import resource
            import sys

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row["memory"] = {
                "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
                "scope": "whole child including exact outcome diagnostics",
            }
            try:
                row["memory"]["device_stats"] = device.memory_stats()
            except (RuntimeError, AttributeError):
                row["memory"]["device_stats"] = None
    except (ValueError, RuntimeError, OSError, MemoryError, FloatingPointError) as exc:
        row.update(
            status="capacity" if isinstance(exc, MemoryError) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    row["runtime_seconds"] = time.perf_counter() - process_start
    artifact = path / "training_result.json"
    row["artifact_path"] = str(artifact)
    artifact.write_text(json.dumps(row))
    return row
