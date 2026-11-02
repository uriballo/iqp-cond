"""Measure conditional IQP operations and a fixed Curie-Weiss training fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.conditional_pipeline.analysis import (  # noqa: E402
    summarize,
    workload_id,
)
from benchmarks.conditional_pipeline.fixtures import (  # noqa: E402
    configs,
    fixture_hash,
    generate_fixture,
    resolve_config,
)
from benchmarks.conditional_pipeline.presentation import render_terminal  # noqa: E402


def print_summary(args, records, summary, report_dir, *, payload=None):
    """Keep child JSON transport separate from user-facing output."""
    if args.json:
        print(json.dumps(payload or summary["counts"]))
    else:
        print(
            render_terminal(
                records,
                summary,
                records_path=args.input or args.output,
                report_path=report_dir / "report.md",
            )
        )


ENV_KEYS = (
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files():
    return sorted(
        [
            *ROOT.glob("src/iqp_cond/**/*.py"),
            *ROOT.glob("benchmarks/conditional_pipeline/*.py"),
        ]
    )


def command_output(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def make_manifest():
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}
    packages = {}
    for name in (
        "jax",
        "jaxlib",
        "numpy",
        "pennylane",
        "iqp-cond",
        "jax-cuda12-plugin",
        "jax-cuda13-plugin",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    stack = {"python": platform.python_version(), "packages": packages}
    manifest = {
        "schema_version": 3,
        "source_hashes": sources,
        "source_identity": digest(sources),
        "library_source_identity": digest(
            {k: v for k, v in sources.items() if k.startswith("src/")}
        ),
        "harness_source_identity": digest(
            {k: v for k, v in sources.items() if k.startswith("benchmarks/")}
        ),
        "stack": stack,
        "stack_identity": digest(stack),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "environment": {k: os.environ.get(k) for k in ENV_KEYS},
        "git_revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "gpu_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,pstate,power.limit,clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ]
        ),
        "runtime_inventory": command_output(["nvidia-smi"]),
        "persistent_cache": "disabled",
    }
    # Dynamic clocks/pstate are observations, not resume identity.
    manifest["environment_identity"] = digest(
        {
            k: manifest[k]
            for k in ("host", "platform", "cpu", "cpu_count", "environment")
        }
    )
    manifest["gpu_static_inventory"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    manifest["environment_identity"] = digest(
        [manifest["environment_identity"], manifest["gpu_static_inventory"]]
    )
    manifest["manifest_id"] = digest(manifest)
    return manifest


def fixture_identity(config):
    excluded = {"dtype", "sample_chunk", "repeats", "warmups", "sigma", "axis", "stage"}
    return digest({k: v for k, v in config.items() if k not in excluded})[:24]


def available_memory():
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                try:
                    root = Path("/sys/fs/cgroup")
                    limit = (root / "memory.max").read_text().strip()
                    if limit != "max":
                        host = min(
                            host,
                            max(
                                0,
                                int(limit) - int((root / "memory.current").read_text()),
                            ),
                        )
                except (OSError, ValueError):
                    pass
                return host
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def preflight(config):
    n, c = config["n_qubits"], config["n_conditions"]
    g = n * (n + 1) // 2 if config["ansatz"] == "complete" else config["n_generators"]
    lower = 8 * (
        g * n
        + g
        + c * n * (config["n_targets"] + config["n_samples"])
        + c * config["n_ops"] * (n + 1)
    )
    available = available_memory()
    if available is not None and lower > 0.8 * available:
        return (
            f"input allocation lower bound {lower} exceeds "
            f"host/container limit {available}"
        )
    return None


def memory(device=None):
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        stats = device.memory_stats() if device is not None else None
    except (RuntimeError, AttributeError):
        stats = None
    stats = stats or {}
    return {
        "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "device_bytes_current": stats.get("bytes_in_use", stats.get("bytes_used")),
        "device_bytes_peak": stats.get(
            "peak_bytes_in_use", stats.get("peak_bytes_used")
        ),
    }


class GPUObserver:
    """Best-effort sampled process memory; may miss short-lived peaks."""

    def __init__(self, enabled):
        self.enabled = enabled and shutil.which("nvidia-smi") is not None
        self.stop = threading.Event()
        self.peak = None
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            out = command_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            for line in (out or "").splitlines():
                try:
                    pid, mib = (int(x.strip()) for x in line.split(","))
                    if pid == os.getpid():
                        self.peak = max(self.peak or 0, mib * 1024 * 1024)
                except ValueError:
                    pass
            self.stop.wait(0.5)

    def __enter__(self):
        if self.enabled:
            self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        if self.enabled:
            self.thread.join(timeout=6)


def run_cell(
    config,
    *,
    backend="cpu",
    artifacts=None,
    kernels=None,
    validate=False,
    trace=False,
    fixtures_dir=None,
):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from benchmarks.conditional_pipeline.validity import independent_checks
    from benchmarks.conditional_pipeline.workloads import (
        build_kernels,
        encoder_from_fixture,
        measure,
        numerical_outputs,
    )

    config = resolve_config(config)
    row = {
        "schema_version": 3,
        "status": "failed",
        "config": config,
        "backend": backend,
        "workload_id": workload_id(config),
        "fixture_identity": fixture_identity(config),
        "validity": {"status": "unverified"},
        "timings_seconds": {},
    }
    start = time.perf_counter()
    device = None
    artifacts = Path(artifacts or ROOT / "benchmarks/conditional_pipeline/results/cell")
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        jax.config.update("jax_enable_compilation_cache", False)
        begin = time.perf_counter()
        device = jax.devices(backend)[0]
        if device.platform != backend:
            raise RuntimeError(f"requested {backend}, got {device.platform}")
        with jax.default_device(device):
            jax.block_until_ready(jnp.zeros((), dtype=jnp.float64))
        row["backend_init_seconds"] = time.perf_counter() - begin
        row["device"] = {
            "kind": device.device_kind,
            "platform": device.platform,
            "id": device.id,
        }
        reason = preflight(config)
        if reason:
            row.update(status="capacity", error=reason)
            return row
        begin = time.perf_counter()
        path = (
            Path(fixtures_dir or artifacts / "fixtures")
            / f"{fixture_identity(config)}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path = path.with_suffix(".sha256")
        if path.exists():
            if not checksum_path.exists() or checksum_path.read_text() != file_hash(
                path
            ):
                raise ValueError(
                    "stored fixture checksum failed; use a fresh artifact directory"
                )
            with np.load(path, allow_pickle=False) as data:
                fixture = {k: data[k] for k in data.files}
        else:
            with jax.default_device(jax.devices("cpu")[0]):
                fixture = generate_fixture(config)
            np.savez(path, **fixture)
            checksum_path.write_text(file_hash(path))
        row.update(
            fixture_hash=fixture_hash(fixture),
            fixture_path=str(path),
            fixture_file_hash=file_hash(path),
        )
        row["fixture_setup_seconds"] = time.perf_counter() - begin
        config["sigma"] = float(fixture["sigma"])
        begin = time.perf_counter()
        data = {k: jax.device_put(v, device) for k, v in fixture.items()}
        jax.block_until_ready(data)
        row["transfer_seconds"] = time.perf_counter() - begin
        with jax.default_device(device):
            begin = time.perf_counter()
            encoder = encoder_from_fixture(config, data)
            jax.block_until_ready(encoder)
            row["encoder_construction_seconds"] = time.perf_counter() - begin
            key = jax.random.wrap_key_data(data["key_data"])
            a = (
                data["params"],
                data["generators"],
                encoder,
                data["targets"],
                data["samples"],
                data["masks"],
                data["weights"],
                key,
            )
            row["array_shapes"] = {k: list(v.shape) for k, v in fixture.items()}
            row["actual_generators"] = len(fixture["generators"])
            row["distinct_masks"] = [
                len(np.unique(m, axis=0)) for m in fixture["masks"]
            ]
            row["mask_weights"] = [
                float(m.sum(axis=1).mean()) for m in fixture["masks"]
            ]
            row["scalar_reductions"] = {
                "phase_gradient": "mean over conditions/samples/masks",
                "prepared_estimator_gradient": "mean over conditions/masks",
            }
            with GPUObserver(backend == "gpu") as observer:
                funcs = build_kernels(config)
                chosen = kernels or list(funcs)
                compiled_outputs = {}
                row["component_memory_observations"] = {}
                for label in chosen:
                    before_memory = memory(device)
                    trace_path = (
                        artifacts / "traces" / row["workload_id"] / label
                        if trace
                        and label in {"amplitude_ratios", "phase", "mmd_gradient"}
                        else None
                    )
                    timings, result = measure(
                        funcs[label],
                        a,
                        1 if validate else config["repeats"],
                        0 if validate else config["warmups"],
                        trace_path,
                        return_result=True,
                    )
                    compiled_outputs[label] = result
                    if not validate:
                        row["timings_seconds"][label] = timings
                    row["component_memory_observations"][label] = {
                        "before": before_memory,
                        "after": memory(device),
                        "scope": "process snapshots; cumulative JAX peaks",
                    }
                outputs = numerical_outputs(config, a)
                output_name = (
                    f"{row['workload_id']}-{config['seed']}-"
                    f"{backend}-{config['dtype']}.npz"
                )
                path = artifacts / "outputs" / output_name
                path.parent.mkdir(exist_ok=True)
                np.savez(path, **outputs)
                row.update(output_path=str(path), output_file_hash=file_hash(path))
                from benchmarks.conditional_pipeline.contracts import (
                    compare_kernel_results,
                    flatten_kernel_outputs,
                    reference_outputs,
                )

                kernel_path = path.with_name(path.stem + "-kernels.npz")
                actual = flatten_kernel_outputs(compiled_outputs)
                np.savez(kernel_path, **actual)
                row.update(
                    kernel_output_path=str(kernel_path),
                    kernel_output_file_hash=file_hash(kernel_path),
                    output_scope="first execution of the actual compiled kernels",
                    kernel_output_shapes={k: list(v.shape) for k, v in actual.items()},
                    kernel_output_dtypes={k: str(v.dtype) for k, v in actual.items()},
                )
                if config["n_qubits"] <= 8:
                    row["independent_checks"] = independent_checks(
                        config, fixture, encoder
                    )
                    oracle = reference_outputs(config, fixture)
                    row["compiled_checks"] = compare_kernel_results(
                        {k: oracle[k] for k in chosen},
                        compiled_outputs,
                    )
                    if config["dtype"] == "fp64":
                        row["compiled_checks"]["status"] = (
                            "passed"
                            if all(
                                x.get("allclose_fp64", False)
                                for x in row["compiled_checks"]
                                .get("errors", {})
                                .values()
                            )
                            and row["compiled_checks"]["status"] != "failed"
                            else "failed"
                        )
            row["memory"] = memory(device)
            row["memory"].update(
                external_gpu_bytes_peak=observer.peak,
                external_observation=(
                    "process memory sampled every 0.5s; may miss peaks"
                ),
            )
        row["status"] = "ok"
        row["measurement_policy"] = {
            "kernels": chosen,
            "check_only": validate,
            "trace": trace,
            "persistent_cache": "disabled",
        }
        row["executed_source_identity"] = digest(
            {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}
        )
    except (
        ValueError,
        RuntimeError,
        MemoryError,
        OSError,
        FloatingPointError,
        KeyError,
    ) as exc:
        message = f"{type(exc).__name__}: {exc}"
        capacity = isinstance(exc, MemoryError) or any(
            s in message.lower()
            for s in ("out of memory", "resource_exhausted", "bad_alloc")
        )
        row.update(
            status="capacity" if capacity else "failed",
            error=message,
            memory=memory(device),
        )
    finally:
        row["runtime_seconds"] = time.perf_counter() - start
    return row


def reconcile(records):
    import numpy as np

    from benchmarks.conditional_pipeline.validity import compare_outputs

    refs = {}
    for r in records:
        if r.get("benchmark_kind") == "training":
            continue
        if (
            r["status"] == "ok"
            and r["config"]["dtype"] == "fp64"
            and r.get("output_path")
        ):
            refs[
                (
                    r["workload_id"],
                    r["config"]["seed"],
                    r["backend"],
                    r.get("fixture_hash"),
                    r.get("source_identity"),
                    r.get("stack_identity"),
                    r.get("environment_identity"),
                )
            ] = r
    for r in records:
        if r.get("benchmark_kind") == "training":
            continue
        if r["status"] != "ok" or not r.get("output_path"):
            continue
        if any(
            r.get(k, {}).get("status") == "failed"
            for k in ("independent_checks", "compiled_checks")
        ):
            r["validity"] = {
                "status": "failed",
                "reason": "independent compiled correctness check failed",
            }
            continue
        backend = "cpu" if r["config"]["dtype"] == "fp64" else r["backend"]
        ref = refs.get(
            (
                r["workload_id"],
                r["config"]["seed"],
                backend,
                r.get("fixture_hash"),
                r.get("source_identity"),
                r.get("stack_identity"),
                r.get("environment_identity"),
            )
        )
        if ref is None:
            r["validity"] = {
                "status": "unverified",
                "reason": "paired fp64 reference unavailable",
            }
            continue
        try:
            for saved in (ref, r):
                if saved.get("output_file_hash") != file_hash(saved["output_path"]):
                    raise ValueError("output checksum mismatch")
            with (
                np.load(ref["output_path"], allow_pickle=False) as x,
                np.load(r["output_path"], allow_pickle=False) as y,
            ):
                r["validity"] = compare_outputs(dict(x), dict(y), r["config"]["dtype"])
            r["validity"]["reference"] = ref["output_path"]
            expected = {
                "moments": (r["config"]["n_conditions"], r["config"]["n_ops"]),
                "standard_errors": (r["config"]["n_conditions"], r["config"]["n_ops"]),
                "loss": (),
                "gradient": (r["config"]["n_generators"],),
            }
            with np.load(r["output_path"], allow_pickle=False) as data:
                bad = [
                    k
                    for k, shape in expected.items()
                    if k not in data or data[k].shape != shape
                ]
            if bad:
                r["validity"].update(status="failed", unexpected_shapes=bad)
            if (
                r.get("independent_checks", {}).get("status") != "passed"
                and r["config"]["n_qubits"] <= 8
            ):
                r["validity"].update(status="failed", independent_failure=True)
            if r.get("compiled_checks", {}).get("status") == "failed":
                r["validity"].update(status="failed", compiled_failure=True)
        except (OSError, ValueError) as exc:
            r["validity"] = {"status": "unverified", "reason": str(exc)}
    reconcile_compiled(records)
    # Mixed-precision comparisons require a passing fp64 reference result.
    by_path = {r.get("output_path"): r for r in records if r.get("output_path")}
    for r in records:
        if r.get("benchmark_kind") == "training":
            continue
        if (
            r["config"]["dtype"] != "fp64"
            and r.get("validity", {}).get("status") == "passed"
        ):
            reference = by_path.get(r["validity"].get("reference"), {})
            status = reference.get("validity", {}).get("status", "unverified")
            if status != "passed":
                r["validity"].update(
                    status=status, reason="fp64 reference did not pass validity"
                )
    return records


def reconcile_compiled(records):
    import numpy as np

    from benchmarks.conditional_pipeline.contracts import compare_kernel_results

    identity = (
        "workload_id",
        "fixture_hash",
        "source_identity",
        "stack_identity",
        "environment_identity",
    )
    for row in records:
        if row.get("status") != "ok" or not row.get("kernel_output_path"):
            continue
        backend = "cpu" if row["config"]["dtype"] == "fp64" else row["backend"]
        chosen = set(row.get("measurement_policy", {}).get("kernels", []))
        matches = [
            r
            for r in records
            if r.get("status") == "ok"
            and r.get("kernel_output_path")
            and r["config"].get("dtype") == "fp64"
            and r.get("backend") == backend
            and all(r.get(k) == row.get(k) for k in identity)
        ]
        matches.sort(
            key=lambda r: (
                set(r.get("measurement_policy", {}).get("kernels", [])) != chosen
            )
        )
        ref = next(
            (
                r
                for r in matches
                if chosen <= set(r.get("measurement_policy", {}).get("kernels", []))
            ),
            None,
        )
        if ref is None:
            row["compiled_comparison"] = {
                "status": "unverified",
                "reason": "actual compiled fp64 outputs unavailable",
            }
            continue
        try:
            for saved in (ref, row):
                if file_hash(saved["kernel_output_path"]) != saved.get(
                    "kernel_output_file_hash"
                ):
                    raise ValueError("compiled output checksum mismatch")
            with (
                np.load(ref["kernel_output_path"], allow_pickle=False) as x,
                np.load(row["kernel_output_path"], allow_pickle=False) as y,
            ):
                comparison = compare_kernel_results(
                    {k: x[k] for k in y.files if k in x}, dict(y)
                )
            comparison["reference"] = ref["kernel_output_path"]
            row["compiled_comparison"] = comparison
            if comparison["status"] == "failed" or (
                row["config"]["dtype"] == "fp64"
                and not all(r["allclose_fp64"] for r in comparison["errors"].values())
            ):
                row["validity"] = {
                    "status": "failed",
                    "reason": "actual compiled outputs disagree with fp64 reference",
                }
            elif (
                ref.get("validity", {}).get("status") != "passed"
                and row.get("validity", {}).get("status") != "failed"
            ):
                row["validity"] = {
                    "status": "unverified",
                    "reason": "compiled fp64 reference did not pass correctness",
                }
        except (ValueError, OSError) as exc:
            row["compiled_comparison"] = {"status": "unverified", "reason": str(exc)}
            row["validity"] = {"status": "unverified", "reason": str(exc)}


def read_records(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def reusable(row, config, backend, session, role, manifest):
    if row.get("status") != "ok" or row.get("validity", {}).get("status") != "passed":
        return False
    actual = {k: v for k, v in row.get("config", {}).items() if k != "sigma"}
    expected = {
        "backend": backend,
        "session_id": session,
        "role": role,
        "source_identity": manifest["source_identity"],
        "stack_identity": manifest["stack_identity"],
        "environment_identity": manifest["environment_identity"],
        "fixture_identity": fixture_identity(config),
    }
    if actual != config or any(row.get(k) != v for k, v in expected.items()):
        return False
    for pk, hk in (
        ("fixture_path", "fixture_file_hash"),
        ("output_path", "output_file_hash"),
    ):
        p = Path(row.get(pk, ""))
        if not p.is_file() or file_hash(p) != row.get(hk):
            return False
    if row.get("schema_version", 0) >= 3:
        p = Path(row.get("kernel_output_path", ""))
        if not p.is_file() or file_hash(p) != row.get("kernel_output_file_hash"):
            return False
    return True


def child_environment(backend):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu" if backend == "cpu" else "cuda,cpu"
    env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_child(
    config,
    backend,
    artifacts,
    timeout,
    kernels=None,
    validate=False,
    trace=False,
    fixtures_dir=None,
):
    artifacts.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.conditional_pipeline.benchmark",
        "--child-config",
        json.dumps(config),
        "--backend",
        backend,
        "--artifacts",
        str(artifacts),
    ]
    if kernels:
        cmd += ["--kernels", ",".join(kernels)]
    if validate:
        cmd.append("--check-only")
    if trace:
        cmd.append("--trace")
    if fixtures_dir:
        cmd += ["--fixtures", str(fixtures_dir)]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=child_environment(backend),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        record = json.loads(stdout.strip().splitlines()[-1])
        if not isinstance(record, dict) or record.get("status") not in {
            "ok",
            "failed",
            "capacity",
        }:
            raise ValueError("invalid child record")
        if proc.returncode and record["status"] == "ok":
            raise ValueError("inconsistent child exit")
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        record = {"status": "failed", "error": f"cell timeout after {timeout:.1f}s"}
    except (ValueError, IndexError):
        record = {
            "status": "failed",
            "error": stderr[-4000:] or "invalid child output",
            "returncode": proc.returncode,
        }
        if any(
            s in stderr.lower()
            for s in ("resource_exhausted", "out of memory", "bad_alloc")
        ):
            record["status"] = "capacity"
    logs = artifacts / "logs"
    logs.mkdir(exist_ok=True)
    precision = config.get("dtype", config.get("precision_policy", "unspecified"))
    name = f"{workload_id(config)}-{config['seed']}-{backend}-{precision}"
    for suffix, content in (("stdout", stdout), ("stderr", stderr)):
        (logs / f"{name}.{suffix}").write_text(content)
        record[f"{suffix}_path"] = str(logs / f"{name}.{suffix}")
    record.update(
        config=record.get("config", config),
        backend=backend,
        workload_id=workload_id(config),
        fixture_identity=fixture_identity(config),
    )
    return record


def schedule(mode, seeds, backends):
    tasks = []
    if mode in {"pilot", "full"}:
        for c in configs("validate"):
            for b in backends:
                tasks.append((c, b, "validation", "validation", True))
    stages = ("pilot", "core", "combined", "frontier") if mode == "full" else (mode,)
    for role in ("control_start", "measurement", "control_end"):
        role_stages = stages if role == "measurement" else (None,)
        for stage in role_stages:
            for i, seed in enumerate(seeds):
                session = f"session-{seed}"
                order = backends if i % 2 == 0 else list(reversed(backends))
                if role != "measurement":
                    if mode in {"smoke", "validate"}:
                        continue
                    groups = [
                        [
                            resolve_config({"encoder": "y", "dtype": d, "seed": seed})
                            for d in ("fp64", "estimator_f32")
                        ]
                    ]
                else:
                    pairs = {}
                    for c in configs(stage):
                        pairs.setdefault(workload_id(c), []).append(c)
                    groups = list(pairs.values())
                    random.Random(seed).shuffle(groups)
                for group in groups:
                    group.sort(key=lambda c: c["dtype"], reverse=i % 2 == 1)
                    for row in group:
                        c = dict(row, seed=seed)
                        for backend in order:
                            tasks.append(
                                (c, backend, session, role, mode == "validate")
                            )
    result, seen = [], set()
    for t in tasks:
        identity = (digest(t[0]), *t[1:])
        if identity not in seen:
            result.append(t)
            seen.add(identity)
    return result


def freeze_bundle(artifacts):
    path = artifacts / "benchmark-reference.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        files = [
            *source_files(),
            ROOT / "benchmarks/__init__.py",
            *ROOT.glob("tests/test_conditional_pipeline*.py"),
        ]
        files += [
            ROOT / n
            for n in ("pyproject.toml", "uv.lock", "README.md", "LICENSE")
            if (ROOT / n).exists()
        ]
        files += list((ROOT / "benchmarks/conditional_pipeline").glob("*.md"))
        for p in files:
            archive.add(p, arcname="source/" + str(p.relative_to(ROOT)))
        for p in artifacts.rglob("*"):
            if p.is_file() and p != path and not p.name.endswith(".tar.gz"):
                archive.add(p, arcname="evidence/" + str(p.relative_to(artifacts)))
    (artifacts / "bundle.sha256").write_text(f"{file_hash(path)}  {path.name}\n")
    return path


_TRAINING_REQUIRED_FIELDS = tuple(
    "n_qubits steps seed batch_size n_train n_validation n_test n_ops mc_samples "
    "validation_n_ops validation_mc_samples validation_interval betas "
    "development_betas evaluation_betas learning_rate final_learning_rate "
    "initialization initial_scale initial_seed_offset encoder_axes ansatz "
    "ancilla".split()
)
_TRAINING_IMPLEMENTATION_FIELDS = {
    "precision_policy",
    "sample_chunk",
    "profile",
    "benchmark_kind",
}


def training_config_differences(reference, candidate):
    """Freeze the fit while permitting precision and implementation chunking."""
    fields = (set(reference) | set(candidate) | set(_TRAINING_REQUIRED_FIELDS)) - (
        _TRAINING_IMPLEMENTATION_FIELDS
    )
    return [
        {
            "field": key,
            "reference": reference.get(key, "missing"),
            "candidate": candidate.get(key, "missing"),
        }
        for key in sorted(fields)
        if key not in reference
        or key not in candidate
        or digest(reference[key]) != digest(candidate[key])
    ]


def compare_records(reference, candidates):
    """Compare actual outputs across forks without requiring equal source hashes."""
    import numpy as np

    from benchmarks.conditional_pipeline.contracts import compare_kernel_results

    def key(row):
        return (
            row.get("benchmark_kind", "operations"),
            row.get("fixture_hash"),
            row.get("backend"),
        )

    for row in candidates:
        if row.get("status") != "ok" or not row.get("fixture_hash"):
            row["reference_comparison"] = {
                "status": "unverified",
                "timing_comparable": False,
                "reason": "candidate execution incomplete or fixture identity missing",
            }
            continue
        matches = [
            r
            for r in reference
            if r.get("status") == "ok"
            and key(r) == key(row)
            and r.get("role", "measurement") == "measurement"
        ]
        fp64 = [r for r in matches if r["config"].get("dtype") == "fp64"]
        if fp64:
            matches = fp64
        if len(matches) != 1:
            row["reference_comparison"] = {
                "status": "unverified",
                "reason": "need exactly one matching fixture/backend reference",
                "matching_rows": len(matches),
            }
            continue
        ref = matches[0]
        training = row.get("benchmark_kind") == "training"
        if training:
            differences = training_config_differences(ref["config"], row["config"])
            if differences or any(
                r.get("completed_updates") != r["config"].get("steps")
                for r in (ref, row)
            ):
                row["reference_comparison"] = {
                    "status": "unverified",
                    "configuration_match": False,
                    "timing_comparable": False,
                    "config_differences": differences,
                    "reason": (
                        "frozen training configuration differs "
                        "or update budget incomplete"
                    ),
                    "reference_source": ref.get("source_identity"),
                    "candidate_source": row.get("source_identity"),
                    "speedups": {},
                }
                continue
        try:
            pk = "output_path" if training else "kernel_output_path"
            hk = "output_file_hash" if training else "kernel_output_file_hash"
            for record in (row, ref):
                if not record.get(pk) or file_hash(record[pk]) != record.get(hk):
                    raise ValueError("reference or candidate output checksum failed")
            with (
                np.load(ref[pk], allow_pickle=False) as x,
                np.load(row[pk], allow_pickle=False) as y,
            ):
                if training:
                    if row["completed_updates"] != ref["completed_updates"]:
                        raise ValueError("training update budgets differ")
                    leaves = (
                        "best_params",
                        "final_params",
                        "train_losses",
                        "validation_losses",
                    )
                    comparison = compare_kernel_results(
                        {f"{k}__0": x[k] for k in leaves},
                        {f"{k}__0": y[k] for k in leaves},
                    )
                    comparison["reference_outcomes"] = ref.get("outcomes")
                    comparison["candidate_outcomes"] = row.get("outcomes")
                else:
                    # Compare only the selected operations.
                    comparison = compare_kernel_results(
                        {k: x[k] for k in y.files if k in x}, dict(y)
                    )
            comparison.update(
                reference_source=ref.get("source_identity"),
                candidate_source=row.get("source_identity"),
                reference_path=ref[pk],
                reference_precision=ref["config"].get(
                    "precision_policy", ref["config"].get("dtype")
                ),
                candidate_precision=row["config"].get(
                    "precision_policy", row["config"].get("dtype")
                ),
                review_status="requires researcher review",
            )
            if training:
                comparison["configuration_match"] = True
                comparison["implementation_changes"] = [
                    key
                    for key in sorted(_TRAINING_IMPLEMENTATION_FIELDS)
                    if ref["config"].get(key) != row["config"].get(key)
                ]
            policy_same = ref.get("measurement_policy") == row.get("measurement_policy")
            if not training:
                policy_same &= all(
                    ref["config"].get(k) == row["config"].get(k)
                    for k in ("repeats", "warmups")
                )
            host_same = bool(ref.get("environment_identity")) and ref.get(
                "environment_identity"
            ) == row.get("environment_identity")
            stack_same = bool(ref.get("stack_identity")) and ref.get(
                "stack_identity"
            ) == row.get("stack_identity")
            comparison["timing_comparable"] = policy_same and host_same and stack_same
            comparison["timing_comparability_reason"] = (
                "matched host, stack and measurement policy"
                if comparison["timing_comparable"]
                else "host, stack or measurement policy differs or is missing"
            )
            comparison["timing_scope"] = (
                "diagnostic comparison; no stability or numerical acceptance implied"
            )
            comparison["speedups"] = {}
            if comparison["timing_comparable"] and comparison["status"] != "failed":
                import statistics

                for label, timing in row.get("timings_seconds", {}).items():
                    old = ref.get("timings_seconds", {}).get(label)
                    if isinstance(timing, dict) and isinstance(old, dict):
                        a, b = old.get("warm", []), timing.get("warm", [])
                        if a and b and all(np.isfinite(t) and t > 0 for t in a + b):
                            comparison["speedups"][label] = statistics.median(
                                a
                            ) / statistics.median(b)
                    elif (
                        training
                        and label == "total_training"
                        and isinstance(timing, (int, float))
                        and isinstance(old, (int, float))
                        and np.isfinite(timing)
                        and np.isfinite(old)
                        and timing > 0
                        and old > 0
                    ):
                        comparison["speedups"][label] = old / timing
            row["reference_comparison"] = comparison
        except (ValueError, KeyError, OSError) as exc:
            row["reference_comparison"] = {"status": "unverified", "reason": str(exc)}
    return candidates


def collect_training(args, artifacts, smoke, *, write=True):
    from benchmarks.conditional_pipeline.training_workload import (
        resolve_training_config,
    )

    c = resolve_training_config(smoke=smoke)
    c.update(
        seed=args.seed if args.seed is not None else 410,
        precision_policy=args.training_precision,
        benchmark_kind="training",
    )
    if args.steps is not None:
        c["steps"] = args.steps
        if args.steps != (2 if smoke else 2000):
            c["profile"] = "custom-smoke" if smoke else "custom-paper"
    c = resolve_training_config(c)
    if args.cell_timeout <= 0 or (
        args.budget_seconds is not None and args.budget_seconds <= 0
    ):
        raise ValueError("budgets/timeouts must be positive")
    if args.dry_run:
        print(
            json.dumps(
                {"config": c, "backend": args.backend, "single_fit": True}, indent=2
            )
        )
        return 0
    if write and args.output.exists() and not args.resume:
        raise ValueError("output exists; use --resume or a new --output")
    manifest = make_manifest()
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / f"manifest-{manifest['manifest_id'][:16]}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Reuse a completed fit only when its inputs and environment match.
    if write and args.resume:
        for saved in reversed(read_records(args.output)):
            saved_config = {
                k: v for k, v in saved.get("config", {}).items() if k != "sigmas"
            }
            if (
                saved.get("benchmark_kind") == "training"
                and saved.get("status") == "ok"
                and saved.get("backend") == args.backend
                and saved.get("completed_updates") == c["steps"]
                and digest(saved_config) == digest(c)
                and all(
                    saved.get(k) == manifest[k]
                    for k in (
                        "source_identity",
                        "stack_identity",
                        "environment_identity",
                    )
                )
                and all(
                    Path(saved.get(pk, "")).is_file()
                    and file_hash(saved[pk]) == saved.get(hk)
                    for pk, hk in (
                        ("fixture_path", "fixture_file_hash"),
                        ("output_path", "output_file_hash"),
                    )
                )
            ):
                rows = read_records(args.output)
                if args.compare_reference:
                    compare_records(read_records(args.compare_reference), rows)
                summary = summarize(rows, artifacts / "report")
                print_summary(
                    args,
                    rows,
                    summary,
                    artifacts / "report",
                    payload={
                        "records": str(args.output),
                        "status": "reused",
                        "completed_updates": saved["completed_updates"],
                    },
                )
                return 0
    timeout = min(args.cell_timeout, args.budget_seconds or args.cell_timeout)
    row = run_child(
        c,
        args.backend,
        artifacts / "runs" / uuid.uuid4().hex[:12],
        timeout,
        fixtures_dir=args.fixtures,
    )
    row.update(
        benchmark_kind="training",
        role="measurement",
        workload_id=digest({k: v for k, v in c.items() if k != "seed"})[:16],
        session_id=f"training-{c['seed']}",
        manifest_id=manifest["manifest_id"],
        manifest_path=str(manifest_path),
    )
    row.update(
        {
            k: manifest[k]
            for k in ("source_identity", "stack_identity", "environment_identity")
        }
    )
    row.update(
        {
            k: manifest.get(k)
            for k in ("library_source_identity", "harness_source_identity")
        }
    )
    if (
        row.get("executed_source_identity", manifest["source_identity"])
        != manifest["source_identity"]
    ):
        row.update(status="failed", error="source changed during training")
    if not write:
        return row
    if args.compare_reference:
        compare_records(read_records(args.compare_reference), [row])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    rows = read_records(args.output)
    summary = summarize(rows, artifacts / "report")
    if args.bundle:
        freeze_bundle(artifacts)
    print_summary(
        args,
        rows,
        summary,
        artifacts / "report",
        payload={
            "records": str(args.output),
            "status": row["status"],
            "completed_updates": row.get("completed_updates"),
            "timings_seconds": row.get("timings_seconds"),
        },
    )
    return int(row["status"] == "failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    names = (
        "cell",
        "smoke",
        "validate",
        "pilot",
        "core",
        "combined",
        "frontier",
        "full",
        "training",
        "training_smoke",
        "cpu_check",
        "summary",
    )
    for mode in names:
        modes.add_argument(f"--{mode.replace('_', '-')}", action="store_true")
    parser.add_argument("--child-config", help=argparse.SUPPRESS)
    parser.add_argument("--check-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--backends", default="cpu,gpu")
    parser.add_argument("--seeds", default="1729,1730,1731")
    parser.add_argument("--kernels")
    parser.add_argument("--encoder", choices=("y", "yz", "yzyz"))
    parser.add_argument("--dtype", choices=("fp64", "estimator_f32"))
    for name in (
        "n-qubits",
        "n-conditions",
        "n-generators",
        "arity",
        "n-ops",
        "n-samples",
        "n-targets",
        "seed",
        "repeats",
        "warmups",
    ):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--sample-chunk")
    parser.add_argument("--mask-band", choices=("low", "mid", "high"))
    parser.add_argument("--ansatz", choices=("uniform", "complete"))
    parser.add_argument(
        "--select", action="append", default=[], help="config key=value filter"
    )
    for flag in ("dry-run", "resume", "trace", "bundle"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--cell-timeout", type=float, default=1200)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/conditional_pipeline/results/measurements.jsonl",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable completion output instead of timing tables",
    )
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--fixtures", type=Path, help="shared frozen fixture directory")
    parser.add_argument(
        "--compare-reference", type=Path, help="reference JSONL from another checkout"
    )
    parser.add_argument("--steps", "--n-epochs", type=int, dest="steps")
    parser.add_argument(
        "--training-precision",
        # iqp_cond.PRECISIONS; listed here so the CLI does not import JAX.
        choices=("fp64", "gemm_f32", "estimator_f32", "fp32"),
        default="fp64",
    )
    args = parser.parse_args(argv)
    result_path = args.input if args.summary and args.input else args.output
    artifacts = (args.artifacts or result_path.parent).resolve()
    kernels = args.kernels.split(",") if args.kernels else None
    if args.child_config:
        child_config = json.loads(args.child_config)
        if child_config.get("benchmark_kind") == "training":
            from benchmarks.conditional_pipeline.training_workload import (
                run_training_cell,
            )

            row = run_training_cell(
                child_config,
                backend=args.backend,
                artifacts=artifacts,
                fixtures_dir=args.fixtures,
            )
            row["executed_source_identity"] = digest(
                {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}
            )
        else:
            row = run_cell(
                child_config,
                backend=args.backend,
                artifacts=artifacts,
                kernels=kernels,
                validate=args.check_only,
                trace=args.trace,
                fixtures_dir=args.fixtures,
            )
        print(json.dumps(row), flush=True)
        return int(row["status"] == "failed")
    if args.summary:
        rows = reconcile(read_records(args.input or args.output))
        if args.compare_reference:
            compare_records(read_records(args.compare_reference), rows)
        summary = summarize(rows, artifacts / "report")
        print_summary(args, rows, summary, artifacts / "report")
        return 0
    mode = next((m for m in names[:-1] if getattr(args, m)), None)
    if not mode:
        parser.error("choose a benchmark mode")
    if mode in {"training", "training_smoke"}:
        return collect_training(args, artifacts, mode == "training_smoke")
    cpu_check = mode == "cpu_check"
    if cpu_check:
        mode = "smoke"
        if args.backend != "cpu":
            parser.error("--cpu-check requires --backend cpu")
    backends = ["cpu"] if cpu_check else args.backends.split(",")
    if any(b not in {"cpu", "gpu"} for b in backends):
        parser.error("backends must be cpu and/or gpu")
    fields = (
        "n_qubits",
        "n_conditions",
        "n_generators",
        "arity",
        "n_ops",
        "n_samples",
        "n_targets",
        "seed",
        "repeats",
        "warmups",
        "encoder",
        "dtype",
        "mask_band",
        "ansatz",
    )
    overrides = {k: getattr(args, k) for k in fields if getattr(args, k) is not None}
    if args.sample_chunk is not None:
        overrides["sample_chunk"] = (
            None if args.sample_chunk == "none" else int(args.sample_chunk)
        )
    if mode == "cell":
        base = dict(n_qubits=8, n_generators=8, n_ops=4, n_samples=16, n_targets=16)
        base.update(overrides)
        if args.n_generators is None:
            base["n_generators"] = min(64, base["n_qubits"])
        c = resolve_config(base)
        tasks = [(c, args.backend, f"session-{c['seed']}", "measurement", False)]
    else:
        seeds = (
            [args.seed]
            if args.seed is not None
            else [int(x) for x in args.seeds.split(",")]
        )
        tasks = schedule(mode, seeds, backends)
        tasks = [
            (resolve_config(dict(c, **overrides)), b, s, r, v)
            for c, b, s, r, v in tasks
        ]
    for selection in args.select:
        k, value = selection.split("=", 1)
        tasks = [t for t in tasks if str(t[0].get(k)).lower() == value.lower()]
    if not tasks:
        parser.error("selection matches no workloads")
    from benchmarks.conditional_pipeline.workloads import build_kernels

    if kernels:
        unknown = set(kernels) - set(build_kernels(tasks[0][0]))
        if unknown:
            parser.error(f"unknown kernels: {sorted(unknown)}")
    budget = (
        args.budget_seconds
        if args.budget_seconds is not None
        else (7200 if mode == "pilot" else 86400)
    )
    if budget <= 0 or args.cell_timeout <= 0:
        parser.error("budgets/timeouts must be positive")
    manifest = make_manifest()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": manifest,
                    "budget_seconds": budget,
                    "cells": [
                        {
                            "config": c,
                            "backend": b,
                            "session_id": s,
                            "role": r,
                            "check_only": v,
                        }
                        for c, b, s, r, v in tasks
                    ],
                    "includes_training_smoke": cpu_check,
                },
                indent=2,
            )
        )
        return 0
    artifacts.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        parser.error("output exists; use --resume or a new --output")
    manifest_path = artifacts / f"manifest-{manifest['manifest_id'][:16]}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    prior = reconcile(read_records(args.output)) if args.resume else []
    records, executed = [], 0
    run_identity = digest(
        [
            manifest[k]
            for k in ("source_identity", "stack_identity", "environment_identity")
        ]
    )[:24]
    start = time.monotonic()
    segment = uuid.uuid4().hex[:12]
    prerequisite_failed = False
    for c, b, session, role, check_only in tasks:
        found = next(
            (r for r in prior if reusable(r, c, b, session, role, manifest)), None
        )
        if found is not None:
            policy = found.get("measurement_policy", {})
            expected_kernels = (
                ["mmd_gradient"]
                if role.startswith("control")
                else kernels or list(build_kernels(c))
            )
            if (
                policy.get("check_only") != check_only
                or (set(expected_kernels) != set(policy.get("kernels", [])))
                or (
                    args.trace
                    and role == "measurement"
                    and c["n_qubits"] == 64
                    and not policy.get("trace")
                )
                or role.startswith("control")
            ):
                found = None
        if found:
            # Historical rows retain their original controls/session segment.
            records.append(found)
            continue
        remaining = budget - (time.monotonic() - start)
        if (
            prerequisite_failed
            or remaining <= 1
            or (args.max_cells is not None and executed >= args.max_cells)
        ):
            row = {
                "status": "deferred",
                "error": "prerequisite validation failed"
                if prerequisite_failed
                else "campaign budget or selection cap",
                "config": c,
                "backend": b,
                "workload_id": workload_id(c),
                "fixture_identity": fixture_identity(c),
            }
        else:
            row = run_child(
                c,
                b,
                artifacts / "runs" / run_identity / segment / session / role,
                min(args.cell_timeout, remaining),
                ["mmd_gradient"] if role.startswith("control") else kernels,
                check_only,
                args.trace and role == "measurement" and c["n_qubits"] == 64,
                *([args.fixtures] if args.fixtures else []),
            )
            executed += 1
            if (
                row.get("executed_source_identity", manifest["source_identity"])
                != manifest["source_identity"]
            ):
                row.update(status="failed", error="source changed during campaign")
        row.update(
            session_id=session,
            measurement_segment=segment,
            role=role,
            manifest_path=str(manifest_path),
            manifest_id=manifest["manifest_id"],
            source_identity=manifest["source_identity"],
            stack_identity=manifest["stack_identity"],
            environment_identity=manifest["environment_identity"],
            library_source_identity=manifest.get("library_source_identity"),
            harness_source_identity=manifest.get("harness_source_identity"),
        )
        records.append(row)
        if check_only:
            reconcile(records)
            prerequisite_failed |= row["status"] != "ok" or any(
                row.get(k, {}).get("status") == "failed"
                for k in ("validity", "independent_checks", "compiled_checks")
            )
        with args.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    reconcile(prior + records)
    # Keep historical rows when a resumed selection is narrower.
    selected = {
        (
            r.get("workload_id"),
            r.get("session_id"),
            r.get("backend"),
            r.get("role"),
            r["config"]["dtype"],
            r.get("source_identity"),
            r.get("stack_identity"),
            r.get("environment_identity"),
            r.get("measurement_segment"),
        )
        for r in records
    }
    history = [
        r
        for r in prior
        if (
            r.get("workload_id"),
            r.get("session_id"),
            r.get("backend"),
            r.get("role"),
            r["config"].get("dtype"),
            r.get("source_identity"),
            r.get("stack_identity"),
            r.get("environment_identity"),
            r.get("measurement_segment"),
        )
        not in selected
    ]
    if cpu_check:
        remaining = budget - (time.monotonic() - start)
        if remaining <= 1 or (
            args.max_cells is not None and executed >= args.max_cells
        ):
            training = {
                "benchmark_kind": "training",
                "status": "deferred",
                "backend": "cpu",
                "config": {"profile": "smoke"},
                "error": "campaign budget or selection cap",
            }
        else:
            args.cell_timeout = min(args.cell_timeout, remaining)
            training = collect_training(args, artifacts / "training", True, write=False)
        records.append(training)
    if args.compare_reference:
        compare_records(read_records(args.compare_reference), records)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        "".join(json.dumps(r) + "\n" for r in history + records)
    )
    temporary_output.replace(args.output)
    report = summarize(records, artifacts / "report")
    if args.bundle:
        freeze_bundle(artifacts)
    print_summary(
        args,
        records,
        report,
        artifacts / "report",
        payload={
            "records": str(args.output),
            "report": str(artifacts / "report/report.md"),
            "counts": report["counts"],
        },
    )
    return int(
        any(
            r["status"] == "failed" or r.get("validity", {}).get("status") == "failed"
            for r in records
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
