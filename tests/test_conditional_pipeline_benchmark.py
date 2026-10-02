"""Collector isolation, durable failures, budgeting and resume provenance."""

import json
from pathlib import Path

import pytest

from benchmarks.conditional_pipeline import benchmark as b
from benchmarks.conditional_pipeline.fixtures import configs


def test_summary_cli_is_readable_and_preserves_raw_records(tmp_path, capsys):
    path = tmp_path / "measurements.jsonl"
    row = {
        "status": "ok",
        "backend": "cpu",
        "role": "measurement",
        "config": configs("smoke")[0],
        "validity": {"status": "unverified"},
        "timings_seconds": {"phase": {"warm": [1e-5, 2e-5], "compile": 0.01}},
    }
    raw = json.dumps(row) + "\n"
    path.write_text(raw)
    assert b.main(["--summary", "--input", str(path)]) == 0
    output = capsys.readouterr().out
    assert "phase" in output.lower()
    assert "unverified" in output
    assert "report.md" in output
    assert path.read_text() == raw
    assert (tmp_path / "report/report.md").is_file()
    assert (
        b.main(
            [
                "--summary",
                "--input",
                str(path),
                "--artifacts",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["records"] == 1


def test_failure_rows_and_budget_deferrals(monkeypatch, tmp_path):
    calls = []

    def fail(config, backend, *_args):
        calls.append(config)
        return dict(
            status="failed",
            config=config,
            backend=backend,
            workload_id=b.workload_id(config),
            error="simulated failure",
        )

    monkeypatch.setattr(b, "run_child", fail)
    output = tmp_path / "rows.jsonl"
    code = b.main(
        [
            "--smoke",
            "--backends",
            "cpu",
            "--seeds",
            "1729",
            "--max-cells",
            "1",
            "--output",
            str(output),
        ]
    )
    rows = b.read_records(output)
    assert code == 1 and len(calls) == 1
    assert len(rows) == 6 and sum(r["status"] == "deferred" for r in rows) == 5
    assert all("source_identity" in r for r in rows)


def test_dryrun_does_not_create_output(tmp_path, capsys):
    p = tmp_path / "new" / "rows.jsonl"
    assert (
        b.main(
            [
                "--core",
                "--backends",
                "cpu",
                "--seeds",
                "1729",
                "--dry-run",
                "--output",
                str(p),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["cells"] and not p.parent.exists()


def test_resume_rejects_changed_source_or_corrupt_fixture(tmp_path):
    c = configs("smoke")[0]
    manifest = dict(source_identity="s", stack_identity="v", environment_identity="e")
    fixture = tmp_path / "fixture"
    fixture.write_text("fixture")
    output = tmp_path / "output"
    output.write_text("output")
    r = dict(
        status="ok",
        validity={"status": "passed"},
        config=c,
        backend="cpu",
        session_id="session-1729",
        role="measurement",
        **manifest,
        fixture_identity=b.fixture_identity(c),
        fixture_path=str(fixture),
        fixture_file_hash=b.file_hash(fixture),
        output_path=str(output),
        output_file_hash=b.file_hash(output),
    )
    assert b.reusable(r, c, "cpu", "session-1729", "measurement", manifest)
    assert not b.reusable(
        r,
        c,
        "cpu",
        "session-1729",
        "measurement",
        dict(manifest, source_identity="new"),
    )
    fixture.write_text("corrupt")
    assert not b.reusable(r, c, "cpu", "session-1729", "measurement", manifest)


def test_backend_selection_and_cache_disabled():
    assert b.child_environment("cpu")["JAX_PLATFORMS"] == "cpu"
    assert b.child_environment("gpu")["JAX_PLATFORMS"] == "cuda,cpu"
    assert b.child_environment("gpu")["JAX_ENABLE_COMPILATION_CACHE"] == "false"


def test_timeout_and_oom_diagnostics(monkeypatch, tmp_path):
    import subprocess

    class Fake:
        returncode = -9

        def __init__(self, *_args, **_kwargs):
            self.first = True

        def communicate(self, timeout=None):
            if self.first:
                self.first = False
                raise subprocess.TimeoutExpired("child", timeout)
            return "", "diagnostic"

        def kill(self):
            pass

    monkeypatch.setattr(b.subprocess, "Popen", Fake)
    r = b.run_child(configs("smoke")[0], "cpu", tmp_path, 1)
    assert r["status"] == "failed" and "timeout" in r["error"]
    assert Path(r["stderr_path"]).read_text() == "diagnostic"


def test_input_lower_bound_capacity(monkeypatch):
    monkeypatch.setattr(b, "available_memory", lambda: 100)
    assert b.preflight(configs("smoke")[0]) is not None


def test_reconciliation_requires_same_host_and_intact_outputs(tmp_path):
    import numpy as np

    c = configs("smoke")[0]
    outputs = dict(
        moments=np.ones((c["n_conditions"], c["n_ops"])),
        standard_errors=np.zeros((c["n_conditions"], c["n_ops"])),
        loss=np.array(0.1),
        gradient=np.ones(c["n_generators"]),
    )
    path = tmp_path / "output.npz"
    np.savez(path, **outputs)
    row = dict(
        status="ok",
        config=dict(c, dtype="fp64"),
        backend="cpu",
        workload_id=b.workload_id(c),
        fixture_hash="fixture",
        source_identity="s",
        stack_identity="stack",
        environment_identity="host",
        output_path=str(path),
        output_file_hash=b.file_hash(path),
        independent_checks={"status": "passed"},
    )
    mixed = dict(
        row, config=dict(c, dtype="estimator_f32"), environment_identity="other"
    )
    b.reconcile([row, mixed])
    assert row["validity"]["status"] == "passed"
    assert mixed["validity"]["status"] == "unverified"
    mixed["environment_identity"] = "host"
    b.reconcile([row, mixed])
    assert mixed["validity"]["status"] == "passed"
    path.write_bytes(b"corrupt")
    b.reconcile([row, mixed])
    assert row["validity"]["status"] == mixed["validity"]["status"] == "unverified"


def test_backend_mismatch_fails_before_fixture(monkeypatch, tmp_path):
    import jax

    class CPU:
        platform = "cpu"

    monkeypatch.setattr(jax, "devices", lambda _backend: [CPU()])
    row = b.run_cell(configs("smoke")[0], backend="gpu", artifacts=tmp_path)
    assert row["status"] == "failed" and "requested gpu, got cpu" in row["error"]
    assert not row.get("fixture_path")


def test_explicit_oom_is_retained_as_capacity(monkeypatch, tmp_path):
    class OOM:
        returncode = 1

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, timeout=None):
            return "", "RESOURCE_EXHAUSTED: out of memory"

    monkeypatch.setattr(b.subprocess, "Popen", OOM)
    row = b.run_child(configs("smoke")[0], "gpu", tmp_path, 1)
    assert row["status"] == "capacity"
    assert "out of memory" in Path(row["stderr_path"]).read_text()


def test_cross_fork_compares_actual_outputs_and_detects_corruption(tmp_path):
    import numpy as np

    refpath = tmp_path / "ref.npz"
    candidatepath = tmp_path / "candidate.npz"
    np.savez(refpath, phase__0=np.array([1.0, 2.0]))
    np.savez(candidatepath, phase__0=np.array([1.0, 3.0]))
    c = configs("smoke")[0]
    base = dict(
        config=c,
        status="ok",
        backend="cpu",
        role="measurement",
        fixture_hash="fixed",
        environment_identity="host",
        stack_identity="stack",
        measurement_policy={"kernels": ["phase"]},
        timings_seconds={"phase": {"warm": [2.0]}},
    )
    ref = dict(
        base,
        source_identity="reference",
        kernel_output_path=str(refpath),
        kernel_output_file_hash=b.file_hash(refpath),
    )
    candidate = dict(
        base,
        source_identity="candidate",
        kernel_output_path=str(candidatepath),
        kernel_output_file_hash=b.file_hash(candidatepath),
    )
    b.compare_records([ref], [candidate])
    comparison = candidate["reference_comparison"]
    assert comparison["status"] == "compared"
    assert comparison["errors"]["phase__0"]["max_abs"] == 1
    assert comparison["review_status"] == "requires researcher review"
    assert comparison["timing_comparable"]
    candidatepath.write_text("corrupt")
    b.compare_records([ref], [candidate])
    assert candidate["reference_comparison"]["status"] == "unverified"


def _training_pair(tmp_path):
    import numpy as np

    from benchmarks.conditional_pipeline.training_workload import (
        resolve_training_config,
    )

    path = tmp_path / "training.npz"
    np.savez(
        path,
        best_params=np.ones(2),
        final_params=np.ones(2),
        train_losses=np.ones(2),
        validation_losses=np.ones(3),
    )
    c = resolve_training_config(smoke=True)
    base = dict(
        config=c,
        benchmark_kind="training",
        status="ok",
        backend="cpu",
        completed_updates=2,
        fixture_hash="fixed",
        role="measurement",
        output_path=str(path),
        output_file_hash=b.file_hash(path),
        environment_identity="host",
        stack_identity="stack",
        timings_seconds={"total_training": 2.0},
    )
    return dict(base, source_identity="reference"), dict(
        base,
        config=dict(c),
        source_identity="candidate",
        timings_seconds={"total_training": 1.0},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 0.02),
        ("final_learning_rate", 0.002),
        ("batch_size", 16),
        ("mc_samples", 32),
        ("validation_interval", 2),
        ("validation_n_ops", 8),
        ("development_betas", [0.9, 1.1]),
        ("initial_seed_offset", 102),
        ("new_optimizer", "sgd"),
    ],
)
def test_training_comparison_rejects_changed_contract(tmp_path, field, value):
    ref, candidate = _training_pair(tmp_path)
    candidate["config"][field] = value
    b.compare_records([ref], [candidate])
    comparison = candidate["reference_comparison"]
    assert comparison["status"] == "unverified"
    assert not comparison["configuration_match"]
    assert not comparison["timing_comparable"] and not comparison["speedups"]
    assert field in [d["field"] for d in comparison["config_differences"]]


def test_training_comparison_allows_precision_and_chunking(tmp_path):
    ref, candidate = _training_pair(tmp_path)
    candidate["config"].update(precision_policy="fp64", gemm_dtype=None, sample_chunk=4)
    b.compare_records([ref], [candidate])
    comparison = candidate["reference_comparison"]
    assert comparison["configuration_match"]
    assert comparison["status"] == "compared"
    assert comparison["speedups"]["total_training"] == 2.0
    assert comparison["reference_precision"] == "gemm_f32"
    assert set(comparison["implementation_changes"]) == {
        "precision_policy",
        "gemm_dtype",
        "sample_chunk",
    }


def test_training_comparison_requires_contract_and_completed_budget(tmp_path):
    ref, candidate = _training_pair(tmp_path)
    del ref["config"]["learning_rate"]
    del candidate["config"]["learning_rate"]
    b.compare_records([ref], [candidate])
    assert candidate["reference_comparison"]["status"] == "unverified"
    ref, candidate = _training_pair(tmp_path)
    candidate["completed_updates"] = 1
    b.compare_records([ref], [candidate])
    assert not candidate["reference_comparison"]["timing_comparable"]


def test_training_comparison_does_not_report_invalid_total_or_missing_host(tmp_path):
    ref, candidate = _training_pair(tmp_path)
    candidate["timings_seconds"]["total_training"] = float("nan")
    b.compare_records([ref], [candidate])
    assert not candidate["reference_comparison"]["speedups"]
    ref, candidate = _training_pair(tmp_path)
    ref.pop("environment_identity")
    candidate.pop("environment_identity")
    b.compare_records([ref], [candidate])
    assert not candidate["reference_comparison"]["timing_comparable"]


def test_cpu_check_dryrun_and_training_budget(tmp_path, capsys):
    output = tmp_path / "none" / "rows.jsonl"
    assert (
        b.main(["--cpu-check", "--seeds", "1729", "--dry-run", "--output", str(output)])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["includes_training_smoke"]
    assert {r["backend"] for r in payload["cells"]} == {"cpu"}
    assert not output.parent.exists()
    assert b.main(["--training", "--dry-run", "--output", str(output)]) == 0
    c = json.loads(capsys.readouterr().out)["config"]
    assert c["steps"] == 2000 and c["n_qubits"] == 16
    assert c["n_ops"] == 256 and c["mc_samples"] == 8192


def test_failed_prerequisite_blocks_large_measurement(monkeypatch, tmp_path):
    import numpy as np

    small = configs("smoke")[0]
    large = dict(small, n_qubits=64)
    monkeypatch.setattr(
        b,
        "schedule",
        lambda *_: [
            (small, "cpu", "validation", "validation", True),
            (large, "cpu", "session-1729", "measurement", False),
        ],
    )
    calls = []

    def child(c, backend, *_):
        calls.append(c)
        p = tmp_path / "bad.npz"
        np.savez(
            p,
            moments=np.zeros((c["n_conditions"], c["n_ops"])),
            standard_errors=np.zeros((c["n_conditions"], c["n_ops"])),
            loss=np.array(0.0),
            gradient=np.zeros(c["n_generators"]),
        )
        return dict(
            status="ok",
            config=c,
            backend=backend,
            workload_id=b.workload_id(c),
            fixture_hash="f",
            output_path=str(p),
            output_file_hash=b.file_hash(p),
            independent_checks={"status": "failed"},
        )

    monkeypatch.setattr(b, "run_child", child)
    p = tmp_path / "rows.jsonl"
    assert b.main(["--pilot", "--backends", "cpu", "--output", str(p)]) == 1
    rows = b.read_records(p)
    assert len(calls) == 1 and rows[1]["status"] == "deferred"
    assert "prerequisite" in rows[1]["error"]
