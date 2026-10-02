import importlib.util
from pathlib import Path

_path = Path(__file__).parents[1] / "benchmarks/conditional_pipeline/analysis.py"
_spec = importlib.util.spec_from_file_location("conditional_pipeline_analysis", _path)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize, write_report = _module.summarize, _module.write_report


def _record(
    backend,
    *,
    fixture="f",
    source="s",
    stack="stack",
    role="measurement",
    warm=(2.0, 2.1),
):
    config = dict(
        encoder="y",
        dtype="fp64",
        n_qubits=64,
        n_conditions=1,
        n_generators=64,
        arity=2,
        ansatz="complete",
        n_ops=64,
        n_samples=1024,
        n_targets=1024,
        mask_band=2,
        sample_chunk=256,
    )
    return dict(
        status="ok",
        config=config,
        backend=backend,
        session_id="session-1729",
        role=role,
        fixture_hash=fixture,
        source_identity=source,
        stack_identity=stack,
        environment_identity="same-host",
        validity={"status": "passed"},
        timings_seconds={"mmd_gradient": {"warm": list(warm)}},
    )


def test_missing_controls_and_fixture_mismatch_are_not_paired():
    result = summarize([_record("cpu"), _record("gpu", fixture="other")])
    assert result["paired_speedups"] == []


def test_control_drift_invalidates_pair_but_retains_rows():
    records = [
        _record("cpu", role="control_start", warm=(1.0,)),
        _record("cpu", role="control_end", warm=(2.0,)),
        _record("gpu", role="control_start", warm=(1.0,)),
        _record("gpu", role="control_end", warm=(1.0,)),
        _record("cpu"),
        _record("gpu"),
    ]
    result = summarize(records)
    assert result["counts"]["records"] == 6
    assert result["paired_speedups"] == []


def test_report_writes_machine_and_human_outputs(tmp_path):
    records = [_record("cpu"), _record("gpu")]
    # Smoke records intentionally have no controls and therefore no accepted pair.
    report = write_report(records, tmp_path)
    assert report.exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()


def test_valid_controls_pair_different_measurement_fixture():
    records = []
    for backend in ("cpu", "gpu"):
        records += [
            _record(backend, role="control_start", fixture="anchor", warm=(1.0,)),
            _record(backend, role="control_end", fixture="anchor", warm=(1.0,)),
            _record(
                backend, fixture="measurement", warm=(2.0 if backend == "cpu" else 1.0,)
            ),
        ]
    result = summarize(records)
    assert result["paired_speedups"][0]["speedup"] == 2.0
    records[1]["validity"]["status"] = "failed"
    assert not summarize(records)["paired_speedups"]


def test_two_repeat_quantiles_bracket_median():
    d = _module._percentiles([1.0, 3.0])
    assert d["p25"] <= d["median"] <= d["p75"]
    assert d["p25"] == 1.5 and d["p75"] == 2.5


def _controlled_pair(*, source="s", environment="same-host", policy=None, scale=1.0):
    records = []
    for backend in ("cpu", "gpu"):
        records += [
            _record(
                backend,
                source=source,
                role="control_start",
                fixture="anchor",
                warm=(1.0,),
            ),
            _record(
                backend,
                source=source,
                role="control_end",
                fixture="anchor",
                warm=(1.0,),
            ),
            _record(
                backend,
                source=source,
                fixture="measurement",
                warm=((2.0 if backend == "cpu" else 1.0) * scale,),
            ),
        ]
        for r in records[-3:]:
            r["environment_identity"] = environment
            if policy is not None:
                r["measurement_policy"] = policy
    return records


def test_speedup_summary_keeps_source_cohorts_separate():
    first = _controlled_pair(source="v1")
    second = _controlled_pair(source="v2")
    gpu = next(
        r for r in second if r["backend"] == "gpu" and r["role"] == "measurement"
    )
    gpu["timings_seconds"]["mmd_gradient"]["warm"] = [0.1]
    result = summarize(first + second)
    assert len(result["paired_speedup_summary"]) == 2
    assert sorted(
        r["distribution"]["median"] for r in result["paired_speedup_summary"]
    ) == [2.0, 20.0]


def test_speedup_summary_keeps_environment_and_policy_separate():
    records = _controlled_pair(environment="host-a", policy="warm-v1")
    records += _controlled_pair(environment="host-b", policy="warm-v2")
    result = summarize(records)
    assert len(result["paired_speedup_summary"]) == 2


def test_invalid_timing_is_not_paired():
    records = _controlled_pair()
    gpu = next(
        r for r in records if r["backend"] == "gpu" and r["role"] == "measurement"
    )
    gpu["timings_seconds"]["mmd_gradient"]["warm"] = [0.0]
    result = summarize(records)
    assert result["paired_speedups"] == []


def test_training_rows_are_reported(tmp_path):
    record = _record("cpu")
    record.update(
        benchmark_kind="training",
        total_time=12.5,
        completed_updates=2000,
        profile="curie-weiss",
        outcome={"status": "completed"},
    )
    summary = summarize([record], output_dir=tmp_path)
    assert summary["training_runs"][0]["total_time"] == 12.5
    report = (tmp_path / "report.md").read_text()
    assert "12.5" in report
    assert "2000" in report


def test_timing_csv_contains_readable_configuration(tmp_path):
    import csv

    write_report([_record("cpu")], tmp_path)
    with (tmp_path / "summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["encoder"] == "y"
    assert row["n_qubits"] == "64"
    assert row["n_samples"] == "1024"
    assert row["sample_chunk"] == "256"


def test_resumed_measurement_cannot_use_previous_segment_controls():
    records = _controlled_pair()
    for r in records:
        r["measurement_segment"] = "old"
        r["measurement_policy"] = {"kernels": ["mmd_gradient"], "trace": False}
    assert summarize(records)["paired_speedups"]
    for r in records:
        if r["role"] == "measurement":
            r["measurement_segment"] = "resumed"
    assert not summarize(records)["paired_speedups"]
