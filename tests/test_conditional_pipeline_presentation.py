from benchmarks.conditional_pipeline.presentation import (
    render_markdown,
    render_terminal,
)


def _record(**kw):
    r = {
        "status": "ok",
        "backend": "cpu",
        "source_identity": "abcdef1234567890",
        "config": {
            "encoder": "yz",
            "dtype": "fp64",
            "n_qubits": 4,
            "n_conditions": 1,
            "n_generators": 4,
            "mask_band": "low",
            "n_samples": 16,
            "n_targets": 16,
            "sample_chunk": 256,
            "seed": 7,
        },
        "validity": {"status": "passed", "errors": {"loss": {"max_abs": 1e-7}}},
        "timings_seconds": {
            "phase": {
                "compile": 0.2,
                "first_execution": 0.01,
                "warm": [0.00001, 0.00003, float("nan")],
            }
        },
    }
    r.update(kw)
    return r


def test_terminal_is_human_readable_and_keeps_failed_rows():
    text = render_terminal(
        [_record(), _record(status="failed", timings_seconds={})],
        {"counts": {"ok": 1, "failed": 1}},
    )
    assert "phase" in text
    assert "Warm median [IQR] (us)" in text
    assert "Execution: failed" in text
    assert "INVALID warm samples" in text
    assert "nan" not in text.lower()
    assert "source=abcdef1234" in text


def test_markdown_preserves_distinct_cells_and_missing_values():
    text = render_markdown(
        [
            _record(),
            _record(
                source_identity="different",
                backend="gpu",
                config={"encoder": "yz"},
                timings_seconds={},
            ),
        ],
        {},
    )
    assert "Cell 1: cpu / yz / fp64" in text
    assert "Cell 2: gpu / yz / n/a" in text
    assert "n/a" in text


def test_training_is_separate_and_timings_are_labeled():
    r = _record(
        benchmark_kind="training",
        profile="smoke",
        completed_updates=2,
        total_time=1.5,
        timings_seconds={
            "training_loop": 0.2,
            "compile": 0.4,
            "validation": 0.3,
            "checkpoint_selection": 0.1,
            "final_diagnostics": 0.2,
        },
    )
    text = render_terminal([r], {})
    assert "Training runs" in text and "Total training (primary)" in text
    assert "1.5" in text
    assert "do not add" in text


def test_two_repeats_have_interpolated_iqr_and_explicit_units():
    r = _record(
        timings_seconds={
            "phase": {
                "warm": [1e-5, 3e-5],
                "compile": 0.2,
                "first_execution": 0.001,
            }
        }
    )
    text = render_markdown([r], {})
    assert "20 [15, 25]" in text
    assert "Compile (ms)" in text and "First (us)" in text
    assert "| 200 | 1000 |" in text


def test_training_nested_diagnostics_preserve_checkpoint_and_beta():
    r = _record(
        benchmark_kind="training",
        config={
            "profile": "smoke",
            "precision_policy": "gemm_f32",
            "steps": 2,
            "n_qubits": 16,
        },
        completed_updates=2,
        outcomes={
            "best": {
                "evaluation": {
                    "betas": [0.4, 1.6],
                    "raw_probability_tvd": [0.1, 0.2],
                    "profile_tvd": [0.03, 0.04],
                }
            },
            "final": {
                "evaluation": {
                    "betas": [1.6, 0.4],
                    "raw_probability_tvd": [0.6, 0.5],
                    "profile_tvd": [0.08, 0.07],
                }
            },
        },
    )
    text = render_markdown([r], {})
    assert "gemm_f32; 16 qubits" in text
    assert "updates: 2/2" in text
    assert "| evaluation | 0.4 | 0.1 | 0.03 | 0.5 | 0.07 |" in text
    assert "| evaluation | 1.6 | 0.2 | 0.04 | 0.6 | 0.08 |" in text
    assert "{'" not in text


def test_failure_and_comparison_reasons_and_paths_are_visible():
    r = _record(
        status="failed",
        error="backend unavailable",
        timings_seconds={},
        compiled_checks={"status": "failed", "failures": ["gradient sign changed"]},
        reference_comparison={"status": "unverified", "reason": "missing reference"},
    )
    text = render_terminal([r], {}, records_path="raw.jsonl", report_path="report.md")
    assert "backend unavailable" in text and "gradient sign changed" in text
    assert "unverified; missing reference" in text
    assert "Full readable report: report.md" in text
    assert "Raw records: raw.jsonl" in text


def _comparison(status="compared", timing_comparable=True, configuration_match=True):
    return {
        "status": status,
        "reason": "same fixture and environment",
        "reference_source": "base-source",
        "candidate_source": "fork-source",
        "review_status": "pending researcher review",
        "timing_comparable": timing_comparable,
        "configuration_match": configuration_match,
        "speedups": {"phase": 1.5},
        "config_differences": []
        if configuration_match
        else [{"field": "dtype", "reference": "fp64", "candidate": "fp32"}],
        "errors": {
            "phase__0": {
                "max_abs": 1e-4,
                "relative_l2": 2e-5,
                "gradient_sign_flips": 1,
                "reference_dtype": "complex128",
                "candidate_dtype": "complex64",
            }
        },
        "reference_outcomes": {
            "best": {
                "evaluation": {
                    "betas": [0.4],
                    "profile_tvd": [0.1],
                    "raw_probability_tvd": [0.2],
                }
            }
        },
        "candidate_outcomes": {
            "best": {
                "evaluation": {
                    "betas": [0.4],
                    "profile_tvd": [0.05],
                    "raw_probability_tvd": [0.15],
                }
            }
        },
    }


def test_comparison_tables_show_sources_speedups_errors_and_outcomes():
    r = _record(reference_comparison=_comparison())
    terminal = render_terminal([r], {})
    markdown = render_markdown([r], {})
    for text in (terminal, markdown):
        assert "base-source" in text and "fork-source" in text
        assert "1.5x" in text
        assert "phase__0" in text and "complex128 / complex64" in text
        assert "Training outcome comparison" in text
        assert "0.05" in text


def test_invalid_comparison_suppresses_speedups_and_shows_config_difference():
    r = _record(
        reference_comparison=_comparison(
            status="unverified", timing_comparable=False, configuration_match=False
        )
    )
    text = render_terminal([r], {})
    assert "Speedups: suppressed" in text
    assert "1.5x" not in text
    assert "dtype" in text and "fp32" in text
