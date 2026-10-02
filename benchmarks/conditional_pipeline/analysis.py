"""Summaries and reports for conditional-pipeline benchmark records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmarks.conditional_pipeline.presentation import render_markdown

_LABELS = (
    "encoder_sampling",
    "amplitude_ratios",
    "phase",
    "phase_gradient",
    "prepared_estimator",
    "prepared_estimator_gradient",
    "sampling_estimator",
    "prepared_mmd",
    "prepared_mmd_gradient",
    "mmd",
    "mmd_gradient",
)
_ID_FIELDS = (
    "encoder",
    "n_qubits",
    "n_conditions",
    "n_generators",
    "arity",
    "ansatz",
    "n_ops",
    "n_samples",
    "n_targets",
    "mask_band",
    "sample_chunk",
)


def _config(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("config", record)


def workload_id(config: dict[str, Any]) -> str:
    payload = {k: config.get(k) for k in _ID_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    vals = sorted(float(x) for x in values)

    def quantile(q):
        position = q * (len(vals) - 1)
        low, high = math.floor(position), math.ceil(position)
        return vals[low] + (vals[high] - vals[low]) * (position - low)

    return {
        "median": statistics.median(vals),
        "p25": quantile(0.25),
        "p75": quantile(0.75),
        "min": vals[0],
        "max": vals[-1],
    }


def _warm(record: dict[str, Any], label: str) -> list[float]:
    timings = record.get("timings_seconds", {})
    value = timings.get(label, {})
    if isinstance(value, dict):
        value = value.get("warm", [])
    return [float(x) for x in value] if isinstance(value, list) else []


def _timing_identity(record: dict[str, Any]) -> Any:
    policy = record.get("measurement_policy", _config(record).get("measurement_policy"))
    return json.dumps(policy, sort_keys=True) if policy is not None else None


def _valid_times(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(x) and x > 0 for x in values)


def _identity(record: dict[str, Any], key: str) -> Any:
    return record.get(key, record.get("manifest", {}).get(key))


def _control_status(records: list[dict[str, Any]]) -> dict[tuple, bool]:
    controls: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in records:
        role = r.get("role", _config(r).get("role"))
        if role in ("control_start", "control_end"):
            key = (
                _identity(r, "session_id"),
                r.get("backend"),
                _config(r).get("dtype"),
                _identity(r, "source_identity"),
                _identity(r, "stack_identity"),
                _identity(r, "environment_identity"),
                r.get("measurement_segment"),
            )
            controls.setdefault(key, {})[role] = r
    result = {}
    for key, pair in controls.items():
        ok = "control_start" in pair and "control_end" in pair
        if ok:
            ok = all(
                pair[name].get("status") == "ok"
                and pair[name].get("validity", {}).get("status") == "passed"
                for name in ("control_start", "control_end")
            )
            ok = ok and _identity(pair["control_start"], "fixture_hash") == _identity(
                pair["control_end"], "fixture_hash"
            )
            a, b = (
                _warm(pair["control_start"], "mmd_gradient"),
                _warm(pair["control_end"], "mmd_gradient"),
            )
            if not _valid_times(a) or not _valid_times(b):
                ok = False
            else:
                ok = ok and all(math.isfinite(x) for x in a + b)
                ok = ok and (
                    abs(statistics.median(a) - statistics.median(b))
                    / max(statistics.median(a), 1e-30)
                    <= 0.15
                )
        result[key] = ok
    return result


def summarize(
    records: list[dict[str, Any]], output_dir: Path | None = None
) -> dict[str, Any]:
    controls = _control_status(records)
    rows, speed_rows = [], []
    training_runs = []
    for r in records:
        c = _config(r)
        backend = r.get("backend", c.get("backend"))
        dtype = c.get("dtype")
        session = _identity(r, "session_id")
        role = r.get("role", c.get("role", "measurement"))
        if r.get("benchmark_kind") == "training":
            training_runs.append(
                {
                    "total_time": r.get(
                        "total_time",
                        r.get(
                            "total_time_seconds",
                            r.get("timings_seconds", {}).get("total_training"),
                        ),
                    ),
                    "completed_updates": r.get("completed_updates"),
                    "outcome": r.get("outcomes", r.get("outcome", r.get("validity"))),
                    "profile": r.get("profile", c.get("profile")),
                    "timings_seconds": r.get("timings_seconds"),
                    "memory": r.get("memory"),
                    "status": r.get("status"),
                    "config": c,
                    "backend": backend,
                    "source_identity": _identity(r, "source_identity"),
                    "stack_identity": _identity(r, "stack_identity"),
                    "environment_identity": _identity(r, "environment_identity"),
                }
            )
            continue
        valid = r.get("validity", {}).get("status", "unverified")
        perf = bool(r.get("performance_valid", True))
        if role.startswith("control"):
            continue
        key = (
            session,
            backend,
            dtype,
            _identity(r, "source_identity"),
            _identity(r, "stack_identity"),
            _identity(r, "environment_identity"),
            r.get("measurement_segment"),
        )
        if key not in controls or not controls[key]:
            perf = False
        if not any(x.get("status") == "ok" for x in [r]):
            perf = False
        if any(
            _warm(r, label) and not _valid_times(_warm(r, label)) for label in _LABELS
        ):
            perf = False
        for label in _LABELS:
            dist = _percentiles(_warm(r, label))
            if dist:
                rows.append(
                    {
                        "label": label,
                        "workload_id": r.get("workload_id", workload_id(c)),
                        "session_id": session,
                        "backend": backend,
                        "status": r.get("status"),
                        "dtype": dtype,
                        "validity": valid,
                        "performance_valid": perf,
                        "distribution": dist,
                        "config": c,
                        "fixture_hash": _identity(r, "fixture_hash"),
                        "source_identity": _identity(r, "source_identity"),
                        "stack_identity": _identity(r, "stack_identity"),
                        "environment_identity": _identity(r, "environment_identity"),
                        "measurement_policy": _timing_identity(r),
                        "memory": r.get("memory", {}),
                        "phase_timings": r.get("timings_seconds", {}).get(label, {}),
                        "setup_timings": {
                            k: r.get(k)
                            for k in (
                                "backend_init_seconds",
                                "fixture_setup_seconds",
                                "transfer_seconds",
                                "encoder_construction_seconds",
                            )
                        },
                    }
                )
    groups: dict[tuple, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if (
            row["validity"] == "passed"
            and row["performance_valid"]
            and all(
                row[k] is not None
                for k in (
                    "fixture_hash",
                    "source_identity",
                    "stack_identity",
                    "environment_identity",
                )
            )
        ):
            groups.setdefault(
                (
                    row["workload_id"],
                    row["session_id"],
                    row["dtype"],
                    row["label"],
                    row["fixture_hash"],
                    row["source_identity"],
                    row["stack_identity"],
                    row["environment_identity"],
                    row["measurement_policy"],
                ),
                {},
            ).setdefault(row["backend"], []).append(row)
    for key, pair in groups.items():
        if "cpu" in pair and "gpu" in pair:
            speed_rows.append(
                {
                    "workload_id": key[0],
                    "session_id": key[1],
                    "dtype": key[2],
                    "label": key[3],
                    "source_identity": key[5],
                    "stack_identity": key[6],
                    "environment_identity": key[7],
                    "measurement_policy": key[8],
                    "speedup": pair["cpu"][0]["distribution"]["median"]
                    / pair["gpu"][0]["distribution"]["median"],
                }
            )
    speed_groups: dict[tuple, list[float]] = {}
    for row in speed_rows:
        speed_groups.setdefault(
            (
                row["workload_id"],
                row["dtype"],
                row["label"],
                row["source_identity"],
                row["stack_identity"],
                row["environment_identity"],
                row["measurement_policy"],
            ),
            [],
        ).append(row["speedup"])
    speed_summary = [
        {
            "workload_id": key[0],
            "dtype": key[1],
            "label": key[2],
            "source_identity": key[3],
            "stack_identity": key[4],
            "environment_identity": key[5],
            "measurement_policy": key[6],
            "distribution": _percentiles(values),
        }
        for key, values in speed_groups.items()
    ]
    memory = [
        {
            "workload_id": row["workload_id"],
            "session_id": row["session_id"],
            "backend": row["backend"],
            "dtype": row["dtype"],
            **row["memory"],
        }
        for row in rows
        if row["memory"]
    ]
    summary = {
        "counts": {
            "records": len(records),
            "ok": sum(r.get("status") == "ok" for r in records),
            "failed": sum(r.get("status") == "failed" for r in records),
            "capacity": sum(r.get("status") == "capacity" for r in records),
            "deferred": sum(r.get("status") == "deferred" for r in records),
            "numerical_errors": sum(
                r.get("validity", {}).get("status") == "failed" for r in records
            ),
        },
        "timings": rows,
        "paired_speedups": speed_rows,
        "paired_speedup_summary": speed_summary,
        "memory": memory,
        "training_runs": training_runs,
        "numerical_comparisons": [
            {
                "workload_id": r.get("workload_id"),
                "backend": r.get("backend"),
                "dtype": r["config"]["dtype"],
                "session_id": r.get("session_id"),
                "validity": r.get("validity"),
                "independent_checks": r.get("independent_checks"),
            }
            for r in records
            if r.get("role") == "measurement" and r.get("benchmark_kind") != "training"
        ],
        "reference_comparisons": [
            {
                "workload_id": r.get("workload_id"),
                "backend": r.get("backend"),
                "comparison": r["reference_comparison"],
            }
            for r in records
            if r.get("reference_comparison")
        ],
        "failures_and_boundaries": [
            {
                "config": r["config"],
                "backend": r.get("backend"),
                "status": r.get("status"),
                "reason": r.get("error"),
                "validity": r.get("validity"),
            }
            for r in records
            if r.get("status") != "ok"
            or r.get("validity", {}).get("status") != "passed"
        ],
        "controls": {str(k): v for k, v in controls.items()},
        "gpu_evidence": any(
            r.get("backend") == "gpu" and r.get("status") == "ok" for r in records
        ),
        "uncertainties": (
            "Speedups exclude failed, unverified, mismatched, and "
            "control-drift sessions."
        ),
    }
    if output_dir is not None:
        write_report(records, output_dir, summary=summary)
    return summary


def write_report(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    summary: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary or summarize(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    config_columns = (
        "encoder",
        "n_qubits",
        "n_conditions",
        "n_generators",
        "arity",
        "n_ops",
        "n_samples",
        "n_targets",
        "sample_chunk",
        "mask_band",
        "seed",
    )
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "label",
                *config_columns,
                "workload_id",
                "session_id",
                "backend",
                "dtype",
                "status",
                "validity",
                "performance_valid",
                "source_identity",
                "stack_identity",
                "environment_identity",
                "measurement_policy",
                "median",
                "p25",
                "p75",
                "lowering",
                "compile",
                "first_execution",
                "peak_rss_bytes",
            )
        )
        for r in summary["timings"]:
            d = r["distribution"]
            writer.writerow(
                (
                    r["label"],
                    *(r["config"].get(k) for k in config_columns),
                    r["workload_id"],
                    r["session_id"],
                    r["backend"],
                    r["dtype"],
                    r.get("status"),
                    r.get("validity"),
                    r.get("performance_valid"),
                    r.get("source_identity"),
                    r.get("stack_identity"),
                    r.get("environment_identity"),
                    r.get("measurement_policy"),
                    d["median"],
                    d["p25"],
                    d["p75"],
                    r["phase_timings"].get("lowering"),
                    r["phase_timings"].get("compile"),
                    r["phase_timings"].get("first_execution"),
                    r["memory"].get("peak_rss_bytes"),
                )
            )
    _plots(summary, output_dir)
    with (output_dir / "training.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("backend", "profile", "status", "completed_updates", "metric", "seconds")
        )
        for run in summary.get("training_runs", []):
            for metric, seconds in (run.get("timings_seconds") or {}).items():
                writer.writerow(
                    (
                        run["backend"],
                        run["profile"],
                        run["status"],
                        run["completed_updates"],
                        metric,
                        seconds,
                    )
                )
    report = output_dir / "report.md"
    report.write_text(render_markdown(records, summary))
    return report


def _plots(summary, output_dir):
    """Plot matching workloads only; warm IQR and session IQR stay distinct."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.prop_cycle": mpl.cycler(
                color=["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
            ),
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "text.usetex": False,
            "mathtext.fontset": "cm",
        }
    )
    rows = summary["timings"]
    if not rows:
        return

    def save(fig, name):
        for ax in fig.axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output_dir / (name + ".png"), dpi=300)
        fig.savefig(output_dir / (name + ".pdf"))
        plt.close(fig)

    # A single exact mathematical configuration, distinguished by encoder,
    # policy and backend. These are diagnostic warm-repeat distributions.
    base = rows[0]["config"]
    cohort_fields = (
        "source_identity",
        "stack_identity",
        "environment_identity",
        "measurement_policy",
    )
    diagnostic_cohort = tuple(rows[0].get(k) for k in cohort_fields)
    shape_keys = [k for k in _ID_FIELDS if k != "encoder"]
    chosen = [
        r
        for r in rows
        if all(r["config"].get(k) == base.get(k) for k in shape_keys)
        and tuple(r.get(k) for k in cohort_fields) == diagnostic_cohort
    ]
    chosen = [
        r for r in chosen if r["label"] in {"mmd", "mmd_gradient", "prepared_estimator"}
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(250 / 25.4, 110 / 25.4), layout="constrained"
    )
    for ax, label in zip(
        axes, ("prepared_estimator", "mmd", "mmd_gradient"), strict=True
    ):
        subset = [r for r in chosen if r["label"] == label]
        for i, row in enumerate(subset):
            d = row["distribution"]
            ax.errorbar(
                d["median"] * 1e6,
                i,
                xerr=[
                    [(d["median"] - d["p25"]) * 1e6],
                    [(d["p75"] - d["median"]) * 1e6],
                ],
                fmt="o",
            )
        ax.set_yticks(
            range(len(subset)),
            [
                (
                    f"{r['config']['encoder']} {r['dtype']} "
                    f"{r['backend']} {r['session_id']}"
                )
                for r in subset
            ],
            fontsize=7,
        )
        ax.set_xlabel("Warm latency [microseconds], median/IQR")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.set_title(label.replace("_", " "))
    fig.suptitle(
        f"Matched workload: n={base.get('n_qubits')}, G={base.get('n_generators')}, "
        f"R={base.get('n_ops')}, S={base.get('n_samples')}; diagnostic timings"
    )
    save(fig, "timing_distribution")

    anchor = dict(
        n_qubits=64,
        n_conditions=1,
        n_generators=64,
        arity=2,
        ansatz="uniform",
        n_ops=512,
        n_samples=1024,
        n_targets=1024,
        mask_band="low",
        sample_chunk=256,
    )
    for axis in (
        "n_qubits",
        "n_conditions",
        "n_generators",
        "arity",
        "n_ops",
        "n_samples",
        "n_targets",
        "mask_band",
        "sample_chunk",
    ):
        matched = [
            r
            for r in rows
            if r["label"] == "mmd_gradient"
            and r["performance_valid"]
            and r["validity"] == "passed"
            and all(r["config"].get(k) == v for k, v in anchor.items() if k != axis)
        ]
        if matched:
            cohort = tuple(matched[0].get(k) for k in cohort_fields)
            matched = [
                r for r in matched if tuple(r.get(k) for k in cohort_fields) == cohort
            ]
        groups = {}
        for r in matched:
            groups.setdefault(
                (r["config"]["encoder"], r["dtype"], r["backend"]), {}
            ).setdefault(str(r["config"][axis]), []).append(r["distribution"]["median"])
        if not any(len(points) > 1 for points in groups.values()):
            continue
        fig, ax = plt.subplots(figsize=(170 / 25.4, 105 / 25.4), layout="constrained")
        labels = sorted(
            {label for points in groups.values() for label in points},
            key=lambda x: (
                x in {"None", "low", "mid", "high"},
                float(x) if x not in {"None", "low", "mid", "high"} else x,
            ),
        )
        positions = {label: i for i, label in enumerate(labels)}
        for key, points in groups.items():
            present = [label for label in labels if label in points]
            xs = np.array([positions[label] for label in present])
            med = np.array([np.median(points[x]) for x in present])
            lo = np.array([np.percentile(points[x], 25) for x in present])
            hi = np.array([np.percentile(points[x], 75) for x in present])
            ax.errorbar(
                xs, med, yerr=[med - lo, hi - med], fmt="o-", label=" ".join(key)
            )
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(axis)
        ax.set_ylabel("MMD plus gradient [s], session median/IQR")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        save(fig, axis + "_scaling")
    if summary["paired_speedup_summary"]:
        fig, ax = plt.subplots(figsize=(170 / 25.4, 105 / 25.4), layout="constrained")
        vals = summary["paired_speedup_summary"]
        for i, r in enumerate(vals):
            d = r["distribution"]
            ax.errorbar(
                i,
                d["median"],
                yerr=[[d["median"] - d["p25"]], [d["p75"] - d["median"]]],
                fmt="o",
            )
        ax.set_xticks(
            range(len(vals)),
            [r["workload_id"] + " " + r["dtype"] + " " + r["label"] for r in vals],
            rotation=90,
            fontsize=6,
        )
        ax.axhline(1, color="0.5", linestyle="--")
        ax.set_ylabel("Paired CPU/GPU speedup, session median/IQR")
        save(fig, "paired_speedups")
