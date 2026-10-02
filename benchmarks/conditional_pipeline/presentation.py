"""Format benchmark records as readable tables and reports."""
# ruff: noqa: E501

from __future__ import annotations

import math

_NAMES = {
    "encoder_sampling": "encoder sampling",
    "amplitude_ratios": "amplitude ratios",
    "phase": "phase",
    "phase_gradient": "phase gradient",
    "prepared_estimator": "prepared estimator",
    "prepared_estimator_gradient": "prepared estimator gradient",
    "sampling_estimator": "sampling estimator",
    "prepared_mmd": "prepared MMD",
    "prepared_mmd_gradient": "prepared MMD gradient",
    "mmd": "fresh MMD",
    "mmd_gradient": "fresh MMD gradient",
}


def _cfg(r):
    return r.get("config", r)


def _fmt(x, unit=""):
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "n/a"
    if x == 0:
        s = "0"
    elif abs(x) >= 100:
        s = f"{x:.0f}"
    elif abs(x) >= 10:
        s = f"{x:.1f}"
    else:
        s = f"{x:.3g}"
    return s + unit


def _short(x):
    return str(x)[:10] if x else "n/a"


def _latencies(r, label):
    t = r.get("timings_seconds", {}).get(label, {})
    raw = t.get("warm", [])
    if not isinstance(raw, list) or not raw:
        return "n/a"
    if any(
        not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in raw
    ):
        return "INVALID warm samples"
    a = sorted(raw)
    med = (
        a[(len(a) - 1) // 2]
        if len(a) % 2
        else (a[len(a) // 2 - 1] + a[len(a) // 2]) / 2
    )

    def q(p):
        position = p * (len(a) - 1)
        low, high = math.floor(position), math.ceil(position)
        return a[low] + (a[high] - a[low]) * (position - low)

    return f"{med * 1e6:.3g} [{q(0.25) * 1e6:.3g}, {q(0.75) * 1e6:.3g}]"


def _cell(r):
    c = _cfg(r)
    return (
        f"encoder={c.get('encoder', 'n/a')} dtype={c.get('dtype', 'n/a')} "
        f"backend={r.get('backend', c.get('backend', 'n/a'))} "
        f"nq={c.get('n_qubits', 'n/a')} conditions={c.get('n_conditions', 'n/a')} "
        f"generators={c.get('n_generators', 'n/a')} masks={c.get('n_ops', 'n/a')} "
        f"samples={c.get('n_samples', 'n/a')} targets={c.get('n_targets', 'n/a')} "
        f"chunk={c.get('sample_chunk', 'n/a')} seed={c.get('seed', 'n/a')} "
        f"source={_short(r.get('source_identity'))}"
    )


def _checks(r):
    vals = []
    for key in ("validity", "independent_checks", "compiled_checks"):
        v = r.get(key)
        s = v.get("status") if isinstance(v, dict) else None
        if s:
            vals.append(f"{key.replace('_', ' ')}={s}")
    compiled = r.get("compiled_checks")
    if isinstance(compiled, dict) and compiled.get("status") == "compared":
        vals.append("compiled comparison requires review")
    errs = r.get("validity", {}).get("errors", {})
    bits = []
    for k, n in (("moments", "moment"), ("loss", "loss"), ("gradient", "gradient")):
        if isinstance(errs.get(k), dict):
            bits.append(
                f"{n} {'relative L2' if k == 'gradient' else 'max abs'}={_fmt(errs[k].get('relative_l2' if k == 'gradient' else 'max_abs'))}"
            )
    return "; ".join(vals) + ("; " + ", ".join(bits) if bits else "")


def _failure(r):
    failures = []
    for key in ("failure_reason", "error", "errors"):
        value = r.get(key)
        if value:
            failures.append(str(value))
    for key in ("validity", "independent_checks", "compiled_checks"):
        value = r.get(key, {})
        if isinstance(value, dict):
            failures.extend(str(x) for x in value.get("failures", []) if x)
            if value.get("reason"):
                failures.append(str(value["reason"]))
    return "; ".join(failures)


def _training_total(r):
    t = r.get("timings_seconds", {})
    return r.get("total_time", r.get("total_time_seconds", t.get("total_training")))


def _outcomes(r):
    """Flatten exact diagnostics without mixing best and final checkpoints."""
    outcomes = r.get("outcomes", r.get("outcome", {}))
    if not isinstance(outcomes, dict):
        return []
    result = []
    for split in ("development", "evaluation"):
        checkpoints = {
            key: outcomes.get(key, {}).get(split, {}) for key in ("best", "final")
        }
        betas = list(
            dict.fromkeys(
                beta for data in checkpoints.values() for beta in data.get("betas", [])
            )
        )
        for beta in betas:
            row = [split, _fmt(beta)]
            for key in ("best", "final"):
                data = checkpoints[key]
                index = (
                    data.get("betas", []).index(beta)
                    if beta in data.get("betas", [])
                    else None
                )
                for metric in ("raw_probability_tvd", "profile_tvd"):
                    values = data.get(metric, [])
                    row.append(
                        _fmt(
                            values[index]
                            if index is not None and index < len(values)
                            else None
                        )
                    )
            result.append(row)
    return result


def _table(headers, rows, markdown):
    def clean(value):
        return str(value).replace("|", "/").replace("\n", " ")

    rows = [[clean(x) for x in row] for row in rows]
    headers = [clean(x) for x in headers]
    if markdown:
        return [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ] + ["| " + " | ".join(row) + " |" for row in rows]
    widths = (
        [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(headers))
        ]
        if rows
        else [len(h) for h in headers]
    )
    return ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))] + [
        "  ".join(value.ljust(w) for value, w in zip(row, widths, strict=True))
        for row in rows
    ]


def _scaled(value, factor):
    return _fmt(value * factor if isinstance(value, (int, float)) else None)


def _rows(records):
    return [
        r
        for r in records
        if not str(r.get("role", "")).startswith("control")
        and r.get("benchmark_kind") != "training"
    ]


def _comparison(r):
    value = r.get("reference_comparison", {})
    if not value:
        return ""
    return f"Reference comparison: {value.get('status', 'n/a')}; {value.get('reason', 'numerical evidence requires review')}"


def _comparison_detail(value, *, markdown, error_cap=None):
    """Render a stored fork comparison without interpreting its evidence."""
    if not isinstance(value, dict) or not value:
        return []
    out = ["", "Comparison details:" if markdown else "Comparison details"]
    status = value.get("status", "n/a")
    comparable = (
        value.get("timing_comparable") is True
        and value.get("configuration_match", True) is True
    )
    out.append(
        f"Status: {status}; timing comparable: {'yes' if value.get('timing_comparable') is True else 'no'}; "
        f"configuration match: {'yes' if value.get('configuration_match', True) is True else 'no'}"
    )
    if value.get("reason"):
        out.append("Comparability reason: " + str(value["reason"]))
    if value.get("timing_comparability_reason"):
        out.append("Timing scope: " + str(value["timing_comparability_reason"]))
    out.append(f"Reference source: {value.get('reference_source', 'n/a')}")
    out.append(f"Candidate source: {value.get('candidate_source', 'n/a')}")
    out.append(
        f"Precision policy: reference={value.get('reference_precision', 'n/a')}; "
        f"candidate={value.get('candidate_precision', 'n/a')}"
    )
    if value.get("failures"):
        out.append("Comparison failures: " + "; ".join(map(str, value["failures"])))
    if value.get("review_status"):
        out.append("Review status: " + str(value["review_status"]))
    differences = value.get("config_differences") or []
    if differences:
        rows = [
            [
                d.get("field", "n/a"),
                d.get("reference", "n/a"),
                d.get("candidate", "n/a"),
            ]
            for d in differences
        ]
        out += ["", "Configuration differences:", ""]
        out += _table(["Field", "Reference", "Candidate"], rows, markdown)
    speeds = value.get("speedups") or {}
    if comparable and status == "compared" and speeds:
        rows = [[label, _fmt(ratio, "x")] for label, ratio in speeds.items()]
        out += [
            "",
            "Diagnostic speedups (reference / candidate; >1 means candidate is faster):",
            "",
        ]
        out += _table(["Measurement", "Ratio"], rows, markdown)
    elif speeds:
        out.append(
            "Speedups: suppressed because comparison is not valid for performance claims."
        )
    errors = value.get("errors") or {}
    if errors:
        items = list(errors.items())
        shown = items if error_cap is None else items[:error_cap]
        rows = []
        for label, data in shown:
            rows.append(
                [
                    label,
                    _fmt(data.get("max_abs")),
                    _fmt(data.get("relative_l2")),
                    _fmt(data.get("gradient_sign_flips")),
                    f"{data.get('reference_dtype', 'n/a')} / {data.get('candidate_dtype', 'n/a')}",
                ]
            )
        out += [
            "",
            "Numerical differences (evidence for review; no automatic approval):",
            "Output suffix __0 denotes the value; __1 denotes the IQP gradient.",
            "",
        ]
        out += _table(
            ["Output", "Max abs", "Relative L2", "Sign flips", "Dtypes ref / cand"],
            rows,
            markdown,
        )
        if len(items) > len(shown):
            out.append(
                f"{len(items) - len(shown)} additional numerical rows are in the full report."
            )
    ref, cand = value.get("reference_outcomes"), value.get("candidate_outcomes")
    if isinstance(ref, dict) and isinstance(cand, dict):
        rows = []
        for split in ("development", "evaluation"):
            for checkpoint in ("best", "final"):
                rd, cd = (
                    ref.get(checkpoint, {}).get(split, {}),
                    cand.get(checkpoint, {}).get(split, {}),
                )
                betas = list(dict.fromkeys(rd.get("betas", []) + cd.get("betas", [])))
                for beta in betas:

                    def metric(data, name, current_beta):
                        bs = data.get("betas", [])
                        index = bs.index(current_beta) if current_beta in bs else None
                        values = data.get(name, [])
                        return (
                            values[index]
                            if index is not None and index < len(values)
                            else None
                        )

                    rows.append(
                        [
                            split,
                            checkpoint,
                            _fmt(beta),
                            _fmt(metric(rd, "profile_tvd", beta)),
                            _fmt(metric(cd, "profile_tvd", beta)),
                            _fmt(metric(rd, "raw_probability_tvd", beta)),
                            _fmt(metric(cd, "raw_probability_tvd", beta)),
                        ]
                    )
        if rows:
            out += ["", "Training outcome comparison (TVD; smaller is better):", ""]
            out += _table(
                [
                    "Split",
                    "Checkpoint",
                    "Beta",
                    "Ref profile",
                    "Cand profile",
                    "Ref raw",
                    "Cand raw",
                ],
                rows,
                markdown,
            )
    return out


def _render(records, summary, *, markdown):
    heading = (
        "# Conditional pipeline benchmark"
        if markdown
        else "Conditional pipeline benchmark"
    )
    out = [heading, ""]
    counts = summary.get("counts", {})
    if counts:
        out.append("Run status: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    out += [
        "Warm = median [25th-75th percentiles] of repeated execution, excluding setup and compilation.",
        "Smoke timings are diagnostics. Numerical checks and performance approval are separate.",
    ]
    paired = summary.get("paired_speedups", [])
    if paired:
        out.append(
            f"Paired CPU/GPU timing ratios: {len(paired)}. See summary.json for matched records."
        )
    rows = _rows(records)
    shown = rows if markdown else rows[:6]
    for i, r in enumerate(shown, 1):
        c = _cfg(r)
        title = f"Cell {i}: {r.get('backend', 'n/a')} / {c.get('encoder', 'n/a')} / {c.get('dtype', 'n/a')}"
        out += [
            "",
            ("## " if markdown else "") + title,
            _cell(r),
            f"Execution: {r.get('status', 'n/a')}. {_checks(r) or 'Checks: n/a'}",
        ]
        reason = _failure(r)
        if reason:
            out.append("Reason: " + reason)
        if _comparison(r):
            out.append(_comparison(r))
        out += _comparison_detail(
            r.get("reference_comparison"),
            markdown=markdown,
            error_cap=None if markdown else 20,
        )
        timings = []
        for label, data in r.get("timings_seconds", {}).items():
            if label not in _NAMES or not isinstance(data, dict):
                continue
            timings.append(
                [
                    _NAMES[label],
                    _latencies(r, label),
                    _scaled(data.get("compile"), 1e3),
                    _scaled(data.get("first_execution"), 1e6),
                    _scaled(data.get("lowering"), 1e3),
                ]
            )
        if timings:
            out.append("")
            out += _table(
                [
                    "Operation",
                    "Warm median [IQR] (us)",
                    "Compile (ms)",
                    "First (us)",
                    "Lowering (ms)",
                ],
                timings,
                markdown,
            )
        else:
            out.append("Operation timings: n/a")
        rss = (r.get("memory") or {}).get("peak_rss_bytes")
        out.append(
            "Peak process memory: "
            + _fmt(rss / 2**20 if isinstance(rss, (int, float)) else None, " MiB")
        )
    if len(rows) > len(shown):
        out.append(
            f"{len(rows) - len(shown)} additional component records are shown in the full report."
        )
    training = [r for r in records if r.get("benchmark_kind") == "training"]
    if training:
        out += ["", ("## " if markdown else "") + f"Training runs: {len(training)}"]
    shown_training = training if markdown else training[:6]
    for i, r in enumerate(shown_training, 1):
        c, timings = _cfg(r), r.get("timings_seconds") or {}
        out += [
            "",
            f"Fit {i}: {r.get('backend', 'n/a')} / {c.get('profile', r.get('profile', 'n/a'))} / {c.get('precision_policy', c.get('dtype', 'n/a'))}; {c.get('n_qubits', 'n/a')} qubits",
            f"Execution: {r.get('status', 'n/a')}; updates: {r.get('completed_updates', 'n/a')}/{c.get('steps', c.get('n_updates', 'n/a'))}; correctness/outcomes: {r.get('validity', {}).get('status', 'unverified')}.",
            "Total time is primary; loop time is supplementary. Final diagnostics are outside total.",
        ]
        if _failure(r):
            out.append("Reason: " + _failure(r))
        if _comparison(r):
            out.append(_comparison(r))
        out += _comparison_detail(
            r.get("reference_comparison"),
            markdown=markdown,
            error_cap=None if markdown else 20,
        )
        timing_rows = [["Total training (primary)", _fmt(_training_total(r))]]
        labels = {
            "preparation": "Preparation",
            "training_call": "Training call",
            "lowering": "Lowering",
            "compile": "Compilation",
            "training_loop": "Training loop (supplementary)",
            "validation": "Validation",
            "checkpoint_selection": "Checkpoint selection",
            "artifact_write": "Artifact writing",
            "final_diagnostics": "Final diagnostics (separate)",
        }
        timing_rows += [
            [label, _fmt(timings.get(key))] for key, label in labels.items()
        ]
        out.append("")
        out += _table(["Training metric", "Seconds"], timing_rows, markdown)
        out.append(
            "Timing scopes overlap; do not add these rows. Outcomes require researcher review."
        )
        outcomes = _outcomes(r)
        if outcomes:
            out += [
                "",
                "Exact outcome diagnostics: TVD is total variation distance; smaller is better.",
                "",
            ]
            out += _table(
                [
                    "Conditions",
                    "Beta",
                    "Best raw TVD",
                    "Best profile TVD",
                    "Final raw TVD",
                    "Final profile TVD",
                ],
                outcomes,
                markdown,
            )
        else:
            out.append("Exact outcome diagnostics: n/a")
    if len(training) > len(shown_training):
        out.append(
            f"{len(training) - len(shown_training)} additional fits are shown in the full report."
        )
    return out


def render_terminal(records, summary, *, records_path=None, report_path=None):
    out = _render(records, summary, markdown=False)
    if report_path:
        out += ["", f"Full readable report: {report_path}"]
    if records_path:
        out.append(f"Raw records: {records_path}")
    return "\n".join(out) + "\n"


def render_markdown(records, summary):
    out = _render(records, summary, markdown=True)
    out += [
        "",
        "[Component timings and configuration](summary.csv), [training timings](training.csv), [full diagnostics and provenance](summary.json).",
        "Memory is the process peak, not the operation allocation. Missing values are shown as n/a.",
    ]
    return "\n".join(out) + "\n"
