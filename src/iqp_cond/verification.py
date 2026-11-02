"""Conformance checks for custom encoders against an independent PennyLane oracle.

An encoder is truthful for training only when its two classically tractable
operations describe one and the same post-Hadamard state: ``sample`` draws from
``|psi_c(z)|**2`` and ``amplitude_ratios`` returns ``psi_c(z xor a) / psi_c(z)``
for ``psi_c(z) = <z|H^n|phi(c)>``.  :func:`check_encoder` derives that state from
the encoder's own preparation callable through PennyLane, compares both
operations against it, and reports contract violations, statistical mismatches,
and observations that are not violations.

The reference is a statevector built on ``default.qubit``, so a check is limited
to small registers; ``max_qubits`` guards that limit.  PennyLane is imported
inside the functions, because importing :mod:`iqp_cond` must not import it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import numpy as np

from .encoders import Encoder
from .encoders.base import _check_condition_pytree, _select_condition

__all__ = ["EncoderCheck", "check_encoder"]

#: Smallest expected outcome count for the full-distribution chi-square test.
_MIN_EXPECTED = 5.0

#: Ratio errors are judged where both amplitudes reach this fraction of the
#: largest one; below it the ratio is rounding noise rather than signal.
_SIGNIFICANCE_FLOOR = 1e-3


@dataclass(frozen=True)
class EncoderCheck:
    """Outcome of :func:`check_encoder`.

    ``failures`` lists contract and comparison violations; ``notes`` lists
    observations that are not violations, such as ratio entries skipped where a
    reference amplitude vanishes.  ``ok`` is true when ``failures`` is empty.
    """

    n_qubits: int
    n_conditions: int
    n_samples: int
    probes: int
    ratio_error: float
    ratio_entries: int
    ratio_skipped: int
    sampling_p_value: float
    tv_distance: float
    preparation_ops: tuple[int, ...]
    preparation_gates: tuple[str, ...]
    preparation_repeat_error: float
    exact_state_error: float | None
    failures: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no check failed."""
        return not self.failures

    def summary(self) -> str:
        """Human-readable report, one line per check plus failures and notes."""
        state = "ok" if self.ok else "FAILED"
        lines = [
            f"check_encoder({self.n_qubits} qubits, {self.n_conditions} "
            f"conditions): {state}",
            f"  ratios      worst relative error {self.ratio_error:.2e} over "
            f"{self.ratio_entries} entries, {self.ratio_skipped} skipped",
            f"  sampling    worst chi-square p-value {self.sampling_p_value:.2e} "
            f"and TV distance {self.tv_distance:.2e} at {self.n_samples} samples",
            f"  preparation {sum(self.preparation_ops)} operations "
            f"[{', '.join(self.preparation_gates) or 'none'}], repeat error "
            f"{self.preparation_repeat_error:.2e}",
        ]
        if self.exact_state_error is not None:
            lines.append(f"  exact state max |dpsi| {self.exact_state_error:.2e}")
        lines.extend(f"  failure: {item}" for item in self.failures)
        lines.extend(f"  note: {item}" for item in self.notes)
        return "\n".join(lines)


def _sci(value: float) -> str:
    """Format a value for failure messages."""
    return f"{value:.3e}"


def _chi_square_sf(statistic: float, dof: int) -> float:
    """Return ``P(X >= statistic)`` for a chi-square variable with ``dof`` degrees.

    Campbell's closed form evaluated in float64, with each series term formed in
    log space so large statistics cannot overflow.
    """
    if dof <= 0:
        return 1.0
    half = 0.5 * statistic
    if dof % 2 == 0:
        total = 0.0
        log_term = -half
        for order in range(dof // 2):
            if order:
                log_term += np.log(half) - np.log(order)
            if np.isfinite(log_term):
                total += float(np.exp(log_term))
        return total
    total = float(math.erfc(math.sqrt(half)))
    exponent = 0.5
    log_gamma = float(np.log(0.5 * np.sqrt(np.pi)))
    for _ in range((dof - 1) // 2):
        log_term = -half + exponent * np.log(half) - log_gamma
        if np.isfinite(log_term):
            total += float(np.exp(log_term))
        exponent += 1.0
        log_gamma += float(np.log(exponent))
    return total


def _goodness_of_fit(observed: np.ndarray, expected: np.ndarray) -> float:
    """Upper-tail p-value of a chi-square comparison, zero for impossible draws."""
    if np.any((expected <= 0.0) & (observed > 0.0)):
        return 0.0
    usable = expected > 0.0
    if int(usable.sum()) <= 1:
        return 1.0
    statistic = float(
        np.sum((observed[usable] - expected[usable]) ** 2 / expected[usable]),
    )
    return _chi_square_sf(statistic, int(usable.sum()) - 1)


def _law_p_values(
    counts: np.ndarray,
    law: np.ndarray,
    n_samples: int,
    n_qubits: int,
) -> list[float]:
    """Chi-square p-values for the full law, the marginals, and pairwise parity.

    The full-distribution test runs only while every expected count reaches
    :data:`_MIN_EXPECTED`; the marginal and pairwise tests are valid at any
    register size.
    """
    p_values = []
    expected = n_samples * law
    if (expected >= _MIN_EXPECTED).all():
        p_values.append(_goodness_of_fit(counts.astype(float), expected))
    index = np.arange(1 << n_qubits)
    codes = np.repeat(index, counts.astype(np.int64))
    for qubit in range(n_qubits):
        bit = (codes >> qubit) & 1
        qubit_bit = (index >> qubit) & 1
        observed = np.array([n_samples - bit.sum(), bit.sum()], dtype=float)
        marginal = n_samples * np.array(
            [law[qubit_bit == 0].sum(), law[qubit_bit == 1].sum()],
        )
        p_values.append(_goodness_of_fit(observed, marginal))
    for first in range(n_qubits):
        for second in range(first + 1, n_qubits):
            pair = (((codes >> first) & 1) << 1) | ((codes >> second) & 1)
            observed = np.bincount(pair, minlength=4).astype(float)
            pair_law = np.array(
                [
                    law[
                        ((((index >> first) & 1) << 1) | ((index >> second) & 1))
                        == value
                    ].sum()
                    for value in range(4)
                ],
            )
            p_values.append(_goodness_of_fit(observed, n_samples * pair_law))
    return p_values


def _condition_state(encoder: Encoder, condition: int, n_qubits: int) -> np.ndarray:
    """Return one condition's post-Hadamard state in little-endian order."""
    import pennylane as qml

    from .simulation import _prepare, _reverse_bits

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit() -> np.ndarray:
        _prepare(encoder, condition)
        for wire in range(n_qubits):
            qml.Hadamard(wire)
        return qml.state()

    state = np.asarray(circuit())
    return state[_reverse_bits(np.arange(1 << n_qubits), n_qubits)]


def _reference_states(
    encoder: Encoder,
    n_qubits: int,
    n_conditions: int,
) -> np.ndarray:
    return np.stack(
        [
            _condition_state(encoder, condition, n_qubits)
            for condition in range(n_conditions)
        ],
    )


def _tape_signature(encoder: Encoder, condition: int) -> tuple:
    """Names and wires of the operations a condition's preparation queues."""
    import pennylane as qml

    operations = qml.tape.make_qscript(encoder.preparation(condition))().operations
    return tuple(
        (operation.name, tuple(int(wire) for wire in operation.wires))
        for operation in operations
    )


def _probe_rows(
    n_qubits: int,
    states: np.ndarray,
    probes: int,
    seed: int,
) -> np.ndarray:
    """Draw ``(conditions, probes, n_qubits)`` bitstrings from the reference law.

    Probing at the reference distribution keeps the ratio denominators typical,
    and it keeps the ratio verdict independent of ``sample``.
    """
    rows = []
    for condition in range(states.shape[0]):
        law = np.abs(states[condition]) ** 2
        indices = np.random.default_rng(seed + condition).choice(
            law.size,
            size=probes,
            p=law / law.sum(),
        )
        rows.append(
            np.stack([(indices >> bit) & 1 for bit in range(n_qubits)], axis=-1),
        )
    return np.stack(rows).astype(np.float64)


def _probe_masks(n_qubits: int, n_masks: int, seed: int) -> np.ndarray:
    """Masks to probe, with the empty mask first and one single-qubit mask each."""
    masks = [np.zeros(n_qubits)]
    for qubit in range(n_qubits):
        mask = np.zeros(n_qubits)
        mask[qubit] = 1.0
        masks.append(mask)
    generator = np.random.default_rng(seed)
    while len(masks) < n_masks:
        masks.append(generator.integers(0, 2, size=n_qubits).astype(np.float64))
    return np.stack(masks)


def _ratio_report(
    encoder: Encoder,
    states: np.ndarray,
    probes: np.ndarray,
    masks: np.ndarray,
    n_qubits: int,
    *,
    floor: float,
) -> tuple[float, int, int, float]:
    """Compare reported ratios with the reference.

    Returns the worst relative error, the number of compared entries, the number
    skipped below the significance floor, and the worst deviation from one for
    the empty mask (``masks[0]``), which must be exactly one.
    """
    weights = 2 ** np.arange(n_qubits)
    ratios = np.asarray(encoder.amplitude_ratios(probes, masks))
    shape = (states.shape[0], probes.shape[1], masks.shape[0])
    if ratios.shape != shape:
        raise ValueError(
            f"amplitude_ratios returned shape {ratios.shape}; expected {shape}",
        )
    if not np.isfinite(ratios).all():
        raise ValueError(
            "amplitude_ratios returned non-finite entries where the reference "
            "state is defined",
        )
    flips = ((masks > 0.5) * weights).sum(-1).astype(np.int64)
    worst = 0.0
    empty_mask_error = 0.0
    compared = 0
    skipped = 0
    for condition in range(states.shape[0]):
        state = states[condition]
        scale = float(np.abs(state).max())
        indices = (probes[condition] * weights).sum(-1).astype(np.int64)
        denominator = state[indices][:, None]
        numerator = state[indices[:, None] ^ flips[None, :]]
        significant = (np.abs(denominator) >= floor * scale) & (
            np.abs(numerator) >= floor * scale
        )
        compared += int(significant.sum())
        skipped += int((~significant).sum())
        if not significant.any():
            continue
        safe = np.where((denominator[:, 0] != 0.0)[:, None], denominator, 1.0)
        expected = numerator / safe
        error = np.abs(ratios[condition][significant] - expected[significant]) / np.abs(
            expected[significant]
        )
        worst = max(worst, float(np.max(error)))
        empty = significant[:, 0]
        if empty.any():
            empty_mask_error = max(
                empty_mask_error,
                float(np.max(np.abs(ratios[condition][empty, 0] - 1.0))),
            )
    return worst, compared, skipped, empty_mask_error


def _sample_failures(samples: np.ndarray, shape: tuple[int, int, int]) -> list[str]:
    """Contract failures for one ``sample`` result."""
    if samples.shape != shape:
        return [f"sample returned shape {samples.shape}; expected {shape}"]
    if np.iscomplexobj(samples):
        return ["sample returned complex values; bitstrings must be real"]
    unexpected = np.unique(samples[~np.isin(samples, (0.0, 1.0))])
    if unexpected.size:
        return [
            f"sample returned values outside {{0, 1}}: {unexpected[:5].tolist()}",
        ]
    return []


def _sampling_report(
    samples: np.ndarray,
    states: np.ndarray,
    n_samples: int,
    n_qubits: int,
    notes: list[str],
) -> tuple[float, float]:
    """Worst chi-square p-value and worst total-variation distance."""
    weights = 2 ** np.arange(n_qubits)
    p_value = 1.0
    distance = 0.0
    full_law_skipped = False
    for condition in range(states.shape[0]):
        law = np.abs(states[condition]) ** 2
        law = law / law.sum()
        if not (n_samples * law >= _MIN_EXPECTED).all():
            full_law_skipped = True
        counts = np.bincount(
            ((samples[condition] * weights).sum(-1)).astype(int),
            minlength=1 << n_qubits,
        )
        p_value = min(
            p_value,
            *_law_p_values(counts, law, n_samples, n_qubits),
        )
        distance = max(distance, 0.5 * float(np.abs(counts / n_samples - law).sum()))
    if full_law_skipped:
        notes.append(
            "the full-distribution chi-square test was skipped because some "
            "expected outcome counts fall below five",
        )
    return p_value, distance


def check_encoder(
    encoder: Encoder,
    *,
    n_samples: int = 100_000,
    probes_per_condition: int = 8_192,
    n_masks: int = 12,
    seed: int = 0,
    rtol: float = 1e-6,
    floor: float = _SIGNIFICANCE_FLOOR,
    p_min: float = 1e-6,
    max_qubits: int = 16,
) -> EncoderCheck:
    """Check a custom encoder's sampling and ratio queries against PennyLane.

    The reference state is built from the encoder's own ``preparation``, so the
    check compares the classically tractable operations with the circuit the
    encoder declares.  All randomness is seeded, so a verdict is reproducible.
    Raises ``ValueError`` only when the check cannot run at all, such as a
    register above ``max_qubits``; encoder misbehaviour is reported in
    ``EncoderCheck.failures``.

    ``rtol`` is the relative ratio tolerance, ``floor`` the amplitude
    significance floor for ratio comparisons, ``p_min`` the smallest acceptable
    chi-square p-value, and ``seed`` the draw behind both the ratio probes and
    the sampling test.
    """
    n_qubits = int(encoder.n_qubits)
    n_conditions = int(encoder.n_conditions)
    if n_qubits < 1 or n_conditions < 1:
        raise ValueError("encoder must declare at least one qubit and condition")
    if n_qubits > max_qubits:
        raise ValueError(
            f"check_encoder builds a 2**{n_qubits} reference statevector; "
            f"n_qubits={n_qubits} exceeds max_qubits={max_qubits}",
        )

    failures: list[str] = []
    notes: list[str] = []

    operations: list[tuple] = []
    for condition in range(n_conditions):
        try:
            operations.append(_tape_signature(encoder, condition))
        except Exception as error:
            failures.append(f"preparation({condition}): {error!r}")
            operations.append(())
    queued = tuple(len(item) for item in operations)
    gates = tuple(sorted({op[0] for item in operations for op in item}))
    if not any(queued):
        notes.append("the preparation queues no operations, so phi(c) is |0>^n")
    try:
        encoder.preparation(n_conditions)
    except Exception:
        pass
    else:
        notes.append("preparation does not reject a condition outside the batch")

    try:
        states = _reference_states(encoder, n_qubits, n_conditions)
    except Exception as error:
        failures.append(f"preparation: the reference circuit failed: {error!r}")
        return EncoderCheck(
            n_qubits=n_qubits,
            n_conditions=n_conditions,
            n_samples=n_samples,
            probes=probes_per_condition,
            ratio_error=float("nan"),
            ratio_entries=0,
            ratio_skipped=0,
            sampling_p_value=float("nan"),
            tv_distance=float("nan"),
            preparation_ops=queued,
            preparation_gates=gates,
            preparation_repeat_error=float("nan"),
            exact_state_error=None,
            failures=tuple(failures),
            notes=tuple(notes),
        )

    try:
        repeated = _reference_states(encoder, n_qubits, n_conditions)
    except Exception as error:
        failures.append(f"preparation: rebuilding the circuit failed: {error!r}")
        repeat_error = float("nan")
    else:
        repeat_error = float(np.abs(states - repeated).max())
        if repeat_error > 1e-9:
            failures.append(
                "the preparation is not reproducible: rebuilding it changed the "
                f"reference state by {_sci(repeat_error)}",
            )

    probes = _probe_rows(n_qubits, states, probes_per_condition, seed)
    masks = _probe_masks(n_qubits, n_masks, seed)
    ratio_error = float("nan")
    ratio_entries = 0
    ratio_skipped = 0
    ratios_available = False
    try:
        ratio_error, ratio_entries, ratio_skipped, empty_error = _ratio_report(
            encoder,
            states,
            probes,
            masks,
            n_qubits,
            floor=floor,
        )
    except Exception as error:
        failures.append(f"amplitude_ratios: {error}")
    else:
        ratios_available = True
        if ratio_error > rtol:
            failures.append(
                "amplitude_ratios disagrees with the preparation: worst relative "
                f"error {_sci(ratio_error)} exceeds rtol={rtol:g}",
            )
        if empty_error > 1e-12:
            failures.append(
                "amplitude_ratios is not one for the empty mask: worst deviation "
                f"{_sci(empty_error)}",
            )
        if ratio_skipped:
            notes.append(
                f"{ratio_skipped} ratio entries were skipped where a reference "
                "amplitude is below the significance floor",
            )

    samples = None
    try:
        samples = np.asarray(encoder.sample(jax.random.PRNGKey(seed), n_samples))
    except Exception as error:
        failures.append(f"sample: {error!r}")
    sampling_p = float("nan")
    tv_distance = float("nan")
    if samples is not None:
        contract = _sample_failures(
            samples,
            (n_conditions, n_samples, n_qubits),
        )
        failures.extend(contract)
        if not contract:
            if samples.dtype.kind != "f":
                notes.append(
                    f"sample returns {samples.dtype} bitstrings; the built-in "
                    "encoder returns float64",
                )
            try:
                sampling_p, tv_distance = _sampling_report(
                    samples,
                    states,
                    n_samples,
                    n_qubits,
                    notes,
                )
            except Exception as error:
                failures.append(f"sample: {error!r}")
            else:
                if sampling_p < p_min:
                    failures.append(
                        "sample does not follow the preparation's Born law: "
                        f"worst chi-square p-value {_sci(sampling_p)} is below "
                        f"p_min={p_min:g}",
                    )

    failures.extend(
        _restriction_report(
            encoder,
            states,
            probes,
            masks,
            n_qubits,
            ratios_available,
            n_samples,
            seed,
        ),
    )

    exact_error = None
    if hasattr(encoder, "post_hadamard_amplitudes"):
        try:
            reported = np.asarray(encoder.post_hadamard_amplitudes())
        except Exception as error:
            failures.append(f"post_hadamard_amplitudes: {error!r}")
        else:
            if reported.shape != states.shape:
                failures.append(
                    "post_hadamard_amplitudes returned shape "
                    f"{reported.shape}; expected {states.shape}",
                )
            else:
                exact_error = float(np.abs(reported - states).max())
                if exact_error > 1e-12:
                    failures.append(
                        "post_hadamard_amplitudes disagrees with the "
                        f"preparation: max |dpsi| {_sci(exact_error)}",
                    )

    return EncoderCheck(
        n_qubits=n_qubits,
        n_conditions=n_conditions,
        n_samples=n_samples,
        probes=probes_per_condition,
        ratio_error=ratio_error,
        ratio_entries=ratio_entries,
        ratio_skipped=ratio_skipped,
        sampling_p_value=sampling_p,
        tv_distance=tv_distance,
        preparation_ops=queued,
        preparation_gates=gates,
        preparation_repeat_error=repeat_error,
        exact_state_error=exact_error,
        failures=tuple(failures),
        notes=tuple(notes),
    )


def _restriction_report(
    encoder: Encoder,
    states: np.ndarray,
    probes: np.ndarray,
    masks: np.ndarray,
    n_qubits: int,
    ratios_available: bool,
    n_samples: int,
    seed: int,
) -> list[str]:
    """Check that row ``c`` of the leaves restricts the encoder to condition ``c``.

    Training restricts the encoder by indexing its leaves, so a leaf whose
    leading axis is not the condition axis silently changes the objective.
    Training calls only ``sample`` and ``amplitude_ratios`` on a restriction,
    so the restricted ratios must match the batched ones for that condition.
    Realized samples are not comparable across the two call shapes, because
    the draws are independent; the restricted sampling law is checked against
    the reference instead.
    """
    try:
        _check_condition_pytree(encoder)
    except TypeError as error:
        return [f"{error}; training rejects it"]
    except Exception as error:
        return [f"condition 0 restriction: {error!r}"]
    failures: list[str] = []

    if ratios_available:
        try:
            batched = np.asarray(encoder.amplitude_ratios(probes, masks))
        except Exception as error:
            failures.append(f"restriction: the batched ratios failed: {error!r}")
            batched = None
    else:
        batched = None

    for condition in range(states.shape[0]):
        try:
            selected = _select_condition(encoder, condition)
            selected_ratios = np.asarray(
                selected.amplitude_ratios(probes[condition : condition + 1], masks),
            )
        except Exception as error:
            failures.append(f"condition {condition} restriction: {error!r}")
            continue
        if batched is not None:
            row = batched[condition : condition + 1]
            if selected_ratios.shape != row.shape or not np.allclose(
                selected_ratios,
                row,
                rtol=0.0,
                atol=1e-9,
            ):
                failures.append(
                    f"condition {condition} restriction disagrees with the "
                    "batched ratios",
                )

    try:
        selected_samples = np.asarray(
            _select_condition(encoder, 0).sample(
                jax.random.PRNGKey(seed + 1),
                n_samples,
            ),
        )
    except Exception as error:
        failures.append(f"condition 0 restriction sample: {error!r}")
        return failures
    if selected_samples.shape != (1, n_samples, n_qubits):
        failures.append(
            "condition 0 restriction sample returned shape "
            f"{selected_samples.shape}; expected {(1, n_samples, n_qubits)}",
        )
        return failures

    law = np.abs(states[0]) ** 2
    law = law / law.sum()
    counts = np.bincount(
        ((selected_samples[0] * 2 ** np.arange(n_qubits)).sum(-1)).astype(int),
        minlength=law.size,
    )
    p_value = min(_law_p_values(counts, law, n_samples, n_qubits))
    if p_value < 1e-6:
        failures.append(
            "condition 0 restriction sample does not follow the preparation's "
            "Born law: worst "
            f"chi-square p-value {p_value:.3e}",
        )
    return failures
