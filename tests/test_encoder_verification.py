"""The encoder conformance check, exercised against custom encoders.

The checker builds its reference from the encoder's own ``preparation`` through
PennyLane, so every control here is an encoder written the way a user would
write one: its sampling and ratio queries come from its own numpy statevector,
never from library internals.  Positive controls cover a product state, entangling
ladders, complex amplitudes, native gates, a state preparation, a PennyLane
template, and the documented cached-operator preparation.  Knockout controls are
the same encoder with exactly one behaviour broken, and each must fail the check
it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

import iqp_cond as ciqp
from iqp_cond.verification import EncoderCheck, check_encoder

N_QUBITS = 4
N_CONDITIONS = 2
BUDGET = {"n_samples": 40_000, "probes_per_condition": 4_096}

_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def bit_reverse(indices: np.ndarray, n_bits: int) -> np.ndarray:
    """Bit-reverse PennyLane's wire-ordered indices into little-endian codes."""
    out = np.zeros_like(indices)
    for bit in range(n_bits):
        out |= ((indices >> bit) & 1) << (n_bits - 1 - bit)
    return out


def apply_1q(
    state: np.ndarray, matrix: np.ndarray, qubit: int, n_bits: int
) -> np.ndarray:
    """Apply a single-qubit matrix to a little-endian register state."""
    tensor = state.reshape([2] * n_bits)
    axis = n_bits - 1 - qubit
    tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n_bits: int) -> np.ndarray:
    out = state.copy()
    for index in range(1 << n_bits):
        if (index >> control) & 1:
            out[index ^ (1 << target)] = state[index]
    return out


def apply_cz(state: np.ndarray, first: int, second: int, n_bits: int) -> np.ndarray:
    out = state.copy()
    for index in range(1 << n_bits):
        if ((index >> first) & 1) and ((index >> second) & 1):
            out[index] = -state[index]
    return out


def rotation(label: str, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle / 2.0), np.sin(angle / 2.0)
    if label == "X":
        return np.array([[cosine, -1j * sine], [-1j * sine, cosine]])
    if label == "Z":
        return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)]).astype(complex)
    return np.array([[cosine, -sine], [sine, cosine]])


def post_hadamard(angles: np.ndarray, axes: tuple[str, ...]) -> np.ndarray:
    """Post-Hadamard state of a rotation-layer product state, little-endian."""
    n_bits = angles.shape[-1]
    state = np.zeros(1 << n_bits, dtype=complex)
    state[0] = 1.0
    for layer, labels in enumerate(axes):
        for qubit, label in enumerate(labels):
            state = apply_1q(
                state, rotation(label, angles[layer, qubit]), qubit, n_bits
            )
    for qubit in range(n_bits):
        state = apply_1q(state, _HADAMARD, qubit, n_bits)
    return state


def entangled_post_hadamard(
    angles: np.ndarray,
    axes: tuple[str, ...],
    *,
    native_gates: bool = False,
    big_endian: bool = False,
) -> np.ndarray:
    """Post-Hadamard state of a layered encoder with a CNOT ladder."""
    n_bits = angles.shape[-1]
    state = np.zeros(1 << n_bits, dtype=complex)
    state[0] = 1.0
    for layer, labels in enumerate(axes):
        for qubit, label in enumerate(labels):
            state = apply_1q(
                state, rotation(label, angles[layer, qubit]), qubit, n_bits
            )
    if native_gates:
        state = apply_1q(state, _HADAMARD, 0, n_bits)
    for control in range(n_bits - 1):
        state = apply_cnot(state, control, control + 1, n_bits)
    if native_gates:
        state = apply_cz(state, 0, n_bits - 1, n_bits)
    for qubit in range(n_bits):
        state = apply_1q(state, _HADAMARD, qubit, n_bits)
    if big_endian:
        state = state[bit_reverse(np.arange(1 << n_bits), n_bits)]
    return state


@dataclass(frozen=True)
class LayeredEncoder:
    """Rotation layers, a CNOT ladder, and its own numpy statevector."""

    angles: np.ndarray
    axes: tuple[str, ...] = ("Y" * N_QUBITS,)
    native_gates: bool = False
    big_endian: bool = False
    flip_sampling: bool = False
    conjugate_ratios: bool = False
    inverse_ratios: bool = False
    scaled_ratios: bool = False
    doubled_empty_mask: bool = False
    prep_mode: str = "normal"
    lying_select: bool = False
    psi: np.ndarray = field(init=False, repr=False, compare=False)
    calls: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.angles, dtype=np.float64)
        if values.ndim == 2:
            values = values[:, None, :]
        states = [
            entangled_post_hadamard(
                row,
                self.axes,
                native_gates=self.native_gates,
                big_endian=self.big_endian,
            )
            for row in values
        ]
        object.__setattr__(self, "angles", values)
        object.__setattr__(self, "psi", np.stack(states))

    @property
    def n_conditions(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.angles.shape[2])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        probs = np.abs(self.psi) ** 2
        if self.flip_sampling:
            probs = probs[:, ::-1]
        logits = np.log(np.maximum(probs, 1e-300))
        draws = np.stack(
            [
                np.asarray(
                    jax.random.categorical(key, jnp.asarray(row), shape=(n_samples,)),
                )
                for row in logits
            ],
        )
        bits = (draws[:, :, None] >> np.arange(self.n_qubits)) & 1
        return jnp.asarray(bits.astype(np.float64))

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        weights = 2 ** np.arange(self.n_qubits)
        codes = (np.asarray(samples) * weights).sum(-1).astype(np.int64)
        flips = ((np.asarray(masks) > 0.5) * weights).sum(-1).astype(np.int64)
        psi = np.conj(self.psi) if self.conjugate_ratios else self.psi
        row = np.arange(self.n_conditions)[:, None, None]
        numerator = psi[row, codes[:, :, None] ^ flips[None, None, :]]
        denominator = psi[np.arange(self.n_conditions)[:, None], codes][:, :, None]
        if self.inverse_ratios:
            numerator, denominator = denominator, numerator
        ratios = np.where(denominator != 0.0, numerator / denominator, 0.0)
        if self.doubled_empty_mask:
            ratios = np.where(flips[None, None, :] == 0, 2.0, ratios)
        return jnp.asarray(ratios * (2.0 if self.scaled_ratios else 1.0))

    def preparation(self, condition: int):
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")

        def queue_layers() -> None:
            for layer, labels in enumerate(self.axes):
                for qubit, label in enumerate(labels):
                    gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[label]
                    gate(float(self.angles[condition, layer, qubit]), wires=qubit)

        def queue_entangler() -> None:
            if self.native_gates:
                qml.Hadamard(wires=0)
            for control in range(self.n_qubits - 1):
                qml.CNOT(wires=[control, control + 1])
            if self.native_gates:
                qml.CZ(wires=[0, self.n_qubits - 1])

        def queue() -> None:
            self.calls.append(condition)  # statefulness is the point
            if self.prep_mode == "empty":
                return
            queue_layers()
            if self.prep_mode == "double":
                queue_entangler()
                queue_layers()
                queue_entangler()
                return
            if self.prep_mode == "accumulating":
                for _ in range(len(self.calls)):
                    queue_layers()
            queue_entangler()

        return queue

    def select(self, condition: int) -> LayeredEncoder:
        index = 0 if self.lying_select else condition
        return LayeredEncoder(
            self.angles[index : index + 1],
            self.axes,
            self.native_gates,
            self.big_endian,
            self.flip_sampling,
            self.conjugate_ratios,
            self.inverse_ratios,
            self.scaled_ratios,
            self.doubled_empty_mask,
            self.prep_mode,
            self.lying_select,
        )


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(11).uniform(0.4, 2.2, size=(N_CONDITIONS, N_QUBITS))


@pytest.fixture
def layered_angles() -> np.ndarray:
    return np.random.default_rng(12).uniform(0.4, 2.2, size=(N_CONDITIONS, 2, N_QUBITS))


def report_for(encoder: ciqp.Encoder) -> EncoderCheck:
    return check_encoder(encoder, seed=5, **BUDGET)


def test_built_in_angle_encoder_passes(
    angles: np.ndarray, layered_angles: np.ndarray
) -> None:
    single = ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * N_QUBITS,))
    layered = ciqp.AngleEncoder(layered_angles, axes=("XYXY", "YZYX"))
    for encoder in (single, layered):
        report = report_for(encoder)
        assert report.ok, report.summary()
        assert report.ratio_error < 1e-9
        assert report.exact_state_error is not None
        assert report.exact_state_error < 1e-12


def test_entangled_custom_encoder_passes(angles: np.ndarray) -> None:
    report = report_for(LayeredEncoder(angles))
    assert report.ok, report.summary()
    assert report.ratio_error == 0.0
    assert report.preparation_repeat_error == 0.0
    assert report.exact_state_error is None


def test_complex_custom_encoder_passes(layered_angles: np.ndarray) -> None:
    encoder = LayeredEncoder(layered_angles, axes=("XYXY", "YZYX"))
    assert np.iscomplexobj(encoder.psi)
    report = report_for(encoder)
    assert report.ok, report.summary()
    assert report.ratio_error < 1e-9


def test_native_gate_preparation_passes(angles: np.ndarray) -> None:
    report = report_for(LayeredEncoder(angles, native_gates=True))
    assert report.ok, report.summary()
    assert "Hadamard" in report.preparation_gates
    assert "CZ" in report.preparation_gates


@dataclass(frozen=True)
class StatePrepEncoder:
    """A declared state preparation: the amplitudes are the interface."""

    amplitudes: np.ndarray

    @property
    def n_conditions(self) -> int:
        return int(self.amplitudes.shape[0])

    @property
    def n_qubits(self) -> int:
        return 2

    @property
    def psi(self) -> np.ndarray:
        states = []
        for vector in self.amplitudes:
            state = np.zeros(4, dtype=complex)
            state[0], state[1] = vector[0], vector[1]
            state = apply_cz(state, 0, 1, 2)
            for qubit in range(2):
                state = apply_1q(state, _HADAMARD, qubit, 2)
            states.append(state)
        return np.stack(states)

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        logits = np.log(np.maximum(np.abs(self.psi) ** 2, 1e-300))
        draws = np.stack(
            [
                np.asarray(
                    jax.random.categorical(key, jnp.asarray(row), shape=(n_samples,)),
                )
                for row in logits
            ],
        )
        bits = (draws[:, :, None] >> np.arange(2)) & 1
        return jnp.asarray(bits.astype(np.float64))

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        weights = 2 ** np.arange(2)
        codes = (np.asarray(samples) * weights).sum(-1).astype(np.int64)
        flips = ((np.asarray(masks) > 0.5) * weights).sum(-1).astype(np.int64)
        psi = self.psi
        row = np.arange(self.n_conditions)[:, None, None]
        numerator = psi[row, codes[:, :, None] ^ flips[None, None, :]]
        denominator = psi[np.arange(self.n_conditions)[:, None], codes][:, :, None]
        ratios = np.where(denominator != 0.0, numerator / denominator, 0.0)
        return jnp.asarray(ratios)

    def preparation(self, condition: int):
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        vector = np.asarray(self.amplitudes[condition], dtype=complex)

        def queue() -> None:
            qml.StatePrep(vector, wires=0)
            qml.CZ(wires=[0, 1])

        return queue

    def select(self, condition: int) -> StatePrepEncoder:
        return StatePrepEncoder(self.amplitudes[condition : condition + 1])


def test_state_preparation_encoder_passes() -> None:
    amplitudes = np.array([[0.8, 0.6], [0.3, float(np.sqrt(1 - 0.09))]])
    encoder = StatePrepEncoder(amplitudes)
    report = report_for(encoder)
    assert report.ok, report.summary()
    assert report.preparation_gates == ("CZ", "StatePrep")


@dataclass(frozen=True)
class TemplateEncoder:
    """A preparation that queues a PennyLane template."""

    weights: np.ndarray

    @property
    def n_conditions(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.weights.shape[2])

    @property
    def psi(self) -> np.ndarray:
        # Derived with PennyLane's own matrix so the control tests the checker's
        # acceptance of templates, not an independent derivation.
        states = []
        for condition in range(self.n_conditions):
            weights = self.weights[condition]
            script = qml.tape.make_qscript(
                lambda weights=weights: qml.StronglyEntanglingLayers(
                    weights,
                    wires=range(self.n_qubits),
                ),
            )
            matrix = qml.matrix(script(), wire_order=list(range(self.n_qubits)))
            state = (matrix @ np.eye(1 << self.n_qubits, dtype=complex)[:, 0])[
                bit_reverse(np.arange(1 << self.n_qubits), self.n_qubits)
            ]
            for qubit in range(self.n_qubits):
                state = apply_1q(state, _HADAMARD, qubit, self.n_qubits)
            states.append(state)
        return np.stack(states)

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        logits = np.log(np.maximum(np.abs(self.psi) ** 2, 1e-300))
        draws = np.stack(
            [
                np.asarray(
                    jax.random.categorical(key, jnp.asarray(row), shape=(n_samples,)),
                )
                for row in logits
            ],
        )
        bits = (draws[:, :, None] >> np.arange(self.n_qubits)) & 1
        return jnp.asarray(bits.astype(np.float64))

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        weights = 2 ** np.arange(self.n_qubits)
        codes = (np.asarray(samples) * weights).sum(-1).astype(np.int64)
        flips = ((np.asarray(masks) > 0.5) * weights).sum(-1).astype(np.int64)
        psi = self.psi
        row = np.arange(self.n_conditions)[:, None, None]
        numerator = psi[row, codes[:, :, None] ^ flips[None, None, :]]
        denominator = psi[np.arange(self.n_conditions)[:, None], codes][:, :, None]
        return jnp.asarray(np.where(denominator != 0.0, numerator / denominator, 0.0))

    def preparation(self, condition: int):
        if not 0 <= condition < self.n_conditions:
            raise IndexError(f"condition {condition} outside the batch")
        return lambda: qml.StronglyEntanglingLayers(
            self.weights[condition],
            wires=range(self.n_qubits),
        )

    def select(self, condition: int) -> TemplateEncoder:
        return TemplateEncoder(self.weights[condition : condition + 1])


def test_template_preparation_passes() -> None:
    weights = np.random.default_rng(13).uniform(
        0.0,
        2.0 * np.pi,
        size=(N_CONDITIONS, 2, 3, 3),
    )
    report = report_for(TemplateEncoder(weights))
    assert report.ok, report.summary()
    assert report.preparation_gates == ("StronglyEntanglingLayers",)


def test_cached_operator_preparation_passes(angles: np.ndarray) -> None:
    cached = tuple(
        tuple(qml.RY(float(value), wires=qubit) for qubit, value in enumerate(row))
        + tuple(
            qml.CNOT(wires=[control, control + 1]) for control in range(N_QUBITS - 1)
        )
        for row in angles
    )
    base = LayeredEncoder(angles)

    @dataclass(frozen=True)
    class CachedEncoder(LayeredEncoder):
        def preparation(self, condition: int):
            if not 0 <= condition < self.n_conditions:
                raise IndexError(f"condition {condition} outside the batch")
            return lambda: [qml.apply(operator) for operator in cached[condition]]

    report = report_for(CachedEncoder(angles))
    assert report.ok, report.summary()
    assert report.preparation_ops[0] == len(cached[0])
    assert base.psi is not None


@dataclass(frozen=True)
class TrivialEncoder:
    """The README ``|+>^n`` encoder: no gates, uniform bits, unit ratios."""

    angles: jnp.ndarray

    @property
    def n_conditions(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.angles.shape[1])

    def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
        return jax.random.bernoulli(
            key,
            p=0.5,
            shape=(self.n_conditions, n_samples, self.n_qubits),
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((self.n_conditions, samples.shape[1], masks.shape[0]))

    def preparation(self, condition: int):
        return lambda: None

    def select(self, condition: int) -> TrivialEncoder:
        return TrivialEncoder(self.angles[condition : condition + 1])


def test_trivial_encoder_passes_with_notes() -> None:
    report = report_for(TrivialEncoder(jnp.zeros((3, N_QUBITS))))
    assert report.ok, report.summary()
    assert report.ratio_error == 0.0
    assert report.preparation_gates == ()
    assert any("queues no operations" in note for note in report.notes)
    assert any("outside the batch" in note for note in report.notes)


def test_sparse_preparation_passes_with_skipped_entries() -> None:
    angles = np.zeros((N_CONDITIONS, N_QUBITS))
    angles[:, 0] = np.pi / 2.0  # |+> on one qubit, then the ladder: half the
    encoder = LayeredEncoder(angles)  # amplitudes of the reference vanish
    report = report_for(encoder)
    assert report.ok, report.summary()
    assert report.ratio_skipped > 0
    assert any("below the significance floor" in note for note in report.notes)
    # Half the outcomes cannot occur, so the full-distribution test is skipped
    # and the marginals and pairwise parities carry the sampling verdict.
    assert any("full-distribution chi-square" in note for note in report.notes)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"flip_sampling": True}, "Born law"),
        ({"big_endian": True}, "disagrees with the preparation"),
        ({"conjugate_ratios": True}, "disagrees with the preparation"),
        ({"inverse_ratios": True}, "disagrees with the preparation"),
        ({"scaled_ratios": True}, "disagrees with the preparation"),
        ({"doubled_empty_mask": True}, "not one for the empty mask"),
        ({"prep_mode": "empty"}, "Born law"),
        ({"prep_mode": "double"}, "disagrees with the preparation"),
        ({"prep_mode": "accumulating"}, "not reproducible"),
        ({"lying_select": True}, "select(1)"),
    ],
)
def test_knockouts_fail_with_the_intended_check(
    kwargs: dict,
    expected: str,
) -> None:
    angles = np.random.default_rng(15).uniform(
        0.4,
        2.2,
        size=(N_CONDITIONS, 2, N_QUBITS),
    )
    report = report_for(LayeredEncoder(angles, ("XYXY", "YZYX"), **kwargs))
    assert not report.ok
    assert any(expected in failure for failure in report.failures), report.summary()


def test_non_finite_ratios_are_reported(angles: np.ndarray) -> None:
    @dataclass(frozen=True)
    class NotFinite(LayeredEncoder):
        def amplitude_ratios(
            self, samples: jnp.ndarray, masks: jnp.ndarray
        ) -> jnp.ndarray:
            return jnp.full(
                (self.n_conditions, samples.shape[1], masks.shape[0]),
                jnp.nan,
            )

    report = report_for(NotFinite(angles))
    assert not report.ok
    assert any("non-finite" in failure for failure in report.failures)


def test_broken_shape_is_reported(angles: np.ndarray) -> None:
    @dataclass(frozen=True)
    class WrongShape(LayeredEncoder):
        def sample(self, key: jax.Array, n_samples: int) -> jnp.ndarray:
            return jnp.zeros((self.n_conditions, n_samples, self.n_qubits + 1))

    report = report_for(WrongShape(angles))
    assert not report.ok
    assert any("returned shape" in failure for failure in report.failures)


def test_report_is_deterministic(angles: np.ndarray) -> None:
    encoder = LayeredEncoder(angles)
    first = report_for(encoder)
    second = report_for(encoder)
    assert first == second


@dataclass(frozen=True)
class Degenerate:
    """An encoder whose declared sizes make the check meaningless."""

    n_qubits: int
    n_conditions: int


def test_register_above_the_cap_is_rejected(angles: np.ndarray) -> None:
    encoder = ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * N_QUBITS,))
    with pytest.raises(ValueError, match="max_qubits"):
        check_encoder(encoder, max_qubits=2)
    with pytest.raises(ValueError, match="at least one"):
        check_encoder(Degenerate(n_qubits=N_QUBITS, n_conditions=0))
    with pytest.raises(ValueError, match="at least one"):
        check_encoder(Degenerate(n_qubits=0, n_conditions=1))


@pytest.mark.slow
def test_default_budget_passes(layered_angles: np.ndarray) -> None:
    encoder = LayeredEncoder(layered_angles, axes=("XYXY", "YZYX"))
    report = check_encoder(encoder)
    assert report.ok, report.summary()
    assert report.sampling_p_value > 1e-4
