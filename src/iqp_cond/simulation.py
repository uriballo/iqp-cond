"""Sampling the conditional model through PennyLane backends.

The trainer never touches this module: it consumes the encoder's sampling and
amplitude-ratio interface directly.  Sampling the trained model, by contrast,
is a statevector or hardware operation, so the model is compiled into a
PennyLane circuit and handed to a device chosen by the caller::

    import pennylane as qml

    import iqp_cond as ciqp

    draws = ciqp.simulation.sample_model(model, params, 1024, seed=0)
    device = qml.device("default.qubit", wires=model.n_qubits)
    qnode = ciqp.simulation.build_qnode(model, params, 0, device=device, shots=1024)

The circuit is the encoder preparation, a Hadamard layer, the shared diagonal
as ``MultiRZ(-2 theta_j)`` per nonzero generator, a second Hadamard layer, and
measurement of the visible wires.  Encoders describe their preparation as
single-qubit rotations
(:meth:`~iqp_cond.encoders.CircuitEncoder.preparation_ops`), so the same
description runs on a classical simulator or on hardware.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .core import ConditionalModel
from .encoders import CircuitEncoder, Operation

__all__ = ["build_qnode", "probabilities", "sample_model"]

_ROTATION_GATES = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}


def _validate_params(model: ConditionalModel, params: np.ndarray) -> np.ndarray:
    values = np.asarray(params, dtype=np.float64)
    n_parameters = model.ansatz.n_parameters
    if values.shape != (n_parameters,):
        raise ValueError(f"params must have shape ({n_parameters},)")
    if not np.isfinite(values).all():
        raise ValueError("params must contain only finite values")
    return values


def _require_circuit_encoder(model: ConditionalModel) -> CircuitEncoder:
    if not isinstance(model.encoders, CircuitEncoder):
        raise TypeError(
            "sampling needs an encoder with a circuit representation; "
            f"{type(model.encoders).__name__} does not implement preparation_ops",
        )
    return model.encoders


def _resolve_device(device: str | qml.device, n_qubits: int, seed: int | None):
    if isinstance(device, str):
        return qml.device(device, wires=n_qubits, seed=seed)
    if seed is not None:
        raise ValueError(
            "seed applies when the device is built from a string; seed a "
            "passed device yourself",
        )
    wires = getattr(device, "wires", None)
    if wires is None or len(wires) != n_qubits:
        raise ValueError(f"device must act on {n_qubits} wires")
    return device


def _prepare(encoders: CircuitEncoder, condition: int) -> None:
    for operation in encoders.preparation_ops(condition):
        if not isinstance(operation, Operation):
            raise TypeError(
                "preparation_ops must yield Operation instances, got "
                f"{type(operation).__name__}",
            )
        gate = _ROTATION_GATES.get(operation.name)
        if gate is None:
            raise ValueError(
                f"unsupported preparation operation {operation.name!r}; "
                f"expected one of {', '.join(sorted(_ROTATION_GATES))}",
            )
        if len(operation.wires) != 1 or len(operation.parameters) != 1:
            raise ValueError(
                f"{operation.name} preparation needs one wire and one angle",
            )
        gate(operation.parameters[0], wires=operation.wires[0])


def _hadamard_layer(n_qubits: int) -> None:
    for wire in range(n_qubits):
        qml.Hadamard(wire)


def _diagonal_layer(params: np.ndarray, generators: np.ndarray) -> None:
    for theta, generator in zip(params, generators, strict=True):
        wires = tuple(int(qubit) for qubit, bit in enumerate(generator) if bit)
        if wires:
            # MultiRZ(phi) = exp(-i phi/2 Z_g), so -2 theta gives exp(+i theta Z_g).
            qml.MultiRZ(-2.0 * float(theta), wires=wires)


def build_qnode(
    model: ConditionalModel,
    params: np.ndarray,
    condition: int,
    device: str | qml.device | None = None,
    *,
    shots: int | None = None,
    seed: int | None = None,
) -> qml.QNode:
    """Compile one condition's model circuit into a PennyLane QNode.

    ``device`` is a PennyLane device name (default ``"lightning.qubit"``) or an
    already-built device, which must act on ``model.n_qubits`` wires.  ``seed``
    applies only to devices built from a name.  The QNode measures the visible
    wires and needs finite shots; pass ``shots`` here or wrap the result with
    ``qml.set_shots``.
    """
    encoders = _require_circuit_encoder(model)
    if not 0 <= condition < encoders.n_conditions:
        raise IndexError(f"condition {condition} outside the batch")
    values = _validate_params(model, params)
    resolved = _resolve_device(device or "lightning.qubit", model.n_qubits, seed)
    generators = np.asarray(model.generators, dtype=np.float64)
    n_qubits = model.n_qubits
    visible = model.visible

    def circuit() -> object:
        _prepare(encoders, condition)
        _hadamard_layer(n_qubits)
        _diagonal_layer(values, generators)
        _hadamard_layer(n_qubits)
        return qml.sample(wires=range(visible))

    qnode = qml.QNode(circuit, resolved)
    if shots is not None:
        qnode = qml.set_shots(qnode, shots=int(shots))
    return qnode


def sample_model(
    model: ConditionalModel,
    params: np.ndarray,
    n_samples: int,
    *,
    seed: int | None = None,
    device: str | qml.device = "lightning.qubit",
) -> np.ndarray:
    """Draw ``(conditions, n_samples, n_visible)`` bitstrings from the model.

    One circuit per condition, sampled with finite shots.  ``device`` may be a
    simulator (default ``"lightning.qubit"``) or a backend plugin; ``seed``
    applies only to devices built from a name and is folded by condition so
    each one has its own stream.
    """
    encoders = _require_circuit_encoder(model)
    _validate_params(model, params)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    draws = []
    for condition in range(encoders.n_conditions):
        condition_seed = None if seed is None else int(seed) + condition
        qnode = build_qnode(
            model,
            params,
            condition,
            device,
            shots=n_samples,
            seed=condition_seed,
        )
        draws.append(np.asarray(qnode(), dtype=np.float64))
    return np.stack(draws)


def _reverse_bits(indices: np.ndarray, n_bits: int) -> np.ndarray:
    """Bit-reverse PennyLane's wire-ordered indices into little-endian."""
    out = np.zeros_like(indices)
    for bit in range(n_bits):
        out |= ((indices >> bit) & 1) << (n_bits - 1 - bit)
    return out


def probabilities(
    model: ConditionalModel,
    params: np.ndarray,
    *,
    device: str | qml.device = "lightning.qubit",
    seed: int | None = None,
) -> np.ndarray:
    """Return the exact visible distribution ``(conditions, 2**n_visible)``.

    Analytic evaluation, so this is a small-system path: the device must
    support analytic execution (a statevector simulator).  For hardware or
    shot-based evaluation use :func:`sample_model`.  Indices are little-endian,
    matching the rest of the package.
    """
    encoders = _require_circuit_encoder(model)
    values = _validate_params(model, params)
    generators = np.asarray(model.generators, dtype=np.float64)
    n_qubits = model.n_qubits
    visible = model.visible
    index = np.arange(2**visible)

    distributions = []
    for condition in range(encoders.n_conditions):
        resolved = _resolve_device(device, n_qubits, seed)

        def circuit(
            encoders: CircuitEncoder = encoders,
            condition: int = condition,
            values: np.ndarray = values,
            generators: np.ndarray = generators,
            n_qubits: int = n_qubits,
            visible: int = visible,
        ) -> object:
            _prepare(encoders, condition)
            _hadamard_layer(n_qubits)
            _diagonal_layer(values, generators)
            _hadamard_layer(n_qubits)
            return qml.probs(wires=range(visible))

        raw = np.asarray(qml.QNode(circuit, resolved)(), dtype=np.float64)
        distributions.append(raw[_reverse_bits(index, visible)])
    return np.stack(distributions)
