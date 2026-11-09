"""Hardware-independent cost of one forward step, by operation and format.

GEMMs are counted as ``2 M N K`` flops, as every backend counts them;
elementwise work (cosines, products, sums) is counted as one operation per
element and reported separately, since its flop cost depends on how a backend
implements each function.  Bytes are element counts times the format's width,
plus the block and tensor scales of the block-scaled formats.  The backward
pass roughly doubles the GEMM work and is not counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from iqp_cond import Precision

_BITS = {
    "fp64": 64,
    "fp32": 32,
    "tf32": 32,  # stored in a 32-bit container
    "bf16": 16,
    "fp16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp6_e2m3": 6,
    "fp6_e3m2": 6,
    "fp4_e2m1": 4,
    "mxfp8": 8,
    "mxfp4": 4,
    "nvfp4": 4,
}
# (block size, scale bits per block, tensor-scale bits) of the block formats.
_SCALES = {"mxfp8": (32, 8, 0), "mxfp4": (32, 8, 0), "nvfp4": (16, 8, 32)}


@dataclass(frozen=True)
class Cost:
    flops: int = 0
    elementwise: int = 0
    bytes: float = 0.0


def tensor_bytes(elements: int, name: str) -> float:
    """Bytes of a tensor of ``elements`` values stored in format ``name``."""
    bits = elements * _BITS[name]
    if name in _SCALES:
        block, scale_bits, tensor_bits = _SCALES[name]
        bits += -(-elements // block) * scale_bits + tensor_bits
    return bits / 8


def step_cost(
    n_qubits: int, n_samples: int, n_masks: int, precision: Precision
) -> dict[str, Cost]:
    """Return the forward cost of one loss evaluation for each operation.

    ``n_samples`` is ``S``, ``n_masks`` is ``R``, and the complete 2-local
    ansatz has ``m = n (n + 1) / 2`` generators.
    """
    n, s, r = n_qubits, n_samples, n_masks
    m = n * (n + 1) // 2
    return {
        "parity_gemms": Cost(
            flops=2 * s * n * m + 2 * r * n * m,
            bytes=tensor_bytes(s * m + r * m, precision.gemm_accumulate),
        ),
        "phase_gemm": Cost(
            flops=2 * s * r * m,
            bytes=tensor_bytes(s * m + r * m, precision.gemm_inputs)
            + tensor_bytes(s * r, precision.gemm_accumulate),
        ),
        "phases": Cost(
            elementwise=2 * s * r,  # storing the phase and its cosine
            bytes=2 * tensor_bytes(s * r, precision.phases),
        ),
        "statistics": Cost(
            elementwise=3 * s * r,  # value, square, and their sums
            bytes=2 * tensor_bytes(s * r, precision.statistics),
        ),
    }
