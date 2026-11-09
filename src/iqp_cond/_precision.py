"""Named precision policies for the estimator, the loss and training.

A policy fixes four dtypes: ``compute`` for the parities, the phases and the
ratio-phase products; ``contraction`` for the final phase matrix product;
``accumulate`` for the per-sample statistics, the loss, the parameters and the
optimizer state; and ``ratios`` for the encoder's amplitude-ratio queries.
Below float64, the estimator casts the encoder's floating leaves, the samples
and the masks to ``ratios`` before the query, so the built-in encoders compute
their ratios in that precision.  Reference sampling stays float64 under every
policy, so every policy sees the same draws.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Precision:
    """The dtypes of one precision policy."""

    compute: jnp.dtype
    contraction: jnp.dtype
    accumulate: jnp.dtype
    ratios: jnp.dtype


PRECISIONS = {
    "fp64": Precision(jnp.float64, jnp.float64, jnp.float64, jnp.float64),
    "gemm_f32": Precision(jnp.float64, jnp.float32, jnp.float64, jnp.float64),
    "estimator_f32": Precision(jnp.float32, jnp.float32, jnp.float64, jnp.float64),
    "fp32": Precision(jnp.float32, jnp.float32, jnp.float32, jnp.float32),
}


@dataclass(frozen=True)
class _ElementFormat:
    """A binary floating-point format by its bit layout.

    ``overflow`` is what a finite value rounding past ``max`` becomes, as in
    the ``ml_dtypes`` casts: ``"inf"`` for IEEE-like formats, ``"nan"`` for the
    e4m3 format without infinities, and ``"saturate"`` for fp6 and fp4.
    ``amax_scaled`` formats rescale each tensor so that its largest magnitude
    lands on ``max`` before rounding, as low-precision hardware recipes do.
    """

    exponent_bits: int
    mantissa_bits: int
    max: float
    overflow: str
    amax_scaled: bool = False

    @property
    def min_exponent(self) -> int:
        return 2 - 2 ** (self.exponent_bits - 1)


_ELEMENT_FORMATS = {
    "bf16": _ElementFormat(8, 7, float(jnp.finfo(jnp.bfloat16).max), "inf"),
    "tf32": _ElementFormat(8, 10, (2 - 2.0**-10) * 2.0**127, "inf"),
    "fp16": _ElementFormat(5, 10, 65504.0, "inf"),
    "fp8_e4m3": _ElementFormat(4, 3, 448.0, "nan", amax_scaled=True),
    "fp8_e5m2": _ElementFormat(5, 2, 57344.0, "inf", amax_scaled=True),
    "fp6_e2m3": _ElementFormat(2, 3, 7.5, "saturate", amax_scaled=True),
    "fp6_e3m2": _ElementFormat(3, 2, 28.0, "saturate", amax_scaled=True),
    "fp4_e2m1": _ElementFormat(2, 1, 6.0, "saturate", amax_scaled=True),
}


def _round(x: jnp.ndarray, fmt: _ElementFormat) -> jnp.ndarray:
    """Round to nearest, ties to even, including subnormals, in ``x``'s dtype.

    Dividing by the power-of-two quantum is exact, so rounding happens once.
    """
    magnitude = jnp.abs(x)
    _, exponent = jnp.frexp(magnitude)
    exponent = jnp.maximum(exponent - 1, fmt.min_exponent)
    quantum = jnp.ldexp(jnp.ones_like(x), exponent - fmt.mantissa_bits)
    rounded = jnp.round(magnitude / quantum) * quantum
    overflow = {"inf": jnp.inf, "nan": jnp.nan, "saturate": fmt.max}[fmt.overflow]
    rounded = jnp.where(rounded > fmt.max, overflow, rounded)
    return jnp.copysign(rounded, x)


_MX_FORMATS = {"mxfp8": "fp8_e4m3", "mxfp4": "fp4_e2m1"}


def _to_blocks(x: jnp.ndarray, size: int) -> jnp.ndarray:
    """Split the last axis into zero-padded blocks of ``size``."""
    padded = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, -x.shape[-1] % size)])
    return padded.reshape(*x.shape[:-1], -1, size)


def _from_blocks(blocks: jnp.ndarray, width: int) -> jnp.ndarray:
    return blocks.reshape(*blocks.shape[:-2], -1)[..., :width]


def _quantize_mx(x: jnp.ndarray, element: _ElementFormat) -> jnp.ndarray:
    """OCP MX v1.0: blocks of 32 along the last axis share a power-of-two scale.

    The scale is ``2^(floor(log2 amax) - emax)``, with ``emax`` the exponent of
    the element format's largest value, and elements saturate.
    """
    blocks = _to_blocks(x, 32)
    amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    _, amax_exponent = jnp.frexp(amax)
    _, element_exponent = jnp.frexp(element.max)
    scale = jnp.where(
        amax > 0,
        jnp.ldexp(jnp.ones_like(amax), amax_exponent - element_exponent),
        1.0,
    )
    rounded = _round(blocks / scale, replace(element, overflow="saturate")) * scale
    return _from_blocks(rounded, x.shape[-1])


def _quantize_nvfp4(x: jnp.ndarray) -> jnp.ndarray:
    """NVFP4: fp4 elements in blocks of 16 with an e4m3 scale per block.

    A per-tensor scale ``6 * 448 / amax`` maps the block scales into e4m3's
    range; each element is ``x * tensor_scale / block_scale`` rounded to fp4.
    """
    fp4 = _ELEMENT_FORMATS["fp4_e2m1"]
    e4m3 = replace(_ELEMENT_FORMATS["fp8_e4m3"], overflow="saturate")
    blocks = _to_blocks(x, 16)
    amax = jnp.max(jnp.abs(blocks))
    tensor_scale = jnp.where(amax > 0, fp4.max * e4m3.max / amax, 1.0)
    block_amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    block_scale = _round(block_amax / fp4.max * tensor_scale, e4m3)
    block_scale = jnp.where(block_scale > 0, block_scale, 1.0)
    elements = _round(
        blocks * tensor_scale / block_scale, replace(fp4, overflow="saturate")
    )
    return _from_blocks(elements * block_scale / tensor_scale, x.shape[-1])


def quantize(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Round ``x`` to the named format and return it in its own dtype."""
    if name in _MX_FORMATS:
        return _quantize_mx(x, _ELEMENT_FORMATS[_MX_FORMATS[name]])
    if name == "nvfp4":
        return _quantize_nvfp4(x)
    fmt = _ELEMENT_FORMATS[name]
    if not fmt.amax_scaled:
        return _round(x, fmt)
    amax = jnp.max(jnp.abs(x))
    scale = jnp.where(amax > 0, fmt.max / amax, 1.0)
    return _round(x * scale, fmt) / scale


def accumulate(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Sum the last axis as hardware accumulating in the named format would.

    Blocks of 32 are summed exactly and rounded to the format, and the running
    sum is rounded after adding each block, following chunk-based accumulation
    (Wang et al., NeurIPS 2018).
    """
    fmt = _ELEMENT_FORMATS[name]
    partials = _round(jnp.sum(_to_blocks(x, 32), axis=-1), fmt)

    def add(running, partial):
        running = _round(running + partial, fmt)
        return running, None

    total, _ = jax.lax.scan(
        add,
        jnp.zeros(x.shape[:-1], x.dtype),
        jnp.moveaxis(partials, -1, 0),
    )
    return total


def resolve_precision(name: str) -> Precision:
    """Return the dtypes of the named policy, or raise for an unknown name."""
    if name not in PRECISIONS:
        raise ValueError(
            f"unknown precision {name!r}; expected one of {sorted(PRECISIONS)}",
        )
    return PRECISIONS[name]
