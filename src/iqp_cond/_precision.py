"""Number formats for the operations of the estimator, the loss and training.

:class:`Precision` names a format for each operation of a training step, and
:data:`PRECISIONS` holds the uniform presets.  :func:`quantize` and
:func:`accumulate` emulate the formats: they round values, and sum them, as
hardware working in the format would, while computing in the input's dtype.

:class:`_Arithmetic` is the seam the estimator and the loss compute through:
an fp64 knob runs the plain float64 expression, and any other knob rounds to
its format and computes in float32.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

import jax
import jax.numpy as jnp


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


_KNOBS = ("gemm_inputs", "gemm_accumulate", "phases", "statistics", "ratios")
_BLOCK_FORMATS = ("mxfp8", "mxfp4", "nvfp4")
FORMATS = ("fp64", "fp32", *_ELEMENT_FORMATS, *_BLOCK_FORMATS)


@dataclass(frozen=True)
class Precision:
    """The number format of each operation in a training step.

    ``gemm_inputs`` rounds the parameters entering the phase contraction (the
    other operand is exactly ``±1``); ``gemm_accumulate`` is the precision in
    which that contraction sums and returns the phases; ``phases`` covers phase
    storage and the cosine and sine; ``statistics`` covers the per-sample
    values, their sums and the loss assembly; and ``ratios`` covers encoder
    amplitude ratios.  ``variant="split"`` runs the forward computations in
    these formats and the backward computations in fp32.
    """

    gemm_inputs: str = "fp64"
    gemm_accumulate: str = "fp64"
    phases: str = "fp64"
    statistics: str = "fp64"
    ratios: str = "fp64"
    variant: str = "uniform"

    def __post_init__(self) -> None:
        for knob in _KNOBS:
            name = getattr(self, knob)
            if name not in FORMATS:
                raise ValueError(
                    f"unknown format {name!r} for {knob}; expected one of {FORMATS}"
                )
            if name in _BLOCK_FORMATS and knob != "gemm_inputs":
                raise ValueError(
                    f"block-scaled {name!r} applies only to gemm_inputs, not {knob}"
                )
        if self.variant not in ("uniform", "split"):
            raise ValueError(
                f"variant must be 'uniform' or 'split', not {self.variant!r}"
            )

    @property
    def loss_scaled(self) -> bool:
        """Whether training scales the loss to keep fp16 gradients in range.

        Only the uniform variant differentiates in the knobs' formats; the
        split variant's backward pass is fp32.
        """
        return self.variant == "uniform" and any(
            getattr(self, knob) == "fp16" for knob in _KNOBS
        )


def precision(
    base: str = "fp64", *, variant: str = "uniform", **knobs: str
) -> Precision:
    """Return a ``Precision`` with every knob in ``base`` except ``knobs``."""
    return Precision(**({knob: base for knob in _KNOBS} | knobs), variant=variant)


PRECISIONS = {name: precision(name) for name in ("fp64", "fp32", *_ELEMENT_FORMATS)}


def resolve_precision(value: str | Precision) -> Precision:
    """Return ``value`` if it is a ``Precision``, else the named preset."""
    if isinstance(value, Precision):
        return value
    if value not in PRECISIONS:
        raise ValueError(
            f"unknown precision {value!r}; expected a Precision or one of "
            f"{sorted(PRECISIONS)}",
        )
    return PRECISIONS[value]


@cache
def _rounder(name: str, variant: str):
    """``quantize`` with the backward pass of the variant.

    The uniform variant rounds cotangents to the same format; the split
    variant passes them through unchanged (a straight-through estimator).
    """

    @jax.custom_vjp
    def rounded(x):
        return quantize(x, name)

    def forward(x):
        return quantize(x, name), None

    def backward(_, cotangent):
        if variant == "split":
            return (cotangent,)
        return (quantize(cotangent, name),)

    rounded.defvjp(forward, backward)
    return rounded


def _blocked_product(a: jnp.ndarray, b: jnp.ndarray, name: str) -> jnp.ndarray:
    """``a @ b.T`` accumulated in blocks of 32 as :func:`accumulate` does."""
    fmt = _ELEMENT_FORMATS[name]
    a_blocks = jnp.moveaxis(_to_blocks(a, 32), -2, 0)
    b_blocks = jnp.moveaxis(_to_blocks(b, 32), -2, 0)

    def add(running, block):
        a_block, b_block = block
        partial = jnp.matmul(a_block, b_block.T, precision=jax.lax.Precision.HIGHEST)
        return _round(running + _round(partial, fmt), fmt), None

    zeros = jnp.zeros((a.shape[0], b.shape[0]), a.dtype)
    total, _ = jax.lax.scan(add, zeros, (a_blocks, b_blocks))
    return total


@cache
def _accumulating_gemm(name: str, variant: str):
    """Phase contraction accumulating in ``name``, characters held constant.

    The uniform variant also accumulates the parameter gradient, a sum over
    samples, in ``name``; the split variant computes it in float32.
    """

    @jax.custom_vjp
    def gemm(characters, flips):
        return _blocked_product(characters, flips, name)

    def forward(characters, flips):
        return gemm(characters, flips), characters

    def backward(characters, cotangent):
        if variant == "split":
            flips_cotangent = jnp.matmul(
                cotangent.T, characters, precision=jax.lax.Precision.HIGHEST
            )
        else:
            flips_cotangent = _blocked_product(cotangent.T, characters.T, name)
        return jnp.zeros_like(characters), flips_cotangent

    gemm.defvjp(forward, backward)
    return gemm


class _Arithmetic:
    """The operations of a training step in the formats of a ``Precision``."""

    def __init__(self, value: str | Precision) -> None:
        self.precision = resolve_precision(value)

    def dtype(self, knob: str) -> jnp.dtype:
        """float64 for an fp64 knob, float32 otherwise."""
        fp64 = getattr(self.precision, knob) == "fp64"
        return jnp.dtype(jnp.float64 if fp64 else jnp.float32)

    def round(self, x: jnp.ndarray, knob: str) -> jnp.ndarray:
        """Round ``x`` to the knob's format, in ``x``'s dtype below fp32."""
        name = getattr(self.precision, knob)
        if name == "fp64":
            return x
        if name == "fp32":
            return x.astype(jnp.float32)
        return _rounder(name, self.precision.variant)(x)

    def gemm(self, characters: jnp.ndarray, flips: jnp.ndarray) -> jnp.ndarray:
        """Return ``characters @ flips.T`` with rounded inputs and accumulation."""
        flips = self.round(flips, "gemm_inputs")
        name = self.precision.gemm_accumulate
        if name == "fp64":
            return characters @ flips.T
        characters = characters.astype(jnp.float32)
        flips = flips.astype(jnp.float32)
        if name == "fp32":
            return jnp.matmul(characters, flips.T, precision=jax.lax.Precision.HIGHEST)
        return _accumulating_gemm(name, self.precision.variant)(characters, flips)

    def sum(self, x: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Sum in the statistics format: block partial sums, fp32 final sum."""
        name = self.precision.statistics
        if name == "fp64":
            return jnp.sum(x, axis=axis)
        x = jnp.moveaxis(x.astype(jnp.float32), axis, -1)
        if name == "fp32":
            return jnp.sum(x, axis=-1)
        partials = jnp.sum(_to_blocks(x, 32), axis=-1)
        return jnp.sum(self.round(partials, "statistics"), axis=-1)
