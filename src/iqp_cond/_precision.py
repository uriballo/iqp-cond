"""Number formats for the operations of the estimator, the loss and training.

:class:`Precision` names a format for each operation of a training step, and
:data:`PRECISIONS` holds the uniform presets and ``"fp8_stochastic"``.
:func:`quantize` and :func:`accumulate` emulate the formats: they round values,
and sum them, as hardware working in the format would, while computing in the
input's dtype.

:class:`_Arithmetic` is the seam the estimator and the loss compute through:
an fp64 knob runs the plain float64 expression, and any other knob rounds to
its format and computes in float32.  With ``arithmetic="native"`` it rounds
through the formats' hardware types and computes the two phase contractions in
them, accumulating in fp32.
"""

from __future__ import annotations

import warnings
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


def _round(
    x: jnp.ndarray, fmt: _ElementFormat, key: jax.Array | None = None
) -> jnp.ndarray:
    """Round to nearest, ties to even, including subnormals, in ``x``'s dtype.

    Dividing by the power-of-two quantum is exact, so rounding happens once.
    With ``key``, round stochastically instead: up with probability equal to
    the distance from the lower neighbour in quanta, so the result is unbiased.
    """
    magnitude = jnp.abs(x)
    _, exponent = jnp.frexp(magnitude)
    exponent = jnp.maximum(exponent - 1, fmt.min_exponent)
    quantum = jnp.ldexp(jnp.ones_like(x), exponent - fmt.mantissa_bits)
    steps = magnitude / quantum
    if key is None:
        steps = jnp.round(steps)
    else:
        lower = jnp.floor(steps)
        up = jax.random.uniform(key, x.shape, x.dtype) < steps - lower
        steps = lower + up
    rounded = steps * quantum
    if key is not None:
        # A value whose nearest rounding is in range (such as an amax element
        # scaled to just above the largest value) must not round up past it.
        in_range = jnp.round(magnitude / quantum) * quantum <= fmt.max
        rounded = jnp.where(in_range, jnp.minimum(rounded, fmt.max), rounded)
    overflow = {"inf": jnp.inf, "nan": jnp.nan, "saturate": fmt.max}[fmt.overflow]
    rounded = jnp.where(rounded > fmt.max, overflow, rounded)
    return jnp.copysign(rounded, x)


_MX_FORMATS = {
    "mxfp8": "fp8_e4m3",
    "mxfp6_e2m3": "fp6_e2m3",
    "mxfp6_e3m2": "fp6_e3m2",
    "mxfp4": "fp4_e2m1",
}


def _to_blocks(x: jnp.ndarray, size: int) -> jnp.ndarray:
    """Split the last axis into zero-padded blocks of ``size``."""
    padded = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, -x.shape[-1] % size)])
    return padded.reshape(*x.shape[:-1], -1, size)


def _from_blocks(blocks: jnp.ndarray, width: int) -> jnp.ndarray:
    return blocks.reshape(*blocks.shape[:-2], -1)[..., :width]


def _quantize_mx(
    x: jnp.ndarray, element: _ElementFormat, key: jax.Array | None = None
) -> jnp.ndarray:
    """OCP MX v1.0: blocks of 32 along the last axis share a power-of-two scale.

    The scale is ``2^(floor(log2 amax) - emax)``, with ``emax`` the exponent of
    the element format's largest value, and elements saturate.
    """
    blocks = _to_blocks(x, 32)
    scale = _mx_scale(blocks, element)
    element = replace(element, overflow="saturate")
    rounded = _round(blocks / scale, element, key) * scale
    return _from_blocks(rounded, x.shape[-1])


def _mx_scale(blocks: jnp.ndarray, element: _ElementFormat) -> jnp.ndarray:
    """The power-of-two scale of each block, with a trailing axis of one."""
    amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    _, amax_exponent = jnp.frexp(amax)
    _, element_exponent = jnp.frexp(element.max)
    # The shared scale is an E8M0 power of two, limited to 2^-127 ... 2^127.
    exponent = jnp.clip(amax_exponent - element_exponent, -127, 127)
    return jnp.where(amax > 0, jnp.ldexp(jnp.ones_like(amax), exponent), 1.0)


def _quantize_nvfp4(x: jnp.ndarray, key: jax.Array | None = None) -> jnp.ndarray:
    """NVFP4: fp4 elements in blocks of 16 with an e4m3 scale per block.

    A per-tensor scale ``6 * 448 / amax`` maps the block scales into e4m3's
    range; each element is ``x * tensor_scale / block_scale`` rounded to fp4.
    """
    fp4 = _ELEMENT_FORMATS["fp4_e2m1"]
    blocks = _to_blocks(x, 16)
    tensor_scale, block_scale = _nvfp4_scales(blocks)
    elements = _round(
        blocks * tensor_scale / block_scale, replace(fp4, overflow="saturate"), key
    )
    return _from_blocks(elements * block_scale / tensor_scale, x.shape[-1])


def _nvfp4_scales(blocks: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The per-tensor scale and the e4m3 scale of each block of 16."""
    fp4 = _ELEMENT_FORMATS["fp4_e2m1"]
    e4m3 = replace(_ELEMENT_FORMATS["fp8_e4m3"], overflow="saturate")
    amax = jnp.max(jnp.abs(blocks))
    tensor_scale = jnp.where(amax > 0, fp4.max * e4m3.max / amax, 1.0)
    block_amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    block_scale = _round(block_amax / fp4.max * tensor_scale, e4m3)
    return tensor_scale, jnp.where(block_scale > 0, block_scale, 1.0)


def quantize(
    x: jnp.ndarray, name: str, key: jax.Array | None = None, scaling: str = "amax"
) -> jnp.ndarray:
    """Round ``x`` to the named format and return it in its own dtype.

    With ``key`` the elements round stochastically; scales always round to
    nearest.  ``scaling="pow2"`` rounds the per-tensor scale down to a power of
    two, which leaves the format's grid for normal values the same for every
    tensor, whatever its largest magnitude; block formats ignore it.
    """
    if name in _MX_FORMATS:
        return _quantize_mx(x, _ELEMENT_FORMATS[_MX_FORMATS[name]], key)
    if name == "nvfp4":
        return _quantize_nvfp4(x, key)
    fmt = _ELEMENT_FORMATS[name]
    if not fmt.amax_scaled:
        return _round(x, fmt, key)
    scale = _tensor_scale(x, fmt, scaling)
    return _round(x * scale, fmt, key) / scale


def _tensor_scale(x: jnp.ndarray, fmt: _ElementFormat, scaling: str) -> jnp.ndarray:
    """The scale mapping ``x``'s largest magnitude onto the format's largest value."""
    amax = jnp.max(jnp.abs(x))
    scale = jnp.where(amax > 0, fmt.max / amax, 1.0)
    if scaling == "pow2":
        scale = jnp.exp2(jnp.floor(jnp.log2(scale)))
    return scale


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
# Unset (None), cosines follow phases and the contraction's backward follows
# the variant.
_OPTIONAL_KNOBS = ("cosines", "phase_gradient", "backward_accumulate")
_BACKWARD_KNOBS = ("phase_gradient", "backward_accumulate")
_BLOCK_FORMATS = (*_MX_FORMATS, "nvfp4")
FORMATS = ("fp64", "fp32", *_ELEMENT_FORMATS, *_BLOCK_FORMATS)


@dataclass(frozen=True)
class Precision:
    """The number format of each operation in a training step.

    ``gemm_inputs`` rounds the parameters entering the phase contraction (the
    other operand, the characters, is exactly ``±1``); ``gemm_accumulate`` is
    the precision in which that contraction sums and returns the phases;
    ``phases`` covers phase storage; ``cosines`` the cosine and sine (unset, it
    follows ``phases``); ``statistics`` covers the per-sample values, their
    sums and the loss assembly; and ``ratios`` covers encoder amplitude ratios.

    The backward pass follows ``variant``: ``"uniform"`` rounds the gradient
    leaving each operation to that operation's format, and ``"split"`` runs
    the backward computations in fp32.  ``phase_gradient`` and
    ``backward_accumulate`` add roundings to the phase contraction's backward:
    the gradient with respect to the phases is rounded to ``phase_gradient``
    where it enters the backward contraction, which sums the gradient with
    respect to the flips in ``backward_accumulate``, following
    ``accumulation``.  Both round within the forward contraction's dtype
    (float64 when ``gemm_accumulate`` is fp64, float32 otherwise).  Under the
    uniform variant the variant's own roundings stay: the gradient reaching
    the contraction is already rounded to ``phases`` (and, with
    ``"hardware"`` accumulation, to ``gemm_accumulate``), the gradient leaving
    it is rounded to ``gemm_inputs``, and an unset ``backward_accumulate``
    sums as the forward contraction does.  Under the split variant only the
    two knobs round, an unset one not at all, so the split variant isolates
    each of them.

    Below fp32, ``gemm_accumulate`` follows ``accumulation``: ``"blocked"``,
    the default, keeps a genuine low-precision accumulator (blocks of 32 and a
    rounded running sum), a pessimistic bound; ``"hardware"`` accumulates in
    fp32 and rounds the contraction's output to the format, as tensor cores
    do.

    ``rounding="stochastic"`` rounds every value below fp32 up or down at
    random, with probability set by its distance to each neighbour, so the
    rounding is unbiased; the uniform variant's cotangents too, with
    independent bits.  A tuple of field names, such as
    ``rounding=("gemm_inputs", "phase_gradient")``, rounds only those
    operations stochastically and the rest to nearest.  Block scales and the
    ``"blocked"`` accumulator still round to nearest.  The bits come from the
    key of each loss evaluation.

    ``scaling`` sets the per-tensor scale of the formats below 16 bits:
    ``"amax"`` maps each tensor's largest magnitude to the format's largest
    value, and ``"pow2"`` rounds that scale down to a power of two, so the grid
    of normal values does not move with the tensor's largest magnitude.

    ``arithmetic="native"`` computes in the formats' hardware types instead of
    emulating them, with the same formats and the same rounded values.  Each
    rounding is XLA's cast to the format's type, and the forward and backward
    phase contractions multiply in the type of ``gemm_inputs`` and
    ``phase_gradient`` and accumulate in fp32 (float64 when
    ``gemm_accumulate`` is fp64).  The formats below 16 bits keep the
    current-maximum scaling of each tensor; delayed scaling is not
    implemented.  Block formats multiply through ``jax.nn.scaled_matmul``.
    Element-wise operations compute in fp32 between roundings, as in
    emulation, so the variants keep their meaning.  A format without a fast
    product on the device (fp6, plain fp4 and mxfp6 on any device, tf32 off
    NVIDIA GPUs), rounds as in emulation and multiplies in fp32, with
    a warning: exact, but not faster.  Native arithmetic refuses stochastic
    rounding and the ``"blocked"`` accumulator below fp32, neither of which
    XLA provides.
    """

    gemm_inputs: str = "fp64"
    gemm_accumulate: str = "fp64"
    phases: str = "fp64"
    statistics: str = "fp64"
    ratios: str = "fp64"
    variant: str = "uniform"
    accumulation: str = "blocked"
    rounding: str | tuple[str, ...] = "nearest"
    scaling: str = "amax"
    arithmetic: str = "emulated"
    cosines: str | None = None
    phase_gradient: str | None = None
    backward_accumulate: str | None = None

    def __post_init__(self) -> None:
        for knob in (*_KNOBS, *_OPTIONAL_KNOBS):
            name = getattr(self, knob)
            if name is None and knob in _OPTIONAL_KNOBS:
                continue
            if name not in FORMATS:
                raise ValueError(
                    f"unknown format {name!r} for {knob}; expected one of {FORMATS}"
                )
            if name in _BLOCK_FORMATS and knob != "gemm_inputs":
                raise ValueError(
                    f"block-scaled {name!r} applies only to gemm_inputs, not {knob}"
                )
        if self.accumulation not in ("hardware", "blocked"):
            raise ValueError(
                f"accumulation must be 'hardware' or 'blocked', "
                f"not {self.accumulation!r}"
            )
        if self.variant not in ("uniform", "split"):
            raise ValueError(
                f"variant must be 'uniform' or 'split', not {self.variant!r}"
            )
        if isinstance(self.rounding, str):
            if self.rounding not in ("nearest", "stochastic"):
                raise ValueError(
                    "rounding must be 'nearest', 'stochastic' or a tuple of field "
                    f"names, not {self.rounding!r}"
                )
        else:
            object.__setattr__(self, "rounding", tuple(self.rounding))
            unknown = set(self.rounding) - {*_KNOBS, *_OPTIONAL_KNOBS}
            if not self.rounding or unknown:
                raise ValueError(
                    "rounding must name at least one format field, and only "
                    f"those; got {self.rounding!r}"
                )
            unrounded = [
                knob
                for knob in self.rounding
                if self.format_of(knob) in (None, "fp64", "fp32")
            ]
            if unrounded:
                raise ValueError(
                    f"rounding names {unrounded}, which do not round below fp32"
                )
        if self.scaling not in ("amax", "pow2"):
            raise ValueError(f"scaling must be 'amax' or 'pow2', not {self.scaling!r}")
        if self.arithmetic not in ("emulated", "native"):
            raise ValueError(
                f"arithmetic must be 'emulated' or 'native', not {self.arithmetic!r}"
            )
        if self.arithmetic == "native":
            if self.rounding != "nearest":
                raise ValueError(
                    "native arithmetic rounds to nearest, as XLA has no stochastic "
                    "rounding; set rounding='nearest', or arithmetic='emulated'"
                )
            for knob in ("gemm_accumulate", "backward_accumulate"):
                name = self.format_of(knob)
                if self.accumulation == "blocked" and name not in (
                    None,
                    "fp64",
                    "fp32",
                ):
                    raise ValueError(
                        f"native products accumulate in fp32, so {knob}={name!r} "
                        "needs accumulation='hardware', which rounds the fp32 sum "
                        f"to {name}; or set arithmetic='emulated'"
                    )

    @property
    def loss_scaled(self) -> bool:
        """Whether training scales the loss to keep fp16 gradients in range.

        Only the uniform variant differentiates in the knobs' formats; the
        split variant's backward pass is fp32, apart from the backward knobs.
        """
        forward = self.variant == "uniform" and any(
            self.format_of(knob) == "fp16" for knob in (*_KNOBS, "cosines")
        )
        return forward or "fp16" in (self.phase_gradient, self.backward_accumulate)

    def format_of(self, knob: str) -> str | None:
        """Return the knob's format, with ``cosines`` following ``phases``."""
        if knob == "cosines" and self.cosines is None:
            return self.phases
        return getattr(self, knob)

    def stochastic(self, knob: str) -> bool:
        """Whether the knob's operation rounds stochastically."""
        if isinstance(self.rounding, str):
            return self.rounding == "stochastic"
        return knob in self.rounding


def precision(
    base: str = "fp64",
    *,
    variant: str = "uniform",
    rounding: str | tuple[str, ...] = "nearest",
    scaling: str = "amax",
    arithmetic: str = "emulated",
    **knobs: str,
) -> Precision:
    """Return a ``Precision`` with every format in ``base`` except ``knobs``.

    ``base`` sets the five forward formats; ``cosines`` and the backward
    fields stay unset unless given in ``knobs``, which may also set
    ``accumulation``.
    """
    return Precision(
        **({knob: base for knob in _KNOBS} | knobs),
        variant=variant,
        rounding=rounding,
        scaling=scaling,
        arithmetic=arithmetic,
    )


PRECISIONS = {name: precision(name) for name in ("fp64", "fp32", *_ELEMENT_FORMATS)}
# fp8 E4M3 with stochastic rounding, the statistics in fp32 and the
# contraction accumulated in fp32 as tensor cores do.
PRECISIONS["fp8_stochastic"] = precision(
    "fp8_e4m3", statistics="fp32", rounding="stochastic", accumulation="hardware"
)


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
def _rounder(name: str, variant: str, scaling: str = "amax", native: bool = False):
    """``quantize``, or with ``native`` :func:`_cast`, with the variant's backward.

    The uniform variant rounds cotangents to the same format; the split
    variant passes them through unchanged (a straight-through estimator).
    """

    def round_(x):
        return _cast(x, name, scaling) if native else quantize(x, name, scaling=scaling)

    @jax.custom_vjp
    def rounded(x):
        return round_(x)

    def forward(x):
        return round_(x), None

    def backward(_, cotangent):
        if variant == "split":
            return (cotangent,)
        return (round_(cotangent),)

    rounded.defvjp(forward, backward)
    return rounded


@cache
def _stochastic_rounder(name: str, variant: str, scaling: str = "amax"):
    """:func:`_rounder` rounding stochastically with bits from ``key``; the
    uniform variant's cotangents use bits independent of the forward ones."""

    @jax.custom_vjp
    def rounded(x, key):
        return quantize(x, name, key, scaling)

    def forward(x, key):
        return quantize(x, name, key, scaling), key

    def backward(key, cotangent):
        if variant == "split":
            return cotangent, None
        return quantize(cotangent, name, jax.random.fold_in(key, 1), scaling), None

    rounded.defvjp(forward, backward)
    return rounded


@cache
def _contraction(
    blocked: str | None,
    phase_gradient: str | None,
    backward_accumulate: str | None,
    accumulation: str,
    scaling: str,
):
    """Phase contraction with the backward set by the two backward knobs.

    The forward accumulates in ``blocked`` when given, else as one matmul.
    The backward rounds the gradient with respect to the phases to
    ``phase_gradient``, and sums the gradient with respect to the flips in
    ``backward_accumulate``; ``None`` leaves either unrounded.  Each key
    rounds its knob stochastically, ``None`` to nearest.
    """

    def rounded(x, name, key):
        if name in (None, "fp64"):
            return x
        if name == "fp32":
            return x.astype(jnp.float32).astype(x.dtype)
        return quantize(x, name, key, scaling)

    def product(characters, flips):
        if blocked is not None:
            return _blocked_product(characters, flips, blocked)
        return jnp.matmul(characters, flips.T, precision=jax.lax.Precision.HIGHEST)

    @jax.custom_vjp
    def contract(characters, flips, gradient_key, accumulate_key):
        return product(characters, flips)

    def forward(characters, flips, gradient_key, accumulate_key):
        return product(characters, flips), (characters, gradient_key, accumulate_key)

    def backward(residual, cotangent):
        characters, gradient_key, accumulate_key = residual
        cotangent = rounded(cotangent, phase_gradient, gradient_key)
        if accumulation == "blocked" and backward_accumulate not in (
            None,
            "fp64",
            "fp32",
        ):
            flips_cotangent = _blocked_product(
                cotangent.T, characters.T, backward_accumulate
            )
        else:
            flips_cotangent = rounded(
                jnp.matmul(
                    cotangent.T, characters, precision=jax.lax.Precision.HIGHEST
                ),
                backward_accumulate,
                accumulate_key,
            )
        return jnp.zeros_like(characters), flips_cotangent, None, None

    contract.defvjp(forward, backward)
    return contract


def _blocked_product(a: jnp.ndarray, b: jnp.ndarray, name: str) -> jnp.ndarray:
    """``a @ b.T`` accumulated in ``name`` exactly as :func:`accumulate` sums.

    The block partial products are rounded, and so is the running sum after
    each block: the accumulator of a GEMM in that format.
    """
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


# Native arithmetic.  The hardware type of each format XLA can cast to; XLA
# fails to compile fp6 casts (JAX 0.11.2) and tf32 has no storage type, so
# those formats round in software.
_DTYPES = {
    "bf16": jnp.bfloat16,
    "fp16": jnp.float16,
    "fp8_e4m3": jnp.float8_e4m3fn,
    "fp8_e5m2": jnp.float8_e5m2,
    "fp4_e2m1": jnp.float4_e2m1fn,
}
# The first CUDA compute capability whose tensor cores multiply in each format:
# Volta for fp16, Ampere for bf16 and tf32, Ada for fp8, Blackwell for the
# block formats that jax.nn.scaled_matmul fuses.  The formats left out
# multiply in fp32.
_FAST_PRODUCT = {
    "fp16": 7.0,
    "bf16": 8.0,
    "tf32": 8.0,
    "fp8_e4m3": 8.9,
    "fp8_e5m2": 8.9,
    "mxfp8": 10.0,
    "nvfp4": 10.0,
}
# The element type, scale type and block size of the block formats that
# jax.nn.scaled_matmul multiplies.
_SCALED = {
    "mxfp8": (jnp.float8_e4m3fn, jnp.float8_e8m0fnu, 32),
    "nvfp4": (jnp.float4_e2m1fn, jnp.float8_e4m3fn, 16),
}
_FORWARD = (((1,), (1,)), ((), ()))  # characters (S, m) with flips (R, m)
_BACKWARD = (((0,), (0,)), ((), ()))  # characters (S, m) with phase cotangents (S, R)


def _cast(x: jnp.ndarray, name: str, scaling: str = "amax") -> jnp.ndarray:
    """``quantize`` through the format's hardware type, in ``x``'s dtype.

    XLA's cast rounds to nearest, ties to even, as :func:`_round` does, so the
    values are the emulated ones.  Formats without a type XLA casts to round in
    software, and so do float64 values, which XLA rounds to bf16 through
    float32, twice.  XLA casts NaN to fp4 as zero, so NaN passes through.
    """
    if name == "fp64":
        return x
    if name == "fp32":
        return x.astype(jnp.float32).astype(x.dtype)
    if name not in _DTYPES or x.dtype == jnp.float64:
        return quantize(x, name, scaling=scaling)
    fmt = _ELEMENT_FORMATS[name]
    scale = _tensor_scale(x, fmt, scaling) if fmt.amax_scaled else 1.0
    cast = (x * scale).astype(_DTYPES[name]).astype(x.dtype) / scale
    return jnp.where(jnp.isnan(x), x, cast)


def _has_product(name: str) -> bool:
    """Whether the default device multiplies in ``name``'s type.

    The CPU computes every type XLA has in software, so the native products
    run there for tests, apart from tf32, which only NVIDIA GPUs have.  NVIDIA
    GPUs follow ``_FAST_PRODUCT``; other devices multiply in fp32.
    """
    if name not in _FAST_PRODUCT:
        return False
    device = jax.devices()[0]
    if device.platform == "cpu":
        return name != "tf32"
    capability = str(getattr(device, "compute_capability", ""))
    if not capability.replace(".", "", 1).isdigit():
        return False
    return float(capability) >= _FAST_PRODUCT[name]


def _warn_upcast(knob: str, name: str) -> None:
    device = jax.devices()[0].device_kind
    warnings.warn(
        f"native arithmetic: {knob}={name!r} has no fast product on {device}; it "
        f"rounds to {name} and multiplies in fp32, exact but not faster",
        stacklevel=4,
    )


def _native_dot(
    characters: jnp.ndarray,
    x: jnp.ndarray,
    dimensions,
    name: str,
    scaling: str,
    fast: bool,
) -> jnp.ndarray:
    """The characters contracted with ``x`` rounded to ``name``, in ``x``'s dtype.

    The characters are exactly ``±1`` in every format.  With ``fast`` the
    product runs in the format's type and accumulates in fp32; otherwise ``x``
    rounds to the format and the product runs in its own dtype.
    """
    wide = x.dtype
    if not fast:
        x = _cast(x, name, scaling)
        return jax.lax.dot_general(
            characters.astype(wide), x, dimensions, precision=jax.lax.Precision.HIGHEST
        )
    if name in _SCALED:
        return _scaled_product(characters, x, name)
    if name == "tf32":
        # Rounded first, the inputs pass the tensor cores' tf32 conversion as they are.
        product = jax.lax.dot_general(
            characters.astype(jnp.float32),
            quantize(x, name).astype(jnp.float32),
            dimensions,
            precision=jax.lax.DotAlgorithmPreset.TF32_TF32_F32,
        )
        return product.astype(wide)
    fmt = _ELEMENT_FORMATS[name]
    scale = _tensor_scale(x, fmt, scaling) if fmt.amax_scaled else 1.0
    dtype = _DTYPES[name]
    product = jax.lax.dot_general(
        characters.astype(dtype),
        (x * scale).astype(dtype),
        dimensions,
        preferred_element_type=jnp.float32,
    )
    return product.astype(wide) / scale


def _scaled_operand(
    x: jnp.ndarray, name: str
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | float]:
    """``x``'s elements and block scales in the block format's types.

    The last axis is zero-padded to whole blocks.  The values are the emulated
    ones: ``quantize(x, name)`` is the elements times their block scale over
    the per-tensor scale (1 for the MX formats), with the padding cut off.
    """
    element, scale_type, size = _SCALED[name]
    blocks = _to_blocks(x, size)
    if name == "nvfp4":
        tensor_scale, scales = _nvfp4_scales(blocks)
        blocks = blocks * tensor_scale
    else:
        tensor_scale = 1.0
        scales = _mx_scale(blocks, _ELEMENT_FORMATS[_MX_FORMATS[name]])
    # The elements saturate; clipped first, the cast cannot overflow.
    limit = float(jnp.finfo(element).max)
    elements = jnp.clip(blocks / scales, -limit, limit).astype(element)
    return (
        elements.reshape(*x.shape[:-1], -1),
        scales[..., 0].astype(scale_type),
        tensor_scale,
    )


def _scaled_product(
    characters: jnp.ndarray, flips: jnp.ndarray, name: str
) -> jnp.ndarray:
    """``characters @ flips.T`` with the flips in a block format.

    The characters are exact with unit scales; the padding of the contracted
    axis is zero in both operands.
    """
    elements, scales, tensor_scale = _scaled_operand(flips, name)
    width = elements.shape[-1]
    signs = jnp.pad(characters, ((0, 0), (0, width - characters.shape[-1])))
    element, scale_type, _ = _SCALED[name]
    ones = jnp.ones((signs.shape[0], scales.shape[-1]), scale_type)
    product = jax.nn.scaled_matmul(
        signs.astype(element)[None],
        elements[None],
        ones[None],
        scales[None],
        preferred_element_type=jnp.float32,
    )[0]
    # fp4 elements cannot hold NaN; a NaN flip makes its phases NaN, as emulated.
    return jnp.where(jnp.isnan(flips).any(axis=-1), jnp.nan, product / tensor_scale)


@cache
def _native_contraction(
    inputs: str,
    phase_gradient: str | None,
    backward_accumulate: str | None,
    variant: str,
    scaling: str,
    forward_fast: bool,
    backward_fast: bool,
):
    """The phase contraction in hardware types, with the emulation's backward.

    The forward multiplies the characters by the flips in ``inputs``.  The
    backward rounds the gradient with respect to the phases to
    ``phase_gradient`` and multiplies in its type (``None`` leaves it
    unrounded), rounds the fp32 sum to ``backward_accumulate``, and under the
    uniform variant rounds the gradient leaving the input rounding to
    ``inputs``, as :class:`Precision` describes.
    """

    def product(characters, flips):
        return _native_dot(characters, flips, _FORWARD, inputs, scaling, forward_fast)

    @jax.custom_vjp
    def contract(characters, flips):
        return product(characters, flips)

    def forward(characters, flips):
        return product(characters, flips), characters

    def backward(characters, cotangent):
        gradient = _native_dot(
            characters,
            cotangent,
            _BACKWARD,
            phase_gradient or "fp64",
            scaling,
            backward_fast,
        ).T
        gradient = _cast(gradient, backward_accumulate or "fp64", scaling)
        if variant == "uniform":
            gradient = _cast(gradient, inputs, scaling)
        return jnp.zeros_like(characters), gradient

    contract.defvjp(forward, backward)
    return contract


# Folded into a loss's or estimator's key to give the stochastic-rounding bits,
# leaving every draw already made from that key unchanged.
_ROUNDING_TAG = 0x5EED


def _rounding_key(key: jax.Array | None) -> jax.Array | None:
    return None if key is None else jax.random.fold_in(key, _ROUNDING_TAG)


class _Arithmetic:
    """The operations of a training step in the formats of a ``Precision``.

    Stochastic rounding needs ``key``; each rounding site draws its bits from
    ``key`` folded with the site's position in the trace.
    """

    def __init__(self, value: str | Precision, key: jax.Array | None = None) -> None:
        self.precision = resolve_precision(value)
        if self.precision.rounding != "nearest" and key is None:
            raise ValueError(
                "stochastic rounding needs a random key; pass one to the estimator "
                "or the loss"
            )
        self._key = key
        self._sites = 0
        self._native = self.precision.arithmetic == "native"

    def dtype(self, knob: str) -> jnp.dtype:
        """float64 for an fp64 knob, float32 otherwise."""
        fp64 = self.precision.format_of(knob) == "fp64"
        return jnp.dtype(jnp.float64 if fp64 else jnp.float32)

    def _site_key(self, knob: str) -> jax.Array | None:
        """A fresh key for a stochastic knob's rounding site, else ``None``."""
        if not self.precision.stochastic(knob):
            return None
        self._sites += 1
        return jax.random.fold_in(self._key, self._sites)

    def round(self, x: jnp.ndarray, knob: str) -> jnp.ndarray:
        """Return ``x`` in the knob's format: float64 for fp64, float32 for fp32.

        Formats below fp32 round in ``x``'s own dtype, so a float64 value is
        rounded once, without passing through float32.
        """
        name = self.precision.format_of(knob)
        if name == "fp64":
            return x.astype(jnp.float64)
        if name == "fp32":
            return x.astype(jnp.float32)
        variant, scaling = self.precision.variant, self.precision.scaling
        if self._native:
            return _rounder(name, variant, scaling, native=True)(x)
        key = self._site_key(knob)
        if key is None:
            return _rounder(name, variant, scaling)(x)
        return _stochastic_rounder(name, variant, scaling)(x, key)

    def gemm(self, characters: jnp.ndarray, flips: jnp.ndarray) -> jnp.ndarray:
        """Return ``characters @ flips.T`` with rounded inputs and accumulation."""
        if self._native:
            return self._native_gemm(characters, flips)
        if any(getattr(self.precision, knob) is not None for knob in _BACKWARD_KNOBS):
            return self._gemm_with_backward(characters, flips)
        flips = self.round(flips, "gemm_inputs")
        name = self.precision.gemm_accumulate
        if name == "fp64":
            return characters.astype(jnp.float64) @ flips.astype(jnp.float64).T
        characters = characters.astype(jnp.float32)
        flips = flips.astype(jnp.float32)
        if name != "fp32" and self.precision.accumulation == "blocked":
            return _accumulating_gemm(name, self.precision.variant)(characters, flips)
        product = jnp.matmul(characters, flips.T, precision=jax.lax.Precision.HIGHEST)
        return product if name == "fp32" else self.round(product, "gemm_accumulate")

    def _native_gemm(self, characters: jnp.ndarray, flips: jnp.ndarray) -> jnp.ndarray:
        """:meth:`gemm` in hardware types; below fp32 the sum rounds to the format."""
        precision = self.precision
        dtype = self.dtype("gemm_accumulate")
        # An fp64 accumulator multiplies the rounded values in float64.
        fast = {}
        for knob in ("gemm_inputs", "phase_gradient"):
            name = precision.format_of(knob)
            low = dtype == jnp.float32 and name not in (None, "fp64", "fp32")
            fast[knob] = low and _has_product(name)
            if low and not fast[knob]:
                _warn_upcast(knob, name)
        contract = _native_contraction(
            precision.gemm_inputs,
            precision.phase_gradient,
            precision.backward_accumulate,
            precision.variant,
            precision.scaling,
            fast["gemm_inputs"],
            fast["phase_gradient"],
        )
        product = contract(characters.astype(dtype), flips.astype(dtype))
        if precision.gemm_accumulate in ("fp64", "fp32"):
            return product
        return self.round(product, "gemm_accumulate")

    def _gemm_with_backward(
        self, characters: jnp.ndarray, flips: jnp.ndarray
    ) -> jnp.ndarray:
        """:meth:`gemm` with the backward knobs' roundings in its backward pass.

        The forward roundings keep the variant's backward, so under the
        uniform variant the knobs add to its roundings.
        """
        precision = self.precision
        dtype = self.dtype("gemm_accumulate")
        flips = self.round(flips, "gemm_inputs").astype(dtype)
        name = precision.gemm_accumulate
        below_fp32 = name not in ("fp64", "fp32")
        blocked = below_fp32 and precision.accumulation == "blocked"
        backward_accumulate = precision.backward_accumulate
        if backward_accumulate is None and blocked and precision.variant == "uniform":
            backward_accumulate = name
        contract = _contraction(
            name if blocked else None,
            precision.phase_gradient,
            backward_accumulate,
            precision.accumulation,
            precision.scaling,
        )
        product = contract(
            characters.astype(dtype),
            flips,
            self._site_key("phase_gradient"),
            self._site_key("backward_accumulate"),
        )
        if below_fp32 and not blocked:
            return self.round(product, "gemm_accumulate")
        return product

    def sum(self, x: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Sum in the statistics format: block partial sums, fp32 final sum.

        Unlike the GEMM accumulator, the statistics keep their per-sample
        values and block partial sums in the format but reduce the partials in
        fp32.
        """
        name = self.precision.statistics
        if name == "fp64":
            return jnp.sum(x, axis=axis)
        x = jnp.moveaxis(x.astype(jnp.float32), axis, -1)
        if name == "fp32":
            return jnp.sum(x, axis=-1)
        partials = jnp.sum(_to_blocks(x, 32), axis=-1)
        return jnp.sum(self.round(partials, "statistics"), axis=-1)
