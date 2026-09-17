"""Fourier-mask law, samplers, and compressed operator banks.

The Hamming-RBF kernel has a product character expansion, so its squared MMD is
a weighted sum over Pauli masks whose law under bandwidth ``sigma`` is Bernoulli
with ``p = (1 - exp(-1/(2 sigma^2))) / 2``.  Training needs to draw those masks —
in NumPy when building a bank once, in JAX when refreshing inside a compiled
step — and to collapse duplicate draws exactly, because a repeated mask is a
bit-identical column that costs compute and adds no information.

Mask draws are conditioned on ``mask != 0``: the identity contributes exactly
zero to the kernel MMD, so spending budget on it only adds variance.  The
remaining mass ``1 - (1 - p)**n`` is folded into the returned weights, and
duplicates are represented once with their exact empirical multiplicity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

__all__ = [
    "OperatorBank",
    "bank_from_ragged",
    "expected_bodyness_sigmas",
    "mmd_operator_probability",
    "operator_bank",
    "sample_nonidentity_masks",
    "sample_nonidentity_masks_jax",
]


def expected_bodyness_sigmas(
    n_visible: int,
    targets: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return bandwidths whose mask law targets the given mean bodyness.

    The mean mask weight under bandwidth ``sigma`` is ``E[K] = n p_sigma`` with
    ``p_sigma = (1 - exp(-1/(2 sigma**2))) / 2``, so a bodyness target ``K``
    inverts to ``sigma = [-2 log(1 - 2K/n)]**-1/2``.  The library default
    targets ``(2, n/4, 0.49 n)``: a low-order band, a mid band at a quarter of
    the register, and a band just inside the high-bodyness limit.

    No bandwidth reaches ``K = n/2``, because ``p_sigma < 1/2`` strictly;
    equality holds only in the degenerate ``sigma -> 0`` limit.  Every target
    is therefore capped at ``0.49 n``, the campaign's convention for staying
    inside that boundary, which matters for narrow registers where ``2`` would
    otherwise exceed ``n/2``.
    """
    if n_visible < 1:
        raise ValueError("n_visible must be positive")
    if targets is None:
        targets = (2.0, n_visible / 4.0, 0.49 * n_visible)
    cap = 0.49 * n_visible
    values = [min(float(target), cap) for target in targets]
    if not values or any(target <= 0.0 for target in values):
        raise ValueError("bodyness targets must be positive")
    return tuple(
        float((-2.0 * math.log(1.0 - 2.0 * target / n_visible)) ** -0.5)
        for target in values
    )


def mmd_operator_probability(sigma: float | jnp.ndarray) -> jnp.ndarray:
    """Return the Bernoulli probability for Fourier observable masks."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    return (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0


def _mask_probability(sigma: float) -> float:
    """Scalar mask probability for the host-side bank path.

    The public :func:`mmd_operator_probability` goes through JAX, whose dispatch
    costs ~0.04 ms per call; the bank path calls it several times per build, so
    it uses the scalar version instead.  The pinned NumPy sampler keeps the JAX
    expression so its draws stay identical to the campaign fixtures.
    """
    return (1.0 - math.exp(-1.0 / (2.0 * sigma * sigma))) / 2.0


def sample_nonidentity_masks_jax(
    key: jax.Array,
    n_ops: int,
    n_visible: int,
    sigma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw ``n_ops`` nonidentity masks from the Hamming-RBF mask law, in JAX.

    The draw is exact and loop-free: the Hamming weight is sampled from
    ``Binomial(n, p)`` conditioned on ``k >= 1`` by inverse CDF, and a
    uniformly random ``k``-subset is produced from one order statistic of
    ``n`` uniforms.  Shapes are static, so the sampler is safe inside a jitted
    training step and masks refresh with the step key.

    Returns ``(masks, weights)`` with ``masks`` of shape ``(n_ops, n_visible)``
    and constant weights summing to the nonidentity mass ``1 - (1 - p)**n``,
    the same law and rescaling as :func:`sample_nonidentity_masks`.
    """
    if n_ops < 1 or n_visible < 1:
        raise ValueError("n_ops and n_visible must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    p = mmd_operator_probability(sigma)
    k = jnp.arange(n_visible + 1, dtype=jnp.float64)
    log_choose = (
        gammaln(n_visible + 1.0) - gammaln(k + 1.0) - gammaln(n_visible - k + 1.0)
    )
    log_pmf = log_choose + k * jnp.log(p) + (n_visible - k) * jnp.log1p(-p)
    cdf = jnp.cumsum(jnp.exp(log_pmf))
    nonidentity_mass = 1.0 - cdf[0]
    uniforms = jax.random.uniform(key, (n_ops,))
    weights = jnp.searchsorted(cdf, cdf[0] + uniforms * nonidentity_mass, side="right")
    bits = jax.random.uniform(jax.random.fold_in(key, 1), (n_ops, n_visible))
    ordered = jnp.sort(bits, axis=1)
    rows = jnp.arange(n_ops)
    threshold = ordered[rows, jnp.clip(weights - 1, 0, n_visible - 1)]
    masks = (bits <= threshold[:, None]).astype(jnp.float64)
    return masks, jnp.full((n_ops,), nonidentity_mass / n_ops, dtype=jnp.float64)


def sample_nonidentity_masks(
    rng: np.random.Generator,
    n_ops: int,
    n_qubits: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_ops`` nonidentity masks and return them compressed with weights.

    The identity mask has exactly zero population-MMD contribution, so drawing
    from ``q(mask | mask != 0)`` costs only the analytic factor
    ``1 - q(0)``; that factor is folded into the returned weights.  Duplicate
    masks are represented once with their exact empirical multiplicity, which
    reproduces the unweighted mean exactly while evaluating far fewer masks.

    Runs outside any JIT: rejection sampling needs a concrete NumPy generator.
    """
    if n_ops < 1 or n_qubits < 1 or sigma <= 0.0:
        raise ValueError("n_ops, n_qubits, and sigma must be positive")
    p = float(mmd_operator_probability(sigma))
    q_nonidentity = 1.0 - (1.0 - p) ** n_qubits
    chunks: list[np.ndarray] = []
    remaining = n_ops
    while remaining:
        proposal = rng.binomial(1, p, size=(max(remaining * 2, 64), n_qubits))
        accepted = proposal[np.any(proposal, axis=1)]
        take = min(remaining, accepted.shape[0])
        if take:
            chunks.append(accepted[:take].astype(np.int32, copy=False))
            remaining -= take
    raw = np.concatenate(chunks, axis=0)
    unique, counts = np.unique(raw, axis=0, return_counts=True)
    weights = q_nonidentity * counts.astype(np.float64) / n_ops
    return unique, weights


def _conditional_weight_cdf(n_qubits: int, sigma: float) -> np.ndarray:
    """CDF of the Hamming weight conditioned on ``k >= 1``."""
    p = _mask_probability(sigma)
    k = np.arange(n_qubits + 1, dtype=np.float64)
    log_binomial = np.asarray(
        [
            math.lgamma(n_qubits + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n_qubits - i + 1)
            for i in range(n_qubits + 1)
        ],
    )
    log_pmf = log_binomial + k * math.log(p) + (n_qubits - k) * math.log1p(-p)
    pmf = np.exp(log_pmf)[1:]
    cdf = np.cumsum(pmf)
    return cdf / cdf[-1]


def _sample_nonidentity_codes(
    rng: np.random.Generator,
    n_ops: int,
    n_qubits: int,
    sigma: float,
    n_conditions: int,
) -> np.ndarray:
    """Draw nonidentity masks for every condition as little-endian codes.

    The draw is exact and rejection-free: the Hamming weight comes from the
    binomial law conditioned on ``k >= 1`` by inverse CDF, and the bitstring is
    a uniformly random ``k``-subset taken from one order statistic of ``n``
    uniforms.  Every draw is accepted, so the cost does not degrade as sigma
    grows and the nonidentity mass concentrates on the identity.

    Codes are integers so compression is a single ``np.unique`` over the whole
    (condition, draw) batch instead of one lexicographic row sort per entry.
    """
    cdf = _conditional_weight_cdf(n_qubits, sigma)
    powers = 1 << np.arange(n_qubits, dtype=np.int64)
    # 32-bit integer ranks: the order statistic only needs a total order, and a
    # tie is a 2**-32 event, so the subset law is unchanged.  Draws and sorts
    # run for every condition at once, which measured ~1.7x faster than float64
    # uniforms drawn per condition.
    weights = np.searchsorted(cdf, rng.random((n_conditions, n_ops)), side="right") + 1
    ranks = rng.integers(
        0,
        2**32,
        size=(n_conditions, n_ops, n_qubits),
        dtype=np.uint32,
    )
    ordered = np.sort(ranks, axis=2)
    threshold = np.take_along_axis(
        ordered,
        (weights - 1)[..., None],
        axis=2,
    )[..., 0]
    bits = ranks <= threshold[..., None]
    return bits.astype(np.int64) @ powers


def _compress_codes(
    codes: np.ndarray,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-wise unique codes with counts: ``(groups, codes, counts)``."""
    n_conditions, n_ops = codes.shape
    groups = np.repeat(np.arange(n_conditions, dtype=np.int64), n_ops)
    keys = (groups << np.int64(n_qubits)) | codes.ravel()
    unique, counts = np.unique(keys, return_counts=True)
    return (
        unique >> np.int64(n_qubits),
        unique & ((1 << n_qubits) - 1),
        counts,
    )


def _decode_codes(codes: np.ndarray, n_qubits: int) -> np.ndarray:
    """Expand little-endian integer codes back into 0/1 mask rows."""
    return ((codes[:, None] >> np.arange(n_qubits, dtype=np.int64)) & 1).astype(
        np.int32
    )


@dataclass(frozen=True)
class OperatorBank:
    """Compressed Fourier-mask bank, one entry per (bandwidth, condition).

    ``masks`` has shape ``(n_bandwidths, n_conditions, n_masks, n_visible)`` and
    ``weights`` the matching ``(n_bandwidths, n_conditions, n_masks)``; each
    weight is ``q_nonidentity * count / n_ops``, so one entry's weights sum to the
    nonidentity mass.  Rows past an entry's distinct-mask count are padding — a
    zero mask with a zero weight, which contributes nothing to the weighted sum.

    Padding exists so bank shape is static.  A bank that is refreshed during
    training is passed into the compiled step as an argument rather than closed
    over, and a change of shape would force a recompilation; the bank is
    registered as a pytree for the same reason.  Build one from ragged per-entry
    arrays with :func:`bank_from_ragged`.

    ``widths`` gives the mask-axis width *evaluated* for each bandwidth; it is
    what keeps bandwidths with small effective support (large sigma) from paying
    the width of the widest one.  An empty tuple means every bandwidth is
    evaluated at the full padded ``n_masks``, which is what a bank built from
    ragged arrays does.
    """

    sigmas: tuple[float, ...]
    masks: jnp.ndarray
    weights: jnp.ndarray
    n_ops: int
    n_visible: int
    n_masks: int
    widths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Static checks only: JAX reconstructs this dataclass while tracing a
        # bank argument, where any Python-level value test would fail.
        if self.masks.ndim != 4 or self.weights.ndim != 3:
            raise ValueError(
                "masks must be (n_bandwidths, n_conditions, n_masks, n_visible) "
                "and weights (n_bandwidths, n_conditions, n_masks)",
            )
        if self.masks.shape[:-1] != self.weights.shape:
            raise ValueError("masks and weights must agree on their leading axes")
        if self.masks.shape[0] != len(self.sigmas):
            raise ValueError("masks must have one entry per bandwidth")
        if self.masks.shape[1] < 1:
            raise ValueError("bank must cover at least one condition")
        if self.masks.shape[-1] != self.n_visible:
            raise ValueError("bank masks must be visible-width")
        if self.masks.shape[2] != self.n_masks:
            raise ValueError("n_masks must match the padded mask axis")
        if self.widths:
            if len(self.widths) != len(self.sigmas):
                raise ValueError("widths must have one entry per bandwidth")
            if any(not 1 <= width <= self.n_masks for width in self.widths):
                raise ValueError("each width must be between 1 and n_masks")

    @property
    def n_bandwidths(self) -> int:
        """Number of bandwidths in the bank."""
        return len(self.sigmas)

    @property
    def n_conditions(self) -> int:
        """Number of conditions in the bank."""
        return int(self.masks.shape[1])

    def width(self, bandwidth: int) -> int:
        """Mask-axis width evaluated for ``bandwidth``."""
        return self.widths[bandwidth] if self.widths else int(self.n_masks)

    def entry(self, bandwidth: int, condition: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(masks, weights)`` for one bandwidth and condition, padded.

        The entry is cut to its bandwidth's width, so entries whose law cannot
        reach many masks are not charged for the widest bandwidth's columns.
        """
        width = self.width(bandwidth)
        return (
            self.masks[bandwidth, condition, :width],
            self.weights[bandwidth, condition, :width],
        )


jax.tree_util.register_dataclass(
    OperatorBank,
    data_fields=["masks", "weights"],
    meta_fields=["sigmas", "n_ops", "n_visible", "n_masks", "widths"],
)


def _pad_bank_entry(
    masks: np.ndarray,
    weights: np.ndarray,
    n_masks: int,
    n_visible: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad one entry's masks and weights to ``n_masks`` rows."""
    if masks.shape[0] > n_masks:
        raise ValueError(
            f"bank entry has {masks.shape[0]} masks, more than the {n_masks} "
            "it is padded to",
        )
    padded_masks = np.zeros((n_masks, n_visible), dtype=np.float64)
    padded_weights = np.zeros(n_masks, dtype=np.float64)
    padded_masks[: masks.shape[0]] = masks
    padded_weights[: weights.shape[0]] = weights
    return padded_masks, padded_weights


def _assemble_bank(
    sigmas: tuple[float, ...],
    drawn: list[list[tuple[np.ndarray, np.ndarray]]],
    n_ops: int,
    n_visible: int,
    n_masks: int,
    widths: tuple[int, ...],
) -> OperatorBank:
    n_conditions = len(drawn[0])
    masks = np.zeros((len(sigmas), n_conditions, n_masks, n_visible), dtype=np.float64)
    weights = np.zeros((len(sigmas), n_conditions, n_masks), dtype=np.float64)
    for bandwidth, row in enumerate(drawn):
        for condition, (masks_c, weights_c) in enumerate(row):
            masks[bandwidth, condition], weights[bandwidth, condition] = (
                _pad_bank_entry(
                    np.asarray(masks_c, dtype=np.float64),
                    np.asarray(weights_c, dtype=np.float64),
                    n_masks,
                    n_visible,
                )
            )
    return OperatorBank(
        sigmas=tuple(float(sigma) for sigma in sigmas),
        masks=jnp.asarray(masks),
        weights=jnp.asarray(weights),
        n_ops=n_ops,
        n_visible=n_visible,
        n_masks=n_masks,
        widths=tuple(int(width) for width in widths),
    )


def bank_from_ragged(
    sigmas: tuple[float, ...],
    masks: tuple[tuple[np.ndarray, ...], ...],
    weights: tuple[tuple[np.ndarray, ...], ...],
    n_ops: int,
    n_visible: int,
) -> OperatorBank:
    """Build a bank from per-(bandwidth, condition) mask and weight arrays.

    The result is padded to one width for every entry and evaluated at that
    width, which is what reproducing an externally built bank requires; use
    :func:`operator_bank` for the per-bandwidth widths the fixed-bank path uses.
    """
    if len(masks) != len(sigmas) or len(weights) != len(sigmas):
        raise ValueError("masks and weights need one entry per bandwidth")
    drawn = [
        [(masks[b][c], weights[b][c]) for c in range(len(masks[b]))]
        for b in range(len(sigmas))
    ]
    n_masks = max(entry[0].shape[0] for row in drawn for entry in row)
    return _assemble_bank(
        sigmas,
        drawn,
        n_ops,
        n_visible,
        n_masks,
        (n_masks,) * len(sigmas),
    )


def _draw_batched(
    root: np.random.Generator,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_conditions: int,
    n_visible: int,
    caps: tuple[int | None, ...],
    max_attempts: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Draw every (bandwidth, condition) entry with one batched compression."""
    drawn: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for bandwidth, sigma in enumerate(sigmas):
        cap = caps[bandwidth]
        p = _mask_probability(sigma)
        nonidentity_mass = 1.0 - (1.0 - p) ** n_visible
        codes = _sample_nonidentity_codes(
            root,
            n_ops,
            n_visible,
            float(sigma),
            n_conditions,
        )
        groups, unique_codes, counts = _compress_codes(codes, n_visible)
        # Keys are sorted, so the groups are nondecreasing: slice each
        # condition out with one search over the boundaries instead of a
        # boolean pass per condition.
        bounds = np.searchsorted(groups, np.arange(n_conditions + 1))
        row = []
        for condition in range(n_conditions):
            lo, hi = int(bounds[condition]), int(bounds[condition + 1])
            condition_codes = unique_codes[lo:hi]
            condition_counts = counts[lo:hi]
            if cap is not None and condition_codes.shape[0] > cap:
                # Rare with calibrated caps: redraw this entry only.
                entry = None
                for _ in range(max_attempts):
                    masks, weights = sample_nonidentity_masks(
                        root,
                        n_ops,
                        n_visible,
                        float(sigma),
                    )
                    if masks.shape[0] <= cap:
                        entry = (masks, weights)
                        break
                if entry is None:
                    raise RuntimeError(
                        f"no bank entry with at most {cap} distinct masks after "
                        f"{max_attempts} draws at sigma={sigma}; raise max_masks",
                    )
                row.append(entry)
                continue
            row.append(
                (
                    _decode_codes(condition_codes, n_visible),
                    nonidentity_mass * condition_counts.astype(np.float64) / n_ops,
                ),
            )
        drawn.append(row)
    return drawn


def _draw_entrywise(
    root: np.random.Generator,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_conditions: int,
    n_visible: int,
    caps: tuple[int | None, ...],
    max_attempts: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Per-entry fallback for registers too wide to key on one integer."""
    seeds = root.integers(
        0,
        2**63 - 1,
        size=(len(sigmas), n_conditions, max(1, max_attempts)),
    )
    drawn: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for bandwidth, sigma in enumerate(sigmas):
        cap = caps[bandwidth]
        row = []
        for condition in range(n_conditions):
            entry = None
            for attempt in range(max_attempts):
                rng = np.random.default_rng(int(seeds[bandwidth, condition, attempt]))
                masks, weights = sample_nonidentity_masks(
                    rng,
                    n_ops,
                    n_visible,
                    float(sigma),
                )
                if cap is None or masks.shape[0] <= cap:
                    entry = (masks, weights)
                    break
            if entry is None:
                raise RuntimeError(
                    f"no bank entry with at most {cap} distinct masks after "
                    f"{max_attempts} draws at sigma={sigma}; raise max_masks",
                )
            row.append(entry)
        drawn.append(row)
    return drawn


def operator_bank(
    sigmas: tuple[float, ...],
    n_ops: int,
    n_conditions: int,
    n_visible: int,
    seed: int,
    *,
    max_masks: int | Sequence[int] | None = None,
    max_attempts: int = 64,
) -> OperatorBank:
    """Draw a compressed operator bank for every (bandwidth, condition).

    ``n_ops`` is the number of nonidentity draws per entry.  Entries are padded
    to the largest distinct count present, or to ``max_masks`` when given — pass
    a fixed width when the bank will be handed to a compiled step as an
    argument, so that refreshed banks keep one shape.  With ``max_masks`` set, an
    entry that draws more distinct masks than that is redrawn, so the result
    never depends on truncation.

    Each bandwidth is evaluated at its own width, the largest distinct count its
    own entries reached, so a bandwidth whose mask law cannot reach many masks
    does not pay for the widest one.  ``max_masks`` may therefore be one width
    for every bandwidth (an int) or one per bandwidth (a sequence) — the latter
    keeps a refreshed bank's *per-bandwidth* widths fixed while each bandwidth
    stays as narrow as its own law allows.
    """
    if not sigmas:
        raise ValueError("sigmas must not be empty")
    if any(not math.isfinite(float(sigma)) or float(sigma) <= 0.0 for sigma in sigmas):
        raise ValueError("sigmas must be positive and finite")
    if n_ops < 1:
        raise ValueError("n_ops must be positive")
    if n_conditions < 1:
        raise ValueError("n_conditions must be positive")
    if n_visible < 1:
        raise ValueError("n_visible must be positive")
    per_bandwidth = isinstance(max_masks, Sequence) and not isinstance(max_masks, str)
    caps: tuple[int | None, ...]
    if max_masks is None:
        caps = (None,) * len(sigmas)
    elif per_bandwidth:
        caps = tuple(int(cap) for cap in max_masks)  # type: ignore[arg-type]
        if len(caps) != len(sigmas) or any(cap < 1 for cap in caps):
            raise ValueError("max_masks must give one positive width per bandwidth")
    else:
        caps = (int(max_masks),) * len(sigmas)

    root = np.random.default_rng(seed)
    if n_visible + max(1, (n_conditions - 1).bit_length()) > 62:
        # Codes are int64; the batched path packs (group, code) into one int and
        # falls back when that key would overflow.
        drawn = _draw_entrywise(
            root,
            sigmas,
            n_ops,
            n_conditions,
            n_visible,
            caps,
            max_attempts,
        )
    else:
        drawn = _draw_batched(
            root,
            sigmas,
            n_ops,
            n_conditions,
            n_visible,
            caps,
            max_attempts,
        )

    natural = tuple(max(entry[0].shape[0] for entry in row) for row in drawn)
    if max_masks is None:
        # Each bandwidth is evaluated at the width its own law reached.
        widths = natural
    elif per_bandwidth:
        # A fixed width per bandwidth: refreshed banks keep every shape.
        widths = caps  # type: ignore[assignment]
    else:
        # One width for the whole bank (the pre-existing contract).
        widths = (int(max_masks),) * len(sigmas)  # type: ignore[arg-type]
    n_masks = max(widths)
    return _assemble_bank(sigmas, drawn, n_ops, n_visible, n_masks, widths)
