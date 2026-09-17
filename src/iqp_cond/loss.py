"""Condition-balanced multiscale Hamming-RBF MMD.

The kernel ``k(x, y) = exp(-d_H(x, y) / (2 sigma^2))`` has a product character
expansion, so the squared MMD is a weighted sum over Fourier masks whose
probability under each bandwidth is Bernoulli with
``p = (1 - exp(-1/(2 sigma^2))) / 2``.  The loss averages over conditions and
bandwidths, uses the unbiased ``(B-1)`` normalization for both target and model
moments, and consumes only quantities the reference makes classically
computable: sampled masks, sampled bitstrings, and amplitude ratios.

Two estimators are provided, and they differ in cost, not in target:

:func:`sampled_mmd` (default for :func:`iqp_cond.train`)
    Draws nonidentity masks once into a fixed :class:`OperatorBank`, collapses
    duplicates to their exact multiplicities, rescales by the nonidentity mass,
    and evaluates one condition at a time.  This is the paper's stochastic
    operator estimator.
:func:`conditional_mmd`
    The reference estimator: draws every mask i.i.d. from the full Bernoulli law
    including the identity, keeps duplicates, and evaluates all conditions at
    once.  It is unbiased but spends a large share of its budget on a term that
    is identically zero, and its peak allocation grows with the condition count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .ancilla import pad_visible_masks
from .estimator import _estimate_pauli_z
from .references import ReferenceStates

__all__ = [
    "COMPRESSED",
    "PLAIN",
    "ConditionalMMDResult",
    "OperatorBank",
    "conditional_mmd",
    "hamming_rbf",
    "mmd_operator_probability",
    "operator_bank",
    "sample_mmd_operators",
    "sample_nonidentity_masks",
    "sampled_mmd",
    "target_moments",
]

COMPRESSED = "compressed"
PLAIN = "plain"


@dataclass(frozen=True)
class ConditionalMMDResult:
    """Conditional MMD estimate with condition- and bandwidth-level diagnostics."""

    loss: jnp.ndarray
    per_condition: jnp.ndarray
    per_bandwidth_condition: jnp.ndarray


def hamming_rbf(x: jnp.ndarray, y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Evaluate ``exp(-d_H(x,y)/(2 sigma^2))``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return jnp.exp(-jnp.sum(jnp.not_equal(x, y), axis=-1) / (2.0 * sigma**2))


def mmd_operator_probability(sigma: float | jnp.ndarray) -> jnp.ndarray:
    """Return the Bernoulli probability for Fourier observable masks."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    return (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0


def sample_mmd_operators(
    key: jax.Array,
    n_ops: int,
    n_qubits: int,
    sigma: float,
) -> jnp.ndarray:
    """Sample Pauli-Z masks for the Hamming-RBF Fourier representation."""
    if n_ops < 1 or n_qubits < 1:
        raise ValueError("n_ops and n_qubits must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return jax.random.bernoulli(
        key,
        p=mmd_operator_probability(sigma),
        shape=(n_ops, n_qubits),
    ).astype(jnp.float64)


def target_moments(
    targets: jnp.ndarray,
    masks: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the target mean character and its unbiased squared mean.

    Shapes are ``(conditions, samples, qubits)`` for ``targets`` and
    ``(masks, qubits)`` for ``masks``.
    """
    target_characters = 1.0 - 2.0 * jnp.mod(
        jnp.einsum("btn,rn->btr", targets, masks),
        2.0,
    )
    n_targets = targets.shape[1]
    means = jnp.mean(target_characters, axis=1)
    if n_targets < 2:
        raise ValueError("each condition needs at least two target samples")
    sums = jnp.sum(target_characters, axis=1)
    unbiased_squares = (sums**2 - n_targets) / (n_targets * (n_targets - 1))
    return means, unbiased_squares


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


@dataclass(frozen=True)
class OperatorBank:
    """Compressed Fourier-mask bank, one entry per (bandwidth, condition).

    ``masks`` has shape ``(n_bandwidths, n_conditions, n_masks, n_visible)`` and
    ``weights`` the matching ``(n_bandwidths, n_conditions, n_masks)``; each
    weight is ``q_nonidentity * count / n_ops``, so one entry's weights sum to the
    nonidentity mass.  Rows past an entry's distinct-mask count are padding — a
    zero mask with a zero weight, which contributes nothing to the weighted sum.

    Padding exists so bank shape is static.  A bank that is *refreshed* during
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
    :func:`operator_bank` for the per-bandwidth widths the trainer uses.
    """
    if len(masks) != len(sigmas) or len(weights) != len(sigmas):
        raise ValueError("masks and weights need one entry per bandwidth")
    drawn = [
        [(masks[b][c], weights[b][c]) for c in range(len(masks[b]))]
        for b in range(len(sigmas))
    ]
    n_masks = max(entry[0].shape[0] for row in drawn for entry in row)
    return _assemble_bank(
        sigmas, drawn, n_ops, n_visible, n_masks, (n_masks,) * len(sigmas),
    )


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
    is what keeps a refreshed bank's *per-bandwidth* widths fixed while each
    bandwidth stays as narrow as its own law allows.
    """
    if not sigmas:
        raise ValueError("sigmas must not be empty")
    if n_conditions < 1:
        raise ValueError("n_conditions must be positive")
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


def sampled_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    references: ReferenceStates,
    targets: jnp.ndarray,
    bank: OperatorBank,
    n_samples: int,
    key: jax.Array,
) -> ConditionalMMDResult:
    """Estimate the multiscale squared MMD with a fixed compressed operator bank.

    One condition is evaluated at a time, so the ``(samples, masks)`` buffer
    covers a single condition instead of the whole batch; combined with the
    compressed bank this is what keeps the default configuration inside a
    sensible memory budget.
    """
    targets = jnp.asarray(targets, dtype=jnp.float64)
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape (n_conditions, n_targets, n_visible)",
        )
    n_conditions, _, n_data_qubits = targets.shape
    if references.n_conditions != n_conditions:
        raise ValueError(
            f"reference covers {references.n_conditions} conditions but targets "
            f"have {n_conditions}",
        )
    if bank.n_conditions != n_conditions:
        raise ValueError(
            f"operator bank covers {bank.n_conditions} conditions but targets "
            f"have {n_conditions}",
        )
    if bank.n_visible != n_data_qubits:
        raise ValueError(
            f"operator bank is built for {bank.n_visible} visible qubits but "
            f"targets have {n_data_qubits}",
        )
    if references.n_qubits < n_data_qubits:
        raise ValueError(
            "reference qubit count cannot be smaller than the target width",
        )
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")

    n_ancillas = references.n_qubits - n_data_qubits
    keys = jax.random.split(key, bank.n_bandwidths * n_conditions)

    bandwidth_values = []
    cursor = 0
    for bandwidth in range(bank.n_bandwidths):
        condition_values = []
        for condition in range(n_conditions):
            visible_masks, weights = bank.entry(bandwidth, condition)
            model_masks = pad_visible_masks(visible_masks, n_ancillas)
            model = _estimate_pauli_z(
                params,
                generators,
                model_masks,
                references.select(condition),
                keys[cursor],
                n_samples,
            )
            cursor += 1
            target_means, target_squares = target_moments(
                targets[condition : condition + 1],
                visible_masks,
            )
            sums = jnp.sum(model.sample_values, axis=1)
            square_sums = jnp.sum(model.sample_values**2, axis=1)
            model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
            per_mask = (
                model_squares - 2.0 * model.values * target_means + target_squares
            )
            condition_values.append(jnp.sum(per_mask[0] * weights))
        bandwidth_values.append(jnp.stack(condition_values))

    per_bandwidth_condition = jnp.stack(bandwidth_values)
    return ConditionalMMDResult(
        loss=jnp.mean(per_bandwidth_condition),
        per_condition=jnp.mean(per_bandwidth_condition, axis=0),
        per_bandwidth_condition=per_bandwidth_condition,
    )


def conditional_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    references: ReferenceStates,
    targets: jnp.ndarray,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_iqp_samples: int,
    key: jax.Array,
) -> ConditionalMMDResult:
    """Estimate the equally weighted, multiscale, unbiased squared MMD.

    ``targets`` holds observed bitstrings of shape
    ``(n_conditions, n_targets, n_visible)``; any extra reference qubits are
    ancillas and their masks stay identity, so the loss compares the visible
    marginal against the target law.
    """
    targets = jnp.asarray(targets, dtype=jnp.float64)
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape (n_conditions, n_targets, n_visible)",
        )
    n_conditions, _, n_data_qubits = targets.shape
    if references.n_conditions != n_conditions:
        raise ValueError(
            f"reference covers {references.n_conditions} conditions but targets "
            f"have {n_conditions}",
        )
    if references.n_qubits < n_data_qubits:
        raise ValueError(
            "reference qubit count cannot be smaller than the target width",
        )
    if not sigmas:
        raise ValueError("sigmas must not be empty")

    n_ancillas = references.n_qubits - n_data_qubits

    bandwidth_keys = jax.random.split(key, len(sigmas))
    bandwidth_losses = []
    for sigma, sigma_key in zip(sigmas, bandwidth_keys, strict=True):
        op_key, model_key = jax.random.split(sigma_key)
        visible_masks = sample_mmd_operators(
            op_key,
            n_ops,
            n_data_qubits,
            float(sigma),
        )
        model_masks = pad_visible_masks(visible_masks, n_ancillas)
        model = _estimate_pauli_z(
            params,
            generators,
            model_masks,
            references,
            model_key,
            n_iqp_samples,
        )
        target_means, target_squares = target_moments(targets, visible_masks)
        model_sums = jnp.sum(model.sample_values, axis=1)
        model_square_sums = jnp.sum(model.sample_values**2, axis=1)
        model_squares = (model_sums**2 - model_square_sums) / (
            n_iqp_samples * (n_iqp_samples - 1)
        )
        per_operator = (
            model_squares - 2.0 * model.values * target_means + target_squares
        )
        bandwidth_losses.append(jnp.mean(per_operator, axis=-1))

    per_bandwidth_condition = jnp.stack(bandwidth_losses, axis=0)
    per_condition = jnp.mean(per_bandwidth_condition, axis=0)
    return ConditionalMMDResult(
        loss=jnp.mean(per_condition),
        per_condition=per_condition,
        per_bandwidth_condition=per_bandwidth_condition,
    )
