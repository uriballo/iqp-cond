"""Mask law, samplers, and compressed operator banks.

Both samplers must draw the conditional Bernoulli law ``P(a | a != 0)``, with
the nonidentity mass divided over the draws as weights. The bank tests check
inert padding, static shapes, per-bandwidth widths, and pytree registration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.operators import (
    OperatorBank,
    bank_from_ragged,
    expected_bodyness_sigmas,
    mmd_operator_probability,
    operator_bank,
    sample_nonidentity_masks,
    sample_nonidentity_masks_jax,
)

SIGMAS = (1.0, 2.0, 4.0)


@pytest.mark.slow
def test_sample_nonidentity_masks_numpy_draws_the_conditional_law() -> None:
    """The NumPy sampler follows the same law as the JAX sampler."""
    n_visible, n_ops, sigma = 5, 100_000, 2.0
    masks, weights = sample_nonidentity_masks(
        np.random.default_rng(0),
        n_ops,
        n_visible,
        sigma,
    )
    assert np.isin(masks, (0, 1)).all()
    assert np.all(masks.sum(axis=1) > 0)
    codes = (masks * 2 ** np.arange(n_visible)).sum(axis=1).astype(int)
    assert len(set(codes.tolist())) == len(codes)  # duplicates are compressed

    p = float(mmd_operator_probability(sigma))
    nonidentity_mass = 1.0 - (1.0 - p) ** n_visible
    assert float(np.asarray(weights).sum()) == pytest.approx(
        nonidentity_mass,
        rel=1e-12,
    )

    analytic = np.array(
        [
            (
                p ** bin(code).count("1")
                * (1.0 - p) ** (n_visible - bin(code).count("1"))
                / nonidentity_mass
                if code
                else 0.0
            )
            for code in range(2**n_visible)
        ],
    )
    observed = np.zeros(2**n_visible)
    observed[codes] = np.asarray(weights) * n_ops / nonidentity_mass
    expected_counts = n_ops * analytic
    sigma_counts = np.sqrt(n_ops * analytic * (1.0 - analytic))
    reachable = expected_counts > 25.0
    assert reachable.any()
    assert np.all(
        np.abs(observed[reachable] - expected_counts[reachable])
        < 5.0 * sigma_counts[reachable] + 1e-9,
    )


@pytest.mark.slow
def test_sample_nonidentity_masks_jax_draws_the_conditional_law() -> None:
    """The JIT sampler must draw exactly ``P(a | a != 0)`` with mass weights.

    Identity never appears, each weight is ``Q+ / n_ops``, and the empirical
    mask frequencies match the analytic conditional Bernoulli law.
    """
    n_visible, n_ops, sigma = 5, 100_000, 2.0
    masks, weights = sample_nonidentity_masks_jax(
        jax.random.PRNGKey(0),
        n_ops,
        n_visible,
        sigma,
    )
    masks = np.asarray(masks)
    assert masks.shape == (n_ops, n_visible)
    assert np.isin(masks, (0.0, 1.0)).all()
    assert np.all(masks.sum(axis=1) > 0.0)

    p = float(mmd_operator_probability(sigma))
    nonidentity_mass = 1.0 - (1.0 - p) ** n_visible
    assert np.allclose(np.asarray(weights), nonidentity_mass / n_ops)

    codes = (masks * 2 ** np.arange(n_visible)).sum(axis=1).astype(int)
    empirical = np.bincount(codes, minlength=2**n_visible) / n_ops
    analytic = np.array(
        [
            (
                p ** bin(code).count("1")
                * (1.0 - p) ** (n_visible - bin(code).count("1"))
                / nonidentity_mass
                if code
                else 0.0
            )
            for code in range(2**n_visible)
        ],
    )
    assert float(np.max(np.abs(empirical - analytic))) < 3e-3


def test_sample_nonidentity_masks_jax_rejects_bad_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        sample_nonidentity_masks_jax(jax.random.PRNGKey(0), 8, 3, 0.0)


def test_samplers_validate_their_arguments() -> None:
    with pytest.raises(ValueError, match="n_ops and n_visible must be positive"):
        sample_nonidentity_masks_jax(jax.random.PRNGKey(0), 0, 3, 1.0)
    for arguments in ((0, 3, 1.0), (4, 3, 0.0)):
        with pytest.raises(
            ValueError,
            match="n_ops, n_qubits, and sigma must be positive",
        ):
            sample_nonidentity_masks(np.random.default_rng(0), *arguments)


def test_operator_bank_validates_conditions_and_caps() -> None:
    with pytest.raises(ValueError, match="n_conditions must be positive"):
        operator_bank(SIGMAS, 8, 0, 4, seed=0)
    with pytest.raises(ValueError, match="max_masks must give one positive width"):
        operator_bank(SIGMAS, 8, 1, 4, seed=0, max_masks=(3, 0, 3))
    with pytest.raises(ValueError, match="max_masks must give one positive width"):
        operator_bank(SIGMAS, 8, 1, 4, seed=0, max_masks=(3, 3))


def test_bank_dataclass_validates_static_shapes() -> None:
    masks = jnp.zeros((1, 1, 2, 4))
    weights = jnp.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="masks must be"):
        OperatorBank(
            sigmas=(1.0,),
            masks=jnp.zeros(3),
            weights=weights,
            n_ops=2,
            n_visible=4,
            n_masks=2,
        )
    with pytest.raises(ValueError, match="agree on their leading axes"):
        OperatorBank(
            sigmas=(1.0,),
            masks=masks,
            weights=jnp.zeros((1, 1, 3)),
            n_ops=2,
            n_visible=4,
            n_masks=2,
        )
    with pytest.raises(ValueError, match="one entry per bandwidth"):
        OperatorBank(
            sigmas=(1.0, 2.0),
            masks=masks,
            weights=weights,
            n_ops=2,
            n_visible=4,
            n_masks=2,
        )
    with pytest.raises(ValueError, match="visible-width"):
        OperatorBank(
            sigmas=(1.0,),
            masks=masks,
            weights=weights,
            n_ops=2,
            n_visible=3,
            n_masks=2,
        )
    with pytest.raises(ValueError, match="n_masks must match"):
        OperatorBank(
            sigmas=(1.0,),
            masks=masks,
            weights=weights,
            n_ops=2,
            n_visible=4,
            n_masks=3,
        )
    with pytest.raises(ValueError, match="at least one condition"):
        OperatorBank(
            sigmas=(1.0,),
            masks=jnp.zeros((1, 0, 2, 4)),
            weights=jnp.zeros((1, 0, 2)),
            n_ops=2,
            n_visible=4,
            n_masks=2,
        )
    with pytest.raises(ValueError, match="one entry per bandwidth"):
        OperatorBank(
            sigmas=(1.0,),
            masks=masks,
            weights=weights,
            n_ops=2,
            n_visible=4,
            n_masks=2,
            widths=(1, 2),
        )
    with pytest.raises(ValueError, match="between 1 and n_masks"):
        OperatorBank(
            sigmas=(1.0,),
            masks=masks,
            weights=weights,
            n_ops=2,
            n_visible=4,
            n_masks=2,
            widths=(3,),
        )


def test_bank_from_ragged_validates_entry_counts() -> None:
    masks = ((np.ones((2, 4)),),)
    weights = ((np.ones(2),),)
    with pytest.raises(ValueError, match="one entry per bandwidth"):
        bank_from_ragged(
            sigmas=(1.0, 2.0),
            masks=masks,
            weights=weights,
            n_ops=2,
            n_visible=4,
        )


def test_operator_bank_handles_registers_wider_than_the_code_packing() -> None:
    """A 63-qubit register falls back to the per-entry sampler."""
    bank = operator_bank((2.0,), 8, 1, 63, seed=0)
    assert bank.masks.shape == (1, 1, bank.n_masks, 63)
    masks, weights = bank.entry(0, 0)
    carried = np.asarray(masks)[np.asarray(weights) > 0.0]
    assert np.all(carried.sum(axis=1) > 0)


def test_operator_bank_reports_an_unreachable_width_cap() -> None:
    with pytest.raises(RuntimeError, match="no bank entry with at most 1"):
        operator_bank((4.0,), 64, 1, 8, seed=0, max_masks=1)
    with pytest.raises(RuntimeError, match="no bank entry with at most 1"):
        operator_bank((2.0,), 8, 1, 63, seed=0, max_masks=1)


def test_bank_records_one_entry_per_bandwidth_and_condition() -> None:
    bank = operator_bank(SIGMAS, 64, 2, 4, seed=0)
    assert bank.n_bandwidths == len(SIGMAS)
    assert bank.n_conditions == 2
    assert bank.masks.shape == (len(SIGMAS), 2, bank.n_masks, 4)
    assert bank.weights.shape == (len(SIGMAS), 2, bank.n_masks)
    for bandwidth in range(bank.n_bandwidths):
        for condition in range(bank.n_conditions):
            masks, weights = bank.entry(bandwidth, condition)
            assert masks.shape == (bank.width(bandwidth), 4)
            # Every mask that carries weight is a nonidentity mask.
            carried = np.asarray(masks)[np.asarray(weights) > 0.0]
            assert np.all(carried.sum(axis=1) > 0)


def test_bank_weights_carry_the_nonidentity_mass() -> None:
    """Weight sums must equal ``1 - q(0)``, the rescaling for skipping identity."""
    bank = operator_bank(SIGMAS, 64, 2, 4, seed=0)
    for bandwidth, sigma in enumerate(SIGMAS):
        expected = 1.0 - float((1.0 - mmd_operator_probability(sigma)) ** 4)
        for condition in range(bank.n_conditions):
            _masks, weights = bank.entry(bandwidth, condition)
            assert float(np.asarray(weights).sum()) == pytest.approx(
                expected,
                rel=1e-12,
            )


def test_operator_bank_is_deterministic_and_skips_the_identity() -> None:
    first = operator_bank(SIGMAS, 32, 2, 6, seed=7)
    second = operator_bank(SIGMAS, 32, 2, 6, seed=7)
    for bandwidth in range(first.n_bandwidths):
        for condition in range(first.n_conditions):
            masks_a, weights_a = first.entry(bandwidth, condition)
            masks_b, _ = second.entry(bandwidth, condition)
            assert np.array_equal(np.asarray(masks_a), np.asarray(masks_b))
            carried = np.asarray(masks_a)[np.asarray(weights_a) > 0.0]
            assert np.all(carried.sum(axis=1) > 0)
            # Padding is inert, so a zero mask implies zero weight.
            padded = np.asarray(weights_a)[np.asarray(masks_a).sum(axis=1) == 0.0]
            assert np.all(padded == 0.0)


def test_max_masks_fixes_the_padded_width() -> None:
    """A fixed width lets refreshed banks reuse one compilation."""
    natural = operator_bank(SIGMAS, 64, 2, 6, seed=5)
    fixed = operator_bank(SIGMAS, 64, 2, 6, seed=5, max_masks=natural.n_masks + 4)
    assert fixed.n_masks == natural.n_masks + 4
    assert fixed.masks.shape[2] == fixed.n_masks
    for bandwidth in range(fixed.n_bandwidths):
        for condition in range(fixed.n_conditions):
            _, weights = fixed.entry(bandwidth, condition)
            assert float(np.asarray(weights).sum()) > 0.0


def test_operator_bank_is_a_pytree() -> None:
    """A bank can be passed to a compiled step as an argument."""
    bank = operator_bank(SIGMAS, 16, 1, 4, seed=6)
    leaves, treedef = jax.tree_util.tree_flatten(bank)
    assert len(leaves) == 2
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(np.asarray(restored.masks), np.asarray(bank.masks))
    assert np.array_equal(np.asarray(restored.weights), np.asarray(bank.weights))
    assert restored.sigmas == bank.sigmas


def test_compressed_bank_shrinks_the_mask_set() -> None:
    """At a small qubit count the bank holds fewer masks than the draw budget."""
    bank = operator_bank((4.0,), 2000, 1, 8, seed=3)
    assert bank.n_masks < 2000  # duplicates collapse
    assert bank.n_ops == 2000


def test_operator_bank_evaluates_each_bandwidth_at_its_own_width() -> None:
    """A bandwidth whose law cannot reach many masks must not widen the bank.

    The three default bandwidths differ by ~5x in how many distinct masks their
    Bernoulli law reaches, so one width for the whole bank carries most of the
    cost.
    """
    bank = operator_bank(SIGMAS, 128, 2, 6, seed=11)
    assert bank.widths[0] > bank.widths[-1]
    assert bank.n_masks == max(bank.widths)
    for bandwidth in range(bank.n_bandwidths):
        for condition in range(bank.n_conditions):
            masks, weights = bank.entry(bandwidth, condition)
            assert masks.shape == (bank.widths[bandwidth], 6)
            assert weights.shape == (bank.widths[bandwidth],)
    evaluated = sum(bank.width(b) for b in range(bank.n_bandwidths))
    assert evaluated < bank.n_masks * bank.n_bandwidths


def test_fixed_per_bandwidth_widths_keep_refreshed_banks_one_shape() -> None:
    """Per-step refresh needs every redraw to reuse one compilation per bandwidth."""
    pilot = operator_bank(SIGMAS, 64, 2, 6, seed=5)
    widths = tuple(int(width * 1.1) + 1 for width in pilot.widths)
    first = operator_bank(SIGMAS, 64, 2, 6, seed=5, max_masks=widths)
    second = operator_bank(SIGMAS, 64, 2, 6, seed=6, max_masks=widths)
    assert first.widths == widths
    assert second.widths == widths
    assert first.masks.shape == second.masks.shape
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)


@pytest.mark.slow
def test_bank_reproduces_the_conditional_mask_law() -> None:
    """Bank weights must recover ``P(a | a != 0)`` after removing the mass factor.

    The test drives the batched sampler through the public path. On a tiny
    register every nonidentity mask is reachable, so the weight of a mask
    divided by the nonidentity mass is its analytic conditional probability.
    """
    import math

    n_visible, n_ops, sigma = 3, 200_000, 2.0
    bank = operator_bank((sigma,), n_ops, 1, n_visible, seed=0)
    masks, weights = bank.entry(0, 0)
    masks = np.asarray(masks)
    weights = np.asarray(weights)
    assert masks.shape[0] == 2**n_visible - 1  # every nonidentity mask, once

    p = float(mmd_operator_probability(sigma))
    nonidentity_mass = 1.0 - (1.0 - p) ** n_visible
    codes = (masks * 2 ** np.arange(n_visible)).sum(axis=1).astype(int)
    empirical = weights / nonidentity_mass
    analytic = np.array(
        [
            p ** bin(code).count("1")
            * (1.0 - p) ** (n_visible - bin(code).count("1"))
            / nonidentity_mass
            for code in codes
        ],
    )
    assert float(np.max(np.abs(empirical - analytic))) < 5e-3
    assert math.isclose(float(weights.sum()), nonidentity_mass, rel_tol=1e-12)


def test_operator_bank_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_ops must be positive"):
        operator_bank(SIGMAS, 0, 2, 4, seed=0)
    with pytest.raises(ValueError, match="n_visible must be positive"):
        operator_bank(SIGMAS, 8, 2, 0, seed=0)
    with pytest.raises(ValueError, match="positive and finite"):
        operator_bank((0.0,), 8, 2, 4, seed=0)
    with pytest.raises(ValueError, match="sigmas must not be empty"):
        operator_bank((), 8, 2, 4, seed=0)


def test_expected_bodyness_sigmas_inverts_the_mask_law() -> None:
    """A bodyness target K maps to the sigma whose mean mask weight is K."""
    n_visible = 12
    sigmas = expected_bodyness_sigmas(n_visible, targets=(2.0, 3.0, 5.88))
    for sigma, target in zip(sigmas, (2.0, 3.0, 5.88), strict=True):
        rebuilt = n_visible * float(mmd_operator_probability(sigma))
        assert rebuilt == pytest.approx(target, rel=1e-12, abs=1e-12)


def test_expected_bodyness_sigmas_defaults_and_cap() -> None:
    """The default targets are (2, n/4, 0.49 n), capped at 0.49 n."""
    n_visible = 12
    default = expected_bodyness_sigmas(n_visible)
    explicit = expected_bodyness_sigmas(n_visible, targets=(2.0, 3.0, 5.88))
    assert default == explicit

    # For n = 3 the low target 2 exceeds n/2 and is capped at 0.49 n.
    capped = expected_bodyness_sigmas(3)
    assert capped == expected_bodyness_sigmas(3, targets=(1.47, 0.75, 1.47))


def test_expected_bodyness_sigmas_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_visible must be positive"):
        expected_bodyness_sigmas(0)
    with pytest.raises(ValueError, match="bodyness targets must be positive"):
        expected_bodyness_sigmas(8, targets=())
    with pytest.raises(ValueError, match="bodyness targets must be positive"):
        expected_bodyness_sigmas(8, targets=(0.0,))
