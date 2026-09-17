"""The fixed-operator estimator: equivalence with the campaign implementation.

``fixtures/golden.npz`` records the compressed-loss value computed by the
research campaign's ``exact_tvd_runner._weighted_loss_functions`` together with
the bank it built (masks and weights per bandwidth and condition).  Feeding that
same bank to :func:`iqp_cond.sampled_mmd` must reproduce the value bit-for-bit:
the library's job is to compute the paper's estimator, not an approximation of
it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as iqc

FIXTURES = Path(__file__).parent / "fixtures" / "golden.npz"
SIGMAS = (1.0, 2.0, 4.0)
N_CONDITIONS = 2
N_VISIBLE = 4
N_OPS = 64
N_SAMPLES = 64

CASES = {"bank_plain": 0, "bank_ancilla": 2}


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(FIXTURES) as data:
        return {name: data[name] for name in data.files}


def rebuild_bank(golden: dict[str, np.ndarray], label: str) -> iqc.OperatorBank:
    """Rebuild the campaign's bank from the recorded masks and weights."""
    masks = tuple(
        tuple(
            jnp.asarray(golden[f"{label}_masks_{bandwidth * N_CONDITIONS + condition}"])
            for condition in range(N_CONDITIONS)
        )
        for bandwidth in range(len(SIGMAS))
    )
    weights = tuple(
        tuple(
            jnp.asarray(
                golden[f"{label}_weights_{bandwidth * N_CONDITIONS + condition}"],
            )
            for condition in range(N_CONDITIONS)
        )
        for bandwidth in range(len(SIGMAS))
    )
    return iqc.bank_from_ragged(
        sigmas=SIGMAS,
        masks=masks,
        weights=weights,
        n_ops=N_OPS,
        n_visible=N_VISIBLE,
    )


@pytest.mark.parametrize("label", sorted(CASES))
def test_sampled_mmd_reproduces_the_campaign_value(
    golden: dict[str, np.ndarray],
    label: str,
) -> None:
    n_ancillas = CASES[label]
    ansatz = iqc.complete_ansatz(N_VISIBLE + n_ancillas)
    family = iqc.PRODUCT if n_ancillas == 0 else iqc.SPIN_FLIP
    references = iqc.reference(family, golden[f"{label}_angles"])
    bank = rebuild_bank(golden, label)
    key = jax.random.wrap_key_data(np.asarray(golden[f"{label}_key"]))

    result = iqc.sampled_mmd(
        golden[f"{label}_params"],
        ansatz.generators,
        references,
        golden[f"{label}_targets"],
        bank,
        N_SAMPLES,
        key,
    )
    deviation = abs(float(result.loss) - float(golden[f"{label}_value"]))
    assert deviation == 0.0, f"{label}: loss differs from the campaign by {deviation!r}"
    assert result.per_bandwidth_condition.shape == (len(SIGMAS), N_CONDITIONS)


def test_bank_records_one_entry_per_bandwidth_and_condition(
    golden: dict[str, np.ndarray],
) -> None:
    bank = rebuild_bank(golden, "bank_plain")
    assert bank.n_bandwidths == len(SIGMAS)
    assert bank.n_conditions == N_CONDITIONS
    assert bank.masks.shape == (len(SIGMAS), N_CONDITIONS, bank.n_masks, N_VISIBLE)
    assert bank.weights.shape == (len(SIGMAS), N_CONDITIONS, bank.n_masks)
    for bandwidth in range(bank.n_bandwidths):
        for condition in range(bank.n_conditions):
            masks, weights = bank.entry(bandwidth, condition)
            assert masks.shape == (bank.n_masks, N_VISIBLE)
            # Every mask that carries weight is a nonidentity mask.
            carried = np.asarray(masks)[np.asarray(weights) > 0.0]
            assert np.all(carried.sum(axis=1) > 0)


def test_bank_weights_carry_the_nonidentity_mass(
    golden: dict[str, np.ndarray],
) -> None:
    """Weight sums must equal ``1 - q(0)``, the rescaling for skipping identity."""
    for bandwidth, sigma in enumerate(SIGMAS):
        expected = 1.0 - float((1.0 - iqc.mmd_operator_probability(sigma)) ** N_VISIBLE)
        for condition in range(N_CONDITIONS):
            index = bandwidth * N_CONDITIONS + condition
            total = float(np.asarray(golden[f"bank_plain_weights_{index}"]).sum())
            assert total == pytest.approx(expected, rel=1e-12)


def test_sampled_mmd_rejects_a_mismatched_bank() -> None:
    rng = np.random.default_rng(0)
    ansatz = iqc.complete_ansatz(4)
    references = iqc.reference(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(float)
    bank = iqc.operator_bank(SIGMAS, 8, 3, 4, seed=0)  # three conditions, not two
    with pytest.raises(ValueError, match="operator bank covers 3 conditions"):
        iqc.sampled_mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            references,
            targets,
            bank,
            16,
            jax.random.PRNGKey(0),
        )
    wrong_width = iqc.operator_bank(SIGMAS, 8, 2, 5, seed=0)
    with pytest.raises(ValueError, match="built for 5 visible qubits"):
        iqc.sampled_mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            references,
            targets,
            wrong_width,
            16,
            jax.random.PRNGKey(0),
        )


def test_operator_bank_is_deterministic_and_skips_the_identity() -> None:
    first = iqc.operator_bank(SIGMAS, 32, 2, 6, seed=7)
    second = iqc.operator_bank(SIGMAS, 32, 2, 6, seed=7)
    for bandwidth in range(first.n_bandwidths):
        for condition in range(first.n_conditions):
            masks_a, weights_a = first.entry(bandwidth, condition)
            masks_b, _ = second.entry(bandwidth, condition)
            assert np.array_equal(np.asarray(masks_a), np.asarray(masks_b))
            carried = np.asarray(masks_a)[np.asarray(weights_a) > 0.0]
            assert np.all(carried.sum(axis=1) > 0)
            # Padding is inert: zero mask implies zero weight.
            padded = np.asarray(weights_a)[np.asarray(masks_a).sum(axis=1) == 0.0]
            assert np.all(padded == 0.0)


def test_max_masks_fixes_the_padded_width() -> None:
    """A fixed width is what lets refreshed banks reuse one compilation."""
    natural = iqc.operator_bank(SIGMAS, 64, 2, 6, seed=5)
    fixed = iqc.operator_bank(SIGMAS, 64, 2, 6, seed=5, max_masks=natural.n_masks + 4)
    assert fixed.n_masks == natural.n_masks + 4
    assert fixed.masks.shape[2] == fixed.n_masks
    for bandwidth in range(fixed.n_bandwidths):
        for condition in range(fixed.n_conditions):
            _, weights = fixed.entry(bandwidth, condition)
            assert float(np.asarray(weights).sum()) > 0.0


def test_operator_bank_is_a_pytree() -> None:
    """The bank must be passable into a compiled step as an argument."""
    bank = iqc.operator_bank(SIGMAS, 16, 1, 4, seed=6)
    leaves, treedef = jax.tree_util.tree_flatten(bank)
    assert len(leaves) == 2
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(np.asarray(restored.masks), np.asarray(bank.masks))
    assert np.array_equal(np.asarray(restored.weights), np.asarray(bank.weights))
    assert restored.sigmas == bank.sigmas


def test_compressed_bank_shrinks_the_mask_set() -> None:
    """At a small qubit count the bank is far smaller than the draw budget."""
    bank = iqc.operator_bank((4.0,), 2000, 1, 8, seed=3)
    assert bank.n_masks < 2000  # duplicates collapse
    assert bank.n_ops == 2000


def test_sample_nonidentity_masks_matches_the_campaign_sampler(
    golden: dict[str, np.ndarray],
) -> None:
    """Same draws as the research sampler for the same generator state.

    The research sampler seeds a NumPy generator from the raw key bytes; the
    library takes the generator directly, so the bridge is made explicit here
    and the resulting masks and weights must match exactly.
    """
    raw_key = np.asarray(golden["sampler_key"], dtype=np.uint32)
    seed = int.from_bytes(raw_key.tobytes()[:8], "little")
    masks, weights = iqc.sample_nonidentity_masks(
        np.random.default_rng(seed), 64, 5, 2.0
    )
    assert np.array_equal(masks, golden["sampler_masks"])
    assert np.allclose(weights, golden["sampler_weights"])


def test_plain_estimator_is_still_available(golden: dict[str, np.ndarray]) -> None:
    """The reference estimator remains callable and unchanged."""
    ansatz = iqc.complete_ansatz(4)
    references = iqc.reference(iqc.PRODUCT, golden["plain_angles"])
    result = iqc.conditional_mmd(
        golden["plain_params"],
        ansatz.generators,
        references,
        golden["plain_targets"],
        SIGMAS,
        8,
        64,
        jax.random.PRNGKey(27),
    )
    assert float(result.loss) == float(golden["plain_loss"])


def test_operator_bank_evaluates_each_bandwidth_at_its_own_width() -> None:
    """A bandwidth whose law cannot reach many masks must not pay the widest one.

    The three default bandwidths differ by ~5x in how many distinct masks their
    Bernoulli law reaches, so one width for the whole bank is most of the cost.
    """
    bank = iqc.operator_bank(SIGMAS, 128, 2, 6, seed=11)
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
    """Epoch refresh needs every redraw to reuse one compilation per bandwidth."""
    pilot = iqc.operator_bank(SIGMAS, 64, 2, 6, seed=5)
    widths = tuple(int(width * 1.1) + 1 for width in pilot.widths)
    first = iqc.operator_bank(SIGMAS, 64, 2, 6, seed=5, max_masks=widths)
    second = iqc.operator_bank(SIGMAS, 64, 2, 6, seed=6, max_masks=widths)
    assert first.widths == widths
    assert second.widths == widths
    assert first.masks.shape == second.masks.shape
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)


def test_per_bandwidth_widths_do_not_change_the_loss() -> None:
    """Cutting a bandwidth's padding changes cost, not the estimate."""
    rng = np.random.default_rng(3)
    ansatz = iqc.complete_ansatz(5)
    references = iqc.reference(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 5)))
    targets = rng.integers(0, 2, size=(2, 32, 5)).astype(float)
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(4)

    entries = tuple(
        tuple(
            iqc.sample_nonidentity_masks(
                np.random.default_rng(100 * bandwidth + condition), 16, 5, sigma,
            )
            for condition in range(2)
        )
        for bandwidth, sigma in enumerate(SIGMAS)
    )
    bank = iqc.bank_from_ragged(
        sigmas=SIGMAS,
        masks=tuple(tuple(masks for masks, _weights in row) for row in entries),
        weights=tuple(tuple(weights for _masks, weights in row) for row in entries),
        n_ops=16,
        n_visible=5,
    )
    widths = tuple(max(masks.shape[0] for masks, _weights in row) for row in entries)
    narrower = dataclasses.replace(bank, widths=widths)

    wide = float(
        iqc.sampled_mmd(
            params, ansatz.generators, references, targets, bank, 32, key,
        ).loss,
    )
    narrow = float(
        iqc.sampled_mmd(
            params, ansatz.generators, references, targets, narrower, 32, key,
        ).loss,
    )
    assert abs(wide - narrow) <= 1e-12 * max(abs(wide), 1.0)
