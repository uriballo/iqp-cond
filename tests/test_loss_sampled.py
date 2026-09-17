"""The fixed-operator estimator: equivalence with the campaign implementation.

``fixtures/golden.npz`` records the compressed-loss value computed by the
research campaign's ``exact_tvd_runner._weighted_loss_functions`` together with
the bank it built (masks and weights per bandwidth and condition).  Feeding that
same bank to :func:`iqp_cond.banked_mmd` must reproduce the value bit-for-bit:
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
    angles = golden[f"{label}_angles"]
    if family == iqc.SPIN_FLIP:
        # The campaign stored a full matrix but spin-flip read column 0.
        angles = np.broadcast_to(angles[:, :1], angles.shape)
    encoders = iqc.encoder(family, angles)
    bank = rebuild_bank(golden, label)
    key = jax.random.wrap_key_data(np.asarray(golden[f"{label}_key"]))

    result = iqc.banked_mmd(
        golden[f"{label}_params"],
        ansatz.generators,
        encoders,
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
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(float)
    bank = iqc.operator_bank(SIGMAS, 8, 3, 4, seed=0)  # three conditions, not two
    with pytest.raises(ValueError, match="operator bank covers 3 conditions"):
        iqc.banked_mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            bank,
            16,
            jax.random.PRNGKey(0),
        )
    wrong_width = iqc.operator_bank(SIGMAS, 8, 2, 5, seed=0)
    with pytest.raises(ValueError, match="built for 5 visible qubits"):
        iqc.banked_mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
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
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 5)))
    targets = rng.integers(0, 2, size=(2, 32, 5)).astype(float)
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(4)

    entries = tuple(
        tuple(
            iqc.sample_nonidentity_masks(
                np.random.default_rng(100 * bandwidth + condition),
                16,
                5,
                sigma,
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
        iqc.banked_mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            bank,
            32,
            key,
        ).loss,
    )
    narrow = float(
        iqc.banked_mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            narrower,
            32,
            key,
        ).loss,
    )
    assert abs(wide - narrow) <= 1e-12 * max(abs(wide), 1.0)


def test_sample_nonidentity_masks_jax_draws_the_conditional_law() -> None:
    """The JIT sampler must draw exactly ``P(a | a != 0)`` with mass weights.

    Same law as the NumPy sampler the campaign pinned: identity never appears,
    each weight is ``Q+ / n_ops``, and the empirical mask frequencies match the
    analytic conditional Bernoulli law.
    """
    n_visible, n_ops, sigma = 5, 100_000, 2.0
    masks, weights = iqc.sample_nonidentity_masks_jax(
        jax.random.PRNGKey(0),
        n_ops,
        n_visible,
        sigma,
    )
    masks = np.asarray(masks)
    assert masks.shape == (n_ops, n_visible)
    assert np.isin(masks, (0.0, 1.0)).all()
    assert np.all(masks.sum(axis=1) > 0.0)

    p = float(iqc.mmd_operator_probability(sigma))
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


def test_sampled_mmd_is_jittable_and_key_driven() -> None:
    """Training masks come from the step key inside the compiled step."""
    rng = np.random.default_rng(11)
    ansatz = iqc.complete_ansatz(4)
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 32, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)

    def loss(key: jax.Array) -> jax.Array:
        return iqc.sampled_mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            key,
            sigmas=(1.0, 2.0),
            n_ops=16,
            n_samples=64,
        ).loss

    jitted = jax.jit(loss)
    first = float(jitted(jax.random.PRNGKey(0)))
    repeat = float(jitted(jax.random.PRNGKey(0)))
    other = float(jitted(jax.random.PRNGKey(1)))
    assert first == repeat
    assert first != other
    assert np.isfinite(first)


def test_sampled_mmd_rejects_a_mismatched_encoder() -> None:
    rng = np.random.default_rng(12)
    ansatz = iqc.complete_ansatz(4)
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(3, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(np.float64)
    with pytest.raises(ValueError, match="covers 3 conditions"):
        iqc.sampled_mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )


def test_sampled_mmd_rejects_nonpositive_ops_and_sigmas() -> None:
    rng = np.random.default_rng(13)
    ansatz = iqc.complete_ansatz(3)
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    with pytest.raises(ValueError, match="n_ops must be positive"):
        iqc.sampled_mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            sigmas=(1.0,),
            n_ops=0,
            n_samples=16,
        )
    with pytest.raises(ValueError, match="sigmas must not be empty"):
        iqc.sampled_mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            sigmas=(),
            n_ops=8,
            n_samples=16,
        )
    with pytest.raises(ValueError, match="sigma must be positive"):
        iqc.sample_nonidentity_masks_jax(jax.random.PRNGKey(0), 8, 3, 0.0)


def test_sampled_mmd_result_survives_jit() -> None:
    """The result dataclass is a pytree, so jitted calls can return it."""
    rng = np.random.default_rng(14)
    ansatz = iqc.complete_ansatz(3)
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)

    def loss(
        current_params: jax.Array,
        target_batch: jax.Array,
        key: jax.Array,
    ) -> iqc.ConditionalMMDResult:
        return iqc.sampled_mmd(
            current_params,
            ansatz.generators,
            encoders,
            target_batch,
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )

    result = jax.jit(loss)(params, targets, jax.random.PRNGKey(0))
    assert result.per_bandwidth_condition.shape == (1, 2)
    assert result.loss.shape == ()


def test_hamming_rbf_matches_the_character_expansion() -> None:
    """The exact kernel equals ``E_a[chi_a(x) chi_a(y)]`` under the mask law."""
    rng = np.random.default_rng(21)
    n_qubits = 5
    x = rng.integers(0, 2, size=n_qubits).astype(np.float64)
    y = rng.integers(0, 2, size=n_qubits).astype(np.float64)
    masks = ((np.arange(2**n_qubits)[:, None] >> np.arange(n_qubits)) & 1).astype(
        np.float64,
    )
    overlap = (masks @ np.abs(x - y)) % 2.0
    characters = 1.0 - 2.0 * overlap

    for sigma in (0.5, 1.0, 2.0, 4.0):
        p = float(iqc.mmd_operator_probability(sigma))
        weights = np.prod(
            np.where(masks == 1.0, p, 1.0 - p),
            axis=1,
        )
        reconstructed = float(np.sum(weights * characters))
        expected = float(
            iqc.hamming_rbf(jnp.asarray(x), jnp.asarray(y), sigma),
        )
        assert reconstructed == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert iqc.hamming_rbf(jnp.asarray(x), jnp.asarray(x), sigma) == 1.0


def test_shared_samples_banked_mmd_agrees_with_per_entry_sampling() -> None:
    """One reference draw per condition is unbiased for the same objective."""
    rng = np.random.default_rng(41)
    ansatz = iqc.complete_ansatz(5)
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 5)))
    targets = rng.integers(0, 2, size=(2, 64, 5)).astype(np.float64)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    bank = iqc.operator_bank(SIGMAS, 32, 2, 5, seed=3)

    def losses(shared: bool, n_keys: int = 240) -> np.ndarray:
        return np.asarray(
            [
                float(
                    iqc.banked_mmd(
                        params,
                        ansatz.generators,
                        encoders,
                        targets,
                        bank,
                        64,
                        jax.random.PRNGKey(key),
                        shared_samples=shared,
                    ).loss,
                )
                for key in range(n_keys)
            ],
        )

    per_entry = losses(False)
    shared = losses(True)
    spread = np.sqrt(
        per_entry.var(ddof=1) / per_entry.size + shared.var(ddof=1) / shared.size,
    )
    assert abs(shared.mean() - per_entry.mean()) < 5.0 * spread + 1e-9


def test_shared_samples_banked_mmd_keeps_the_result_shape() -> None:
    rng = np.random.default_rng(42)
    ansatz = iqc.complete_ansatz(4)
    encoders = iqc.encoder(iqc.PRODUCT, rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 32, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    bank = iqc.operator_bank(SIGMAS, 16, 2, 4, seed=4)

    shared = iqc.banked_mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        bank,
        32,
        jax.random.PRNGKey(0),
        shared_samples=True,
    )
    per_entry = iqc.banked_mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        bank,
        32,
        jax.random.PRNGKey(0),
    )
    assert shared.per_bandwidth_condition.shape == (3, 2)
    assert per_entry.per_bandwidth_condition.shape == (3, 2)
    assert np.all(np.isfinite(np.asarray(shared.per_bandwidth_condition)))
    # Different reference layouts, same objective: the values need not match.
    assert np.allclose(
        np.asarray(shared.per_condition),
        np.asarray(per_entry.per_condition),
        atol=0.05,
    )
