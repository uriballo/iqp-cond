"""Estimator semantics and exact identities of the conditional MMD objective.

The PennyLane cross-check owns the exact-expectation validation. These tests
cover the two mask sources, the shared-draw layout, result shapes, JIT behavior,
chunking, and the target moments without recorded reference values.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.loss import (
    ConditionalMMDResult,
    _mmd_entries,
    _mmd_from_entries,
    hamming_rbf,
    mmd,
    target_moments,
)
from iqp_cond.operators import (
    bank_from_ragged,
    mmd_operator_probability,
    operator_bank,
    sample_nonidentity_masks,
)


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit, the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


SIGMAS = (1.0, 2.0, 4.0)


def test_mmd_rejects_a_mismatched_bank() -> None:
    rng = np.random.default_rng(0)
    ansatz = ciqp.Ansatz.complete(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(float)
    bank = operator_bank(SIGMAS, 8, 3, 4, seed=0)  # three conditions, not two
    with pytest.raises(ValueError, match="operator bank covers 3 conditions"):
        mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            bank=bank,
            n_samples=16,
        )
    wrong_width = operator_bank(SIGMAS, 8, 2, 5, seed=0)
    with pytest.raises(ValueError, match="built for 5 visible qubits"):
        mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            bank=wrong_width,
            n_samples=16,
        )


def test_per_bandwidth_widths_do_not_change_the_loss() -> None:
    """Cutting a bandwidth's padding changes cost, not the estimate."""
    rng = np.random.default_rng(3)
    ansatz = ciqp.Ansatz.complete(5)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 5)))
    targets = rng.integers(0, 2, size=(2, 32, 5)).astype(float)
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(4)

    entries = tuple(
        tuple(
            sample_nonidentity_masks(
                np.random.default_rng(100 * bandwidth + condition),
                16,
                5,
                sigma,
            )
            for condition in range(2)
        )
        for bandwidth, sigma in enumerate(SIGMAS)
    )
    bank = bank_from_ragged(
        sigmas=SIGMAS,
        masks=tuple(tuple(masks for masks, _weights in row) for row in entries),
        weights=tuple(tuple(weights for _masks, weights in row) for row in entries),
        n_ops=16,
        n_visible=5,
    )
    widths = tuple(max(masks.shape[0] for masks, _weights in row) for row in entries)
    narrower = dataclasses.replace(bank, widths=widths)

    wide = float(
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            key,
            bank=bank,
            n_samples=32,
        ).loss,
    )
    narrow = float(
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            key,
            bank=narrower,
            n_samples=32,
        ).loss,
    )
    assert abs(wide - narrow) <= 1e-12 * max(abs(wide), 1.0)


def test_key_masks_are_jittable_and_key_driven() -> None:
    """Fresh-mask training draws masks from the step key inside the jit."""
    rng = np.random.default_rng(11)
    ansatz = ciqp.Ansatz.complete(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 32, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)

    def loss(key: jax.Array) -> jax.Array:
        return mmd(
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


def test_mmd_rejects_a_mismatched_encoder() -> None:
    rng = np.random.default_rng(12)
    ansatz = ciqp.Ansatz.complete(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(3, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(np.float64)
    with pytest.raises(ValueError, match="covers 3 conditions"):
        mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )


def test_mmd_rejects_generator_and_params_shape_mismatches() -> None:
    """The loss reports width failures itself instead of failing in an einsum."""
    rng = np.random.default_rng(15)
    ansatz = ciqp.Ansatz.complete(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(0)

    narrow = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    with pytest.raises(ValueError, match="generators cover 4"):
        mmd(
            params,
            ansatz.generators,
            narrow,
            targets[:, :, :3],
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )
    with pytest.raises(ValueError, match="params must have shape"):
        mmd(
            np.zeros(3),
            ansatz.generators,
            encoders,
            targets,
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )
    with pytest.raises(ValueError, match="generators must have shape"):
        mmd(
            params,
            ansatz.generators[0],
            encoders,
            targets,
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )

    def loss(current_params: jax.Array, current_generators: jax.Array) -> jax.Array:
        return mmd(
            current_params,
            current_generators,
            narrow,
            jnp.asarray(targets[:, :, :3]),
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        ).loss

    small = ciqp.Ansatz.complete(3)
    small_encoder = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    wide_targets = rng.integers(0, 2, size=(2, 16, 5)).astype(np.float64)
    with pytest.raises(ValueError, match="target width"):
        mmd(
            np.zeros(small.n_parameters),
            small.generators,
            small_encoder,
            wide_targets,
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )
    with pytest.raises(ValueError, match="targets must have shape"):
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets.reshape(-1, 4),
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        )
    with pytest.raises(ValueError, match="n_samples must be at least two"):
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=1,
        )

    with pytest.raises(ValueError, match="generators cover 4"):
        jax.jit(loss)(params, ansatz.generators)


def test_mmd_rejects_mixed_or_missing_mask_sources() -> None:
    rng = np.random.default_rng(13)
    ansatz = ciqp.Ansatz.complete(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 8, 2, 3, seed=1)
    with pytest.raises(ValueError, match="or a prebuilt bank"):
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            n_samples=16,
        )
    with pytest.raises(ValueError, match="key mask source only"):
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            bank=bank,
            sigmas=(1.0,),
            n_samples=16,
        )
    with pytest.raises(ValueError, match="n_ops must be positive"):
        mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            sigmas=(1.0,),
            n_ops=0,
            n_samples=16,
        )


def test_mmd_result_survives_jit() -> None:
    """The result dataclass is a pytree, so jitted calls can return it."""
    rng = np.random.default_rng(14)
    ansatz = ciqp.Ansatz.complete(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)

    def loss(
        current_params: jax.Array,
        target_batch: jax.Array,
        key: jax.Array,
    ) -> ConditionalMMDResult:
        return mmd(
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


def test_target_moments_are_unbiased_for_an_exact_product_law() -> None:
    """The target U-statistic squares converge to the exact squared moments."""
    n_visible, n_samples, n_batches = 3, 256, 200
    probabilities = np.array([0.2, 0.5, 0.8])
    masks = ((np.arange(2**n_visible)[:, None] >> np.arange(n_visible)) & 1).astype(
        np.float64
    )
    # Rows enumerate the little-endian bit patterns, identity first.
    exact = np.prod(
        np.where(masks == 1.0, 1.0 - 2.0 * probabilities, 1.0),
        axis=1,
    )
    rng = np.random.default_rng(22)
    sampled_means = []
    sampled_squares = []
    for _ in range(n_batches):
        batch = (rng.uniform(size=(1, n_samples, n_visible)) < probabilities).astype(
            np.float64,
        )
        means, squares = target_moments(batch, masks)
        sampled_means.append(np.asarray(means)[0])
        sampled_squares.append(np.asarray(squares)[0])
    sampled_means = np.stack(sampled_means)
    sampled_squares = np.stack(sampled_squares)

    mean_error = sampled_means.mean(axis=0) - exact
    square_error = sampled_squares.mean(axis=0) - exact**2
    mean_spread = sampled_means.std(axis=0, ddof=1) / np.sqrt(n_batches)
    square_spread = sampled_squares.std(axis=0, ddof=1) / np.sqrt(n_batches)
    assert np.all(np.abs(mean_error) < 5.0 * mean_spread + 1e-12)
    assert np.all(np.abs(square_error) < 5.0 * square_spread + 1e-12)
    assert np.allclose(sampled_squares[:, 0], 1.0, atol=1e-12)

    with pytest.raises(ValueError, match="at least two target samples"):
        target_moments(np.zeros((1, 1, n_visible)), masks)


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
        p = float(mmd_operator_probability(sigma))
        weights = np.prod(
            np.where(masks == 1.0, p, 1.0 - p),
            axis=1,
        )
        reconstructed = float(np.sum(weights * characters))
        expected = float(
            hamming_rbf(jnp.asarray(x), jnp.asarray(y), sigma),
        )
        assert reconstructed == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert hamming_rbf(jnp.asarray(x), jnp.asarray(x), sigma) == 1.0
    with pytest.raises(ValueError, match="sigma must be positive"):
        hamming_rbf(jnp.asarray(x), jnp.asarray(y), 0.0)


@pytest.mark.slow
def test_shared_samples_agrees_with_per_entry_sampling() -> None:
    """One reference draw per condition is unbiased for the same objective."""
    rng = np.random.default_rng(41)
    ansatz = ciqp.Ansatz.complete(5)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 5)))
    targets = rng.integers(0, 2, size=(2, 64, 5)).astype(np.float64)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 32, 2, 5, seed=3)

    def evaluate(shared: bool, key: int = 0) -> ConditionalMMDResult:
        return mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(key),
            bank=bank,
            n_samples=64,
            shared_samples=shared,
        )

    def losses(shared: bool, n_keys: int = 240) -> np.ndarray:
        return np.asarray([float(evaluate(shared, key).loss) for key in range(n_keys)])

    per_entry = losses(False)
    shared = losses(True)
    spread = np.sqrt(
        per_entry.var(ddof=1) / per_entry.size + shared.var(ddof=1) / shared.size,
    )
    assert abs(shared.mean() - per_entry.mean()) < 5.0 * spread + 1e-9

    shared_result = evaluate(True)
    per_entry_result = evaluate(False)
    assert shared_result.per_bandwidth_condition.shape == (len(SIGMAS), 2)
    assert per_entry_result.per_bandwidth_condition.shape == (len(SIGMAS), 2)
    assert np.all(np.isfinite(np.asarray(shared_result.per_bandwidth_condition)))


def test_shared_samples_reuses_one_draw_per_condition_with_fresh_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh-mask path honors shared_samples, one encoder draw per condition."""
    rng = np.random.default_rng(43)
    ansatz = ciqp.Ansatz.complete(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    calls: list[int] = []

    encoder_class = type(encoders)
    original_sample = encoder_class.sample

    def counting_sample(
        self: object,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        calls.append(1)
        return original_sample(self, key, n_samples)

    monkeypatch.setattr(encoder_class, "sample", counting_sample)

    def evaluate(shared: bool) -> float:
        calls.clear()
        return float(
            mmd(
                params,
                ansatz.generators,
                encoders,
                targets,
                jax.random.PRNGKey(0),
                sigmas=(1.0, 2.0, 4.0),
                n_ops=8,
                n_samples=16,
                shared_samples=shared,
            ).loss,
        )

    # Pytree encoders trace one condition body for every condition, so count
    # calls relative to the shared layout: one draw per bandwidth there.
    assert np.isfinite(evaluate(True))
    shared_calls = len(calls)
    assert shared_calls in (1, 2)
    assert np.isfinite(evaluate(False))
    assert len(calls) == 3 * shared_calls


def test_sample_chunk_matches_the_monolithic_objective() -> None:
    """Streaming the sample axis changes only the summation order."""
    rng = np.random.default_rng(61)
    ansatz = ciqp.Ansatz.complete(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 64, 4)).astype(np.float64)
    params = rng.normal(0.0, 0.3, size=ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 32, 2, 4, seed=12)
    key = jax.random.PRNGKey(6)

    monolithic = mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        key,
        bank=bank,
        n_samples=512,
        shared_samples=True,
    )
    chunked = mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        key,
        bank=bank,
        n_samples=512,
        shared_samples=True,
        sample_chunk=128,
    )
    assert np.allclose(
        np.asarray(chunked.per_bandwidth_condition),
        np.asarray(monolithic.per_bandwidth_condition),
        rtol=1e-9,
        atol=1e-12,
    )


def test_sample_chunk_must_be_positive() -> None:
    rng = np.random.default_rng(62)
    ansatz = ciqp.Ansatz.complete(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    with pytest.raises(ValueError, match="sample_chunk must be positive"):
        mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            bank=operator_bank(SIGMAS, 8, 2, 3, seed=1),
            n_samples=16,
            sample_chunk=0,
        )


def test_parameter_gradient_matches_finite_differences() -> None:
    """Autodiff through the sampled objective is exact for fixed randomness."""
    rng = np.random.default_rng(71)
    ansatz = ciqp.Ansatz.complete(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 32, 4)).astype(np.float64)
    params = jnp.asarray(rng.normal(0.0, 0.3, size=ansatz.n_parameters))
    bank = operator_bank(SIGMAS, 16, 2, 4, seed=13)
    key = jax.random.PRNGKey(7)

    def loss(current: jax.Array) -> jax.Array:
        return mmd(
            current,
            ansatz.generators,
            encoders,
            targets,
            key,
            bank=bank,
            n_samples=256,
            shared_samples=True,
        ).loss

    gradient = np.asarray(jax.grad(loss)(params))
    epsilon = 1e-6
    finite = np.array(
        [
            float(loss(params + step) - loss(params - step)) / (2.0 * epsilon)
            for step in epsilon * np.eye(params.size)
        ],
    )
    assert np.max(np.abs(gradient - finite)) < 1e-8


@pytest.mark.parametrize("axes", ["Y", "YZ"])
@pytest.mark.parametrize(
    "precision",
    [{}, {"estimator_dtype": jnp.float32}, {"gemm_dtype": jnp.float32}],
)
def test_drawn_entries_match_the_fresh_mask_objective(axes, precision) -> None:
    """Entries drawn once reproduce the fresh-mask MMD at any parameters."""
    rng = np.random.default_rng(83)
    n_qubits, n_visible, n_conditions = 6, 4, 3
    encoders = ciqp.AngleEncoder(
        rng.uniform(0.4, 2.4, size=(n_conditions, len(axes), n_qubits)),
        axes=tuple(axis * n_qubits for axis in axes),
    )
    ansatz = ciqp.Ansatz.complete(n_qubits)
    targets = rng.integers(0, 2, size=(n_conditions, 32, n_visible)).astype(float)
    params = jnp.asarray(rng.normal(0.0, 0.3, size=ansatz.n_parameters))
    key = jax.random.PRNGKey(17)
    options = {"sigmas": SIGMAS, "n_ops": 32, "n_samples": 128}

    sampled = mmd(
        params, ansatz.generators, encoders, targets, key, **options, **precision
    )
    entries = _mmd_entries(
        encoders,
        targets,
        key,
        bank=None,
        shared_samples=False,
        **options,
    )
    planned = jax.jit(
        lambda current, entries: _mmd_from_entries(
            current, ansatz.generators, encoders, entries, **precision
        )
    )(params, entries)

    # The float32 estimator agrees to single precision between the eager
    # reference and the compiled evaluation, which may fuse differently.
    rtol = 1e-5 if "estimator_dtype" in precision else 1e-12
    np.testing.assert_allclose(
        planned.per_bandwidth_condition,
        sampled.per_bandwidth_condition,
        rtol=rtol,
    )
