"""The conditional MMD objective: campaign equivalence and estimator semantics.

``fixtures/golden.npz`` records the compressed-loss value computed by the
research campaign's ``exact_tvd_runner._weighted_loss_functions`` together with
the bank it built (masks and weights per bandwidth and condition).  Feeding that
same bank to :func:`iqp_cond.mmd` must reproduce the value bit-for-bit: the
library's job is to compute the paper's estimator, not an approximation of it.
The mask samplers and banks themselves are covered by ``test_operators.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.loss import hamming_rbf
from iqp_cond.operators import (
    OperatorBank,
    bank_from_ragged,
    mmd_operator_probability,
    operator_bank,
    sample_nonidentity_masks,
)


def y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """One ``RY`` layer per qubit: the plain product-state encoding."""
    angles = np.asarray(angles)
    return ciqp.angle_encoder(angles[:, None, :], axes=("Y" * angles.shape[1],))


FIXTURES = Path(__file__).parent / "fixtures" / "golden.npz"
SIGMAS = (1.0, 2.0, 4.0)
N_CONDITIONS = 2
N_VISIBLE = 4
N_OPS = 64
N_SAMPLES = 64


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(FIXTURES) as data:
        return {name: data[name] for name in data.files}


def rebuild_bank(golden: dict[str, np.ndarray], label: str) -> OperatorBank:
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
    return bank_from_ragged(
        sigmas=SIGMAS,
        masks=masks,
        weights=weights,
        n_ops=N_OPS,
        n_visible=N_VISIBLE,
    )


def test_mmd_reproduces_the_campaign_value(
    golden: dict[str, np.ndarray],
) -> None:
    label = "bank_plain"
    ansatz = ciqp.complete_ansatz(N_VISIBLE)
    encoders = y_layer(golden[f"{label}_angles"])
    bank = rebuild_bank(golden, label)
    key = jax.random.wrap_key_data(np.asarray(golden[f"{label}_key"]))

    result = ciqp.mmd(
        golden[f"{label}_params"],
        ansatz.generators,
        encoders,
        golden[f"{label}_targets"],
        key,
        bank=bank,
        n_samples=N_SAMPLES,
    )
    deviation = abs(float(result.loss) - float(golden[f"{label}_value"]))
    assert deviation == 0.0, f"{label}: loss differs from the campaign by {deviation!r}"
    assert result.per_bandwidth_condition.shape == (len(SIGMAS), N_CONDITIONS)


def test_mmd_rejects_a_mismatched_bank() -> None:
    rng = np.random.default_rng(0)
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(float)
    bank = operator_bank(SIGMAS, 8, 3, 4, seed=0)  # three conditions, not two
    with pytest.raises(ValueError, match="operator bank covers 3 conditions"):
        ciqp.mmd(
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
        ciqp.mmd(
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
    ansatz = ciqp.complete_ansatz(5)
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
        ciqp.mmd(
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
        ciqp.mmd(
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
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 32, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)

    def loss(key: jax.Array) -> jax.Array:
        return ciqp.mmd(
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
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(3, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(np.float64)
    with pytest.raises(ValueError, match="covers 3 conditions"):
        ciqp.mmd(
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
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 16, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    key = jax.random.PRNGKey(0)

    narrow = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    with pytest.raises(ValueError, match="generators cover 4"):
        ciqp.mmd(
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
        ciqp.mmd(
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
        ciqp.mmd(
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
        return ciqp.mmd(
            current_params,
            current_generators,
            narrow,
            jnp.asarray(targets[:, :, :3]),
            key,
            sigmas=(1.0,),
            n_ops=8,
            n_samples=16,
        ).loss

    with pytest.raises(ValueError, match="generators cover 4"):
        jax.jit(loss)(params, ansatz.generators)


def test_mmd_rejects_mixed_or_missing_mask_sources() -> None:
    rng = np.random.default_rng(13)
    ansatz = ciqp.complete_ansatz(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 8, 2, 3, seed=1)
    with pytest.raises(ValueError, match="or a prebuilt bank"):
        ciqp.mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            n_samples=16,
        )
    with pytest.raises(ValueError, match="key mask source only"):
        ciqp.mmd(
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
        ciqp.mmd(
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
    ansatz = ciqp.complete_ansatz(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)

    def loss(
        current_params: jax.Array,
        target_batch: jax.Array,
        key: jax.Array,
    ) -> ciqp.ConditionalMMDResult:
        return ciqp.mmd(
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


def test_shared_samples_agrees_with_per_entry_sampling() -> None:
    """One reference draw per condition is unbiased for the same objective."""
    rng = np.random.default_rng(41)
    ansatz = ciqp.complete_ansatz(5)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 5)))
    targets = rng.integers(0, 2, size=(2, 64, 5)).astype(np.float64)
    params = rng.normal(0.0, 0.4, size=ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 32, 2, 5, seed=3)

    def losses(shared: bool, n_keys: int = 240) -> np.ndarray:
        return np.asarray(
            [
                float(
                    ciqp.mmd(
                        params,
                        ansatz.generators,
                        encoders,
                        targets,
                        jax.random.PRNGKey(key),
                        bank=bank,
                        n_samples=64,
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


def test_shared_samples_keeps_the_result_shape() -> None:
    rng = np.random.default_rng(42)
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 32, 4)).astype(np.float64)
    params = np.zeros(ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 16, 2, 4, seed=4)

    def evaluate(shared: bool) -> ciqp.ConditionalMMDResult:
        return ciqp.mmd(
            params,
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            bank=bank,
            n_samples=32,
            shared_samples=shared,
        )

    shared = evaluate(True)
    per_entry = evaluate(False)
    assert shared.per_bandwidth_condition.shape == (3, 2)
    assert per_entry.per_bandwidth_condition.shape == (3, 2)
    assert np.all(np.isfinite(np.asarray(shared.per_bandwidth_condition)))
    # Different reference layouts, same objective: the values need not match.
    assert np.allclose(
        np.asarray(shared.per_condition),
        np.asarray(per_entry.per_condition),
        atol=0.05,
    )


def test_shared_samples_reuses_one_draw_per_condition_with_fresh_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh-mask path honors shared_samples: one encoder draw per condition."""
    rng = np.random.default_rng(43)
    ansatz = ciqp.complete_ansatz(3)
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
            ciqp.mmd(
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

    assert np.isfinite(evaluate(True))
    assert len(calls) == 2
    assert np.isfinite(evaluate(False))
    assert len(calls) == 2 * 3


def test_sample_chunk_matches_the_monolithic_objective() -> None:
    """Streaming the sample axis changes only the summation order."""
    rng = np.random.default_rng(61)
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 64, 4)).astype(np.float64)
    params = rng.normal(0.0, 0.3, size=ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 32, 2, 4, seed=12)
    key = jax.random.PRNGKey(6)

    monolithic = ciqp.mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        key,
        bank=bank,
        n_samples=512,
        shared_samples=True,
    )
    chunked = ciqp.mmd(
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
    ansatz = ciqp.complete_ansatz(3)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 3)))
    targets = rng.integers(0, 2, size=(2, 16, 3)).astype(np.float64)
    with pytest.raises(ValueError, match="sample_chunk must be positive"):
        ciqp.mmd(
            np.zeros(ansatz.n_parameters),
            ansatz.generators,
            encoders,
            targets,
            jax.random.PRNGKey(0),
            bank=operator_bank(SIGMAS, 8, 2, 3, seed=1),
            n_samples=16,
            sample_chunk=0,
        )


def test_mixed_precision_gemm_stays_close_to_float64() -> None:
    """The mixed-precision knob does not move the objective."""
    rng = np.random.default_rng(51)
    ansatz = ciqp.complete_ansatz(4)
    encoders = y_layer(rng.uniform(0.4, 2.4, size=(2, 4)))
    targets = rng.integers(0, 2, size=(2, 64, 4)).astype(np.float64)
    params = rng.normal(0.0, 0.3, size=ansatz.n_parameters)
    bank = operator_bank(SIGMAS, 64, 2, 4, seed=9)
    key = jax.random.PRNGKey(5)

    reference = ciqp.mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        key,
        bank=bank,
        n_samples=512,
    )
    mixed = ciqp.mmd(
        params,
        ansatz.generators,
        encoders,
        targets,
        key,
        bank=bank,
        n_samples=512,
        gemm_dtype=jnp.float32,
    )
    deviation = abs(float(mixed.loss) - float(reference.loss))
    assert 0.0 < deviation < 1e-5
