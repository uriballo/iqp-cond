"""The phase-dressed matrix-product-state encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqp_cond as ciqp
from iqp_cond.verification import check_encoder

N_QUBITS = 6


def make_encoder(
    bond: int,
    *,
    dressed: bool,
    n_qubits: int = N_QUBITS,
    n_conditions: int = 3,
    seed: int = 0,
) -> ciqp.MPSEncoder:
    rng = np.random.default_rng(seed)
    width = bond.bit_length()
    angles = rng.uniform(
        -np.pi,
        np.pi,
        (n_conditions, n_qubits - width + 1, 2, width, 2),
    )
    n_generators = n_qubits * (n_qubits + 1) // 2
    dressing = rng.normal(0.0, 0.7, (n_conditions, n_generators)) if dressed else None
    return ciqp.MPSEncoder(
        angles,
        dressing,
        n_qubits=n_qubits,
        bond_dimension=bond,
    )


def cut_ranks(state: np.ndarray, n_qubits: int) -> list[int]:
    """Schmidt rank across each cut between qubits ``j - 1`` and ``j``."""
    ranks = []
    for cut in range(1, n_qubits):
        values = np.linalg.svd(state.reshape(-1, 1 << cut), compute_uv=False)
        ranks.append(int((values > 1e-10 * values[0]).sum()))
    return ranks


@pytest.mark.slow
@pytest.mark.parametrize("dressed", [False, True])
@pytest.mark.parametrize("bond", [1, 2, 4])
def test_encoder_passes_the_conformance_check(bond: int, dressed: bool) -> None:
    encoder = make_encoder(bond, dressed=dressed, n_qubits=12, n_conditions=4)
    report = check_encoder(encoder, seed=5)
    assert report.ok, report.summary()
    assert report.exact_state_error is not None
    assert report.exact_state_error < 1e-12


def test_dressing_leaves_the_sampling_law_unchanged() -> None:
    plain = make_encoder(4, dressed=False)
    dressed = make_encoder(4, dressed=True)
    np.testing.assert_allclose(
        np.abs(np.asarray(dressed.post_hadamard_amplitudes())) ** 2,
        np.abs(np.asarray(plain.post_hadamard_amplitudes())) ** 2,
        atol=1e-14,
    )
    key = jax.random.key(7)
    np.testing.assert_array_equal(
        np.asarray(dressed.sample(key, 4096)),
        np.asarray(plain.sample(key, 4096)),
    )


def test_bond_one_is_a_product_state() -> None:
    state = np.asarray(make_encoder(1, dressed=False).post_hadamard_amplitudes())
    for row in state:
        assert cut_ranks(row, N_QUBITS) == [1] * (N_QUBITS - 1)


@pytest.mark.parametrize("bond", [2, 4])
def test_window_circuit_reaches_but_never_exceeds_the_bond(bond: int) -> None:
    state = np.asarray(make_encoder(bond, dressed=False).post_hadamard_amplitudes())
    for row in state:
        ranks = cut_ranks(row, N_QUBITS)
        assert max(ranks) == bond


def test_bond_one_reference_yields_a_deterministic_model() -> None:
    """``A(z) = 2**(-n/2) (-1)**(x . z)`` with ``lambda = -theta`` is a point mass."""
    n = 12
    rng = np.random.default_rng(3)
    target = rng.integers(0, 2, (2, n))
    ansatz = ciqp.Ansatz.complete(n)
    theta = rng.normal(0.0, 1.0, ansatz.n_parameters)
    # RY(+-pi/2)|0> = (|0> +- |1>)/sqrt(2) puts the sign (-1)**x_q on |1>.
    angles = np.zeros((2, n, 1, 1, 2))
    angles[..., 0, 0, 0] = np.where(target == 1, -np.pi / 2, np.pi / 2)
    encoder = ciqp.MPSEncoder(
        angles,
        np.broadcast_to(-theta, (2, ansatz.n_parameters)),
        n_qubits=n,
        bond_dimension=1,
    )
    model = ciqp.ConditionalModel(ansatz, encoder)
    probabilities = model.probabilities(theta)
    index = (target << np.arange(n)).sum(axis=1)
    np.testing.assert_allclose(probabilities[[0, 1], index], 1.0, atol=1e-10)


def test_head_absorbs_the_dressing() -> None:
    plain = make_encoder(2, dressed=False)
    dressed = make_encoder(2, dressed=True)
    ansatz = ciqp.Ansatz.complete(N_QUBITS)
    theta = np.random.default_rng(4).normal(0.0, 0.5, ansatz.n_parameters)
    samples = plain.sample(jax.random.key(2), 256)
    index = np.arange(1, 1 << N_QUBITS)
    masks = jnp.asarray((index[:, None] >> np.arange(N_QUBITS)) & 1)
    for condition in range(plain.n_conditions):
        rows = samples[condition : condition + 1]
        direct = ciqp.ConditionalModel(ansatz, dressed.select(condition))
        absorbed = ciqp.ConditionalModel(ansatz, plain.select(condition))
        np.testing.assert_allclose(
            direct.estimate_pauli_z(theta, masks, samples=rows).values,
            absorbed.estimate_pauli_z(
                theta + np.asarray(dressed.dressing[condition]),
                masks,
                samples=rows,
            ).values,
            atol=1e-12,
        )


def test_encoder_runs_under_the_vectorized_loss() -> None:
    encoder = make_encoder(4, dressed=True)
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(N_QUBITS), encoder)
    targets = np.random.default_rng(6).integers(0, 2, (3, 64, N_QUBITS)).astype(float)
    params = np.zeros(model.ansatz.n_parameters)
    result = model.mmd(
        params,
        targets,
        jax.random.key(1),
        sigmas=(1.0, 2.0),
        n_ops=32,
        n_samples=128,
    )
    assert np.isfinite(np.asarray(result.per_bandwidth_condition)).all()
    leaves, treedef = jax.tree_util.tree_flatten(encoder)
    assert type(jax.tree_util.tree_unflatten(treedef, leaves)) is ciqp.MPSEncoder


def test_encoder_validates_its_inputs() -> None:
    angles = np.zeros((1, 5, 1, 2, 2))
    with pytest.raises(ValueError, match="power of two"):
        ciqp.MPSEncoder(angles, n_qubits=6, bond_dimension=3)
    with pytest.raises(ValueError, match="power of two"):
        ciqp.MPSEncoder(np.zeros((1, 1, 1, 3, 2)), n_qubits=2, bond_dimension=4)
    with pytest.raises(ValueError, match="angles must have shape"):
        ciqp.MPSEncoder(angles, n_qubits=6, bond_dimension=1)
    with pytest.raises(ValueError, match="dressing must have shape"):
        ciqp.MPSEncoder(angles, np.zeros((1, 6)), n_qubits=6, bond_dimension=2)
    with pytest.raises(ValueError, match="finite"):
        ciqp.MPSEncoder(np.full_like(angles, np.nan), n_qubits=6, bond_dimension=2)
