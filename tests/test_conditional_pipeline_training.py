"""Focused checks for the end-to-end training benchmark workload."""

import numpy as np

from benchmarks.conditional_pipeline.training_workload import (
    generate_training_fixture,
    resolve_training_config,
    run_training_cell,
)


def test_smoke_fixture_is_reproducible_and_normalized():
    config = resolve_training_config({}, smoke=True)
    first = generate_training_fixture(config)
    second = generate_training_fixture(config)
    names = ("train_targets", "validation_targets", "test_targets", "initial_params")
    for name in names:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["train_targets"].shape == (4, 32, 16)
    assert np.all((first["train_targets"] == 0) | (first["train_targets"] == 1))


def test_smoke_cpu_training_persists_outputs(tmp_path):
    config = resolve_training_config({"n_qubits": 4}, smoke=True)
    fixture = generate_training_fixture(config)
    row = run_training_cell(config, backend="cpu", fixture=fixture, artifacts=tmp_path)
    assert row["status"] == "ok", row.get("error")
    assert row["completed_updates"] == 2
    assert row["fixture_hash"]
    assert (tmp_path / "training_result.json").exists()
    assert (tmp_path / "training_history.npz").exists()


def test_training_observer_preserves_trainer_results(tmp_path):
    import jax.numpy as jnp

    import iqp_cond as ciqp
    from benchmarks.conditional_pipeline.training_workload import _diagnostics
    from iqp_cond.operators import expected_bodyness_sigmas

    c = resolve_training_config({"n_qubits": 4}, smoke=True)
    f = generate_training_fixture(c)
    row = run_training_cell(c, fixture=f, artifacts=tmp_path)
    assert row["status"] == "ok", row.get("error")
    e = ciqp.AngleEncoder(
        np.stack([np.tile([np.pi / 2, 0.75 * b], (4, 1)).T for b in c["betas"]]),
        axes=("YYYY", "ZZZZ"),
    )
    config = ciqp.TrainingConfig(
        steps=2,
        batch_size=8,
        n_ops=4,
        mc_samples=16,
        validation_n_ops=4,
        validation_mc_samples=16,
        validation_interval=1,
        sample_chunk=8,
        gemm_dtype=jnp.float32,
        sigmas=expected_bodyness_sigmas(4)[:2],
    )
    result = ciqp.train(
        ciqp.ConditionalModel(ciqp.Ansatz.complete(4), e),
        f["train_targets"],
        f["validation_targets"],
        seed=c["seed"],
        config=config,
        initial_params=f["initial_params"],
        selection_metric=lambda p: float(
            np.mean(_diagnostics(p, c, c["development_betas"])["profile_tvd"])
        ),
    )
    with np.load(row["output_path"]) as out:
        np.testing.assert_allclose(
            out["final_params"], result.final_params, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            out["train_losses"], result.train_losses, atol=1e-12, rtol=1e-12
        )
    t = row["timings_seconds"]
    assert 0 < t["training_loop"] < t["training_call"] < t["total_training"]
    assert row["measurement_policy"]["selection"] == "exact_development_profile_tvd"
    assert row["outcomes"]["final"]["evaluation"]["betas"] == [0.4, 1.0, 1.6]


def test_training_backend_failure_precedes_fixture(monkeypatch, tmp_path):
    import jax

    class CPU:
        platform = "cpu"

    monkeypatch.setattr(jax, "devices", lambda _: [CPU()])
    row = run_training_cell(
        resolve_training_config(smoke=True), backend="gpu", artifacts=tmp_path
    )
    assert row["status"] == "failed" and "requested gpu, got cpu" in row["error"]
    assert not row.get("fixture_path")
