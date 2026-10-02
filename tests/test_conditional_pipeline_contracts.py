"""Independent tests of the actual compiled benchmark operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.conditional_pipeline import benchmark, workloads
from benchmarks.conditional_pipeline.contracts import (
    compare_kernel_results,
    flatten_kernel_outputs,
    reference_outputs,
)
from benchmarks.conditional_pipeline.fixtures import generate_fixture, resolve_config


def case(encoder="yz"):
    c = resolve_config(
        dict(
            n_qubits=4,
            n_conditions=2,
            n_generators=4,
            arity=2,
            n_ops=4,
            n_samples=9,
            n_targets=7,
            sample_chunk=3,
            encoder=encoder,
            repeats=1,
            warmups=0,
        )
    )
    f = generate_fixture(c)
    c["sigma"] = float(f["sigma"])
    a = (
        jnp.asarray(f["params"]),
        jnp.asarray(f["generators"]),
        workloads.encoder_from_fixture(c, f),
        jnp.asarray(f["targets"]),
        jnp.asarray(f["samples"]),
        jnp.asarray(f["masks"]),
        jnp.asarray(f["weights"]),
        jax.random.wrap_key_data(f["key_data"]),
    )
    return c, f, a


@pytest.mark.parametrize("encoder", ["y", "yz", "yzyz"])
def test_all_compiled_operations_against_independent_oracle(encoder):
    c, f, a = case(encoder)
    actual = {k: jax.jit(fn)(*a) for k, fn in workloads.build_kernels(c).items()}
    comparison = compare_kernel_results(reference_outputs(c, f), actual)
    assert comparison["status"] == "compared", comparison
    assert all(x["allclose_fp64"] for x in comparison["errors"].values()), comparison


def test_compiled_loss_reacts_to_changed_inputs():
    c, f, a = case()
    fn = jax.jit(workloads.build_kernels(c)["prepared_mmd_gradient"])
    fn(*a)
    changed = dict(
        f, params=f["params"] + 0.23, targets=1 - f["targets"], samples=1 - f["samples"]
    )
    b = (
        jnp.asarray(changed["params"]),
        *a[1:3],
        jnp.asarray(changed["targets"]),
        jnp.asarray(changed["samples"]),
        *a[5:],
    )
    ref = reference_outputs(c, changed)["prepared_mmd_gradient"]
    for x, y in zip(fn(*b), ref, strict=True):
        np.testing.assert_allclose(x, y, atol=1e-9, rtol=1e-8)


def test_bogus_finite_timed_kernel_fails_correctness(monkeypatch, tmp_path):
    original = workloads.build_kernels

    def corrupt(c):
        funcs = original(c)
        funcs["mmd_gradient"] = lambda *a: (jnp.asarray(42.0), jnp.ones_like(a[0]))
        return funcs

    monkeypatch.setattr(workloads, "build_kernels", corrupt)
    c, _, _ = case()
    row = benchmark.run_cell(c, artifacts=tmp_path, kernels=["mmd_gradient"])
    assert row["status"] == "ok", row.get("error")
    assert row["compiled_checks"]["status"] == "failed"
    benchmark.reconcile([row])
    assert row["validity"]["status"] == "failed"
    with np.load(row["kernel_output_path"]) as outputs:
        assert float(outputs["mmd_gradient__0"]) == 42.0


def test_complex_error_and_gradient_sign_diagnostics():
    ref = {
        "amplitude_ratios": np.array([1j]),
        "phase_gradient": (0.0, np.array([1.0, 1e-12])),
    }
    got = {
        "amplitude_ratios": np.array([-1j]),
        "phase_gradient": (0.0, np.array([-1.0, -1e-12])),
    }
    result = compare_kernel_results(ref, got)
    assert result["errors"]["amplitude_ratios__0"]["max_abs"] == 2
    assert result["errors"]["phase_gradient__1"]["gradient_sign_flips"] == 1
    bad = flatten_kernel_outputs(got)
    bad["amplitude_ratios__0"] = np.array([np.nan])
    assert (
        compare_kernel_results(flatten_kernel_outputs(ref), bad)["status"] == "failed"
    )


def test_sampling_law_and_support_boundary():
    from benchmarks.conditional_pipeline.contracts import local_amplitudes

    c, f, a = case()
    _, one = local_amplitudes(f["angles"], c["encoder"])
    drawn = np.asarray(jax.jit(lambda e, k: e.sample(k, 20000))(a[2], a[-1]))
    p = abs(one) ** 2
    assert np.all(
        abs(drawn.mean(axis=1) - p) < 6 * np.sqrt(p * (1 - p) / 20000) + 1 / 20000
    )
    joint = np.mean(drawn[:, :, 0] * drawn[:, :, 1], axis=1)
    assert np.all(abs(joint - p[:, 0] * p[:, 1]) < 0.025)
    c, f, _ = case("y")
    f["angles"][:] = np.pi / 2
    f["samples"][:] = 0
    f["masks"][:] = np.eye(4)
    e = workloads.encoder_from_fixture(c, f)
    actual = np.asarray(
        jax.jit(e.amplitude_ratios)(
            jnp.asarray(f["samples"]), jnp.asarray(f["masks"][0])
        )
    )
    assert np.isfinite(actual).all() and np.max(abs(actual)) < 1e-12
