"""Independent numerical acceptance gates, beyond finite execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.conditional_pipeline.fixtures import (
    configs,
    generate_fixture,
    resolve_config,
)
from benchmarks.conditional_pipeline.validity import compare_outputs, independent_checks
from benchmarks.conditional_pipeline.workloads import encoder_from_fixture, prepared_mmd


def outputs():
    return dict(
        moments=np.array([[0.2, 0.4]]),
        standard_errors=np.array([[0.01, 0.02]]),
        loss=np.asarray(0.01),
        gradient=np.array([0.2, -0.1]),
    )


def test_shape_nonfinite_and_precision_failures():
    ref = outputs()
    assert compare_outputs(ref, ref)["status"] == "passed"
    got = outputs()
    got["moments"][0, 0] += 2e-5
    assert compare_outputs(ref, got, "estimator_f32")["status"] == "failed"
    got = outputs()
    got["gradient"] = np.array([np.nan, 0.0])
    assert compare_outputs(ref, got)["status"] == "failed"
    got = outputs()
    got["moments"] = np.zeros((1, 3))
    assert compare_outputs(ref, got)["status"] == "failed"


@pytest.mark.parametrize("encoder", ["y", "yz", "yzyz"])
@pytest.mark.parametrize("arity", [1, 2, 4])
def test_independent_circuit_and_gradient_checks(encoder, arity):
    c = resolve_config(
        dict(
            n_qubits=4,
            n_conditions=2,
            n_generators=1 if arity == 4 else 4,
            arity=arity,
            encoder=encoder,
            n_ops=4,
            n_samples=8,
            n_targets=8,
            sample_chunk=3,
        )
    )
    f = generate_fixture(c)
    e = encoder_from_fixture(c, f)
    c["sigma"] = float(f["sigma"])
    check = independent_checks(c, f, e)
    assert check["status"] == "passed", check


def test_support_zero_is_a_valid_zero_numerator():
    c = configs("smoke")[0]
    f = generate_fixture(c)
    f["angles"][:] = np.pi / 2
    e = encoder_from_fixture(c, f)
    z = jnp.zeros((1, 2, 4))
    a = jnp.eye(4)
    ratio = np.asarray(e.amplitude_ratios(z, a))
    assert np.all(np.abs(ratio) < 1e-12)
    assert np.isfinite(ratio).all()


def test_partial_chunk_preserves_loss_and_gradient():
    c = configs("smoke")[2]
    c["n_samples"] = 17
    f = generate_fixture(c)
    e = encoder_from_fixture(c, f)

    def loss(p, chunk):
        return prepared_mmd(
            p,
            jnp.asarray(f["generators"]),
            e,
            jnp.asarray(f["targets"]),
            jnp.asarray(f["samples"]),
            jnp.asarray(f["masks"]),
            jnp.asarray(f["weights"]),
            dict(c, sample_chunk=chunk),
        )

    p = jnp.asarray(f["params"])
    v, g = jax.value_and_grad(lambda x: loss(x, None))(p)
    w, h = jax.value_and_grad(lambda x: loss(x, 6))(p)
    np.testing.assert_allclose(v, w, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(g, h, atol=1e-12, rtol=1e-12)
