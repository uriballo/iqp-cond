"""Build measurements for the library's encoder, IQP, estimator, and MMD operations."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond._precision import resolve_precision
from iqp_cond.encoders import AngleEncoder
from iqp_cond.estimator import _phase_differences, estimate_pauli_z_unchecked
from iqp_cond.loss import _Entries, _mmd_from_entries, mmd, target_moments

AXES = {"y": ("Y",), "yz": ("Y", "Z"), "yzyz": ("Y", "Z", "Y", "Z")}


def encoder_from_fixture(config, fixture):
    return AngleEncoder(
        fixture["angles"],
        axes=tuple(a * config["n_qubits"] for a in AXES[config["encoder"]]),
    )


def prepared_mmd(
    params,
    generators,
    encoder,
    targets,
    samples,
    masks,
    weights,
    config,
    precision="fp64",
):
    """Compute the per-condition U-statistic from stored random draws."""
    means, squares = jax.vmap(target_moments)(targets[:, None], masks)
    entries = (_Entries(masks, weights, samples[:, None], means[:, 0], squares[:, 0]),)
    return _mmd_from_entries(
        params,
        generators,
        encoder,
        entries,
        precision=precision,
        sample_chunk=config["sample_chunk"],
    ).loss


def build_kernels(config: dict[str, Any]):
    precision = config["dtype"]
    policy = resolve_precision(precision)

    def sampling(p, g, e, t, s, a, w, k):
        return e.sample(k, config["n_samples"])

    def ratios(p, g, e, t, s, a, w, k):
        return jnp.stack(
            [
                e.select(c).amplitude_ratios(s[c : c + 1], a[c])[0]
                for c in range(config["n_conditions"])
            ]
        )

    def phase(p, g, e, t, s, a, w, k):
        return jnp.stack(
            [
                _phase_differences(p, g, a[c], s[c : c + 1], policy)[0]
                for c in range(config["n_conditions"])
            ]
        )

    def estimate(p, g, e, t, s, a, w, k):
        return jnp.stack(
            [
                estimate_pauli_z_unchecked(
                    p,
                    g,
                    e.select(c),
                    a[c],
                    samples=s[c : c + 1],
                    precision=precision,
                ).values[0]
                for c in range(config["n_conditions"])
            ]
        )

    def sampled_estimate(p, g, e, t, s, a, w, k):
        drawn = e.sample(k, config["n_samples"])
        return estimate(p, g, e, t, drawn, a, w, k)

    def fixed_loss(p, g, e, t, s, a, w, k):
        return prepared_mmd(p, g, e, t, s, a, w, config, precision)

    def fresh_loss(p, g, e, t, s, a, w, k):
        return mmd(
            p,
            g,
            e,
            t,
            k,
            sigmas=(config["sigma"],),
            n_ops=config["n_ops"],
            n_samples=config["n_samples"],
            shared_samples=True,
            sample_chunk=config["sample_chunk"],
            precision=precision,
        ).loss

    def scalar_phase(*args):
        return jnp.mean(phase(*args))

    def scalar_estimate(*args):
        return jnp.mean(estimate(*args))

    return {
        "encoder_sampling": sampling,
        "amplitude_ratios": ratios,
        "phase": phase,
        "phase_gradient": jax.value_and_grad(scalar_phase),
        "prepared_estimator": estimate,
        "prepared_estimator_gradient": jax.value_and_grad(scalar_estimate),
        "sampling_estimator": sampled_estimate,
        "prepared_mmd": fixed_loss,
        "prepared_mmd_gradient": jax.value_and_grad(fixed_loss),
        "mmd": fresh_loss,
        "mmd_gradient": jax.value_and_grad(fresh_loss),
    }


def measure(fn, args, repeats=10, warmups=2, trace_dir=None, *, return_result=False):
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    lowering = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter() - start
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(*args))
    first = time.perf_counter() - start
    # Validate outside the timing interval; do not compile extra checks in it.
    for leaf in jax.tree_util.tree_leaves(result):
        if not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError("nonfinite kernel output")
    for _ in range(warmups):
        jax.block_until_ready(compiled(*args))
    warm = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        warm.append(time.perf_counter() - start)
    if trace_dir is not None:
        with jax.profiler.trace(str(trace_dir)):
            jax.block_until_ready(compiled(*args))
    timings = {
        "lowering": lowering,
        "compile": compilation,
        "first_execution": first,
        "warm": warm,
    }
    return (timings, result) if return_result else timings


def numerical_outputs(config, args):
    p, g, e, _t, s, a, _w, _k = args
    precision = config["dtype"]
    moments, errors = [], []
    for c in range(config["n_conditions"]):
        result = estimate_pauli_z_unchecked(
            p,
            g,
            e.select(c),
            a[c],
            samples=s[c : c + 1],
            precision=precision,
            return_samples=True,
        )
        moments.append(result.values[0])
        errors.append(result.standard_errors[0])
    loss, gradient = build_kernels(config)["prepared_mmd_gradient"](*args)
    return {
        "moments": np.asarray(jnp.stack(moments)),
        "standard_errors": np.asarray(jnp.stack(errors)),
        "loss": np.asarray(loss),
        "gradient": np.asarray(gradient),
    }
