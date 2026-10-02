"""Numerical acceptance checks for the conditional-pipeline benchmark.

The functions in this module deliberately operate on plain mappings so that
benchmark workers can serialize their results without importing the model
runner.  They do not alter production tolerances or APIs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    return np.asarray(value)


def compare_outputs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: str = "fp64",
) -> dict[str, Any]:
    """Compare benchmark outputs and return JSON-serializable diagnostics."""
    if policy not in {"fp64", "estimator_f32"}:
        raise ValueError("policy must be 'fp64' or 'estimator_f32'")
    errors: dict[str, Any] = {}
    failures: list[str] = []
    required = ("moments", "standard_errors", "loss", "gradient")
    for name in required:
        if name not in reference or name not in candidate:
            failures.append(f"missing:{name}")
            continue
        ref, got = _array(reference[name]), _array(candidate[name])
        if ref.shape != got.shape:
            failures.append(f"shape:{name}")
            errors[name] = {
                "reference_shape": list(ref.shape),
                "candidate_shape": list(got.shape),
            }
            continue
        if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(got))):
            failures.append(f"nonfinite:{name}")
            continue
        delta = np.abs(got - ref)
        scale = np.maximum(np.abs(ref), 1e-300)
        errors[name] = {
            "max_abs": float(np.max(delta, initial=0.0)),
            "rms": float(np.sqrt(np.mean(delta * delta))),
            "max_rel": float(np.max(delta / scale, initial=0.0)),
        }
    if (
        "moments" in errors
        and "shape:moments" not in failures
        and "nonfinite:moments" not in failures
    ):
        if policy == "fp64":
            ok = np.allclose(
                reference["moments"], candidate["moments"], atol=1e-9, rtol=1e-8
            )
        else:
            ok = errors["moments"]["max_abs"] <= 1e-5
        if not ok:
            failures.append("gate:moments")
    if (
        "loss" in errors
        and "shape:loss" not in failures
        and "nonfinite:loss" not in failures
    ):
        tol = (
            1e-9 + 1e-8 * abs(float(np.asarray(reference["loss"])))
            if policy == "fp64"
            else 1e-5 + 1e-4 * abs(float(np.asarray(reference["loss"])))
        )
        if errors["loss"]["max_abs"] > tol:
            failures.append("gate:loss")
    if (
        "gradient" in errors
        and "shape:gradient" not in failures
        and "nonfinite:gradient" not in failures
    ):
        if policy == "fp64":
            ok = np.allclose(
                reference["gradient"], candidate["gradient"], atol=1e-9, rtol=1e-8
            )
        else:
            refnorm = max(float(np.linalg.norm(_array(reference["gradient"]))), 1e-8)
            errors["gradient"]["relative_l2"] = float(
                np.linalg.norm(
                    _array(candidate["gradient"]) - _array(reference["gradient"])
                )
                / refnorm
            )
            ok = errors["gradient"]["relative_l2"] <= 1e-3
        if not ok:
            failures.append("gate:gradient")
    if (
        "moments" in errors
        and "standard_errors" in errors
        and "shape:moments" not in failures
        and "nonfinite:moments" not in failures
    ):
        ref_se = _array(reference["standard_errors"])
        if ref_se.shape == _array(reference["moments"]).shape:
            errors["moments"]["precision_over_reference_se_rms"] = float(
                np.sqrt(
                    np.mean(
                        (
                            (
                                _array(candidate["moments"])
                                - _array(reference["moments"])
                            )
                            / np.maximum(ref_se, 1e-12)
                        )
                        ** 2
                    )
                )
            )
    return {
        "status": "failed" if failures else "passed",
        "errors": errors,
        "failures": failures,
        "policy": policy,
    }


def independent_checks(config, fixture, encoder):
    """Independent circuit moments, ratio identities and fixed-draw gradients."""
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    from iqp_cond.estimator import estimate_pauli_z_unchecked
    from iqp_cond.loss import mmd

    from .workloads import AXES, prepared_mmd

    n = config["n_qubits"]
    if n > 8:
        return {"status": "unverified", "reason": "exact checks limited to <=8 qubits"}
    failures, checks = [], {}
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    # QNode state vectors use wire zero as most-significant; fixtures use little endian.
    indices = (bits @ (2 ** np.arange(n - 1, -1, -1))).astype(int)
    p, g = jnp.asarray(fixture["params"]), jnp.asarray(fixture["generators"])
    t, samples = jnp.asarray(fixture["targets"]), jnp.asarray(fixture["samples"])
    masks, weights = jnp.asarray(fixture["masks"]), jnp.asarray(fixture["weights"])
    moment_error, ratio_error = 0.0, 0.0
    for c in range(config["n_conditions"]):
        dev = qml.device("default.qubit", wires=n)
        angles = np.asarray(fixture["angles"])[c]

        @qml.qnode(dev)
        def circuit(theta, full, angles=angles):
            for layer, axis in zip(angles, AXES[config["encoder"]], strict=True):
                gate = {"Y": qml.RY, "Z": qml.RZ}[axis]
                for wire, angle in enumerate(layer):
                    gate(float(angle), wires=wire)
            for wire in range(n):
                qml.Hadamard(wires=wire)
            if full:
                for theta_j, support in zip(theta, np.asarray(g), strict=True):
                    wires = np.flatnonzero(support).tolist()
                    qml.MultiRZ(-2 * float(theta_j), wires=wires)
                for wire in range(n):
                    qml.Hadamard(wires=wire)
            return qml.state()

        psi = np.asarray(circuit(np.asarray(p), False))[indices]
        final = np.asarray(circuit(np.asarray(p), True))[indices]
        normalization = abs(np.sum(abs(psi) ** 2) - 1)
        checks[f"normalization_{c}"] = float(normalization)
        if normalization > 1e-10:
            failures.append(f"normalization_{c}")
        selected_masks = np.asarray(masks[c])
        chars = 1 - 2 * np.mod(bits @ selected_masks.T, 2)
        exact = (abs(final) ** 2) @ chars
        est = estimate_pauli_z_unchecked(
            p,
            g,
            encoder.select(c),
            masks[c],
            samples=jnp.asarray(bits[None]),
            return_samples=True,
        )
        enumerated = (abs(psi) ** 2) @ np.asarray(est.sample_values)[0]
        moment_error = max(moment_error, float(np.max(abs(exact - enumerated))))
        if not np.allclose(exact, enumerated, atol=1e-10, rtol=1e-9):
            failures.append(f"exact_moments_{c}")
        supported = np.flatnonzero(abs(psi) > 1e-12)
        flipped = (
            supported[:, None]
            ^ (selected_masks @ (2 ** np.arange(n))).astype(int)[None]
        )
        expected = psi[flipped] / psi[supported, None]
        actual = np.asarray(
            encoder.select(c).amplitude_ratios(
                jnp.asarray(bits[supported][None]), masks[c]
            )
        )[0]
        ratio_error = max(ratio_error, float(np.max(abs(expected - actual))))
        if not np.allclose(actual, expected, atol=1e-10, rtol=1e-9):
            failures.append(f"ratios_{c}")
        if not np.allclose(
            np.asarray(encoder.post_hadamard_amplitudes())[c],
            psi,
            atol=1e-10,
            rtol=1e-9,
        ):
            failures.append(f"reference_state_{c}")
    checks.update(
        exact_moments_max_abs=moment_error, amplitude_ratios_max_abs=ratio_error
    )

    def fixed(theta):
        return prepared_mmd(theta, g, encoder, t, samples, masks, weights, config)

    value, gradient = jax.value_and_grad(fixed)(p)
    fd = np.zeros_like(np.asarray(p))
    for i in range(len(fd)):
        lo, hi = np.asarray(p).copy(), np.asarray(p).copy()
        lo[i] -= 1e-5
        hi[i] += 1e-5
        fd[i] = (float(fixed(jnp.asarray(hi))) - float(fixed(jnp.asarray(lo)))) / 2e-5
    checks["finite_difference_gradient_max_abs"] = float(
        np.max(abs(np.asarray(gradient) - fd))
    )
    if not np.allclose(gradient, fd, atol=1e-7, rtol=1e-5):
        failures.append("finite_difference_gradient")
    key = jax.random.wrap_key_data(jnp.asarray(fixture["key_data"], dtype=jnp.uint32))

    def public(theta):
        return mmd(
            theta,
            g,
            encoder,
            t,
            key,
            sigmas=(float(fixture["sigma"]),),
            n_ops=config["n_ops"],
            n_samples=config["n_samples"],
            shared_samples=True,
            sample_chunk=config["sample_chunk"],
        ).loss

    public_value, public_gradient = jax.value_and_grad(public)(p)
    checks["prepared_public_loss_abs"] = float(abs(value - public_value))
    checks["prepared_public_gradient_max_abs"] = float(
        np.max(abs(np.asarray(gradient - public_gradient)))
    )
    if not np.allclose(value, public_value, atol=1e-10, rtol=1e-9):
        failures.append("prepared_public_loss")
    if not np.allclose(gradient, public_gradient, atol=1e-10, rtol=1e-9):
        failures.append("prepared_public_gradient")
    return {
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
    }
