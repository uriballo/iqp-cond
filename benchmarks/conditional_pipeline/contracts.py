"""Independent fixed-draw references and diagnostics for compiled operations.

The mathematical oracle uses NumPy, not the estimator or loss implementation.
Sampling pins JAX's Bernoulli key mapping for the current reference sampler.
"""

from __future__ import annotations

import numpy as np

LABELS = (
    "encoder_sampling",
    "amplitude_ratios",
    "phase",
    "phase_gradient",
    "prepared_estimator",
    "prepared_estimator_gradient",
    "sampling_estimator",
    "prepared_mmd",
    "prepared_mmd_gradient",
    "mmd",
    "mmd_gradient",
)


def local_amplitudes(angles, encoder):
    angles = np.asarray(angles)
    zero = np.ones(angles.shape[::2], dtype=np.complex128)
    one = np.zeros_like(zero)
    for layer, axis in enumerate({"y": "Y", "yz": "YZ", "yzyz": "YZYZ"}[encoder]):
        a = angles[:, layer]
        if axis == "Y":
            c, s = np.cos(a / 2), np.sin(a / 2)
            zero, one = c * zero - s * one, s * zero + c * one
        else:
            zero, one = zero * np.exp(-0.5j * a), one * np.exp(0.5j * a)
    return (zero + one) / np.sqrt(2), (zero - one) / np.sqrt(2)


def _condition(params, generators, zero, one, samples, masks):
    samples, masks = np.asarray(samples), np.asarray(masks)
    denominator = np.where(samples[:, None, :] == 0, zero, one)
    numerator = np.where(samples[:, None, :] == 0, one, zero)
    local = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )
    ratios = np.prod(np.where(masks[None, :, :] == 1, local, 1), axis=-1)
    characters = 1 - 2 * ((samples @ generators.T) % 2)
    flips = (masks @ generators.T) % 2
    derivative = -2 * characters[:, None, :] * flips[None, :, :]
    phase = np.einsum("srg,g->sr", derivative, params)
    complex_values = ratios * np.exp(1j * phase)
    values = np.real(complex_values)
    gradients = np.real(1j * complex_values)[..., None] * derivative
    return ratios, phase, values, gradients, derivative


def reference_outputs(config, fixture):
    """Array/tuple outputs for all eleven operations at the same fixed draws."""
    import jax
    import jax.numpy as jnp

    p, g = np.asarray(fixture["params"]), np.asarray(fixture["generators"])
    samples, masks = np.asarray(fixture["samples"]), np.asarray(fixture["masks"])
    zero, one = local_amplitudes(fixture["angles"], config["encoder"])
    condition = [
        _condition(p, g, zero[c], one[c], samples[c], masks[c])
        for c in range(len(zero))
    ]
    ratios, phases, values, gradients, derivatives = (
        np.stack(x) for x in zip(*condition, strict=True)
    )
    moments = values.mean(axis=1)
    weights, targets = np.asarray(fixture["weights"]), np.asarray(fixture["targets"])
    target_values = 1 - 2 * (np.einsum("ctn,crn->ctr", targets, masks) % 2)
    ns, nt = samples.shape[1], targets.shape[1]
    sums, gsums = values.sum(axis=1), gradients.sum(axis=1)
    model_u = (sums**2 - (values**2).sum(axis=1)) / (ns * (ns - 1))
    model_du = (
        2 * sums[..., None] * gsums - 2 * (values[..., None] * gradients).sum(axis=1)
    ) / (ns * (ns - 1))
    target_u = (target_values.sum(axis=1) ** 2 - (target_values**2).sum(axis=1)) / (
        nt * (nt - 1)
    )
    loss = np.mean(
        np.sum(
            weights * (model_u - 2 * moments * target_values.mean(axis=1) + target_u),
            axis=1,
        )
    )
    loss_gradient = np.mean(
        np.sum(
            weights[..., None]
            * (
                model_du
                - 2 * gradients.mean(axis=1) * target_values.mean(axis=1)[..., None]
            ),
            axis=1,
        ),
        axis=0,
    )
    # The prepared draws reproduce the one-band public MMD split.
    key = jax.random.wrap_key_data(jnp.asarray(fixture["key_data"]))
    drawn = np.asarray(
        jax.random.bernoulli(
            key,
            p=jnp.asarray(abs(one) ** 2)[:, None, :],
            shape=(len(zero), config["n_samples"], config["n_qubits"]),
        ),
        dtype=float,
    )
    sampled_values = np.stack(
        [
            _condition(p, g, zero[c], one[c], drawn[c], masks[c])[2].mean(axis=0)
            for c in range(len(zero))
        ]
    )
    return {
        "encoder_sampling": drawn,
        "amplitude_ratios": ratios,
        "phase": phases,
        "phase_gradient": (np.mean(phases), np.mean(derivatives, axis=(0, 1, 2))),
        "prepared_estimator": moments,
        "prepared_estimator_gradient": (
            np.mean(moments),
            np.mean(gradients, axis=(0, 1, 2)),
        ),
        "sampling_estimator": sampled_values,
        "prepared_mmd": np.asarray(loss),
        "prepared_mmd_gradient": (np.asarray(loss), loss_gradient),
        "mmd": np.asarray(loss),
        "mmd_gradient": (np.asarray(loss), loss_gradient),
    }


def flatten_kernel_outputs(outputs):
    flat = {}
    for label, value in outputs.items():
        for i, leaf in enumerate(value if isinstance(value, tuple) else (value,)):
            flat[f"{label}__{i}"] = np.asarray(leaf)
    return flat


def compare_kernel_results(reference, candidate):
    """Report precision error; numerical acceptance is left to the reviewer."""
    x = (
        reference
        if all("__" in k for k in reference)
        else flatten_kernel_outputs(reference)
    )
    y = (
        candidate
        if all("__" in k for k in candidate)
        else flatten_kernel_outputs(candidate)
    )
    errors, failures = {}, []
    for key in sorted(set(x) | set(y)):
        if key not in x or key not in y:
            failures.append(f"missing:{key}")
            continue
        ref, got = np.asarray(x[key]), np.asarray(y[key])
        if ref.shape != got.shape:
            failures.append(f"shape:{key}")
            continue
        if not (np.isfinite(ref).all() and np.isfinite(got).all()):
            failures.append(f"nonfinite:{key}")
            continue
        delta = abs(got - ref)
        gradient = key.endswith("__1")
        meaningful = abs(ref) > 1e-8
        real = not (np.iscomplexobj(ref) or np.iscomplexobj(got))
        errors[key] = {
            "max_abs": float(np.max(delta, initial=0)),
            "rms": float(np.sqrt(np.mean(delta**2))),
            "relative_l2": float(
                np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-8)
            ),
            "reference_dtype": str(ref.dtype),
            "candidate_dtype": str(got.dtype),
            "gradient_sign_flips": int(
                np.sum(meaningful & (np.signbit(got) != np.signbit(ref)))
            )
            if gradient and real
            else None,
            "gradient_sign_floor": 1e-8 if gradient else None,
            "allclose_fp64": bool(np.allclose(got, ref, rtol=1e-8, atol=1e-9)),
        }
    return {
        "status": "failed" if failures else "compared",
        "failures": failures,
        "errors": errors,
    }
