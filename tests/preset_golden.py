"""Reference outputs of the reduced-precision presets and the chunked loss.

``python tests/preset_golden.py OUT.npz`` records the outputs of the installed
``iqp_cond``; the stored file was recorded from main at f37cc07, and the
regression test requires the presets to reproduce it bit for bit, and the
fp64 loss streamed in sample chunks to its last bits.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from fp64_golden import inputs

from iqp_cond import precision
from iqp_cond.loss import mmd

PRESETS = ("fp32", "bf16", "fp16", "fp8_e4m3")


def outputs() -> dict[str, np.ndarray]:
    data = inputs()
    model, params = data["model"], jnp.asarray(data["params"])

    def loss(current, recipe, n_samples=64, sample_chunk=None):
        return mmd(
            current,
            model.ansatz.generators,
            model.encoder,
            data["targets"],
            jax.random.key(3),
            sigmas=(1.0, 2.0),
            n_ops=16,
            n_samples=n_samples,
            precision=recipe,
            sample_chunk=sample_chunk,
        ).loss

    out = {}
    for name in PRESETS:
        for variant in ("uniform", "split"):
            value, gradient = jax.value_and_grad(loss)(
                params, precision(name, variant=variant)
            )
            out[f"{name}_{variant}_loss"] = np.asarray(value)
            out[f"{name}_{variant}_gradient"] = np.asarray(gradient)
    # Four whole chunks and a remainder of eight.
    value, gradient = jax.value_and_grad(loss)(params, "fp64", 72, 16)
    out["fp64_chunked_loss"] = np.asarray(value)
    out["fp64_chunked_gradient"] = np.asarray(gradient)
    return out


if __name__ == "__main__":
    np.savez(sys.argv[1], **outputs())
