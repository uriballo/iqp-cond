"""Training targets given as exact moments instead of samples."""

from __future__ import annotations

import jax
import numpy as np
from fp64_golden import inputs

from iqp_cond.loss import TargetMoments, mmd
from iqp_cond.operators import operator_bank


def test_moment_targets_drop_only_the_finite_sample_correction():
    data = inputs()
    model, targets = data["model"], data["targets"]
    bank = operator_bank((1.0, 2.0), 16, targets.shape[0], targets.shape[2], seed=4)
    # Empirical moments of the target samples, one per bank mask.
    parities = np.einsum("ctn,wcmn->wctm", targets, np.asarray(bank.masks)) % 2
    characters = 1.0 - 2.0 * parities
    moments = characters.mean(axis=2)
    n = targets.shape[1]

    def loss(target_source, **options):
        return float(
            mmd(
                data["params"],
                model.ansatz.generators,
                model.encoder,
                target_source,
                jax.random.key(3),
                n_samples=64,
                **options,
            ).loss
        )

    from_moments = loss(TargetMoments(bank, moments))
    from_samples = loss(targets, bank=bank)
    # Exact moments enter as mu^2; samples as the unbiased (n mu^2 - 1)/(n - 1).
    # The difference, (1 - mu^2) / (n - 1) per mask, is weighted and averaged
    # over the (bandwidth, condition) cells.
    weights = np.asarray(bank.weights)
    correction = np.sum(weights * (1.0 - moments**2) / (n - 1), axis=-1).mean()
    np.testing.assert_allclose(from_moments - from_samples, correction, rtol=1e-10)
