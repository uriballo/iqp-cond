"""Pass 1: rounding error against Monte Carlo noise at fixed parameters.

For each parameter point, operation and format (uniform variant, every other
operation in fp64) and each sample count ``S``, the loss and its gradient are
evaluated at ``K`` independent reference-sample draws, and so is fp64 at the
same draws.  ``metrics.noise_ratio`` turns the differences into ``rho_p`` and
its bias and spread, for the loss and for every gradient component.  The masks
are the cached arm's fixed bank.

Usage::

    python -m experiments.precision_study.screening OUT --n-qubits 12 \\
        --teacher-seed 1 --init-seed 1 [--run RUN_DIR ...] \\
        [--samples 1000 4000 16000 64000 256000] [--draws 32]

``--run`` adds the middle and final checkpoints of a saved fp64 run as
parameter points.  Results go to ``OUT/screening.csv`` and the settings, with
provenance, to ``OUT/screening.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import jax
import numpy as np
import yaml

import iqp_cond as ciqp
from iqp_cond.loss import mmd

from . import study
from .metrics import NoiseRatio, noise_ratio

KNOBS = ("gemm_inputs", "gemm_accumulate", "phases", "statistics", "ratios")
BLOCK_FORMATS = ("mxfp8", "mxfp4", "nvfp4")


def recipes():
    """Every (operation, format) pair below fp64, the rest of the step in fp64."""
    for knob in KNOBS:
        for name in ciqp.FORMATS:
            if name == "fp64" or (name in BLOCK_FORMATS and knob != "gemm_inputs"):
                continue
            yield knob, name, ciqp.precision(**{knob: name})


def parameter_points(config: study.RunConfig, runs: list[Path]):
    yield "start", study.initial_params(config)
    for folder in runs:
        _, arrays = study.load(folder)
        checkpoints = arrays["checkpoint_params"]
        yield f"{folder.name}:middle", checkpoints[len(checkpoints) // 2]
        yield f"{folder.name}:final", checkpoints[-1]


def loss_and_gradient(setup, recipe, n_samples):
    model = setup.model

    @jax.jit
    def evaluate(params, key):
        return jax.value_and_grad(
            lambda current: (
                mmd(
                    current,
                    model.generators,
                    model.encoder,
                    setup.targets,
                    key,
                    n_samples=n_samples,
                    precision=recipe,
                ).loss
            )
        )(params)

    return evaluate


def screen(config, runs, samples, draws, out: Path) -> None:
    setup = study.build(dataclasses.replace(config, mask_refresh="cache"))
    keys = [jax.random.key(index) for index in range(draws)]
    rows = []
    for point, params in parameter_points(config, runs):
        for n_samples in samples:
            start = time.time()
            reference = loss_and_gradient(setup, "fp64", n_samples)
            ref_loss, ref_grad = zip(
                *(reference(params, key) for key in keys), strict=True
            )
            ref_loss, ref_grad = np.array(ref_loss), np.array(ref_grad)
            for knob, name, recipe in recipes():
                evaluate = loss_and_gradient(setup, recipe, n_samples)
                loss, grad = zip(*(evaluate(params, key) for key in keys), strict=True)
                loss, grad = np.array(loss, float), np.array(grad, float)
                # A quantity without noise across the draws has no ratio; it is
                # recorded as NaN so one such case cannot stop the screening.
                nan = NoiseRatio(np.nan, np.nan, np.nan)
                noisy_loss = np.std(ref_loss, ddof=1) > 0
                loss_ratio = noise_ratio(loss, ref_loss) if noisy_loss else nan
                noisy = np.std(ref_grad, axis=0, ddof=1) > 0
                grad_ratios = [
                    noise_ratio(grad[:, j], ref_grad[:, j])
                    for j in np.flatnonzero(noisy)
                ] or [nan]
                rows.append(
                    {
                        "point": point,
                        "n_samples": n_samples,
                        "knob": knob,
                        "format": name,
                        "loss_rho": loss_ratio.rho,
                        "loss_bias": loss_ratio.bias,
                        "loss_spread": loss_ratio.spread,
                        "grad_rho_median": float(
                            np.median([r.rho for r in grad_ratios])
                        ),
                        "grad_rho_max": float(np.max([r.rho for r in grad_ratios])),
                        "grad_bias_rms": float(
                            np.sqrt(np.mean([r.bias**2 for r in grad_ratios]))
                        ),
                        "grad_spread_median": float(
                            np.median([r.spread for r in grad_ratios])
                        ),
                    }
                )
            print(
                f"screened {point} at S={n_samples} in {time.time() - start:.0f} s",
                flush=True,
            )
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "screening.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "config": dataclasses.asdict(config),
        "runs": [str(folder) for folder in runs],
        "samples": list(samples),
        "draws": draws,
        "provenance": study._provenance(),
    }
    (out / "screening.yaml").write_text(yaml.safe_dump(record, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--teacher-seed", type=int, default=1)
    parser.add_argument("--init-seed", type=int, default=1)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument(
        "--samples", type=int, nargs="+", default=[1000, 4000, 16000, 64000, 256000]
    )
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument(
        "--plan",
        type=Path,
        help="take the teacher and bank settings from this plan's defaults",
    )
    args = parser.parse_args()
    defaults = {}
    if args.plan:
        defaults = yaml.safe_load(args.plan.read_text()).get("defaults", {})
    config = study.RunConfig(
        **{
            **defaults,
            "n_qubits": args.n_qubits,
            "teacher_seed": args.teacher_seed,
            "init_seed": args.init_seed,
            "seed": 0,
        }
    )
    screen(config, args.run, args.samples, args.draws, args.out)


if __name__ == "__main__":
    main()
