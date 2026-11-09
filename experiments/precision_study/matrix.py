"""Passes 2-4: expand a plan into training runs and run one, or all.

A plan is a YAML file::

    name: smoke
    defaults: {teacher_seed: 1, steps: 200}     # RunConfig fields for every run
    grid: {n_qubits: [12, 64], mc_samples: [1000], mask_refresh: [step, cache],
           init_seed: [1, 2]}                    # one run per combination ...
    recipes:                                     # ... per recipe ...
      fp64: {}
      bf16: {base: bf16}
      bf16_fp32_stats: {base: bf16, statistics: fp32}
    variants: [uniform, split]                   # ... per variant (fp64: uniform)
    baselines: true                              # plus the two fp64 chaos baselines
    eval_every: {128: 4}                         # fp64 evaluation stride by n

A recipe gives ``base`` (the format of every operation) and per-operation
overrides.  Runs use the init seed as their sampling seed; the baselines are
fp64 runs from the same initial parameters with the sampling seed moved by
1000, and with every initial parameter nudged by one float32 ulp.

Usage::

    python -m experiments.precision_study.matrix PLAN --list
    python -m experiments.precision_study.matrix PLAN --prepare
    python -m experiments.precision_study.matrix PLAN --index I
    python -m experiments.precision_study.matrix PLAN --all

``--prepare`` builds and caches every teacher's banks, moments and samples.
Runs are saved under ``runs/<plan name>/<run name>/`` and skipped if present.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import time
from pathlib import Path

import yaml

import iqp_cond as ciqp

from . import study

RUNS = Path(__file__).parent / "runs"
_PRECISION_FIELDS = ("gemm_inputs", "gemm_accumulate", "phases", "statistics", "ratios")


def _precision(spec: dict, variant: str) -> dict:
    spec = dict(spec)
    recipe = ciqp.precision(spec.pop("base", "fp64"), variant=variant, **spec)
    fields = dataclasses.asdict(recipe)
    if variant == "uniform":
        del fields["variant"]
    return {key: value for key, value in fields.items() if value != "fp64"}


def expand(plan: dict) -> list[tuple[str, study.RunConfig]]:
    """Return every run of the plan as ``(name, RunConfig)``."""
    grid = plan.get("grid", {})
    eval_every = {int(n): every for n, every in plan.get("eval_every", {}).items()}
    runs = []
    for values in itertools.product(*grid.values()):
        cell = dict(zip(grid, values, strict=True))
        settings = {**plan.get("defaults", {}), **cell}
        n = settings["n_qubits"]
        settings.setdefault("eval_every", eval_every.get(n, 1))
        settings.setdefault("seed", settings["init_seed"])
        tag = (
            f"n{n}-S{settings.get('mc_samples', 1000)}-"
            f"{settings.get('mask_refresh', 'step')}-init{settings['init_seed']}"
        )
        for recipe, spec in plan["recipes"].items():
            fp64 = not _precision(spec, "uniform")
            for variant in ["uniform"] if fp64 else plan.get("variants", ["uniform"]):
                config = study.RunConfig(
                    **settings, precision=_precision(spec, variant)
                )
                runs.append((f"{recipe}-{variant}-{tag}", config))
        if plan.get("baselines", True):
            base = study.RunConfig(**settings)
            shifted = dataclasses.replace(base, seed=base.seed + 1000)
            runs.append((f"fp64-uniform-{tag}-seedshift", shifted))
            runs.append(
                (f"fp64-uniform-{tag}-ulp", dataclasses.replace(base, theta0_ulp=True))
            )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plan", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--index", type=int)
    action.add_argument("--all", action="store_true")
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    runs = expand(plan)
    if args.list:
        for index, (name, _) in enumerate(runs):
            print(index, name)
        print(f"{len(runs)} runs")
        return
    if args.prepare:
        for setup in {study._shared(config) for _, config in runs}:
            start = time.time()
            study.build(setup)
            print(
                f"prepared {setup.n_qubits} qubits, {setup.mask_refresh} "
                f"in {time.time() - start:.0f} s",
                flush=True,
            )
        return
    selected = runs if args.all else [runs[args.index]]
    for name, config in selected:
        folder = RUNS / plan["name"] / name
        if (folder / "run.npz").exists():
            print(f"skip {name}: done", flush=True)
            continue
        start = time.time()
        result = study.run(config)
        trained = time.time() - start
        study.save(folder, config, result)
        print(
            f"{name}: trained in {trained:.0f} s, saved in "
            f"{time.time() - start - trained:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
