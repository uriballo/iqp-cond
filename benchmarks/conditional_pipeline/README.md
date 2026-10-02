# Conditional IQP pipeline benchmark

This benchmark measures conditional IQP operations and one fixed training fit so you can compare library changes in a fork. It covers encoder construction and sampling, IQP phases, fused estimation, MMD and IQP gradients, and the total time for the paper's 16-qubit Curie-Weiss fit. Correctness tests are separate from performance measurements.

Read the [computational guide](COMPUTATIONAL_GUIDE.md) for the algorithm and array shapes, and the [comparison workflow](HANDOFF.md) for shared fixtures and reference runs.

## Workloads

The operation suite uses product encoders `y`, `yz`, and `yzyz`, two precision policies, `fp64` and `estimator_f32`, and 11 timed operations. The full matrix has 198 distinct configurations. Its stages contain 144 core, 32 combined, 16 frontier, and 12 pilot configurations, with some overlap between stages.

The component anchor uses 64 qubits, one condition, 64 arity-2 generators, 512 masks, 1,024 reference samples, 1,024 targets, and sample chunk size 256. Core sweeps vary one setting at a time:

| Setting | Values |
|---|---|
| Qubits | 16, 32, 64, 128, 256 |
| Conditions | 1, 4, 16 |
| Generators | 16, 64, 256, 1,024 |
| Generator arity | 1, 2, 4, 8 |
| Masks | 128, 512, 2,048 |
| Reference samples | 256, 1,024, 4,096 |
| Targets | 256, 1,024, 4,096 |
| Mask bandwidth targets | Mean weights 2, `n/4`, `0.49n` before omitting identity masks |
| Sample chunk size | None, 64, 256, 1,024 |

Combined probes vary several settings together. Frontier probes use 512 or 1,000 qubits with uniform or complete ansatzes. Sample chunking applies to the MMD operations and their gradients; the standalone phase and estimator operations use the full sample array. Use `--dry-run` to inspect the resolved configurations.

## Check on CPU

Verify all benchmark items on CPU, including a short training fit:

```bash
.venv/bin/python -m pytest tests/test_conditional_pipeline*.py -q
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --cpu-check --seeds 1729 \
  --output benchmarks/conditional_pipeline/results/cpu-check/measurements.jsonl
```

The CLI prints operation timing tables and a separate training summary. Warm timings use microseconds, compilation uses milliseconds, and training uses seconds. Each cell retains its configuration and numerical-check status. The JSONL is the raw record. Read `report/report.md` for the full tables, `report/summary.csv` for component timings and configuration, and `report/training.csv` for training timings. Add `--json` for machine-readable CLI completion output.

Regenerate readable results from an existing run without executing benchmarks:

```bash
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --summary --input benchmarks/conditional_pipeline/results/cpu-check/measurements.jsonl \
  --artifacts benchmarks/conditional_pipeline/results/cpu-check
```

## Training

`--training --backend cpu|gpu` runs one full 2,000-update fit with the paper configuration. The default precision policy is `gemm_f32`; `--training-precision` also accepts `fp64` and `estimator_f32`. `--training-smoke` keeps the 16-qubit model but reduces the data and sampling budgets and runs two updates. `--steps N`, also `--n-epochs N`, means optimizer updates. Changed update budgets are labelled custom. Use `--training --dry-run` to inspect the configuration.

Training records retain total, loop, compile, and validation timings, histories, selected and final parameters, and exact outcome diagnostics. The primary total includes preparation, transfers, compilation, updates, prescribed validation and checkpoint writing. Final outcome diagnostics have a separate timer.

## Campaigns and comparisons

Run a local smoke or validation pass with:

```bash
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --smoke --backends cpu --seeds 1729 \
  --output benchmarks/conditional_pipeline/results/smoke.jsonl
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --validate --backends cpu --seeds 1729
```

`--pilot` uses a two-hour budget and `--full` uses a 24-hour budget. The default timing policy is two warmups, ten timed repetitions, three fixture seeds, and a 20-minute cell timeout. Smoke mode uses one warmup and two repeats. `--cell` measures one configuration and writes a JSONL record.

Use `--backend cpu|gpu` for a cell or fit and `--backends cpu,gpu` for a campaign. `--seeds`, `--select key=value`, and `--kernels` choose the work to measure. `--max-cells`, `--budget-seconds`, and `--cell-timeout` limit execution. `--output` sets the JSONL path and `--artifacts` sets the artifact directory. An unavailable backend produces a failure record.

`--fixtures DIRECTORY` shares frozen inputs across forks. `--compare-reference /absolute/path/reference.jsonl` compares actual compiled outputs and reports numerical errors, gradient sign changes, and diagnostic timing ratios. Precision choices require researcher review. Timing comparisons require the same host, software stack, and measurement policy.

`--trace` records selected profiler traces and `--bundle` packages source and evidence. `--resume` reuses matching completed records and runs fresh campaign controls. `--summary` regenerates reports from recorded results.

Records retain resolved configurations, fixture and source hashes, environment metadata, timings, memory observations, numerical checks, and failure or capacity status. The optional analysis layer uses Matplotlib when installed; measurement collection does not require it.

The CPU checks cover the operations and short training profile. The full 2,000-update fit and GPU measurements remain unverified.
