# Benchmark comparison workflow

Fork this repository, modify the library implementations, and run the supplied correctness tests and benchmarks. Keep the benchmark code and fixture definitions fixed when comparing forks. Record any intentional benchmark or workload changes separately.

Read [the computational guide](COMPUTATIONAL_GUIDE.md) for the algorithm, array shapes, rules for reusable preparation, gradients, and training protocol. [README.md](README.md) documents the CLI.

## Verify on CPU

Use the existing repository environment or `uv sync --frozen`. Run:

```bash
.venv/bin/python -m pytest tests/test_conditional_pipeline*.py -q
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --cpu-check --seeds 1729 \
  --fixtures /tmp/iqp-cond-reference-fixtures \
  --output benchmarks/conditional_pipeline/results/cpu-check/measurements.jsonl
```

`--cpu-check` runs all eleven operations for three encoder families and two precision policies, followed by one short 16-qubit training fit. It records metrics for every item. Repeat correctness checks on the GPU used for measurements before accepting its results.

## Measure a selected workload and the full fit

```bash
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --cell --backend cpu --kernels phase,prepared_estimator,mmd_gradient \
  --fixtures /tmp/iqp-cond-reference-fixtures \
  --output benchmarks/conditional_pipeline/results/reference/operations.jsonl
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --training --backend cpu --seed 410 \
  --fixtures /tmp/iqp-cond-reference-fixtures \
  --output benchmarks/conditional_pipeline/results/reference/training.jsonl
```

Use `--backend gpu` for a cell or fit and `--backends gpu` or `cpu,gpu` for a campaign. `--training --dry-run` prints the paper fit's configuration without running it. `--training-smoke` runs the short fit. `--steps N` and its alias `--n-epochs N` specify optimizer updates; changed budgets are labelled custom.

The training default is `gemm_f32`, which reduces precision only in the final phase contraction. Use `--training-precision fp64` or `--training-precision estimator_f32` to measure the other policies. The operation suite uses `--dtype fp64|estimator_f32`.

## Compare a fork

Retain absolute paths to the reference JSONL and its output artifacts. Run the same command in the fork using the shared fixture directory and a fresh output:

```bash
PYTHONPATH=src .venv/bin/python -m benchmarks.conditional_pipeline.benchmark \
  --cell --backend cpu --kernels phase,prepared_estimator,mmd_gradient \
  --fixtures /tmp/iqp-cond-reference-fixtures \
  --compare-reference /absolute/path/to/reference/operations.jsonl \
  --output benchmarks/conditional_pipeline/results/candidate/operations.jsonl
```

The same comparison flag works for training and `--summary`. It matches fixture and backend identities, verifies artifact checksums, and compares compiled outputs across library versions. Missing or ambiguous references appear in the report. Precision errors require researcher review. Timing comparisons require the same host, software stack, and measurement policy, so a CPU reference cannot establish a GPU implementation speedup. The reported ratios are diagnostics and do not establish numerical approval or timing stability.

Save a separate reference output for each set of matched comparisons. Preserve inputs, initial parameters, update count, optimizer schedule, target pools, and validation and checkpoint rules. You may change dtypes, operations, and internal chunking; report the precision used and provide numerical evidence. Keep sampling inside the timed sampling-inclusive operations. Report any new reusable preparation separately.

Training comparisons reject differences in the fixed fit configuration, including optimizer learning rates, batch size, sampling budgets, conditions, initialization and validation schedule. Precision policy and sample chunking may differ and are recorded as implementation changes. Incomplete fits or missing configuration fields cannot produce training speedup comparisons.

The CLI and Markdown report show reference and candidate identities, diagnostic speedup ratios, numerical errors, and gradient sign changes. Training reports also compare exact outcome TVD values. A ratio above one means the candidate is faster. Review the numerical evidence before accepting a change.

Campaign selection, budgets, traces, and bundles are available. Resuming a campaign runs fresh controls and records them as a new run; earlier rows retain their original run. Keep the reference and candidate outputs, fixtures, and environment metadata together so comparisons can be checked.
