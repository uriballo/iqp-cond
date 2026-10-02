# Computational guide for implementation work

This suite provides library operations and one complete training workload. The reference implementation is ordinary JAX. Correctness tests and performance benchmarks are separate.

## The computation

A condition selects a product-state encoder. The same IQP parameter vector is shared across conditions. The computational circuit is encoder preparation, Hadamards, a diagonal phase, and final Hadamards. For a bitstring `z`, the phase is `f(z) = sum_j theta[j] * (-1) ** dot(g[j], z)`, with binary dot products interpreted modulo two. A generator row specifies a qubit support. Generators can have arbitrary arity; the complete ansatz contains all single supports followed by all two-qubit supports.

The encoder provides samples from the post-Hadamard state's Born distribution and amplitude ratios `psi(z XOR a) / psi(z)`. The IQP unitary contributes `exp(i * (f(z XOR a) - f(z)))`. Averaging the real part of their product over reference samples estimates the output Pauli-Z moment for binary mask `a`. Masks identify Fourier characters of binary data, rather than physical hardware operations. An identity mask contributes zero to MMD and is omitted.

Maximum mean discrepancy, or MMD, compares model and target moments. Its model and target square terms use finite-sample U-statistic corrections. For per-sample estimates `q[s]`, the model square term is `((sum_s q[s])**2 - sum_s q[s]**2) / (S * (S - 1))`. The target term uses the same correction for its binary Fourier characters. Each mask contributes the model square term, minus twice the product of model and target means, plus the target square term. Mask weights account for the omitted identity, and the objective averages over conditions and bandwidths. Replacing this with squared empirical means changes the computation.

## Array and operation contracts

Let `C` be conditions, `N` qubits, `G` generators, `S` reference samples, `R` masks, and `T` target examples. Parameters have shape `(G,)`, generator supports `(G,N)`, samples `(C,S,N)`, masks `(C,R,N)`, weights `(C,R)`, and targets `(C,T,N)`. Bitstrings use little-endian indexing for exact state arrays. The rotation families are `y`, `yz`, and `yzyz`.

| Benchmark item | Result | Work included |
|---|---|---|
| `encoder_sampling` | `(C,S,N)` binary samples | Sampling from fixed encoder and supplied key |
| `amplitude_ratios` | `(C,S,R)` ratios | Stored samples and masks |
| `phase` | `(C,S,R)` phase differences | Stored samples and masks |
| `phase_gradient` | scalar and `(G,)` gradient | Differentiate mean of phase tensor |
| `prepared_estimator` | `(C,R)` moments | Ratios, phases and reduction, fused |
| `prepared_estimator_gradient` | scalar and `(G,)` gradient | Differentiate mean of moments |
| `sampling_estimator` | `(C,R)` moments | Draw samples, then fused estimation |
| `prepared_mmd` | scalar | Stored draws, target moments and U-statistic |
| `prepared_mmd_gradient` | scalar and `(G,)` gradient | Differentiate prepared MMD |
| `mmd` | scalar | Draw masks/samples and compute MMD |
| `mmd_gradient` | scalar and `(G,)` gradient | Differentiate sampling-inclusive MMD |
| training | histories, selected and final parameters | One public-library training fit |

Only IQP parameters are differentiated in this baseline. Encoder construction is timed separately. Inputs remain arguments to compiled operations; parameters, targets and draws must be changeable without stale results. Fixed generator supports and encoders may have reusable preparation, with its time and memory reported. Diagnostic component times do not add up to fused execution time.

The baseline applies `sample_chunk` to `prepared_mmd`, `mmd`, and their gradients, and to training. It accumulates sample sums and squared sums across chunks before applying the U-statistic correction. The standalone phase and estimator operations use the full sample array.

## Correctness and precision

The small-case tests compare actual JIT-compiled operation results against independent NumPy formulas and PennyLane circuits. They also change inputs, exercise chunk boundaries, and detect finite but incorrect output. Large performance cases are not independent exact correctness proofs.

The benchmark retains the first execution's actual compiled outputs in a checked NPZ artifact. Cross-fork comparisons report maximum absolute error, RMS error, relative L2 error, actual output dtypes, and gradient sign changes. Sign changes exclude reference gradients below a documented magnitude floor of `1e-8`. `allclose_fp64` is an informational strict-reference diagnostic. Precision choices and their numerical evidence require researcher review; a low error on one fixture does not automatically approve a precision policy.

Sampling with the existing JAX Bernoulli algorithm has a fixed-key contract. Tests also check the product sampling law. A replacement RNG algorithm that changes individual draws needs distributional evidence; its sample-dependent outputs cannot be interpreted as a pure paired roundoff comparison.

Historical `validity` fields retain the original fp64/mixed-precision checks for the campaign collector. New per-operation `compiled_checks` and `reference_comparison` describe the actual executable outputs. These scopes must not be conflated.

## One training fit

The paper workload has 16 qubits, no ancillas, a complete 136-parameter ansatz, and fixed `RY(pi/2), RZ(0.75 * beta)` encoders. Training conditions are `0.6, 0.8, 1.2, 1.4`; development conditions are `0.7, 1.3`; outcome diagnostics also use `0.4, 1.0, 1.6`. Curie-Weiss targets have shell masses proportional to `choose(N,k) * exp(beta * (2k-N)^2 / (2N))`, with uniform bitstrings within a shell.

The fit uses 2,000 updates, Adam with a cosine learning rate from `0.01` to `0.001`, batch size 450, 256 masks, 8,192 reference samples, chunk size 256, and mask bandwidths targeting mean weights 2 and 4 before omitting identity masks. Each training condition has 10,000 target examples and 2,000 validation examples. Initial parameters are normal draws with standard deviation 0.5 and RNG seed `seed + 101`. Validation occurs every 25 updates using 256 masks and 16,384 samples. Exact development-profile TVD selects checkpoints, matching the paper protocol. One seed, one initialization, and one restart are used. The default is seed 410 and the paper's `gemm_f32` precision policy.

`total_training` includes fixture preparation/loading, transfers, compilation, updates, prescribed validation/selection, and checkpoint artifact writing. Backend initialization is recorded in process runtime outside that total. Final best/final raw-probability and profile TVD diagnostics are timed separately. The benchmark observer measures loop wall time without synchronizing every update, subtracting compilation, validation and selection. It is supplementary and falls back to unavailable when a fork changes the trainer internals.

Raw-probability TVD compares complete bitstring distributions. Profile TVD compares their total probability at each Hamming weight. Both use exact probabilities; a close profile alone does not establish a close bitstring distribution.

The smoke profile retains the 16-qubit model and condition sets but uses two updates and small sampling/data budgets. It verifies execution and metrics; it is not a measurement of the full paper configuration. These runs use the current library, not the archived paper binary, and do not claim to reproduce its historical selected checkpoint or TVD.
