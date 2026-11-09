"""Precision-study runs can be saved, reloaded and re-evaluated."""

from __future__ import annotations

from experiments.precision_study.study import RunConfig, evaluate, load, run, save

TINY = RunConfig(
    n_qubits=6,
    teacher_seed=1,
    seed=2,
    precision={"phases": "bf16"},
    mc_samples=16,
    mask_refresh="step",
    steps=4,
    validation_interval=2,
    n_ops=8,
    pool_draws=64,
    eval_masks=32,
    eval_samples=64,
)


def test_a_saved_run_reloads_and_reproduces_its_final_fp64_loss(tmp_path):
    result = run(TINY)
    save(tmp_path, TINY, result)
    config, arrays = load(tmp_path)
    assert config == TINY
    assert evaluate(config, arrays["final_params"]) == arrays["fp64_losses"][-1]
    assert (tmp_path / "NOTE.md").exists()
