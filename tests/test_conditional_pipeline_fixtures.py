"""Focused tests for the conditional pipeline benchmark fixtures."""

import numpy as np
import pytest

from benchmarks.conditional_pipeline.fixtures import (
    configs,
    fixture_hash,
    generate_fixture,
    resolve_config,
)


def test_default_and_aliases_are_resolved() -> None:
    config = resolve_config({"sample_chunk": None})
    assert config["n_qubits"] == 64
    assert config["sample_chunk"] is None
    assert resolve_config({"encoder": "yz"})["encoder"] == "yz"


@pytest.mark.parametrize(
    "mode", ("smoke", "validate", "pilot", "core", "combined", "frontier", "full")
)
def test_campaign_configs_are_valid_and_deduplicated(mode: str) -> None:
    rows = configs(mode)
    identities = [tuple(sorted(row.items())) for row in rows]
    assert len(identities) == len(set(identities))
    for row in rows:
        resolve_config(row)
        if row["ansatz"] == "uniform":
            assert row["n_generators"] <= np.iinfo(np.int64).max


def test_fixture_is_deterministic_and_layers_are_independent() -> None:
    config = resolve_config(
        {
            "n_qubits": 8,
            "n_conditions": 2,
            "n_generators": 8,
            "arity": 2,
            "n_ops": 8,
            "n_samples": 12,
            "n_targets": 7,
            "encoder": "yz",
        }
    )
    first = generate_fixture(config)
    second = generate_fixture(config)
    assert fixture_hash(first) == fixture_hash(second)
    assert not np.array_equal(first["angles"][:, 0], first["angles"][:, 1])
    assert first["targets"].shape[1] == 7
    assert first["samples"].shape[1] == 12


def test_uniform_supports_are_unique_and_have_requested_arity() -> None:
    config = resolve_config({"n_qubits": 8, "n_generators": 20, "arity": 3})
    generators = generate_fixture(config)["generators"]
    assert np.all(generators.sum(axis=1) == 3)
    assert len({tuple(row) for row in generators}) == 20


def test_impossible_uniform_support_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique supports"):
        resolve_config({"n_qubits": 4, "n_generators": 5, "arity": 1})
