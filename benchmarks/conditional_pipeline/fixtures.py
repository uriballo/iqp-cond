"""Deterministic workload definitions and host-side benchmark fixtures."""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any

import jax
import numpy as np

import iqp_cond as ciqp
from iqp_cond.operators import expected_bodyness_sigmas, sample_nonidentity_masks_jax

ENCODERS = {"y": ("Y",), "yz": ("Y", "Z"), "yzyz": ("Y", "Z", "Y", "Z")}
PRECISIONS = ("fp64", "estimator_f32")
_DEFAULTS = dict(
    n_qubits=64,
    n_conditions=1,
    n_generators=64,
    arity=2,
    ansatz="uniform",
    encoder="y",
    dtype="fp64",
    n_ops=512,
    n_samples=1024,
    n_targets=1024,
    sample_chunk=256,
    mask_band="low",
    seed=1729,
    repeats=10,
    warmups=2,
)


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(_DEFAULTS)
    out.update(config or {})
    if out["sample_chunk"] is not None:
        out["sample_chunk"] = int(out["sample_chunk"])
    for key in (
        "n_qubits",
        "n_conditions",
        "n_generators",
        "arity",
        "n_ops",
        "n_samples",
        "n_targets",
        "seed",
        "repeats",
        "warmups",
    ):
        out[key] = int(out[key])
    if out["n_qubits"] < 1 or out["n_conditions"] < 1 or out["n_generators"] < 1:
        raise ValueError("sizes must be positive")
    if out["n_ops"] < 1 or out["n_samples"] < 1 or out["n_targets"] < 1:
        raise ValueError("sample counts must be positive")
    if out["repeats"] < 1 or out["warmups"] < 0:
        raise ValueError("invalid repeat counts")
    if out["sample_chunk"] is not None and out["sample_chunk"] < 1:
        raise ValueError("sample_chunk must be positive or None")
    if out["arity"] < 1 or out["arity"] > out["n_qubits"]:
        raise ValueError("arity must be between one and n_qubits")
    if out["ansatz"] == "uniform" and out["n_generators"] > _comb(
        out["n_qubits"], out["arity"]
    ):
        raise ValueError("n_generators exceeds the number of unique supports")
    if out["encoder"] not in ENCODERS or out["dtype"] not in PRECISIONS:
        raise ValueError("unknown encoder or dtype")
    if out["mask_band"] not in {"low", "mid", "high"}:
        raise ValueError("unknown mask band")
    if out["ansatz"] not in {"uniform", "complete"}:
        raise ValueError("unknown ansatz")
    if out["ansatz"] == "complete":
        out["n_generators"] = out["n_qubits"] * (out["n_qubits"] + 1) // 2
    return out


def _comb(n: int, k: int) -> int:
    return math.comb(n, k)


def _base(**kwargs: Any) -> dict[str, Any]:
    out = dict(_DEFAULTS)
    out.update(kwargs)
    return resolve_config(out)


def configs(mode: str = "full") -> list[dict[str, Any]]:
    """Return deterministic, deduplicated configurations for a campaign mode."""
    if mode == "smoke":
        return [
            _base(
                n_qubits=4,
                n_generators=4,
                arity=1,
                n_ops=8,
                n_samples=16,
                n_targets=16,
                repeats=2,
                warmups=1,
                encoder=e,
                dtype=d,
            )
            for e in ENCODERS
            for d in PRECISIONS
        ]
    if mode == "validate":
        return [
            _base(
                n_qubits=n,
                n_conditions=c,
                n_generators=min(4, _comb(n, k)),
                arity=k,
                n_ops=8,
                n_samples=16,
                n_targets=16,
                repeats=2,
                warmups=1,
                encoder=e,
                dtype=d,
            )
            for n in (4, 8)
            for c in (1, 2)
            for k in (1, 2, 4)
            if k <= n
            for e in ENCODERS
            for d in PRECISIONS
        ]
    anchor = [_base(encoder=e, dtype=d) for e in ENCODERS for d in PRECISIONS]
    if mode == "pilot":
        return anchor + [
            _base(
                n_qubits=16,
                n_generators=16,
                arity=2,
                n_ops=128,
                n_samples=256,
                n_targets=256,
                encoder=e,
                dtype=d,
            )
            for e in ENCODERS
            for d in PRECISIONS
        ]
    if mode == "combined":
        return [
            _base(
                n_qubits=128,
                n_conditions=c,
                n_generators=128,
                arity=2,
                n_ops=r,
                n_samples=s,
                n_targets=1024,
                encoder=e,
                dtype=d,
            )
            for c in (1, 4)
            for r in (512, 2048)
            for s in (1024, 4096)
            for e in ("y", "yz")
            for d in PRECISIONS
        ]
    if mode == "frontier":
        return [
            _base(
                n_qubits=n,
                n_generators=n if ans == "uniform" else _comb(n, 1) + _comb(n, 2),
                arity=2,
                ansatz=ans,
                encoder=e,
                dtype=d,
            )
            for n in (512, 1000)
            for ans in ("uniform", "complete")
            for e in ("y", "yz")
            for d in PRECISIONS
        ]
    if mode == "core":
        groups = []
        for key, values in (
            ("n_qubits", (16, 32, 64, 128, 256)),
            ("n_conditions", (1, 4, 16)),
            ("n_generators", (16, 64, 256, 1024)),
            ("arity", (1, 2, 4, 8)),
            ("n_ops", (128, 512, 2048)),
            ("n_samples", (256, 1024, 4096)),
            ("n_targets", (256, 1024, 4096)),
            ("mask_band", ("low", "mid", "high")),
            ("sample_chunk", (None, 64, 256, 1024)),
        ):
            for value in values:
                kw = {key: value}
                groups.extend(
                    _base(**kw, encoder=e, dtype=d)
                    for e in ENCODERS
                    for d in PRECISIONS
                    if (key != "arity" or value <= 16)
                )
        return _dedup(groups)
    if mode == "full":
        all_configs: list[dict[str, Any]] = []
        for campaign in ("pilot", "core", "combined", "frontier"):
            all_configs.extend(configs(campaign))
        return _dedup(all_configs)
    raise ValueError(f"unknown mode: {mode}")


def _dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = tuple(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _supports(config: dict[str, Any]) -> np.ndarray:
    n, k, g = config["n_qubits"], config["arity"], config["n_generators"]
    rng = np.random.default_rng(config["seed"] + 991)
    choices = (
        list(itertools.combinations(range(n), k)) if _comb(n, k) <= 100_000 else None
    )
    selected = (
        rng.choice(len(choices), g, replace=False) if choices is not None else None
    )
    rows = np.zeros((g, n), dtype=np.float64)
    if choices is not None:
        for i, index in enumerate(selected):
            rows[i, list(choices[int(index)])] = 1
    else:
        used = set()
        i = 0
        while i < g:
            support = tuple(sorted(rng.choice(n, k, replace=False)))
            if support in used:
                continue
            used.add(support)
            rows[i, list(support)] = 1
            i += 1
    return rows


def generate_fixture(config: dict[str, Any]) -> dict[str, Any]:
    """Create every array on CPU so backend pairs use bit-identical fixtures."""
    with jax.default_device(jax.devices("cpu")[0]):
        return _generate_fixture_cpu(config)


def _generate_fixture_cpu(config: dict[str, Any]) -> dict[str, Any]:
    c = resolve_config(config)
    rng = np.random.default_rng(c["seed"])
    layers = len(ENCODERS[c["encoder"]])
    angles = rng.normal(0.0, 0.7, (c["n_conditions"], layers, c["n_qubits"]))
    params = np.random.default_rng(c["seed"] + 1).normal(0.0, 0.1, c["n_generators"])
    generators = (
        ciqp.Ansatz.complete(c["n_qubits"]).generators
        if c["ansatz"] == "complete"
        else _supports(c)
    )
    key = jax.random.key(c["seed"])
    keys = jax.random.split(key, 2 * c["n_conditions"])
    encoder = ciqp.AngleEncoder(
        angles,
        axes=tuple(
            tuple(layer[0] for _ in range(c["n_qubits"]))
            for layer in ENCODERS[c["encoder"]]
        ),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        samples = np.stack(
            [
                np.asarray(
                    encoder.select(i).sample(
                        keys[c["n_conditions"] + i], c["n_samples"]
                    )[0]
                )
                for i in range(c["n_conditions"])
            ]
        )
    sigmas = expected_bodyness_sigmas(c["n_qubits"])
    sigma = {"low": sigmas[0], "mid": sigmas[1], "high": sigmas[2]}[c["mask_band"]]
    masks = []
    weights = []
    for i in range(c["n_conditions"]):
        m, w = sample_nonidentity_masks_jax(keys[i], c["n_ops"], c["n_qubits"], sigma)
        masks.append(np.asarray(m))
        weights.append(np.asarray(w))
    targets = (
        np.random.default_rng(c["seed"] + 2)
        .integers(
            0, 2, (c["n_conditions"], c["n_targets"], c["n_qubits"]), dtype=np.int8
        )
        .astype(np.float64)
    )
    return {
        "angles": angles,
        "params": params,
        "generators": np.asarray(generators),
        "targets": targets,
        "samples": samples,
        "masks": np.stack(masks),
        "weights": np.stack(weights),
        "sigma": np.float64(sigma),
        "key_data": np.asarray(jax.random.key_data(key), dtype=np.uint32),
    }


def fixture_hash(arrays: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
