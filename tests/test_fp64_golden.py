"""The fp64 path reproduces 0.8.1 bit for bit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fp64_golden import outputs

GOLDEN = np.load(Path(__file__).parent / "data" / "fp64_golden.npz")


@pytest.fixture(scope="module")
def current() -> dict[str, np.ndarray]:
    return outputs()


@pytest.mark.parametrize("name", GOLDEN.files)
def test_fp64_outputs_match_0_8_1(current, name):
    np.testing.assert_array_equal(current[name], GOLDEN[name])
