"""The reduced-precision presets reproduce main bit for bit, the chunked loss
to the last bits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from preset_golden import outputs

GOLDEN = np.load(Path(__file__).parent / "data" / "preset_golden.npz")


@pytest.fixture(scope="module")
def current() -> dict[str, np.ndarray]:
    return outputs()


@pytest.mark.parametrize("name", GOLDEN.files)
def test_outputs_match_main(current, name):
    if name.startswith("fp64_chunked"):
        # The chunks now run in a scan, which XLA compiles apart from the
        # surrounding sums: the same additions, but last-bit differences.
        np.testing.assert_allclose(current[name], GOLDEN[name], rtol=1e-14, atol=0)
    else:
        np.testing.assert_array_equal(current[name], GOLDEN[name])
