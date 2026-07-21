"""
Golden-value regression: the MANTA scan must reproduce the frozen reference
output. A failure here means the physics changed. If that was intentional,
regenerate the golden file (see generate_golden.py) and commit the new values.
"""
import json

import numpy as np
import pytest

from helpers import GOLDEN_FIELDS, GOLDEN_PATH


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


@pytest.mark.parametrize("field", GOLDEN_FIELDS)
def test_field_matches_golden(solved_manta, golden, field):
    got = np.asarray(solved_manta.output[field].values)
    ref = np.asarray(golden["fields"][field], dtype=float)
    assert got.shape == ref.shape, f"{field}: shape {got.shape} != golden {ref.shape}"
    # rtol is loose enough to survive libm/BLAS differences across platforms but
    # far tighter than any real physics change, which moves results by percents.
    np.testing.assert_allclose(
        got, ref, rtol=1e-6, atol=1e-8,
        err_msg=(f"{field} differs from the golden reference. If this change is "
                 f"intentional, regenerate with `uv run python tests/generate_golden.py`."),
    )
