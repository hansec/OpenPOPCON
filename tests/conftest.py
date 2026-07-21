"""
Shared test fixtures
"""
import matplotlib
matplotlib.use("Agg")

import pytest

from helpers import solve_small_manta


@pytest.fixture(scope="session")
def solved_manta():
    return solve_small_manta(parallel=False)
