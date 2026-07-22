"""
Regenerate the golden regression file.

Run whenever a change intentionally alters the computed results:

    uv run python tests/generate_golden.py

review the diff to tests/data/golden_manta.json and commit it together with
the change that caused it.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from helpers import GOLDEN_FIELDS, GOLDEN_PATH, SMALL_GRID, solve_small_manta

import openpopcon as op
import numba
import numpy


def main():
    pc = solve_small_manta(parallel=False)
    payload = {
        "meta": {
            "grid": SMALL_GRID,
            "openpopcon": op.__version__,
            "numpy": numpy.__version__,
            "numba": numba.__version__,
            "note": "regenerate with `uv run python tests/generate_golden.py`",
        },
        "fields": {f: np.asarray(pc.output[f].values).tolist() for f in GOLDEN_FIELDS},
    }
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(
        f"Wrote {GOLDEN_PATH} ({len(GOLDEN_FIELDS)} fields, "
        f"{SMALL_GRID['Nn']}x{SMALL_GRID['NTi']} grid)."
    )


if __name__ == "__main__":
    main()
