"""
Shared setup for the test suite.

Solves the MANTA example.
"""

import os

import matplotlib

matplotlib.use("Agg")

import openpopcon as op

MANTA = op.example_dir("MANTA")
SETTINGS = os.path.join(MANTA, "POPCON_input_example.yaml")
PLOTSETTINGS = os.path.join(MANTA, "plotsettings.yml")

SMALL_GRID = dict(Nn=8, NTi=8, nr=60)

GOLDEN_FIELDS = [
    "Paux",
    "Q",
    "Pheat",
    "Pconf",
    "Psol",
    "f_rad",
    "tauE",
    "betaN",
    "Pfusion",
    "Pohmic",
    "n_i_20_avg",
    "vloop",
]

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "golden_manta.json")

UNPHYSICAL = 99998.0


def solve_small_manta(parallel=False):
    """
    Solve the MANTA example on the small grid. parallel=False keeps the run
    deterministic so golden comparisons are exact up to floating point.
    """
    pc = op.POPCON(settingsfile=SETTINGS, plotsettingsfile=PLOTSETTINGS)
    pc.settings.Nn = SMALL_GRID["Nn"]
    pc.settings.NTi = SMALL_GRID["NTi"]
    pc.settings.nr = SMALL_GRID["nr"]
    pc.settings.parallel = parallel
    pc.settings.verbosity = 0
    pc.run_POPCON()
    return pc
