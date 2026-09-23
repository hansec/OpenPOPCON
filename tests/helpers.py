"""
Shared setup for the test suite.

Solves the MANTA example.
"""

import os

import matplotlib

matplotlib.use("Agg")

import yaml

import openpopcon as op

MANTA = op.example_dir("MANTA")
SETTINGS = os.path.join(MANTA, "POPCON_input_example.yaml")
PLOTSETTINGS = os.path.join(MANTA, "plotsettings.yml")

# SPARC has no gEQDSK, so its geometry settings are actually scannable
SPARC = op.example_dir("SPARC")
SPARC_SETTINGS = os.path.join(SPARC, "POPCON_input_example.yaml")
SPARC_PLOTSETTINGS = os.path.join(SPARC, "plotsettings.yml")

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


def write_settings(tmp_path, base=SETTINGS, name="settings.yaml", **overrides):
    """
    A settings file built from an example with some keys replaced, detached
    from any gEQDSK/profile files so it is self-contained, and from the
    example's scan block so a test's scan is the only one in play.
    """
    with open(base) as fh:
        data = yaml.safe_load(fh)
    data["gfilename"] = ""
    data["profsfilename"] = ""
    data.pop("scan", None)
    data.update(overrides)
    path = tmp_path / name
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return str(path)


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
    # the golden values pin the plain gEQDSK path, where the equilibrium's own
    # R, a and current are used
    pc.settings.gfile_rescale = False
    pc.settings.verbosity = 0
    pc.run_POPCON()
    return pc
