"""
Settings validation. Two guarantees:
  - every example passes validation (the code must never reject its own
    examples, several of which use edge-case values on purpose), and
  - obviously bad settings are rejected at construction with a clear ValueError.
"""
import glob
import os

import yaml
import pytest

import openpopcon as op
from openpopcon.lib.openpopcon_util import package_resource
from helpers import PLOTSETTINGS, SETTINGS


def _settings_file(example):
    # settings files are *.yaml; plotsettings are *.yml
    matches = glob.glob(os.path.join(op.example_dir(example), "*.yaml"))
    assert len(matches) == 1, f"{example}: expected one settings .yaml, found {matches}"
    return matches[0]


@pytest.mark.parametrize("example", op.list_examples())
def test_shipped_examples_validate(example):
    settings = _settings_file(example)
    plotsettings = os.path.join(op.example_dir(example), "plotsettings.yml")
    # construction runs __check_settings; it must not raise for shipped examples
    op.POPCON(settingsfile=settings, plotsettingsfile=plotsettings)


def _write_settings(tmp_path, **overrides):
    with open(SETTINGS) as fh:
        data = yaml.safe_load(fh)
    # detach from the gEQDSK/profile files: validation doesn't need them and it
    # keeps these settings self-contained
    data["gfilename"] = ""
    data["profsfilename"] = ""
    data.update(overrides)
    path = tmp_path / "bad.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return str(path)


BAD_SETTINGS = {
    "Tmin_ge_Tmax": dict(Tmin_keV=12.0, Tmax_keV=11.0),
    "nmin_ge_nmax": dict(nmin_frac=0.9, nmax_frac=0.3),
    "a_ge_R": dict(a=5.0),                      # R is 4.55
    "delta_out_of_range": dict(delta=1.5),
    "grid_too_small": dict(Nn=1),
    "bad_fuel": dict(fuel=7),
    "impurity_index_out_of_range": dict(impurity=9),
    "wrong_impurity_length": dict(impurityfractions=[0.1, 0.2]),
}


@pytest.mark.parametrize("name", list(BAD_SETTINGS))
def test_bad_settings_rejected(tmp_path, name):
    settings = _write_settings(tmp_path, **BAD_SETTINGS[name])
    with pytest.raises(ValueError):
        op.POPCON(settingsfile=settings, plotsettingsfile=PLOTSETTINGS)


def test_singular_scaling_law_rejected(tmp_path):
    # Pheat_alpha = -1 makes the closed-form solver (W/K)**(1/(1+alpha)) singular.
    laws = yaml.safe_load(open(package_resource("scalinglaws.yml")))
    laws["H98y2"]["Pheat_alpha"] = -1
    slfile = tmp_path / "scalinglaws.yml"
    with open(slfile, "w") as fh:
        yaml.safe_dump(laws, fh)

    with pytest.raises(ValueError, match="singular"):
        op.POPCON(settingsfile=SETTINGS, plotsettingsfile=PLOTSETTINGS,
                  scalinglawfile=str(slfile))
