"""
One-parameter sensitivity scans on every shipped example: each machine
parameter varied by +-25% on its own, and the major radius varied under the
three usual constraints. These are what a first POPCON study asks for, so
every example set up for it has to support all of them, gEQDSK ones
included.
"""

import os

import numpy as np
import pytest
import yaml

import openpopcon as op
from openpopcon.core import _qstar

from helpers import SMALL_GRID

SCALE = [0.75, 1.0, 1.25]

SENSITIVITIES = {
    "B_0": {"parameter": "B_0", "scale": SCALE},
    "H_fac": {"parameter": "H_fac", "scale": SCALE},
    "I_P": {"parameter": "I_P", "scale": SCALE},
    "Zeff_target": {"parameter": "Zeff_target", "scale": SCALE},
    "R_fixed_aspect_and_current": {
        "parameter": "R",
        "scale": SCALE,
        "hold": ["aspect_ratio", "I_P"],
    },
    "R_fixed_aspect_and_qstar": {
        "parameter": "R",
        "scale": SCALE,
        "hold": ["aspect_ratio", "qstar"],
    },
    "R_fixed_minor_radius_and_qstar": {
        "parameter": "R",
        "scale": SCALE,
        "hold": ["a", "qstar"],
    },
}


def _example_settings(tmp_path, example):
    """
    An example's settings on the small grid, with its gEQDSK and profiles
    files made absolute so the copy still finds them.
    """
    d = op.example_dir(example)
    [name] = [f for f in os.listdir(d) if f.endswith(".yaml")]
    with open(os.path.join(d, name)) as fh:
        data = yaml.safe_load(fh)
    data.pop("scan", None)
    data.update(verbosity=0, parallel=False, **SMALL_GRID)
    for key in ("gfilename", "profsfilename"):
        if data.get(key):
            data[key] = os.path.join(d, data[key])
    path = tmp_path / "settings.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return str(path), os.path.join(d, "plotsettings.yml")


# the NSF examples keep their original settings, which have no Zeff_target
# and use the equilibrium's own geometry, so these scans do not apply to them
SENSITIVITY_EXAMPLES = [e for e in op.list_examples() if not e.startswith("NSF")]


@pytest.mark.parametrize("sensitivity", list(SENSITIVITIES))
@pytest.mark.parametrize("example", SENSITIVITY_EXAMPLES)
def test_sensitivity_scan_runs_and_changes_something(tmp_path, example, sensitivity):
    settings, plotsettings = _example_settings(tmp_path, example)
    sc = op.POPCON_scan(
        settingsfile=settings,
        plotsettingsfile=plotsettings,
        scan={"rows": SENSITIVITIES[sensitivity]},
    )
    assert sc.shape == (3, 1)
    sc.run_scan(progress=False)

    cells = [sc.cells[i][0] for i in range(3)]
    # the failure worth guarding: a scan whose cells silently came out the
    # same. P_aux is the field every one of these moves; P_fusion at a given
    # grid point does not depend on the field or the confinement at all
    arrays = [np.nan_to_num(c.output.Paux.values) for c in cells]
    assert not np.allclose(arrays[0], arrays[2])

    base = sc.base_settings
    st = [c.settings for c in cells]
    hold = SENSITIVITIES[sensitivity].get("hold", [])
    if SENSITIVITIES[sensitivity]["parameter"] == "R":
        np.testing.assert_allclose([s.R for s in st], np.multiply(SCALE, base.R))
    if "aspect_ratio" in hold:
        np.testing.assert_allclose([s.R / s.a for s in st], base.R / base.a)
    if "a" in hold:
        np.testing.assert_allclose([s.a for s in st], base.a)
    if "I_P" in hold:
        np.testing.assert_allclose([s.Ip for s in st], base.Ip)
    if "qstar" in hold:
        np.testing.assert_allclose([_qstar(s) for s in st], _qstar(base))

    table = sc.operating_point(T_i_avg=cells[1].output.T_i_avg.values[3], n_G_frac=0.5)
    assert len(table) == 3


def test_scale_axis_is_relative_to_the_base(tmp_path):
    settings, plotsettings = _example_settings(tmp_path, "SPARC")
    sc = op.POPCON_scan(
        settingsfile=settings,
        plotsettingsfile=plotsettings,
        scan={"cols": {"parameter": "B_0", "scale": SCALE}},
    )
    assert sc.shape == (1, 3)
    np.testing.assert_allclose(sc.col.values, np.multiply(SCALE, 12.2))


def test_zeff_scan_without_impurity_rejected(tmp_path):
    settings, plotsettings = _example_settings(tmp_path, "SPARC")
    with open(settings) as fh:
        data = yaml.safe_load(fh)
    del data["impurity"]
    with open(settings, "w") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="impurity"):
        op.POPCON_scan(
            settingsfile=settings,
            plotsettingsfile=plotsettings,
            scan={"rows": {"parameter": "Zeff_target", "scale": SCALE}},
        )


def test_operating_point_by_density(tmp_path):
    settings, plotsettings = _example_settings(tmp_path, "SPARC")
    sc = op.POPCON_scan(
        settingsfile=settings,
        plotsettingsfile=plotsettings,
        scan={"rows": {"parameter": "I_P", "scale": SCALE}},
    )
    sc.run_scan(progress=False)
    table = sc.operating_point(T_i_avg=8.0, n_e_20_avg=3.0)
    np.testing.assert_allclose(table["I_P"], np.multiply(SCALE, 8.7))
    # a fixed density is a different Greenwald fraction at each current
    assert table["n_G_frac"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="exactly one"):
        sc.operating_point(T_i_avg=8.0)
