"""
POPCON_scan: an N x M grid of POPCONs over two machine parameters.

The failure mode worth guarding hardest is a scan that silently produces
identical cells, because it looks exactly like a scan that worked.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

import openpopcon as op
from openpopcon.core import DIM_N, DIM_T

from helpers import (
    PLOTSETTINGS,
    SETTINGS,
    SMALL_GRID,
    SPARC_PLOTSETTINGS,
    SPARC_SETTINGS,
    UNPHYSICAL,
    write_settings,
)

ROWS = [8.0, 10.0]
COLS = [11.0, 13.0]


def _scan_settings(tmp_path, base=SPARC_SETTINGS, **overrides):
    return write_settings(
        tmp_path, base=base, verbosity=0, parallel=False, **SMALL_GRID, **overrides
    )


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    path = _scan_settings(tmp_path_factory.mktemp("scan"))
    sc = op.POPCON_scan(
        settingsfile=path,
        plotsettingsfile=SPARC_PLOTSETTINGS,
        scan={"rows": ("I_P", ROWS), "cols": ("B_0", COLS)},
    )
    sc.run_scan(progress=False)
    return sc


# -------------------------------------------------------------------
# Running
# -------------------------------------------------------------------


def test_shape_and_cells(scan):
    assert scan.shape == (2, 2)
    for row in scan.cells:
        for cell in row:
            assert dict(cell.output.sizes) == {
                DIM_N: SMALL_GRID["Nn"],
                DIM_T: SMALL_GRID["NTi"],
            }


def test_scanned_values_actually_reach_the_solver(scan):
    # the check that catches a scan silently doing nothing
    for i, ip in enumerate(ROWS):
        for j, b0 in enumerate(COLS):
            assert scan.cells[i][j].algorithms.Ip == pytest.approx(ip)
            assert scan.cells[i][j].algorithms.B0 == pytest.approx(b0)


def test_cells_differ(scan):
    a = scan.cells[0][0].output.Paux.values
    b = scan.cells[1][1].output.Paux.values
    assert not np.allclose(a, b)


def test_cell_matches_a_standalone_run(tmp_path, scan):
    # proves the raw-dict re-derivation reproduces an ordinary run exactly,
    # and that nothing leaks between cells
    path = _scan_settings(tmp_path, I_P=ROWS[1], B_0=COLS[0])
    pc = op.POPCON(settingsfile=path, plotsettingsfile=SPARC_PLOTSETTINGS)
    pc.run_POPCON()
    np.testing.assert_allclose(
        scan.cells[1][0].output.Paux.values, pc.output.Paux.values, rtol=1e-12
    )


def test_greenwald_axis_is_shared_but_absolute_density_is_not(scan):
    # n_G = Ip / (pi a^2), so scanning I_P moves the absolute density axis
    # while the Greenwald fraction stays pinned. The storage design rests on
    # this, and so does the sharey decision in plot()
    np.testing.assert_allclose(
        scan.cells[0][0].output.n_G_frac.values,
        scan.cells[1][0].output.n_G_frac.values,
    )
    assert not np.allclose(
        scan.cells[0][0].output.n_e_20_max.values,
        scan.cells[1][0].output.n_e_20_max.values,
    )


def test_base_settings_are_not_mutated(tmp_path):
    path = _scan_settings(tmp_path)
    original = yaml.safe_load(open(path))
    sc = op.POPCON_scan(
        settingsfile=path,
        plotsettingsfile=SPARC_PLOTSETTINGS,
        scan={"rows": ("I_P", ROWS), "cols": ("B_0", COLS)},
    )
    sc.run_scan(progress=False)
    assert sc.base_settings.Ip == pytest.approx(original["I_P"])
    assert sc.base_settings.B0 == pytest.approx(original["B_0"])


# -------------------------------------------------------------------
# Combined output
# -------------------------------------------------------------------


def test_combined_dataset(scan):
    ds = scan.output
    assert ds.Paux.dims == ("scan_I_P", "scan_B_0", DIM_N, DIM_T)
    assert ds.Paux.shape == (2, 2, SMALL_GRID["Nn"], SMALL_GRID["NTi"])
    np.testing.assert_allclose(ds.scan_I_P.values, ROWS)
    np.testing.assert_allclose(ds.scan_B_0.values, COLS)
    # a coordinate that differs per cell must be promoted, not collapsed
    assert "scan_I_P" in ds.n_e_20_max.dims


def test_positional_indexing_matches_cells(scan):
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                scan.output.Paux.values[i, j], scan.cells[i][j].output.Paux.values
            )


def test_metric_orientation(scan):
    m = scan.metric("Q", "max")
    assert m.dims == ("scan_I_P", "scan_B_0")
    assert m.shape == (2, 2)
    # reduced over valid points only, so the sentinel never leaks in
    assert np.all(np.isfinite(m.values))
    assert float(m.max()) < UNPHYSICAL


def test_combine_before_running_is_an_error(tmp_path):
    sc = op.POPCON_scan(
        settingsfile=_scan_settings(tmp_path),
        plotsettingsfile=SPARC_PLOTSETTINGS,
        scan={"rows": ("I_P", ROWS), "cols": ("B_0", COLS)},
    )
    with pytest.raises(RuntimeError, match="run_scan"):
        sc.output


# -------------------------------------------------------------------
# Specification and validation
# -------------------------------------------------------------------


def test_yaml_scan_block_is_read(tmp_path):
    path = _scan_settings(
        tmp_path,
        scan={
            "rows": {"parameter": "I_P", "min": 8.0, "max": 10.0, "N": 2},
            "cols": {"parameter": "B_0", "values": COLS},
        },
    )
    sc = op.POPCON_scan(settingsfile=path, plotsettingsfile=SPARC_PLOTSETTINGS)
    assert sc.shape == (2, 2)
    np.testing.assert_allclose(sc.row.values, [8.0, 10.0])
    np.testing.assert_allclose(sc.col.values, COLS)


def test_python_scan_overrides_the_yaml_block(tmp_path):
    path = _scan_settings(
        tmp_path,
        scan={
            "rows": {"parameter": "I_P", "values": [1.0, 2.0]},
            "cols": {"parameter": "B_0", "values": [3.0, 4.0]},
        },
    )
    sc = op.POPCON_scan(
        settingsfile=path,
        plotsettingsfile=SPARC_PLOTSETTINGS,
        scan={"rows": ("I_P", ROWS), "cols": ("B_0", COLS)},
    )
    np.testing.assert_allclose(sc.row.values, ROWS)


def test_shipped_examples_with_scan_blocks_still_load_as_plain_popcons():
    # 'scan' is in KNOWN_SETTINGS_KEYS, so it must not be reported as a typo
    for name in ("CENTAUR", "ITER"):
        d = op.example_dir(name)
        pc = op.POPCON(
            settingsfile=f"{d}/POPCON_input_example.yaml",
            plotsettingsfile=f"{d}/plotsettings.yml",
        )
        assert pc.settings.scan


BAD_SCANS = {
    "unknown_parameter": {"rows": ("Bfield", ROWS), "cols": ("B_0", COLS)},
    "unscannable_Nn": {"rows": ("Nn", [8, 16]), "cols": ("B_0", COLS)},
    "unscannable_gfilename": {"rows": ("gfilename", ["a", "b"]), "cols": ("B_0", COLS)},
    "same_parameter": {"rows": ("B_0", [9.0, 11.0]), "cols": ("B_0", COLS)},
    "only_one_axis": {"rows": ("I_P", ROWS)},
    "empty_values": {"rows": ("I_P", []), "cols": ("B_0", COLS)},
}


@pytest.mark.parametrize("name", list(BAD_SCANS))
def test_bad_scan_specifications_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        op.POPCON_scan(
            settingsfile=_scan_settings(tmp_path),
            plotsettingsfile=SPARC_PLOTSETTINGS,
            scan=BAD_SCANS[name],
        )


def test_no_scan_at_all_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No scan specified"):
        op.POPCON_scan(
            settingsfile=_scan_settings(tmp_path), plotsettingsfile=SPARC_PLOTSETTINGS
        )


@pytest.mark.parametrize("parameter", ["R", "a", "kappa", "delta", "I_P", "qstar"])
def test_geqdsk_geometry_scan_rejected(parameter):
    # MANTA has a gfilename, and __get_geometry takes the geometry and the
    # ohmic current from the equilibrium regardless of the settings
    with pytest.raises(ValueError, match="gfilename"):
        op.POPCON_scan(
            settingsfile=SETTINGS,
            plotsettingsfile=PLOTSETTINGS,
            scan={"rows": (parameter, [1.0, 2.0]), "cols": ("B_0", COLS)},
        )


@pytest.mark.parametrize("parameter", ["B_0", "H_fac"])
def test_geqdsk_safe_parameters_accepted(parameter):
    op.POPCON_scan(
        settingsfile=SETTINGS,
        plotsettingsfile=PLOTSETTINGS,
        scan={"rows": (parameter, [1.0, 2.0]), "cols": ("Tmax_keV", [10.0, 12.0])},
    )


def test_invalid_cell_settings_reported_up_front(tmp_path):
    # a >= R at one corner: caught when the cells are built, before any solve
    with pytest.raises(ValueError, match=r"scan cells have invalid settings"):
        op.POPCON_scan(
            settingsfile=_scan_settings(tmp_path),
            plotsettingsfile=SPARC_PLOTSETTINGS,
            scan={"rows": ("a", [0.5, 90.0]), "cols": ("B_0", COLS)},
        )


def test_shadowed_key_is_dropped(tmp_path):
    # SPARC gives I_P, which shadows qstar in read(). Scanning qstar has to
    # drop I_P or every cell comes out identical
    path = _scan_settings(tmp_path)
    sc = op.POPCON_scan(
        settingsfile=path,
        plotsettingsfile=SPARC_PLOTSETTINGS,
        scan={"rows": ("qstar", [2.0, 4.0]), "cols": ("B_0", COLS)},
    )
    assert sc.cells[0][0].settings.Ip != pytest.approx(sc.cells[1][0].settings.Ip)


def test_scanning_R_redrives_Ip_from_qstar(tmp_path):
    # the case field mutation cannot express: with qstar given instead of
    # I_P, changing R must change the derived Ip
    path = _scan_settings(tmp_path, qstar=2.5)
    # I_P shadows qstar in read(), so strip it to make qstar load-bearing
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw.pop("I_P")
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh)

    sc = op.POPCON_scan(
        settingsfile=path,
        plotsettingsfile=SPARC_PLOTSETTINGS,
        scan={"rows": ("R", [1.6, 2.2]), "cols": ("B_0", COLS)},
    )
    assert sc.cells[0][0].settings.Ip != pytest.approx(sc.cells[1][0].settings.Ip)


# -------------------------------------------------------------------
# Plotting and I/O
# -------------------------------------------------------------------


def test_plot_grid(scan):
    fig, axs = scan.plot(show=False)
    assert axs.shape == (2, 2)
    for row in axs:
        for ax in row:
            assert ax.collections
    plt.close("all")


def test_plot_metric(scan):
    fig, ax = scan.plot_metric("Q", show=False)
    assert ax.images
    plt.close("all")


def test_harmonized_levels_are_shared(scan):
    scan._harmonize_levels(["Q"])
    bounds = {
        (c.plotsettings.plotoptions["Q"]["min"], c.plotsettings.plotoptions["Q"]["max"])
        for row in scan.cells
        for c in row
    }
    assert len(bounds) == 1


def test_write_and_read_roundtrip(scan, tmp_path):
    scan.write_output(name="rt", directory=str(tmp_path), archive=False)
    back = op.POPCON_scan.read_output("rt", directory=str(tmp_path))
    assert back.shape == scan.shape
    np.testing.assert_allclose(back.row.values, scan.row.values)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                back.cells[i][j].output.Paux.values,
                scan.cells[i][j].output.Paux.values,
            )
    plt.close("all")
