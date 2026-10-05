"""
Physics/consistency invariants that should hold regardless of the exact numbers,
so they keep protecting the code even after the golden file is regenerated.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import openpopcon as op
from helpers import UNPHYSICAL, solve_small_manta, SETTINGS, PLOTSETTINGS


def _valid_mask(pc):
    return np.asarray(pc.output["Paux"].values) < UNPHYSICAL


def test_pheat_equals_confinement_loss(solved_manta):
    valid = _valid_mask(solved_manta)
    pheat = np.asarray(solved_manta.output["Pheat"].values)[valid]
    pconf = np.asarray(solved_manta.output["Pconf"].values)[valid]
    np.testing.assert_allclose(pheat, pconf, rtol=1e-6, atol=1e-6)


def test_masked_points_use_sentinel(solved_manta):
    paux = np.asarray(solved_manta.output["Paux"].values)
    masked = paux >= UNPHYSICAL
    assert masked.any(), "expected some unphysical points in this scan"
    assert (~masked).any(), "expected some solved points in this scan"
    assert np.all(paux[masked] == 99999.0)


def test_solved_paux_nonnegative(solved_manta):
    paux = np.asarray(solved_manta.output["Paux"].values)
    valid = paux < UNPHYSICAL
    assert np.all(paux[valid] >= 0.0)


def test_parallel_matches_serial(solved_manta):
    par = solve_small_manta(parallel=True)
    for field in ("Paux", "Q", "Pheat", "betaN"):
        np.testing.assert_allclose(
            np.asarray(par.output[field].values),
            np.asarray(solved_manta.output[field].values),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"parallel and serial disagree on {field}",
        )


def test_single_point_runs(solved_manta):
    solved_manta.single_point(0.6, 8.0, plot=True, show=False)
    plt.close("all")


def test_output_roundtrip(solved_manta, tmp_path):
    solved_manta.write_output(name="rt", directory=str(tmp_path), archive=False)

    restored = op.POPCON(settingsfile=SETTINGS, plotsettingsfile=PLOTSETTINGS)
    restored.read_output("rt", directory=str(tmp_path))
    for field in ("Paux", "betaN", "Q"):
        np.testing.assert_allclose(
            np.asarray(restored.output[field].values),
            np.asarray(solved_manta.output[field].values),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"roundtrip changed {field}",
        )
    plt.close("all")


def test_parabolic_current_profile_integrates_to_ip():
    """
    Without a gEQDSK the current density used for the ohmic power is
    Itot * j_prof. Its poloidal cross-section integral must be the plasma
    current, as it already is on the gEQDSK branch (Jrms_norm).
    """
    from helpers import SPARC_SETTINGS, SPARC_PLOTSETTINGS

    pc = op.POPCON(settingsfile=SPARC_SETTINGS, plotsettingsfile=SPARC_PLOTSETTINGS)
    pc.settings.verbosity = 0
    pc.run_POPCON(setuponly=True)
    al = pc.algorithms
    rho = al.sqrtpsin
    J = al.state.Itot * 1e6 * al.get_profile(rho, 0)  # A/m^2
    ip_from_j = al.volume_integral(rho, J) / (2 * np.pi * al.state.R) / 1e6  # MA
    np.testing.assert_allclose(ip_from_j, al.state.Ip, rtol=1e-3)
