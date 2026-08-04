"""
Plot axes. The xax/yax settings resolve through one registry, and either
axis takes either family, so a density can go on x. The output arrays are
stored (n_index, T_index), so that case only works if they are transposed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import openpopcon as op
from openpopcon.core import DIM_N, DIM_T, PLOT_AXES

from helpers import PLOTSETTINGS, SETTINGS

T_AXES = [k for k, v in PLOT_AXES.items() if v[1] == DIM_T]
N_AXES = [k for k, v in PLOT_AXES.items() if v[1] == DIM_N]


@pytest.fixture(autouse=True)
def restore_axes(solved_manta):
    # solved_manta is session-scoped and these tests set xax/yax on it, so
    # put them back or the next test inherits whatever the last one left
    xax, yax = solved_manta.plotsettings.xax, solved_manta.plotsettings.yax
    yield
    solved_manta.plotsettings.xax = xax
    solved_manta.plotsettings.yax = yax
    plt.close("all")


def test_registry_covers_the_documented_axes():
    # the seven keys the plotsettings files have always accepted
    assert set(PLOT_AXES) == {
        "T_i_av",
        "T_i_ax",
        "T_e_av",
        "T_e_ax",
        "n20_av",
        "n20_ax",
        "nG",
    }
    # every one names a coordinate that build_dataset actually produces
    for var, dim, label in PLOT_AXES.values():
        assert dim in (DIM_N, DIM_T)
        assert label


@pytest.mark.parametrize("xax", T_AXES)
@pytest.mark.parametrize("yax", N_AXES)
def test_every_conventional_pairing_plots(solved_manta, xax, yax):
    solved_manta.plotsettings.xax = xax
    solved_manta.plotsettings.yax = yax
    fig, ax = solved_manta.plot(show=False)
    assert ax.collections
    assert ax.get_xlabel() and ax.get_ylabel()
    plt.close(fig)


@pytest.mark.parametrize("xax", N_AXES)
@pytest.mark.parametrize("yax", T_AXES)
def test_density_on_x_plots(solved_manta, xax, yax):
    # the case that never ran before the registry: the 2-D arrays have to be
    # transposed for this to line up with the meshgrid at all
    solved_manta.plotsettings.xax = xax
    solved_manta.plotsettings.yax = yax
    fig, ax = solved_manta.plot(show=False)
    assert ax.collections
    plt.close(fig)


def test_swapping_axes_transposes_the_data(solved_manta):
    solved_manta.plotsettings.xax = "T_i_av"
    solved_manta.plotsettings.yax = "nG"
    _, _, _, _, normal = solved_manta._resolve_axes()
    a = solved_manta._grid("Q", normal)

    solved_manta.plotsettings.xax = "nG"
    solved_manta.plotsettings.yax = "T_i_av"
    _, _, _, _, swapped = solved_manta._resolve_axes()
    b = solved_manta._grid("Q", swapped)

    assert normal == (DIM_N, DIM_T)
    assert swapped == (DIM_T, DIM_N)
    np.testing.assert_allclose(a, b.T)


@pytest.mark.parametrize(
    "xax,yax",
    [("nG", "n20_av"), ("T_i_av", "T_e_ax")],
)
def test_two_axes_on_one_dimension_rejected(solved_manta, xax, yax):
    solved_manta.plotsettings.xax = xax
    solved_manta.plotsettings.yax = yax
    with pytest.raises(ValueError, match="both labels for"):
        solved_manta.plot(show=False)


@pytest.mark.parametrize("which", ["xax", "yax"])
def test_unknown_axis_reports_the_choices(solved_manta, which):
    setattr(solved_manta.plotsettings, which, "not_an_axis")
    with pytest.raises(ValueError, match="Available:"):
        solved_manta.plot(show=False)


def test_plot_draws_into_a_supplied_ax(solved_manta):
    fig, ax = plt.subplots()
    before = len(plt.get_fignums())
    outfig, outax = solved_manta.plot(ax=ax, show=False)
    # no new figure, and it drew into the one it was given
    assert outax is ax
    assert outfig is fig
    assert len(plt.get_fignums()) == before
    assert ax.collections
    plt.close("all")


def test_plot_without_ax_is_unchanged(solved_manta):
    fig, ax = solved_manta.plot(show=False)
    assert fig is ax.get_figure()
    # the legend and infobox are on by default
    assert ax.get_legend() is not None
    assert any(t.get_text().startswith("$I_p$") for t in ax.texts)
    plt.close(fig)


def test_legend_and_infobox_can_be_turned_off(solved_manta):
    fig, ax = plt.subplots()
    solved_manta.plot(ax=ax, show=False, legend=False, infobox=False)
    assert ax.get_legend() is None
    assert not any(t.get_text().startswith("$I_p$") for t in ax.texts)
    plt.close("all")


def test_savefig_writes_the_right_figure(solved_manta, tmp_path):
    # plt.savefig would write whichever figure happens to be current
    fig, ax = plt.subplots()
    plt.figure()  # make a different figure current
    out = tmp_path / "p.png"
    solved_manta.plot(ax=ax, show=False, savefig=str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close("all")
