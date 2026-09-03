# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 2024

OpenPOPCON v2.0
This is the Columbia University fork of the OpenPOPCON project developed
for MIT course 22.63. This project is a refactor of the contributions
made to the original project in the development of MANTA. Contributors
to the original project are listed in the README.md file.

If you make changes, append your name and the date to this list.

Contributors:
- Matthew Pharr (2024-07-09)
- Alexei Zhurba (2025-04-04)
- Evan Bursch (2026-08-26)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import json
import os
import re
import xarray as xr
from .lib.openpopcon_util import *
import numba as nb
from collections import namedtuple
from .lib import phys_lib as phys
import shutil
import datetime
import io
import contextlib
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# resistivity_model in the settings file -> resistivity_alg on the State
RESISTIVITY_MODELS = {"jardin": 0, "paz-soldan": 1, "maximum": 2, "max": 2}

try:
    __version__ = _pkg_version("openpopcon")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

# P_aux at a grid point with no physical solution. Plotting masks on this.
UNPHYSICAL = 99999.0

# plotting masks at slightly less than UNPHYSICAL, so that a point that solved
# to a legitimately enormous P_aux is still distinguishable from a failure
UNPHYSICAL_PLOT_CUTOFF = 99998.0

# why a grid point has no physical solution. P_aux is set to UNPHYSICAL for all
# of these; the flag survives so the reason can be reported
INVALID_OK = 0
INVALID_IMPRAD = 1
INVALID_NAN = 2
INVALID_REASONS = {
    INVALID_IMPRAD: "impurity radiation exceeds the confinement loss",
    INVALID_NAN: "no real solution to power balance",
}

# a solved point that needs no auxiliary heating
NOAUX = 3

# every key POPCON_settings.read understands; anything else in a settings file
# is ignored, so it gets reported as a likely typo
KNOWN_SETTINGS_KEYS = {
    "name",
    "R",
    "a",
    "kappa",
    "delta",
    "B_0",
    "B_coil",
    "wall_thickness",
    "I_P",
    "qstar",
    "tipeak_over_tepeak",
    "fuel",
    "impurityfractions",
    "Zeff_target",
    "impurity",
    "scalinglaw",
    "H_fac",
    "nr",
    "gfilename",
    "profsfilename",
    "j_alpha1",
    "j_alpha2",
    "j_offset",
    "ne_alpha1",
    "ne_alpha2",
    "ne_offset",
    "ni_alpha1",
    "ni_alpha2",
    "ni_offset",
    "Ti_alpha1",
    "Ti_alpha2",
    "Ti_offset",
    "Te_alpha1",
    "Te_alpha2",
    "Te_offset",
    "Nn",
    "NTi",
    "nmax_frac",
    "nmin_frac",
    "Tmax_keV",
    "Tmin_keV",
    "resistivity_model",
    "verbosity",
    "parallel",
    "scan",
}

# keys that used to do something. They are still accepted so that older
# settings files keep working, but they are ignored, and saying so is more
# useful than reporting them as typos
DEPRECATED_SETTINGS_KEYS = {
    "maxit": "the solver is closed-form and does not iterate",
    "accel": "the solver is closed-form and does not iterate",
    "err": "the solver is closed-form and has no convergence tolerance",
}

# settings-file keys a scan may vary. These are the YAML-facing names, not the
# attribute names, because a scan re-derives the whole settings object from the
# raw file contents rather than patching derived fields
SCANNABLE_SETTINGS_KEYS = {
    "R",
    "a",
    "kappa",
    "delta",
    "B_0",
    "B_coil",
    "wall_thickness",
    "I_P",
    "qstar",
    "H_fac",
    "scalinglaw",
    "tipeak_over_tepeak",
    "Zeff_target",
    "fuel",
    "j_alpha1",
    "j_alpha2",
    "j_offset",
    "ne_alpha1",
    "ne_alpha2",
    "ne_offset",
    "ni_alpha1",
    "ni_alpha2",
    "ni_offset",
    "Ti_alpha1",
    "Ti_alpha2",
    "Ti_offset",
    "Te_alpha1",
    "Te_alpha2",
    "Te_offset",
    "nmin_frac",
    "nmax_frac",
    "Tmin_keV",
    "Tmax_keV",
    "resistivity_model",
}

# scanning these would change array shapes between cells, or means nothing
UNSCANNABLE_REASONS = {
    "Nn": "the density grid must be the same size in every cell",
    "NTi": "the temperature grid must be the same size in every cell",
    "nr": "the radial grid must be the same size in every cell",
    "name": "it only labels the run",
    "gfilename": "the geometry file is what a scan holds fixed",
    "profsfilename": "the profiles file is what a scan holds fixed",
    "verbosity": "it only controls printing",
    "parallel": "it only controls how the solve is threaded",
    "impurityfractions": "it is an array; scan Zeff_target instead",
    "impurity": "it selects which species Zeff_target applies to",
}

# read() prefers the first key of each pair, so overriding the second without
# removing the first would silently do nothing and produce identical cells
SCAN_SHADOWS = {
    "qstar": ("I_P",),
    "B_coil": ("B_0",),
    "wall_thickness": ("B_0",),
}

# geometry that __get_geometry takes from the gEQDSK when one is supplied,
# overriding whatever the settings file says
GEQDSK_OWNED_KEYS = {"R", "a", "kappa", "delta", "I_P", "qstar"}


# Everything __get_geometry takes from a gEQDSK depends only on the file and
# nr, but it costs contour tracing plus half a dozen spline fits, all in plain
# Python. A scan repeats it identically for every cell, so it is cached.
_GEOMETRY_CACHE = {}


def _compute_gfile_geometry(gfilename: str, nr: int) -> dict:
    gfile = read_eqdsk(gfilename)
    psin, volgrid, agrid, fs = get_fluxvolumes(gfile, nr)
    sqrtpsin = np.linspace(0.001, 0.98, nr)
    volgrid = np.interp(sqrtpsin, np.sqrt(psin), volgrid)
    _, jrms, jtoravg, cross_sec_areas = get_current_density(gfile, nr)
    qpsi = np.asarray(gfile["qpsi"])
    psiq = np.linspace(0, 1, qpsi.shape[0])
    qr = np.interp(sqrtpsin, np.sqrt(psiq), qpsi)

    Ipint = np.abs(np.trapezoid(y=jtoravg, x=cross_sec_areas)) / 1e6
    Jrmsint = np.abs(np.trapezoid(y=jrms, x=cross_sec_areas))
    Jrms_norm = jrms / Jrmsint

    lcfs = fs[-1]
    geq_a = (np.max(lcfs[:, 0]) - np.min(lcfs[:, 0])) / 2
    geq_R = np.max(lcfs[:, 0]) - geq_a
    geq_z0 = (np.max(lcfs[:, 1]) + np.min(lcfs[:, 1])) / 2
    geq_kappa = np.abs(np.max(lcfs[:, 1]) - np.min(lcfs[:, 1])) / (2 * geq_a)
    geq_Rtop = lcfs[np.argmax(lcfs[:, 1]), 0]
    geq_Rbot = lcfs[np.argmin(lcfs[:, 1]), 0]
    geq_delta = ((geq_R - geq_Rtop) / geq_a + (geq_R - geq_Rbot) / geq_a) / 2

    psin_ft, ftrapped_profile = get_trapped_particle_fraction(gfile)
    ftrapped_profile = np.interp(sqrtpsin, np.sqrt(psin_ft), ftrapped_profile)

    return {
        "sqrtpsin": sqrtpsin,
        "volgrid": volgrid,
        "agrid": agrid,
        "qr": qr,
        "Jrms_norm": Jrms_norm,
        "ftrapped_profile": ftrapped_profile,
        "Ipint": Ipint,
        "geq_a": geq_a,
        "geq_R": geq_R,
        "geq_z0": geq_z0,
        "geq_kappa": geq_kappa,
        "geq_delta": geq_delta,
    }


def _gfile_geometry(gfilename: str, nr: int) -> dict:
    """
    Cached gEQDSK geometry, keyed on the file's identity and mtime so an
    edited equilibrium is never served stale.
    """
    key = (os.path.abspath(gfilename), os.stat(gfilename).st_mtime_ns, int(nr))
    if key not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[key] = _compute_gfile_geometry(gfilename, nr)
    cached = _GEOMETRY_CACHE[key]
    # copies, because _addextprof stores the array by reference and
    # np.ascontiguousarray in build_state will not copy an already-contiguous
    # float64 array. Without this, one run could mutate another's geometry
    return {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in cached.items()
    }


def _prepare_output_dir(name, directory, overwrite, default_name):
    """
    Works out where write_output should write, and makes an empty directory
    there. Shared by POPCON and POPCON_scan.
    """
    if name == "":
        stamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", default_name) + "_" + stamp

    if directory is None:
        outputsdir = pathlib.Path.cwd().joinpath("OpenPOPCON_outputs")
    else:
        outputsdir = pathlib.Path(directory)

    direxists = outputsdir.joinpath(name).exists()
    zipexists = outputsdir.joinpath(name + ".zip").exists()
    if not (direxists or zipexists):
        outputsdir.joinpath(name).mkdir(parents=True)
    elif overwrite:
        if zipexists:
            outputsdir.joinpath(name + ".zip").unlink()
        if direxists:
            shutil.rmtree(outputsdir.joinpath(name))
        outputsdir.joinpath(name).mkdir(parents=True)
    else:
        raise ValueError(
            f"{'Archive' * zipexists}{' and ' * zipexists * direxists}{'Directory' * direxists} already exist{'s' * (not (direxists and zipexists))}. Set overwrite=True to overwrite."
        )

    return name, outputsdir, outputsdir.joinpath(name)


def _finalize_output(savedir, outputsdir, name, archive):
    """
    Zips up a finished output directory if asked. Returns what was written.
    """
    if archive:
        written = outputsdir.joinpath(name + ".zip")
        shutil.make_archive(str(outputsdir.joinpath(name)), "zip", savedir)
        shutil.rmtree(savedir)
    else:
        written = savedir
    return written


def _apply_overrides(data: dict, overrides: dict) -> dict:
    """
    Replaces raw settings-file keys, dropping any key that would shadow one
    being set. Without the drop, overriding qstar in a file that also gives
    I_P would be ignored and every cell of a scan would come out identical.
    """
    for key, value in overrides.items():
        for shadowed in SCAN_SHADOWS.get(key, ()):
            data.pop(shadowed, None)
        data[key] = value
    return data


DEFAULT_PLOTSETTINGS = package_resource("default_plotsettings.yml")
DEFAULT_SCALINGLAWS = package_resource("scalinglaws.yml")

# core of calculations
# jit compiled. Module level so cache=True works; jitclasses can't cache

# Parameters the functions below operate on. namedtuple fields can't start
# with an underscore, so the _*defined flags drop it. V is constant after setup
State = namedtuple(
    "State",
    [
        "R",
        "a",
        "kappa",
        "delta",
        "B0",
        "Ip",
        "Itot",
        "M_i",
        "tipeak_over_tepeak",
        "fuel",
        "nr",
        "sqrtpsin",
        "volgrid",
        "agrid",
        "impurityfractions",
        "rdefined",
        "volgriddefined",
        "agriddefined",
        "jdefined",
        "nedefined",
        "nidefined",
        "Tidefined",
        "Tedefined",
        "qdefined",
        "ftrappeddefined",
        "extradefined",
        "j_alpha1",
        "j_alpha2",
        "j_offset",
        "ne_alpha1",
        "ne_alpha2",
        "ne_offset",
        "ni_alpha1",
        "ni_alpha2",
        "ni_offset",
        "Ti_alpha1",
        "Ti_alpha2",
        "Ti_offset",
        "Te_alpha1",
        "Te_alpha2",
        "Te_offset",
        "j_prof",
        "ne_prof",
        "ni_prof",
        "Te_prof",
        "Ti_prof",
        "q_prof",
        "ftrapped_prof",
        "extraprof",
        "H_fac",
        "scaling_const",
        "M_i_alpha",
        "Ip_alpha",
        "R_alpha",
        "a_alpha",
        "kappa_alpha",
        "B0_alpha",
        "Pheat_alpha",
        "n20_alpha",
        "resistivity_alg",
        "verbosity",
        "V",
    ],
)

# -------------------------------------------------------------------
# Profiles and integration
# -------------------------------------------------------------------


@nb.njit(cache=True)
def get_profile(s, rho, profid: int):
    """
    Returns the profile for a given profile ID at a given rho.
    Allows for easy interpolation at runtime and avoids implementing
    many profile functions.
    -3: Area grid
    -2: Sqrt(psin) for geometry
    -1: Volume grid
    0: J profile
    1: n_e profile
    2: n_i profile
    3: Ti profile
    4: Te profile
    5: q profile
    6: ftrapped profile
    7: bavg profile
    """
    if profid == -3:
        if not s.agriddefined:
            raise ValueError("Area grid not defined.")
        return np.interp(rho, s.sqrtpsin, s.agrid)
    elif profid == -2:
        if not s.rdefined:
            raise ValueError("Geometry profiles not defined.")
        return np.interp(rho, s.sqrtpsin, s.sqrtpsin)
    elif profid == -1:
        if not s.volgriddefined:
            raise ValueError("Volume grid not defined.")
        return np.interp(rho, s.sqrtpsin, s.volgrid)
    elif profid == 0:
        if not s.jdefined:
            raise ValueError("J profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.j_prof)
    elif profid == 1:
        if not s.nedefined:
            raise ValueError("n_e profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.ne_prof)
    elif profid == 2:
        if not s.nidefined:
            raise ValueError("n_i profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.ni_prof)
    elif profid == 3:
        if not s.Tidefined:
            raise ValueError("Ti profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.Ti_prof)
    elif profid == 4:
        if not s.Tedefined:
            raise ValueError("Te profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.Te_prof)
    elif profid == 5:
        if not s.qdefined:
            raise ValueError("q profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.q_prof)
    elif profid == 6:
        if not s.ftrappeddefined:
            raise ValueError("ftrapped profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.ftrapped_prof)
    elif profid == 7:
        if not s.extradefined:
            raise ValueError("extra profile not defined.")
        return np.interp(rho, s.sqrtpsin, s.extraprof)
    else:
        raise ValueError("Invalid profile ID.")


@nb.njit(cache=True)
def volume_integral(s, rho, func) -> float:
    r"""
    Integrates a function of rho dV-like:

    $\int_0^1 func(\rho) \frac{dV}{d\rho} d\rho$
    """
    # NOTE: profile must be an array of the same length as rho.
    # Integrates functions of rho dV-like
    V_interp = np.interp(rho, s.sqrtpsin, s.volgrid)
    return np.trapezoid(func, V_interp)


# -------------------------------------------------------------------
# Physical Quantities
# -------------------------------------------------------------------


@nb.njit(cache=True)
def Zeff(s, T_e_keV) -> float:
    """
    Effective ion charge.

    Zeff = sum ( n_i * Z_i^2 ) / n_e
    """
    # n_i = n20 * species fraction
    # n_e = n20 / dilution
    # n20 cancels

    # Hydrogen isotopes + impurities
    return (
        (1 - np.sum(s.impurityfractions))
        + np.sum(s.impurityfractions * phys.get_Zeffs(T_e_keV) ** 2)
    ) * plasma_dilution(s, T_e_keV)


@nb.njit(cache=True)
def plasma_dilution(s, T_e_keV) -> float:
    """
    Plasma dilution factor; number of ions per electron. Always <=1.
    """
    dil = 1 / (1 + np.sum(s.impurityfractions * phys.get_Zeffs(T_e_keV)))
    return dil


# coefficient for Spitzer conductivity, necessary to obtain ohmic power
# def get_Cspitz(self, volavgcurr:bool, T0):
#     Fz    = (1+1.198*self.Zeff(T0) + 0.222*self.Zeff(T0)**2)/(1+2.966*self.Zeff(T0) + 0.753*self.Zeff(T0)**2)
#     eta1  = 1.03e-4*self.Zeff(T0)*Fz
#     j0avg = self.Ip/(np.pi*self.a**2*self.kappa)*1.0e6
#     if (volavgcurr == True):
#         # TODO: Change this to use volgrid for averaging
#         Cspitz = eta1*self.q_a*j0avg**2
#     else:
#         Cspitz = eta1
#     Cspitz /= 1.6e-16*1.0e20 #unit conversion to keV 10^20 m^-3
#     return Cspitz

# def get_eta_spitzer(self, rho, T0, n20):
#     # Calculate the Spitzer resistivity in Ohm-m
#     # eta_spitzer = 4 sqrt(2pi)/3 Z_eff e^2 sqrt(m_e) ln(Lambda) / (4 pi epsilon_0)^2 T_e^(3/2)
#     # Const = 4 sqrt(2pi)/3 * 1.602e-19^2 * 9.109e-31 / ( (4 pi * 8.854e-12)^2 * 1.602e-16^3/2 )
#     # eta = const*Z_eff*ln(Lambda)*(T_e (keV))^(-3/2)
#     return


@nb.njit(cache=True)
def eta_NC(s, rho, T_e_keV: float, n_e_20: float):
    """
    Neoclassical resistivity in Ohm-m.

    Equations 16-17 from [1] Jardin et al. 1993, or equation 6 from [8] Paz-Soldan et al. 2016
    """

    if np.any(rho <= 0):
        raise ValueError(
            "Invalid rho value. Neoclassical resistivity not defined at rho=0."
        )
    T_e_r = T_e_keV * get_profile(s, rho, 4)
    n_e_r = 1e20 * n_e_20 * get_profile(s, rho, 1)
    q = get_profile(s, rho, 5)
    Zeffprof = np.empty_like(T_e_r)
    for i in np.arange(T_e_r.shape[0]):
        Zeffprof[i] = Zeff(s, T_e_r[i])
    # logLambda = 17.1-np.log(np.sqrt(n_e_r)/(T_e_r*1e3)) # From Jardin
    logLambda = np.empty_like(Zeffprof)
    where = T_e_r * 1e3 > 10 * Zeffprof**2
    elsewhere = np.logical_not(where)
    logLambda[where] = (
        24 - np.log(np.sqrt(n_e_r) / (T_e_r * 1e3))[where]
    )  # Plasma Formulary
    logLambda[elsewhere] = (
        23 - np.log(np.sqrt(n_e_r) * Zeffprof / (T_e_r * 1e3) ** (3 / 2))[elsewhere]
    )  # Plasma Formulary
    logLambda[logLambda < 5] = 5
    eta_C = 1.03e-4 * logLambda * (T_e_r * 1e3) ** (-3 / 2)

    Lambda_E = 3.4 / Zeffprof * (1.13 + Zeffprof) / (2.67 + Zeffprof)
    C_R = 0.56 / Zeffprof * (3.0 - Zeffprof) / (3.0 + Zeffprof)
    xi = 0.58 + 0.2 * Zeffprof
    invaspect = s.a / s.R

    if not s.ftrappeddefined:
        # Cap possible trapped particle fraction at 1
        f_t = np.minimum(np.sqrt(2 * rho * invaspect), 1.0)
        nu_star_e = (
            1
            / 10.2e16
            * s.R
            * q
            * n_e_r
            * np.exp(logLambda)
            / (f_t * invaspect * (T_e_r * 1e3) ** 2)
        )
        eta_C_eta_NC_ratio = (
            Lambda_E
            * (1 - f_t / (1 + xi * nu_star_e))
            * (1 - C_R * f_t / (1 + xi * nu_star_e))
        )
    else:
        f_t_prof = get_profile(s, rho, 6)
        nu_star_e = (
            1
            / 10.2e16
            * s.R
            * q
            * n_e_r
            * np.exp(logLambda)
            / (f_t_prof * invaspect * (T_e_r * 1e3) ** 2)
        )
        eta_C_eta_NC_ratio = (
            Lambda_E
            * (1 - f_t_prof / (1 + xi * nu_star_e))
            * (1 - C_R * f_t_prof / (1 + xi * nu_star_e))
        )

    eta_NC = eta_C / eta_C_eta_NC_ratio

    if s.resistivity_alg == 0:
        return eta_NC

    f_CL = (1 + 1.2 * Zeffprof + 0.22 * Zeffprof**2) / (
        1 + 3 * Zeffprof + 0.75 * Zeffprof**2
    )

    f_NC = 1 / (1 - np.sqrt(2 * invaspect / (1 + invaspect)))

    eta_NC_2 = eta_C * f_CL * f_NC

    if s.resistivity_alg == 1:
        return eta_NC_2

    if s.resistivity_alg == 2:
        eta_NC_max = np.empty_like(eta_NC)

        for i in range(len(eta_NC)):
            eta_NC_max[i] = max(eta_NC[i], eta_NC_2[i])

        return eta_NC_max

    else:
        raise ValueError("Invalid resistivity algorithm. Must be 0, 1, or 2.")


@nb.njit(cache=True)
def Vloop(s, T_e_keV, n_e_20) -> float:
    """
    Loop Voltage in Volts.

    V_loop = P_ohmic / Ip
    """
    P_OH = volume_integral(s, s.sqrtpsin, _P_OH_prof(s, s.sqrtpsin, T_e_keV, n_e_20))
    return P_OH / s.Ip


@nb.njit(cache=True)
def BetaN(s, T_i_keV, n_e_20) -> float:
    """
    Normalized beta, beta*a*B0/Ip.

    beta = 2 mu0 <P> / (B^2)
    <P> = int P dV / V (average pressure)
    Since Pressure per m^3 is n_eT_e+n_iT_i, 
    and W_tot per m^3 is 3/2 (n_eT_e+n_iT_i),
    P = 2/3 W_tot
    beta_N = beta a B0 / (Ip)
    """
    P_avg = (
        1e6
        * volume_integral(s, s.sqrtpsin, (2/3) * _W_tot_prof(s, s.sqrtpsin, T_i_keV, n_e_20))
        / s.V
    )
    beta = 2 * (4e-7 * np.pi) * P_avg / (s.B0**2)
    return beta * s.a * s.B0 / s.Ip


@nb.njit(cache=True)
def tauE_scalinglaw(s, Pheat, n_e_20) -> float:
    """
    User-chosen confinement time scaling law
    """
    tauE = s.H_fac * s.scaling_const
    tauE *= s.M_i**s.M_i_alpha
    tauE *= s.Ip**s.Ip_alpha
    tauE *= s.R**s.R_alpha
    tauE *= s.a**s.a_alpha
    tauE *= s.kappa**s.kappa_alpha
    tauE *= s.B0**s.B0_alpha
    tauE *= Pheat**s.Pheat_alpha
    tauE *= n_e_20**s.n20_alpha
    return tauE


@nb.njit(cache=True)
def tauE_H98(s, Pheat, n_e_20) -> float:
    """
    H98y2 scaling law for comparison
    """
    tauE = 0.145
    # M_i**(0.19)*Ip**(0.93)*R**(1.39)*a**(0.58)*kappa**(0.78)*B0**(0.15)*Pheat**(-0.69)*n20**(0.41)
    tauE *= s.M_i**0.19
    tauE *= s.Ip**0.93
    tauE *= s.R**1.39
    tauE *= s.a**0.58
    tauE *= s.kappa**0.78
    tauE *= s.B0**0.15
    tauE *= Pheat ** (-0.69)
    tauE *= n_e_20**0.41
    return tauE


@nb.njit(cache=True)
def tauE_H89(s, Pheat, n_e_20) -> float:
    """
    H89 scaling law for comparison
    """
    # 0.048*M_i**(0.5)*Ip**(0.85)*R**(1.2)*a**(0.3)*kappa**(0.5)*B0**(0.2)*Pheat**(-0.5)*n20**(0.1)
    tauE = 0.048
    tauE *= s.M_i**0.5
    tauE *= s.Ip**0.85
    tauE *= s.R**1.2
    tauE *= s.a**0.3
    tauE *= s.kappa**0.5
    tauE *= s.B0**0.2
    tauE *= Pheat ** (-0.5)
    tauE *= n_e_20**0.1
    return tauE


# -------------------------------------------------------------------
# Power profiles
# -------------------------------------------------------------------


@nb.njit(cache=True)
def _W_tot_prof(s, rho, T_i_keV: float, n_e_20: float):
    """
    Plasma energy per cubic meter.
    """
    n_e_r = 1e20 * n_e_20 * get_profile(s, rho, 1)
    T_e_r = T_i_keV * get_profile(s, rho, 4) / s.tipeak_over_tepeak
    dil = plasma_dilution(s, T_i_keV / s.tipeak_over_tepeak)
    n_i_r = 1e20 * n_e_20 * get_profile(s, rho, 2) * dil
    T_i_r = T_i_keV * get_profile(s, rho, 3)
    # W_density = 3/2 * n_i * T_i
    # = 3/2 * (n_i_r) ( 1.60218e-22 * T_i_r (keV) ) (MJ/m^3)
    return 3 / 2 * 1.60218e-22 * (n_i_r * T_i_r + n_e_r * T_e_r)


@nb.njit(cache=True)
def _P_DDpT_prof(s, rho, T_i_keV: float, n_i_20: float):
    """
    D(d,p)T power per cubic meter
    """
    if s.fuel == 1:
        dfrac = 1
    elif s.fuel == 2:
        dfrac = 0.5
    else:
        raise ValueError("Invalid fuel cycle.")

    n_i_r = 1e20 * n_i_20 * get_profile(s, rho, 2)
    T_i_r = T_i_keV * get_profile(s, rho, 3)

    # reaction frequency f = n_D/sqrt(2) * n_D/sqrt(2) * <sigma v> (1/s)
    # 1.60218e-22 is the conversion factor from keV to MJ
    # <sigma v> is in cm^3/s so we need to convert to m^3/s, hence 1e-6
    # P_DD_density = 1.60218e-22 * f(DD->pT) * (T_tritium + T_proton)  (MW/m^3)

    f_ddpt = (
        np.power((dfrac * n_i_r / (np.sqrt(2))), 2)
        * phys.get_reactivity(T_i_r, 2)
        * 1e-6
    )
    return 1.60218e-22 * f_ddpt * (1.01e3 + 3.02e3)


@nb.njit(cache=True)
def _P_DDnHe3_prof(s, rho, T_i_keV: float, n_i_20: float):
    """
    D(d,n)He3 power per cubic meter
    """
    if s.fuel == 1:
        dfrac = 1 - np.sum(s.impurityfractions)
    elif s.fuel == 2:
        dfrac = 0.5 - np.sum(s.impurityfractions) / 2
    else:
        raise ValueError("Invalid fuel cycle.")

    n_i_r = 1e20 * n_i_20 * get_profile(s, rho, 2)
    T_i_r = T_i_keV * get_profile(s, rho, 3)

    # See _P_DDpT_prof for explanation.
    # Note, for heating, only the He3 heats the plasma. For heating purposes,
    # multiply by 0.82e3/(2.45e3 + 0.82e3).

    f_ddnhe3 = (
        np.power((dfrac * n_i_r / (np.sqrt(2))), 2)
        * phys.get_reactivity(T_i_r, 3)
        * 1e-6
    )
    return 1.60218e-22 * f_ddnhe3 * (2.45e3 + 0.82e3)


@nb.njit(cache=True)
def _P_DTnHe4_prof(s, rho, T_i_keV: float, n_i_20: float):
    """
    T(d,n)He4 power density in MW/m^3
    """
    if s.fuel == 1:
        dfrac = 1 - np.sum(s.impurityfractions)
        tfrac = 0
    elif s.fuel == 2:
        dfrac = 0.5 - np.sum(s.impurityfractions) / 2
        tfrac = 0.5 - np.sum(s.impurityfractions) / 2
    else:
        raise ValueError("Invalid fuel cycle.")

    n_i_r = 1e20 * n_i_20 * get_profile(s, rho, 2)
    T_i_r = T_i_keV * get_profile(s, rho, 3)

    # See _P_DDpT_prof for explanation.
    # Note, for heating, only the He4 / alpha heats the plasma. For heating purposes,
    # multiply by 3.52e3/(3.52e3 + 14.06e3).

    f_dtnhe4 = dfrac * (n_i_r) * tfrac * (n_i_r) * phys.get_reactivity(T_i_r, 1) * 1e-6
    return 1.60218e-22 * f_dtnhe4 * (3.52e3 + 14.06e3)


@nb.njit(cache=True)
def _P_fusion_heating(s, rho, T_i_keV: float, n_i_20: float):
    """
    D-D and D-T heating power density in MW/m^3
    """
    return (
        _P_DDpT_prof(s, rho, T_i_keV, n_i_20)
        + _P_DDnHe3_prof(s, rho, T_i_keV, n_i_20) * (0.82e3 / (2.45e3 + 0.82e3))
        + _P_DTnHe4_prof(s, rho, T_i_keV, n_i_20) * (3.52e3 / (3.52e3 + 14.06e3))
    )


@nb.njit(cache=True)
def _P_fusion(s, rho, T_i_keV: float, n_i_20: float):
    """
    Fusion power density in MW/m^3
    """
    return (
        _P_DDpT_prof(s, rho, T_i_keV, n_i_20)
        + _P_DDnHe3_prof(s, rho, T_i_keV, n_i_20)
        + _P_DTnHe4_prof(s, rho, T_i_keV, n_i_20)
    )


@nb.njit(cache=True)
def _P_brem_rad(s, rho, T_e_keV: float, n_e_20: float):
    """
    Radiative power density in MW/m^3; See formulary.
    """

    T_e_r = T_e_keV * get_profile(s, rho, 4)
    n_e_r = 1e20 * n_e_20 * get_profile(s, rho, 1)

    total_Zeff = np.empty(T_e_r.shape[0], dtype=np.float64)
    for i in np.arange(T_e_r.shape[0]):
        total_Zeff[i] = Zeff(s, T_e_r[i])
    G = 1.1  # Gaunt factor
    # P_brem = 5.35e-3*1e-6*G*total_Zeff*(1e-20*n_e_r)**2*T_e_r**0.5
    # From Plasma Formulary
    P_brem = G * 1e-6 * np.sqrt(1000 * T_e_r) * total_Zeff * (n_e_r / 7.69e18) ** 2
    return P_brem


@nb.njit(cache=True)
def _P_synch(s, rho, T_e_keV: float, n_e_20: float):
    """
    Synchrotron radiation power density in MW/m^3; see Zohm 2019.
    """
    T_e_r = T_e_keV * get_profile(s, rho, 4)
    n_e_r = n_e_20 * get_profile(s, rho, 1)
    P_synch = (
        1.32e-7
        * (s.B0 * T_e_r) ** 2.5
        * np.sqrt(n_e_r / s.a)
        * (1 + 18 * s.a / (s.R * np.sqrt(T_e_r)))
    )

    return P_synch


@nb.njit(cache=True)
def _P_impurity_rad(s, rho, T_e_keV: float, n_e_20: float):
    """
    Radiative power density from impurities in MW/m^3; see Zohm 2019.
    """
    T_e_r = T_e_keV * get_profile(s, rho, 4)
    n_e_r = 1e20 * n_e_20 * get_profile(s, rho, 1)
    T_e_r[T_e_r < 0.05] = 0.05
    Lz = np.empty((T_e_r.shape[0], 6), dtype=np.float64)
    for i in np.arange(T_e_r.shape[0]):
        Lz[i, :] = phys.get_rads(T_e_r[i])

    P_line = np.sum(1e-6 * (Lz.T * (n_e_r) ** 2).T * s.impurityfractions, axis=1)
    return P_line


@nb.njit(cache=True)
def _P_rad(s, rho, T_e_keV: float, n_e_20: float):
    """
    Total radiative power density in MW/m^3.
    """
    return (
        _P_brem_rad(s, rho, T_e_keV, n_e_20)
        + _P_impurity_rad(s, rho, T_e_keV, n_e_20)
        + _P_synch(s, rho, T_e_keV, n_e_20)
    )


@nb.njit(cache=True)
def _P_OH_prof(s, rho, T_e_keV: float, n_e_20: float):
    """
    Ohmic power density in MW/m^3
    """

    eta_nc = eta_NC(s, rho, T_e_keV, n_e_20)
    J = s.Itot * 1e6 * get_profile(s, rho, 0)
    return 1e-6 * eta_nc * J**2


@nb.njit(cache=True)
def Q_fusion(s, T_i_keV: float, n_e_20: float, Paux: float) -> float:
    """
    Physical fusion gain factor.
    """
    T_e_keV = T_i_keV / s.tipeak_over_tepeak
    dil = plasma_dilution(s, T_e_keV)
    n_i_20 = n_e_20 * dil
    P_fusion = volume_integral(s, s.sqrtpsin, _P_fusion(s, s.sqrtpsin, T_i_keV, n_i_20))
    P_OH = volume_integral(s, s.sqrtpsin, _P_OH_prof(s, s.sqrtpsin, T_e_keV, n_e_20))
    if Paux + P_OH <= 0.0:
        return 9999999.0
    else:
        return P_fusion / (Paux + P_OH)


# -----------------------------------------------------------------------
# Power balance solver
# -----------------------------------------------------------------------


@nb.njit(cache=True)
def P_aux_impfrac(s, n_e_20, T_i_keV):
    """
    Closed-form power balance solver for the impurity-fraction-fixed mode.

    Returns (P_aux, flag), where flag is INVALID_OK, NOAUX, INVALID_IMPRAD
    or INVALID_NAN. The caller reports these in aggregate; printing per point
    would be one line per grid cell, out of order under prange.
    """
    T_e_keV = T_i_keV / s.tipeak_over_tepeak
    dil = plasma_dilution(s, T_e_keV)
    n_i_20 = n_e_20 * dil
    line_average_fac = np.average(get_profile(s, s.sqrtpsin, 1))
    P_fusion_heating = volume_integral(
        s, s.sqrtpsin, _P_fusion_heating(s, s.sqrtpsin, T_i_keV, n_i_20)
    )
    P_ohmic_heating = volume_integral(
        s, s.sqrtpsin, _P_OH_prof(s, s.sqrtpsin, T_e_keV, n_e_20)
    )
    W_tot = volume_integral(s, s.sqrtpsin, _W_tot_prof(s, s.sqrtpsin, T_i_keV, n_e_20))
    P_brem = volume_integral(s, s.sqrtpsin, _P_brem_rad(s, s.sqrtpsin, T_e_keV, n_e_20))
    P_synch = volume_integral(s, s.sqrtpsin, _P_synch(s, s.sqrtpsin, T_e_keV, n_e_20))
    P_imp = volume_integral(
        s, s.sqrtpsin, _P_impurity_rad(s, s.sqrtpsin, T_e_keV, n_e_20)
    )
    alpha = s.Pheat_alpha
    K = tauE_scalinglaw(s, 1.0, n_e_20 * line_average_fac)
    # Balance P_heat = W_tot / tauE = (W_tot/K) * P_heat^(-alpha)
    P_heat = (W_tot / K) ** (1.0 / (1.0 + alpha))
    # P_heat = P_aux + P_OH + P_fusion_heating - P_brem - P_synch
    P_aux = P_heat - P_ohmic_heating - P_fusion_heating + P_brem + P_synch

    flag = INVALID_OK

    # A negative P_aux means the point already heats itself past the
    # confinement loss with no auxiliary power at all
    if P_aux < 0.0:
        P_aux = 0.0
        flag = NOAUX

    # P_SOL < 0: impurity radiation exceeds the confinement loss, which at
    # balance equals P_heat. The impurity power is assumed to displace what
    # would otherwise be conductive/turbulent/instability-driven loss, so it
    # radiates power away that would otherwise cross the separatrix. Such a
    # state is not physical; flag it.
    if P_imp > P_heat:
        P_aux = UNPHYSICAL
        flag = INVALID_IMPRAD

    if np.isnan(P_aux):
        P_aux = UNPHYSICAL
        flag = INVALID_NAN

    return P_aux, flag


# NOT jit compiled
class POPCON_algorithms:
    """
    Class POPCON_algorithms

    This class is the mathematical backbone of the OpenPOPCON code. Its
    calculations are compiled with numba to provide fast calculations for
    scans. It is best used in Jupyter notebooks or scripts where the user
    wants to conduct a scan, as it is compiled at runtime; this means that
    the first time the code executes, it will take a bit longer to compile,
    but the subsequent runs will be much faster than raw Python code. The
    compiled functions are cached to disk, so that first compile happens
    once per machine; later sessions load it back.

    See __init__ for a list of parameters and their descriptions.

    Setup writes to the attributes below; build_state then freezes them
    into the State passed to the jit compiled functions.

    Properties:
    - n_GR: Greenwald density in 10^20/m^3
    - V: Plasma volume in m^3
    - A: Last closed flux surface area in m^2

    Profiles and integration:
    - get_profile(rho, profid): Returns J, ne, Ti, etc profiles
      for an array or single point, rho. Profile is interpolated.
    - volume_integral(rho, func): Integrates a function of rho dV-like

    Physical Quantities:
    - rho: ~sqrt(psi_norm), normalized radial coordinate
    - Zeff(T_e_keV): Effective ion charge
    - plasma_dilution(T_e_keV): Plasma dilution factor; number of ions per electron
    - eta_NC(rho, T_e_keV, n_e_20): Neoclassical resistivity in Ohm-m
    - Vloop(T_e_keV, n_e_20): Loop voltage, P_ohmic / Ip
    - BetaN(T_i_keV, n_e_20): Normalized beta, beta*a*B0/Ip
    - tauE_scalinglaw(Pheat, n_e_20): Chosen confinement time scaling law
    - tauE_H98(Pheat, n_e_20): H98y2 scaling law for comparison
    - tauE_H89(Pheat, n_e_20): H89 scaling law for comparison

    Power profiles:
    - _W_tot_prof(rho, T_i_keV, n_e_20): Plasma energy per cubic meter
    - _P_DDpT_prof(rho, T_i_keV, n_i_20): D-D -> p, T fusion power per cubic meter
    - _P_DDnHe3_prof(rho, T_i_keV, n_i_20): D-D -> n, He3 fusion power per cubic meter
    - _P_DTnHe4_prof(rho, T_i_keV, n_i_20): D-T -> n, He4 fusion power per cubic meter
    - _P_fusion_heating(rho, T_i_keV, n_i_20): Fusion power absorbed by plasma per cubic meter (excluding neutrons)
    - _P_fusion(rho, T_i_keV, n_i_20): Total fusion power per cubic meter (including neutrons)
    - _P_brem_rad(rho, T_e_keV, n_e_20): Bremsstrahlung radiation power per cubic meter
    - _P_impurity_rad(rho, T_e_keV, n_e_20): Impurity radiation power per cubic meter
    - _P_rad(rho, T_e_keV, n_e_20): Total radiation power per cubic meter
    - _P_OH_prof(rho, T_e_keV, n_e_20): Ohmic power per cubic meter
    - Q_fusion(T_i_keV, n_e_20, Paux): Physical fusion gain factor
    """

    def __init__(self) -> None:

        # ---------------------------------------------------------------
        # Machine parameters
        # ---------------------------------------------------------------

        self.R: float  # Major radius in meters
        self.a: float  # Minor radius in meters
        self.kappa: float  # Elongation
        self.delta: float  # Triangularity
        self.B0: float  # Magnetic field at axis in Tesla
        self.Ip: float  # Plasma current in MA
        self.Itot: float  # Total current in MA
        self.M_i: float  # Ion mass in amu
        self.tipeak_over_tepeak: float  # Ti/Te peak ratio
        # *** 1 = D-D, 2 = D-T, 3 = D-He3 ***
        self.fuel: int  # Fuel cycle

        # ---------------------------------------------------------------
        # Geometry/profile parameters
        # ---------------------------------------------------------------
        self.sqrtpsin = np.empty(
            0, dtype=np.float64
        )  # sqrt(psin) corresponding to profiles
        self.volgrid = np.empty(0, dtype=np.float64)  # Flux surface volumes
        self.agrid = np.empty(0, dtype=np.float64)  # Flux surface surface areas
        self.nr: int  # Number of radial points
        # 0 = He, 1 = Ne, 2 = Ar, 3 = Kr, 4 = Xe, 5 = W
        self.impurityfractions = np.empty(
            6, dtype=np.float64
        )  # Impurity fractions for each impurity, relative to ion density
        self.rdefined: bool = False  # Whether radial grid is defined
        self.volgriddefined: bool = False  # Whether volume grid is defined
        self.agriddefined: bool = False  # Whether area grid is defined

        # Impurity profiles are not implemented yet
        self.extprof_imps: bool = (
            False  # Whether impurity profiles are defined (not implemented)
        )
        self.impsdefined: bool = False  # Whether impurity fractions are defined

        self._jdefined: bool = False  # Whether J profile is defined
        self._nedefined: bool = False  # Whether ne profile is defined
        self._nidefined: bool = False  # Whether ni profile is defined
        self._Tidefined: bool = False  # Whether Ti profile is defined
        self._Tedefined: bool = False  # Whether Te profile is defined
        self._qdefined: bool = False  # Whether q profile is defined
        self._ftrappeddefined: bool = (
            False  # Whether b_max profile is defined (not implemented)
        )
        self._extradefined: bool = (
            False  # Whether b_avg profile is defined (not implemented)
        )

        # ---------------------------------------------------------------
        # Parameters for parabolic profiles case
        # ---------------------------------------------------------------
        """
        Normalized parabolic profiles (or polynomial-like) are defined:
        f(rho) = (1 - offset) * (1 + rho^alpha1)^alpha2 + offset
        """

        self.j_alpha1: float = 2.0
        self.j_alpha2: float = 3.0
        self.j_offset: float = 0.0

        self.ne_alpha1: float = 2.0
        self.ne_alpha2: float = 1.5
        self.ne_offset: float = 0.3

        self.ni_alpha1: float = 2.0
        self.ni_alpha2: float = 1.5
        self.ni_offset: float = 0.3

        self.Ti_alpha1: float = 2.0
        self.Ti_alpha2: float = 1.5
        self.Ti_offset: float = 0.0

        self.Te_alpha1: float = 2.0
        self.Te_alpha2: float = 1.5
        self.Te_offset: float = 0.0

        # ---------------------------------------------------------------
        # Profiles
        # ---------------------------------------------------------------
        """
        All profiles are defined as arrays of the same length as
        sqrtpsin, and **normalized**!
        """
        self.j_prof = np.empty(0, dtype=np.float64)  # Current density profile
        self.ne_prof = np.empty(0, dtype=np.float64)  # Electron density profile
        self.ni_prof = np.empty(0, dtype=np.float64)  # Ion density profile
        self.Te_prof = np.empty(0, dtype=np.float64)  # Electron temperature profile
        self.Ti_prof = np.empty(0, dtype=np.float64)  # Ion temperature profile
        self.q_prof = np.empty(0, dtype=np.float64)  # Safety factor profile
        self.ftrapped_prof = np.empty(
            0, dtype=np.float64
        )  # Maximum B field on flux surface (not implemented)
        self.extraprof = np.empty(
            0, dtype=np.float64
        )  # Average B field on flux surface (not implemented)
        self.extprof_impfracs = np.empty(
            0, dtype=np.float64
        )  # Impurity fractions profile (not implemented)

        # ---------------------------------------------------------------
        # Scaling law parameters
        # ---------------------------------------------------------------

        self.H_fac: float = 1.0  # H (scaling law enhancement) factor
        self.scaling_const: float = 0.145  # Scaling law coefficient
        self.M_i_alpha: float = 0.19  # Ion mass scaling law exponent
        self.Ip_alpha: float = 0.93  # Plasma current scaling law exponent
        self.R_alpha: float = 1.39  # Major radius scaling law exponent
        self.a_alpha: float = 0.58  # Minor radius scaling law exponent
        self.kappa_alpha: float = 0.78  # Elongation scaling law exponent
        self.B0_alpha: float = 0.15  # Toroidal field scaling law exponent
        self.Pheat_alpha: float = -0.69  # Power degradation scaling law exponent
        self.n20_alpha: float = 0.41  # Greenwald density scaling law exponent

        # ---------------------------------------------------------------
        # Etc
        # ---------------------------------------------------------------
        self.resistivity_alg: int = (
            0  # Resistivity algorithm. 0 = Jardin, 1 = Paz-Soldan, 2 = local maximum
        )
        self.verbosity: int = 0  # Verbosity level. 0 = silent, 1 = normal, 2 = debug, 3 = print all matrices

        self.state = None  # Set by build_state after setup

        pass

    # -------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------

    def build_state(self) -> State:
        """
        Freezes the setup parameters into the State the jit compiled
        functions take. Casts every field explicitly so the type
        signature stays the same between sessions; the compile cache
        keys on it.
        """

        def arr(x):
            return np.ascontiguousarray(x, dtype=np.float64)

        s = State(
            R=np.float64(self.R),
            a=np.float64(self.a),
            kappa=np.float64(self.kappa),
            delta=np.float64(self.delta),
            B0=np.float64(self.B0),
            Ip=np.float64(self.Ip),
            Itot=np.float64(self.Itot),
            M_i=np.float64(self.M_i),
            tipeak_over_tepeak=np.float64(self.tipeak_over_tepeak),
            fuel=int(self.fuel),
            nr=int(self.nr),
            sqrtpsin=arr(self.sqrtpsin),
            volgrid=arr(self.volgrid),
            agrid=arr(self.agrid),
            impurityfractions=arr(self.impurityfractions),
            rdefined=bool(self.rdefined),
            volgriddefined=bool(self.volgriddefined),
            agriddefined=bool(self.agriddefined),
            jdefined=bool(self._jdefined),
            nedefined=bool(self._nedefined),
            nidefined=bool(self._nidefined),
            Tidefined=bool(self._Tidefined),
            Tedefined=bool(self._Tedefined),
            qdefined=bool(self._qdefined),
            ftrappeddefined=bool(self._ftrappeddefined),
            extradefined=bool(self._extradefined),
            j_alpha1=np.float64(self.j_alpha1),
            j_alpha2=np.float64(self.j_alpha2),
            j_offset=np.float64(self.j_offset),
            ne_alpha1=np.float64(self.ne_alpha1),
            ne_alpha2=np.float64(self.ne_alpha2),
            ne_offset=np.float64(self.ne_offset),
            ni_alpha1=np.float64(self.ni_alpha1),
            ni_alpha2=np.float64(self.ni_alpha2),
            ni_offset=np.float64(self.ni_offset),
            Ti_alpha1=np.float64(self.Ti_alpha1),
            Ti_alpha2=np.float64(self.Ti_alpha2),
            Ti_offset=np.float64(self.Ti_offset),
            Te_alpha1=np.float64(self.Te_alpha1),
            Te_alpha2=np.float64(self.Te_alpha2),
            Te_offset=np.float64(self.Te_offset),
            j_prof=arr(self.j_prof),
            ne_prof=arr(self.ne_prof),
            ni_prof=arr(self.ni_prof),
            Te_prof=arr(self.Te_prof),
            Ti_prof=arr(self.Ti_prof),
            q_prof=arr(self.q_prof),
            ftrapped_prof=arr(self.ftrapped_prof),
            extraprof=arr(self.extraprof),
            H_fac=np.float64(self.H_fac),
            scaling_const=np.float64(self.scaling_const),
            M_i_alpha=np.float64(self.M_i_alpha),
            Ip_alpha=np.float64(self.Ip_alpha),
            R_alpha=np.float64(self.R_alpha),
            a_alpha=np.float64(self.a_alpha),
            kappa_alpha=np.float64(self.kappa_alpha),
            B0_alpha=np.float64(self.B0_alpha),
            Pheat_alpha=np.float64(self.Pheat_alpha),
            n20_alpha=np.float64(self.n20_alpha),
            resistivity_alg=int(self.resistivity_alg),
            verbosity=int(self.verbosity),
            V=np.float64(0.0),
        )
        # V through the same jit compiled path as before
        v = volume_integral(s, s.sqrtpsin, np.ones_like(s.sqrtpsin))
        self.state = s._replace(V=np.float64(v))
        return self.state

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def n_GR(self):
        """
        Greenwald density in 10^20/m^3
        """
        return self.Ip / (np.pi * self.a**2)

    @property
    def V(self):
        """
        Plasma volume (in last closed flux surface) in m^3
        """
        return self.state.V

    @property
    def A(self):
        """
        Plasma surface area (in last closed flux surface) in m^2
        """
        return self.agrid[-1]

    # -------------------------------------------------------------------
    # Calls to the jit compiled functions above
    # -------------------------------------------------------------------

    def get_profile(self, rho, profid: int):
        return get_profile(self.state, rho, profid)

    def volume_integral(self, rho, func) -> float:
        return volume_integral(self.state, rho, func)

    def Zeff(self, T_e_keV) -> float:
        return Zeff(self.state, T_e_keV)

    def plasma_dilution(self, T_e_keV) -> float:
        return plasma_dilution(self.state, T_e_keV)

    def eta_NC(self, rho, T_e_keV, n_e_20):
        return eta_NC(self.state, rho, T_e_keV, n_e_20)

    def Vloop(self, T_e_keV, n_e_20) -> float:
        return Vloop(self.state, T_e_keV, n_e_20)

    def BetaN(self, T_i_keV, n_e_20) -> float:
        return BetaN(self.state, T_i_keV, n_e_20)

    def tauE_scalinglaw(self, Pheat, n_e_20) -> float:
        return tauE_scalinglaw(self.state, Pheat, n_e_20)

    def tauE_H98(self, Pheat, n_e_20) -> float:
        return tauE_H98(self.state, Pheat, n_e_20)

    def tauE_H89(self, Pheat, n_e_20) -> float:
        return tauE_H89(self.state, Pheat, n_e_20)

    def _W_tot_prof(self, rho, T_i_keV, n_e_20):
        return _W_tot_prof(self.state, rho, T_i_keV, n_e_20)

    def _P_DDpT_prof(self, rho, T_i_keV, n_i_20):
        return _P_DDpT_prof(self.state, rho, T_i_keV, n_i_20)

    def _P_DDnHe3_prof(self, rho, T_i_keV, n_i_20):
        return _P_DDnHe3_prof(self.state, rho, T_i_keV, n_i_20)

    def _P_DTnHe4_prof(self, rho, T_i_keV, n_i_20):
        return _P_DTnHe4_prof(self.state, rho, T_i_keV, n_i_20)

    def _P_fusion_heating(self, rho, T_i_keV, n_i_20):
        return _P_fusion_heating(self.state, rho, T_i_keV, n_i_20)

    def _P_fusion(self, rho, T_i_keV, n_i_20):
        return _P_fusion(self.state, rho, T_i_keV, n_i_20)

    def _P_brem_rad(self, rho, T_e_keV, n_e_20):
        return _P_brem_rad(self.state, rho, T_e_keV, n_e_20)

    def _P_synch(self, rho, T_e_keV, n_e_20):
        return _P_synch(self.state, rho, T_e_keV, n_e_20)

    def _P_impurity_rad(self, rho, T_e_keV, n_e_20):
        return _P_impurity_rad(self.state, rho, T_e_keV, n_e_20)

    def _P_rad(self, rho, T_e_keV, n_e_20):
        return _P_rad(self.state, rho, T_e_keV, n_e_20)

    def _P_OH_prof(self, rho, T_e_keV, n_e_20):
        return _P_OH_prof(self.state, rho, T_e_keV, n_e_20)

    def Q_fusion(self, T_i_keV, n_e_20, Paux) -> float:
        return Q_fusion(self.state, T_i_keV, n_e_20, Paux)

    def P_aux_impfrac(self, n_e_20, T_i_keV):
        return P_aux_impfrac(self.state, n_e_20, T_i_keV)[0]

    # -------------------------------------------------------------------
    # Setup functions
    # -------------------------------------------------------------------

    def _addextprof(self, extprofvals, profid):
        """
        Adds an input profile to the class.
        """
        if profid == -3:
            if self.agriddefined:
                raise ValueError("Area grid already defined.")
            self.agrid = extprofvals
            self.agriddefined = True
        elif profid == -2:
            if self.rdefined:
                raise ValueError("Radius profile already defined.")
            self.sqrtpsin = extprofvals
            self.nr = extprofvals.shape[0]
            self.rdefined = True
        elif profid == -1:
            if self.volgriddefined:
                raise ValueError("Volume grid already defined.")
            self.volgrid = extprofvals
            self.volgriddefined = True
        elif profid == 0:
            if self._jdefined:
                raise ValueError("J profile already defined.")
            self.j_prof = extprofvals
            self._jdefined = True
        elif profid == 1:
            if self._nedefined:
                raise ValueError("n_e profile already defined.")
            self.ne_prof = extprofvals
            self._nedefined = True
        elif profid == 2:
            if self._nidefined:
                raise ValueError("n_i profile already defined.")
            self.ni_prof = extprofvals
            self._nidefined = True
        elif profid == 3:
            if self._Tidefined:
                raise ValueError("Ti profile already defined.")
            self.Ti_prof = extprofvals
            self._Tidefined = True
        elif profid == 4:
            if self._Tedefined:
                raise ValueError("Te profile already defined.")
            self.Te_prof = extprofvals
            self._Tedefined = True
        elif profid == 5:
            if self._qdefined:
                raise ValueError("q profile already defined.")
            self.q_prof = extprofvals
            self._qdefined = True
        elif profid == 6:
            if self._ftrappeddefined:
                raise ValueError("ftrapped profile already defined.")
            self.ftrapped_prof = extprofvals
            self._ftrappeddefined = True
        elif profid == 7:
            if self._extradefined:
                raise ValueError("extra profile already defined." + " This is unused.")
            self.extraprof = extprofvals
            self._extradefined = True
        else:
            raise ValueError("Invalid profile ID.")

    def _set_alpha_and_offset(self, alpha1, alpha2, offset, profid: int):
        """
        If the profile is not externally defined, this function sets the
        parabolic profile parameters.
        """
        if profid == 0:
            if self._jdefined:
                raise ValueError("J profile already defined.")
            self.j_alpha1 = alpha1
            self.j_alpha2 = alpha2
            self.j_offset = offset
        elif profid == 1:
            if self._nedefined:
                raise ValueError("n_e profile already defined.")
            self.ne_alpha1 = alpha1
            self.ne_alpha2 = alpha2
            self.ne_offset = offset
        elif profid == 2:
            if self._nidefined:
                raise ValueError("n_i profile already defined.")
            self.ni_alpha1 = alpha1
            self.ni_alpha2 = alpha2
            self.ni_offset = offset
        elif profid == 3:
            if self._Tidefined:
                raise ValueError("Ti profile already defined.")
            self.Ti_alpha1 = alpha1
            self.Ti_alpha2 = alpha2
            self.Ti_offset = offset
        elif profid == 4:
            if self._Tedefined:
                raise ValueError("Te profile already defined.")
            self.Te_alpha1 = alpha1
            self.Te_alpha2 = alpha2
            self.Te_offset = offset
        else:
            raise ValueError("Invalid profile ID.")

    # def _addextprof_imps(self, extprofvals):
    #     # TODO: implement this
    #     pass

    def _setup_profs(self) -> None:
        """
        Sets up profiles for the first time. If external profiles have
        not been added, it will use parabolic profiles.
        """
        rho = self.sqrtpsin
        if not self.rdefined:
            raise ValueError("Geometry profile not defined.")
        if not self.agriddefined:
            w07 = 1.0
            Lp = (
                2
                * np.pi
                * self.a
                * (1 + 0.55 * (self.kappa - 1))
                * (1 + 0.08 * self.delta**2)
                * (1 + 0.2 * (w07 - 1))
            )
            epsilon = rho * self.a / self.R
            self.agrid = (
                2 * np.pi * self.R * (1 - 0.32 * self.delta * epsilon) * Lp
            )  # Sauter eqs 36
        if not self.volgriddefined:
            epsilon = rho * self.a / self.R
            w07 = 1.0
            S_phi = np.pi * ((rho * self.a) ** 2) * self.kappa * (1 + 0.52 * (w07 - 1))
            self.volgrid = (
                2 * np.pi * self.R * (1 - 0.25 * self.delta * epsilon) * S_phi
            )  # Sauter eqs 36
            self.volgriddefined = True
        if not self._jdefined:
            self.j_prof = (1 - self.j_offset) * (
                1 - rho**self.j_alpha1
            ) ** self.j_alpha2 + self.j_offset
            self._jdefined = True
        if not self._nedefined:
            self.ne_prof = (1 - self.ne_offset) * (
                1 - rho**self.ne_alpha1
            ) ** self.ne_alpha2 + self.ne_offset
            self._nedefined = True
        if not self._nidefined:
            self.ni_prof = (1 - self.ni_offset) * (
                1 - rho**self.ni_alpha1
            ) ** self.ni_alpha2 + self.ni_offset
            self._nidefined = True
        if not self._Tidefined:
            self.Ti_prof = (1 - self.Ti_offset) * (
                1 - rho**self.Ti_alpha1
            ) ** self.Ti_alpha2 + self.Ti_offset
            self._Tidefined = True
        if not self._Tedefined:
            self.Te_prof = (1 - self.Te_offset) * (
                1 - rho**self.Te_alpha1
            ) ** self.Te_alpha2 + self.Te_offset
            self._Tedefined = True
        if not self._qdefined:
            # self.q_a = 2*np.pi*self.a**2*self.B0*(self.kappa**2+1)/(2*self.R*(4e-7*np.pi)*self.Ip*1e6)
            self.q_prof = (
                2
                * np.pi
                * (self.a * rho) ** 2
                * self.B0
                * (self.kappa**2 + 1)
                / (2 * self.R * (4e-7 * np.pi) * self.Ip * 1e6)
            )
            self._qdefined = True
        # profile 6 is the trapped particle fraction; eta_NC falls back to
        # f_t = sqrt(2*rho*a/R) on its own, so no need to fill it
        if not self._extradefined:
            self.extraprof = np.sqrt(
                (self.B0) ** 2
                + ((4e-7 * np.pi) * self.Ip * 1e6 / (2 * np.pi * self.a * rho)) ** 2
            )
            self._extradefined = True
        pass


# NOT jit compiled
class POPCON_settings:
    """
    Class POPCON_settings

    This class reads a YAML file and stores the settings for a POPCON run.
    Uses safe_get where appropriate to allow for missing keys in an outdated
    settings file. When applicable, sets to the default value.
    """

    def __init__(
        self,
        filename: str = "",
        data: dict = None,
        settingsfile: str = "",
        overrides: dict = None,
    ) -> None:
        """
        Normally reads a YAML file. A scan instead passes the raw contents of
        one as `data` plus a dict of `overrides`, so that every derived
        quantity is recomputed from scratch for each cell rather than patched
        after the fact.
        """
        if data is not None:
            merged = _apply_overrides(dict(data), overrides or {})
            self._load(merged, settingsfile or "<dict>")
        else:
            self.read(filename)
        pass

    def with_overrides(self, **overrides):
        """
        A copy of these settings with some raw settings-file keys replaced.
        The whole derivation chain is re-run, so overriding 'R' on a file that
        specifies 'qstar' correctly re-derives Ip, and overriding 'Tmax_keV'
        re-derives the impurity fraction from Zeff_target.
        """
        return POPCON_settings(
            data=self.rawdata, settingsfile=self.settingsfile, overrides=overrides
        )

    def _resolve(self, path: str) -> str:
        """
        Resolves a filename given in the settings file against the directory
        that settings file lives in. '' means "not specified" and is left
        alone, as are absolute paths, so that read_output can hand back
        absolute paths without them being mangled.
        """
        if path == "" or os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.settingsdir, path))

    def read(self, filename: str) -> None:
        """
        Reads a YAML file and sets the settings.
        """
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filename, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Filename must end with .yaml or .yml")

        self._load(data, os.path.abspath(filename))

    def _load(self, data: dict, settingsfile: str) -> None:
        """
        Sets the settings from an already-parsed settings file. Split out from
        read so that a scan can re-derive a whole settings object from raw
        contents it has modified, rather than mutating derived fields.
        """
        self.settingsfile = settingsfile
        self.settingsdir = os.path.dirname(self.settingsfile)
        self.rawkeys = set(data.keys())
        # kept so that with_overrides can re-run the derivation chain below.
        # a dict, so build_dataset's (int, float, str, bool, ndarray) filter
        # leaves it out of the saved attributes
        self.rawdata = dict(data)

        try:
            # -----------------------------------------------------------
            # Params
            # -----------------------------------------------------------
            self.name = str(data["name"])
            self.R = float(data["R"])
            self.a = float(data["a"])
            self.kappa = float(data["kappa"])
            self.delta = float(data["delta"])
            # Ip optional
            # B0 optional
            if "B_0" in data:
                self.B0 = float(data["B_0"])
            elif "B_coil" in data and "wall_thickness" in data:
                self.B0 = phys.get_B0(
                    data["B_coil"], data["wall_thickness"], self.R, self.a
                )
                print(f"B0 = {self.B0} calculated from B_coil and wall_thickness.")
            else:
                raise KeyError(
                    "B0 or B_coil and wall_thickness not found in settings file."
                )

            if "I_P" in data:
                self.Ip = float(data["I_P"])
            elif "qstar" in data:
                self.Ip = phys.get_Ip(
                    data["qstar"], self.R, self.a, self.B0, self.kappa
                )
                print(f"Ip = {self.Ip} calculated from qstar.")
            else:
                raise KeyError("I_P or qstar not found in settings file.")
            self.tipeak_over_tepeak = float(data["tipeak_over_tepeak"])
            self.fuel = int(data["fuel"])
            if self.fuel == 3:
                raise ValueError(
                    "D-He3 fuel cycle not implemented yet. Please use D-D or D-T."
                )
            self.impurityfractions = np.array(
                data["impurityfractions"], dtype=np.float64
            )
            if self.impurityfractions.shape != (6,):
                raise ValueError(
                    f"impurityfractions must have 6 entries "
                    f"(He, Ne, Ar, Kr, Xe, W), got {self.impurityfractions.size}."
                )
            if "Zeff_target" in data and "impurity" in data:
                impurity = int(data["impurity"])
                if not 0 <= impurity <= 5:
                    raise ValueError(
                        f"impurity must be 0-5 "
                        f"(0 He, 1 Ne, 2 Ar, 3 Kr, 4 Xe, 5 W), got {impurity}."
                    )
                self.impurityfractions[impurity] = phys.get_impurity_fraction(
                    data["Zeff_target"],
                    self.impurityfractions[0],
                    impurity,
                    (float(data["Tmax_keV"]) - float(data["Tmin_keV"])) / 2
                    + float(data["Tmin_keV"]),
                )
                print(
                    f"Impurity fractions = {self.impurityfractions} calculated from Zeff_target."
                )

            if self.fuel == 1:
                f_D = 1 - np.sum(self.impurityfractions)
                f_T = 0
            elif self.fuel == 2:
                f_D = 0.5 - np.sum(self.impurityfractions) / 2
                f_T = 0.5 - np.sum(self.impurityfractions) / 2
            else:
                raise ValueError("Invalid fuel cycle.")

            self.M_i = (
                f_D * 2.014
                + f_T * 3.016
                + np.dot(
                    self.impurityfractions,
                    np.array([4.002, 20.18, 39.948, 83.80, 131.29, 183.84]),
                )
            )

            self.scalinglaw = str(data["scalinglaw"])
            self.H_fac = float(data["H_fac"])
            self.nr = int(data["nr"])

            self.gfilename = self._resolve(str(safe_get(data, "gfilename", "")))
            self.profsfilename = self._resolve(str(safe_get(data, "profsfilename", "")))

            self.j_alpha1 = float(safe_get(data, "j_alpha1", 1))
            self.j_alpha2 = float(safe_get(data, "j_alpha2", 2))
            self.j_offset = float(safe_get(data, "j_offset", 0))

            self.ne_alpha1 = float(safe_get(data, "ne_alpha1", 2))
            self.ne_alpha2 = float(safe_get(data, "ne_alpha2", 1.5))
            self.ne_offset = float(safe_get(data, "ne_offset", 0))

            self.ni_alpha1 = float(safe_get(data, "ni_alpha1", 2))
            self.ni_alpha2 = float(safe_get(data, "ni_alpha2", 1.5))
            self.ni_offset = float(safe_get(data, "ni_offset", 0))

            self.Ti_alpha1 = float(safe_get(data, "Ti_alpha1", 2))
            self.Ti_alpha2 = float(safe_get(data, "Ti_alpha2", 1.5))
            self.Ti_offset = float(safe_get(data, "Ti_offset", 0))

            self.Te_alpha1 = float(safe_get(data, "Te_alpha1", 2))
            self.Te_alpha2 = float(safe_get(data, "Te_alpha2", 1.5))
            self.Te_offset = float(safe_get(data, "Te_offset", 0))

            # -----------------------------------------------------------
            # Settings
            # -----------------------------------------------------------
            self.Nn = int(data["Nn"])
            self.NTi = int(data["NTi"])
            self.nmax_frac = float(data["nmax_frac"])
            self.nmin_frac = float(data["nmin_frac"])
            self.Tmax_keV = float(data["Tmax_keV"])
            self.Tmin_keV = float(data["Tmin_keV"])
            self.resistivity_model = str(
                safe_get(data, "resistivity_model", "jardin")
            ).lower()
            self.verbosity = int(data["verbosity"])
            self.parallel = bool(data["parallel"])
            # left raw; POPCON_scan validates it. A plain POPCON ignores it
            self.scan = safe_get(data, "scan", {})
        except KeyError as e:
            raise KeyError(f"Key {e} not found in {self.settingsfile}")


POPCON_data_spec = [
    ("n_G_frac", nb.float64[:]),
    ("n_e_20_max", nb.float64[:]),
    ("n_e_20_avg", nb.float64[:]),
    ("n_i_20_max", nb.float64[:, :]),
    ("n_i_20_avg", nb.float64[:, :]),
    ("T_i_max", nb.float64[:]),
    ("T_i_avg", nb.float64[:]),
    ("T_e_max", nb.float64[:]),
    ("T_e_avg", nb.float64[:]),
    ("Paux", nb.float64[:, :]),
    ("Pfusion", nb.float64[:, :]),
    ("Pfusionheating", nb.float64[:, :]),
    ("Pohmic", nb.float64[:, :]),
    ("Pbrems", nb.float64[:, :]),
    ("Psynch", nb.float64[:, :]),
    ("Pimprad", nb.float64[:, :]),
    ("Prad", nb.float64[:, :]),
    ("Pheat", nb.float64[:, :]),
    ("Wtot", nb.float64[:, :]),
    ("tauE", nb.float64[:, :]),
    ("Pconf", nb.float64[:, :]),
    ("Ploss", nb.float64[:, :]),
    ("Pdd", nb.float64[:, :]),
    ("Pdt", nb.float64[:, :]),
    ("Palpha", nb.float64[:, :]),
    ("Psol", nb.float64[:, :]),
    ("f_rad", nb.float64[:, :]),
    ("Q", nb.float64[:, :]),
    ("H89", nb.float64[:, :]),
    ("H98", nb.float64[:, :]),
    ("vloop", nb.float64[:, :]),
    ("betaN", nb.float64[:, :]),
]

# What populate_outputs returns; POPCON_data_spec minus the 1-D axes and Paux,
# which are inputs. They have to be returned: with parallel=True, a prange
# write through a namedtuple field never reaches the caller
COMPUTED_FIELDS = [
    "n_i_20_max",
    "n_i_20_avg",
    "Pfusion",
    "Pfusionheating",
    "Pohmic",
    "Pbrems",
    "Psynch",
    "Pimprad",
    "Prad",
    "Pheat",
    "Wtot",
    "tauE",
    "Pconf",
    "Ploss",
    "Pdd",
    "Pdt",
    "Palpha",
    "Psol",
    "f_rad",
    "Q",
    "H89",
    "H98",
    "vloop",
    "betaN",
]
ComputedOutputs = namedtuple("ComputedOutputs", COMPUTED_FIELDS)


# Output is an xarray Dataset over these two dimensions. The seven 1-D arrays
# are the different ways of labelling the two axes, so they are all coordinates;
# everything else is 2-D. The dimensions are not called 'n'/'T' because Dataset.T
# is transpose.
DIM_N, DIM_T = "n_index", "T_index"
AXES_N = ("n_G_frac", "n_e_20_max", "n_e_20_avg")
AXES_T = ("T_i_max", "T_i_avg", "T_e_max", "T_e_avg")

# Every axis a POPCON can be plotted against: the xax/yax value in a
# plotsettings file -> (output variable, which grid dimension it indexes, axis
# label). Adding an entry here is all it takes to add an axis. Either of xax
# and yax takes any entry, as long as the two index different dimensions, so
# a density-on-x plot is just a plotsettings change. Only the 1-D coordinates
# can appear: contours need a rectilinear grid, so a 2-D output like Q cannot
# be an axis.
PLOT_AXES = {
    "T_i_av": ("T_i_avg", DIM_T, r"$\langle T_i\rangle$ (keV)"),
    "T_i_ax": ("T_i_max", DIM_T, r"$T_i$ (keV, On-axis)"),
    "T_e_av": ("T_e_avg", DIM_T, r"$\langle T_e\rangle$ (keV)"),
    "T_e_ax": ("T_e_max", DIM_T, r"$T_e$ (keV, On-axis)"),
    "n20_av": ("n_e_20_avg", DIM_N, r"$\langle n_{20}\rangle$ ($10^{20} m^{-3}$)"),
    "n20_ax": ("n_e_20_max", DIM_N, r"$n_{20}(0)$"),
    "nG": ("n_G_frac", DIM_N, r"$\langle n\rangle /n_G$"),
}

UNITS = {
    "n_G_frac": "",
    "n_e_20_max": "1e20 m^-3",
    "n_e_20_avg": "1e20 m^-3",
    "n_i_20_max": "1e20 m^-3",
    "n_i_20_avg": "1e20 m^-3",
    "T_i_max": "keV",
    "T_i_avg": "keV",
    "T_e_max": "keV",
    "T_e_avg": "keV",
    "Paux": "MW",
    "Pfusion": "MW",
    "Pfusionheating": "MW",
    "Pohmic": "MW",
    "Pbrems": "MW",
    "Psynch": "MW",
    "Pimprad": "MW",
    "Prad": "MW",
    "Pheat": "MW",
    "Pconf": "MW",
    "Ploss": "MW",
    "Pdd": "MW",
    "Pdt": "MW",
    "Palpha": "MW",
    "Psol": "MW",
    "Wtot": "MJ",
    "tauE": "s",
    "f_rad": "",
    "Q": "",
    "H89": "",
    "H98": "",
    "vloop": "V",
    "betaN": "",
}


def build_dataset(grids: dict, settings=None, scalinglaws=None):
    """
    Packs the solved arrays into an xarray Dataset. The 1-D axis arrays become
    coordinates so that results can be selected against any of them, e.g.
    ds.sel(T_index=ds.T_i_avg.searchsorted(10)); everything else is a 2-D
    variable over (density, temperature).
    """
    coords = {}
    for name in AXES_N:
        coords[name] = (DIM_N, np.asarray(grids[name]))
    for name in AXES_T:
        coords[name] = (DIM_T, np.asarray(grids[name]))

    data_vars = {
        name: ((DIM_N, DIM_T), np.asarray(arr))
        for name, arr in grids.items()
        if name not in AXES_N and name not in AXES_T
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    for name, var in list(ds.variables.items()):
        if name in UNITS and UNITS[name]:
            var.attrs["units"] = UNITS[name]
    ds["Paux"].attrs["note"] = f"{UNPHYSICAL} marks a point with no physical solution"

    if settings is not None:
        ds.attrs["machine"] = str(getattr(settings, "name", ""))
        ds.attrs["settings"] = json.dumps(
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in vars(settings).items()
                if isinstance(v, (int, float, str, bool, np.ndarray))
            }
        )
    if scalinglaws is not None:
        ds.attrs["scalinglaws"] = json.dumps(scalinglaws)
    ds.attrs["openpopcon_version"] = __version__
    return ds


# NOT jit compiled
class POPCON_plotsettings:
    """
    Class POPCON_plotsettings

    This class reads a YAML file and stores the settings for a POPCON plot.
    Uses safe_get where appropriate to allow for missing keys in an outdated
    settings file. When applicable, sets to the default value.
    """

    def __init__(
        self,
        filename: str,
    ) -> None:

        self.plotoptions: dict
        self.figsize: tuple = (8, 6)
        self.yax: str = "nG"
        self.xax: str = "T_i_av"
        self.fill_invalid: bool = True
        self.read(filename)

        pass

    def read(self, filename: str) -> None:
        """
        Reads a YAML file and sets the settings.
        """
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            data = yaml.safe_load(open(filename, "r"))
        else:
            raise ValueError("Filename must end with .yaml or .yml")

        with open(DEFAULT_PLOTSETTINGS, "r") as f:
            defaults = yaml.safe_load(f)

        self.plotoptions = {}
        for key in defaults["plotoptions"]:
            self.plotoptions[key] = safe_get(
                data["plotoptions"], key, defaults["plotoptions"][key]
            )
        self.figsize = tuple(safe_get(data, "figsize", defaults["figsize"]))
        self.yax = safe_get(data, "yax", defaults["yax"])
        self.xax = safe_get(data, "xax", defaults["xax"])
        self.fill_invalid = safe_get(data, "fill_invalid", defaults["fill_invalid"])

        pass


# NOT jit compiled
class POPCON:
    """
    The Primary class for the OpenPOPCON project.

    This class is the primary interface for the OpenPOPCON project. It
    contains the settings, parameters, and output data for a POPCON run.


    """

    def __init__(
        self,
        settingsfile=None,
        plotsettingsfile=None,
        scalinglawfile=None,
        settings=None,
    ) -> None:
        self.algorithms: POPCON_algorithms
        self.settings: POPCON_settings
        self.plotsettings: POPCON_plotsettings
        self.output: xr.Dataset

        # these are kept as absolute paths so that write_output can still copy
        # them if the working directory has changed since construction
        if settings is not None:
            # a scan cell, built from raw settings contents it has modified
            self.settings = settings
            self.settingsfile = settings.settingsfile
        elif settingsfile is not None:
            self.settings = POPCON_settings(settingsfile)
            self.settingsfile = os.path.abspath(settingsfile)
        else:
            pass

        if plotsettingsfile is None:
            plotsettingsfile = DEFAULT_PLOTSETTINGS

        self.plotsettings = POPCON_plotsettings(plotsettingsfile)
        self.plotsettingsfile = os.path.abspath(plotsettingsfile)

        if scalinglawfile is None:
            scalinglawfile = DEFAULT_SCALINGLAWS

        self.__get_scaling_laws(scalinglawfile)
        self.scalinglawfile = os.path.abspath(scalinglawfile)

        self.__check_settings()

        pass

    # -------------------------------------------------------------------
    # Solving
    # -------------------------------------------------------------------

    def run_POPCON(self, setuponly=False) -> None:
        """
        Wrapper function that sets up and solves the power balance
        equations at each point in the grid.
        """

        self.__check_settings()
        self.__setup_params()
        self.__get_geometry()
        self.__get_profiles()

        self.algorithms._setup_profs()
        scalinglaw = self.settings.scalinglaw
        slparam = self.scalinglaws[scalinglaw]
        self.algorithms.H_fac = self.settings.H_fac
        self.algorithms.scaling_const = slparam["scaling_const"]
        self.algorithms.M_i_alpha = slparam["M_i_alpha"]
        self.algorithms.Ip_alpha = slparam["Ip_alpha"]
        self.algorithms.R_alpha = slparam["R_alpha"]
        self.algorithms.a_alpha = slparam["a_alpha"]
        self.algorithms.kappa_alpha = slparam["kappa_alpha"]
        self.algorithms.B0_alpha = slparam["B0_alpha"]
        self.algorithms.Pheat_alpha = slparam["Pheat_alpha"]
        self.algorithms.n20_alpha = slparam["n20_alpha"]
        self.algorithms.build_state()
        if not setuponly:
            self.__solve()

    def __solve(self) -> None:
        """
        Does the legwork for run_POPCON.
        """

        if self.settings.verbosity > 0:
            print("Setting up algorithm object")

        params = self.algorithms
        rho = params.sqrtpsin
        n_G = params.n_GR
        n_e_avg_fac = params.volume_integral(rho, params.get_profile(rho, 1)) / params.V
        T_i_avg_fac = params.volume_integral(rho, params.get_profile(rho, 3)) / params.V
        T_e_avg_fac = params.volume_integral(rho, params.get_profile(rho, 4)) / params.V
        n_e_20 = np.linspace(
            self.settings.nmin_frac * n_G / n_e_avg_fac,
            self.settings.nmax_frac * n_G / n_e_avg_fac,
            self.settings.Nn,
        )
        T_i_keV = np.linspace(
            self.settings.Tmin_keV / T_i_avg_fac,
            self.settings.Tmax_keV / T_i_avg_fac,
            self.settings.NTi,
        )
        T_e_keV = T_i_keV / self.settings.tipeak_over_tepeak

        axes = {
            "n_G_frac": np.linspace(
                self.settings.nmin_frac, self.settings.nmax_frac, self.settings.Nn
            ),
            "n_e_20_max": n_e_20,
            "n_e_20_avg": n_e_20 * n_e_avg_fac,
            "T_i_max": T_i_keV,
            "T_i_avg": T_i_keV * T_i_avg_fac,
            "T_e_max": T_e_keV,
            "T_e_avg": T_e_keV * T_e_avg_fac,
        }

        if self.settings.verbosity > 0:
            print("Solving power balance equations")

        if self.settings.parallel:
            Paux, flags = solve_nT_par(
                params.state, self.settings.Nn, self.settings.NTi, n_e_20, T_i_keV
            )
        else:
            Paux, flags = solve_nT(
                params.state, self.settings.Nn, self.settings.NTi, n_e_20, T_i_keV
            )

        if self.settings.verbosity > 0:
            print("Power balance solutions found. Populating output arrays.")
        self.__report_invalid(flags, verbose=self.settings.verbosity > 0)
        if self.settings.parallel:
            computed = populate_outputs_par(
                params.state,
                n_e_20,
                T_i_keV,
                T_e_keV,
                Paux,
                self.settings.Nn,
                self.settings.NTi,
            )
        else:
            computed = populate_outputs(
                params.state,
                n_e_20,
                T_i_keV,
                T_e_keV,
                Paux,
                self.settings.Nn,
                self.settings.NTi,
            )

        grids = dict(axes, Paux=Paux, **computed._asdict())
        self.output = build_dataset(grids, self.settings, self.scalinglaws)

        pass

    def __report_invalid(self, flags, verbose: bool = True) -> None:
        """
        Says how much of the grid has no physical solution, and why. These
        points are set to P_aux = UNPHYSICAL and masked out of the plot.
        """
        total = flags.size
        counts = {
            code: int(np.count_nonzero(flags == code)) for code in INVALID_REASONS
        }
        ninvalid = sum(counts.values())
        nnoaux = int(np.count_nonzero(flags == NOAUX))

        if ninvalid:
            print(
                f"{ninvalid} of {total} grid points ({100 * ninvalid / total:.1f}%) "
                f"have no physical solution and are masked in the plot:"
            )
            for code, reason in INVALID_REASONS.items():
                if counts[code]:
                    print(f"  {counts[code]:>6d}  {reason}")
        elif verbose:
            print(f"All {total} grid points solved.")

        if nnoaux and verbose:
            print(
                f"{nnoaux} of {total} grid points need no auxiliary heating "
                f"(P_aux cannot be less than zero, setting P_aux = 0)."
            )

    def single_point(
        self, n_G_frac: float, Ti_av: float, plot: bool = True, show: bool = True
    ) -> None:
        """
        Solves power balance at a single n, T point. Prints the results
        plots the profiles of the solution if plot=True.
        """

        try:
            self.algorithms
        except AttributeError:
            self.run_POPCON(setuponly=True)

        n_G = self.algorithms.n_GR
        rho = self.algorithms.sqrtpsin
        n_e_avg_fac = (
            self.algorithms.volume_integral(rho, self.algorithms.get_profile(rho, 1))
            / self.algorithms.V
        )
        n_e_20 = n_G_frac * n_G / n_e_avg_fac
        T_i_avg_fac = (
            self.algorithms.volume_integral(rho, self.algorithms.get_profile(rho, 3))
            / self.algorithms.V
        )
        n_i_avg_fac = (
            self.algorithms.volume_integral(rho, self.algorithms.get_profile(rho, 2))
            / self.algorithms.V
        )
        T_i_keV = Ti_av / T_i_avg_fac
        T_e_keV = T_i_keV / self.algorithms.tipeak_over_tepeak
        dil = self.algorithms.plasma_dilution(T_e_keV)
        n_i_20 = n_e_20 * dil
        line_avg_fac = np.average(self.algorithms.get_profile(rho, 1))

        Paux, flag = P_aux_impfrac(self.algorithms.state, n_e_20, T_i_keV)
        if flag in INVALID_REASONS:
            print(
                f"This point has no physical solution: {INVALID_REASONS[flag]}. "
                f"The numbers below are not meaningful."
            )
        elif flag == NOAUX:
            print(
                "This point needs no auxiliary heating, so P_aux is zero. "
                "Check Q below for whether that is ignition or just ohmic "
                "heating outrunning the loss at low temperature."
            )

        Pfusion = self.algorithms.volume_integral(
            rho, self.algorithms._P_fusion(rho, T_i_keV, n_i_20)
        )
        Pfusion_heating = self.algorithms.volume_integral(
            rho, self.algorithms._P_fusion_heating(rho, T_i_keV, n_i_20)
        )
        Pohmic = self.algorithms.volume_integral(
            rho, self.algorithms._P_OH_prof(rho, T_e_keV, n_e_20)
        )
        Pbrems = self.algorithms.volume_integral(
            rho, self.algorithms._P_brem_rad(rho, T_e_keV, n_e_20)
        )
        Psynch = self.algorithms.volume_integral(
            rho, self.algorithms._P_synch(rho, T_e_keV, n_e_20)
        )
        Pimprad = self.algorithms.volume_integral(
            rho, self.algorithms._P_impurity_rad(rho, T_e_keV, n_e_20)
        )
        Prad = self.algorithms.volume_integral(
            rho, self.algorithms._P_rad(rho, T_e_keV, n_e_20)
        )
        Pheat = Pfusion_heating + Pohmic + Paux - Pbrems - Psynch
        Palpha = (
            self.algorithms.volume_integral(
                rho, self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20)
            )
            * 3.52e3
            / (3.52e3 + 14.06e3)
        )
        Pdd = self.algorithms.volume_integral(
            rho, self.algorithms._P_DDnHe3_prof(rho, T_i_keV, n_i_20)
        )
        Pdd += self.algorithms.volume_integral(
            rho, self.algorithms._P_DDpT_prof(rho, T_i_keV, n_i_20)
        )
        Pdt = self.algorithms.volume_integral(
            rho, self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20)
        )
        tauE = self.algorithms.tauE_scalinglaw(Pheat, n_e_20 * line_avg_fac)
        Wtot = self.algorithms.volume_integral(
            rho, self.algorithms._W_tot_prof(rho, T_i_keV, n_e_20)
        )
        Pconf = Wtot / tauE
        Ploss = Pconf + Psynch + Pbrems
        f_rad = Prad / Ploss
        Q = self.algorithms.Q_fusion(T_i_keV, n_e_20, Paux)
        H89 = tauE / self.algorithms.tauE_H89(Pheat, n_e_20 * line_avg_fac)
        H98 = tauE / self.algorithms.tauE_H98(Pheat, n_e_20 * line_avg_fac)
        vloop = self.algorithms.Vloop(T_e_keV, n_e_20)
        betaN = 100 * self.algorithms.BetaN(T_i_keV, n_e_20)  # in percent
        pstring = f"""
Params:
n_i_average = {n_i_avg_fac * n_i_20} x 10^20 m^-3
n_e_average = {n_G_frac * n_G:1.3f} x 10^20 m^-3
n_G = {n_G:1.3f} x 10^20 m^-3
n_i_axis = {n_i_20:1.3f} x 10^20 m^-3
n_e_axis = {n_e_20:1.3f} x 10^20 m^-3
Ti_average = {Ti_av:.1f} keV
Ti_axis = {T_i_keV:.1f} keV
Solution:
P_aux = {Paux:.2f} MW
P_fusion = {Pfusion:.2f} MW
P_SOL = {Ploss - Prad:.2f} MW
P_load = {(Ploss - Prad) / self.algorithms.A:.3f} MW/m^2
P_ohmic = {Pohmic:.3f} MW
P_brems = {Pbrems:.3f} MW
P_synch = {Psynch:.3f} MW
P_imprad = {Pimprad:.3f} MW
P_rad = {Prad:.2f} MW
P_heat = {Pheat:.2f} MW
P_alpha = {Palpha:.2f} MW
P_dd = {Pdd:.3f} MW
P_dt = {Pdt:.2f} MW
Wtot/TauE = {Pconf:.2f} MW
f_rad = {f_rad:.3f} 
tauE = {tauE:.3f} s
Q = {Q:.3f} 
H89 = {H89:.2f}
H98 = {H98:.2f}
vloop = {vloop:.4f} V
betaN = {betaN:.3f}
"""
        print(pstring)

        if plot:
            Pfusion_prof = self.algorithms._P_fusion(rho, T_i_keV, n_i_20)
            Pfusion_heating_prof = self.algorithms._P_fusion_heating(
                rho, T_i_keV, n_i_20
            )
            Pohmic_prof = self.algorithms._P_OH_prof(rho, T_e_keV, n_e_20)
            Prad_prof = self.algorithms._P_rad(rho, T_e_keV, n_e_20)
            Pbrems_prof = self.algorithms._P_brem_rad(rho, T_e_keV, n_e_20)
            Psynch_prof = self.algorithms._P_synch(rho, T_e_keV, n_e_20)
            # match the scalar Pheat above: brems and synch are removed before transport
            Pheat_prof = (
                Pfusion_heating_prof
                + Pohmic_prof
                + Paux / self.algorithms.V
                - Pbrems_prof
                - Psynch_prof
            )
            Palpha_prof = (
                self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20)
                * 3.52e3
                / (3.52e3 + 14.06e3)
            )
            Pdt_prof = self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20)
            niprof = n_i_20 * self.algorithms.get_profile(rho, 2)
            neprof = n_e_20 * self.algorithms.get_profile(rho, 1)
            Tiprof = T_i_keV * self.algorithms.get_profile(rho, 3)
            Teprof = T_e_keV * self.algorithms.get_profile(rho, 4)
            qprof = self.algorithms.get_profile(rho, 5)

            # Density and temperatures
            #
            fig, ax1 = plt.subplots(figsize=(4, 3))
            ax2 = ax1.twinx()
            ax1.plot(rho, niprof, "b-", label=r"$n_i$")
            ax1.plot(rho, neprof, "b--", label=r"$n_e$")
            ax2.plot(rho, Tiprof, "r-", label=r"$T_i$")
            ax2.plot(rho, Teprof, "r--", label=r"$T_e$")
            ax1.set_xlabel(r"$\sqrt{\psi_N}$")
            ax1.set_ylabel(r"$n$ ($m^{-3}$)", color="b")
            ax2.set_ylabel(r"$T$ (keV)", color="r")
            ax1.legend(loc="lower left")
            ax2.legend(loc="upper right")
            if show:
                plt.show()

            # Power profiles
            fig, ax = plt.subplots(figsize=(8, 6))
            # V_enclosed = self.algorithms.get_profile(rho, -1)
            radius = rho * self.algorithms.a
            ax.plot(radius, Pfusion_prof, "k-", label=r"$P_{\text{fusion}}$")
            ax.plot(radius, Pohmic_prof, "r-", label=r"$P_{\text{ohmic}}$")
            ax.plot(radius, Prad_prof, "-", color="purple", label=r"$P_{\text{rad}}$")
            ax.plot(radius, Pheat_prof, "b-", label=r"$P_{\text{heat}}$")
            ax.plot(radius, Palpha_prof, "g-", label=r"$P_{\alpha}$")
            ax.set_xlabel(r"$\rho$ ($m$)")
            ax.set_ylabel(r"dP/dV (MW/$m^3$)")
            # ax.set_yscale('log')
            ax.legend(loc="lower left")
            if show:
                plt.show()
        pass

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    def _resolve_axes(self):
        """
        Turns the xax/yax settings into the meshgrid to plot on, the axis
        labels, and the dimension order that 2-D output arrays have to be
        transposed into. That last part is what lets either axis be either
        family: the arrays are stored (n_index, T_index), so putting a density
        on x means every array has to come out transposed.
        """

        def look_up(key, which):
            try:
                return PLOT_AXES[key]
            except KeyError:
                raise ValueError(
                    f"Invalid {which}-axis '{key}'. Change {which}ax in "
                    f"plotsettings. Available: {', '.join(sorted(PLOT_AXES))}."
                )

        xvar, xdim, xlabel = look_up(self.plotsettings.xax, "x")
        yvar, ydim, ylabel = look_up(self.plotsettings.yax, "y")
        if xdim == ydim:
            raise ValueError(
                f"xax '{self.plotsettings.xax}' and yax '{self.plotsettings.yax}' "
                f"are both labels for the {xdim} axis, so they cannot be the two "
                f"axes of one plot. Pick one density and one temperature."
            )
        xx, yy = np.meshgrid(self.output[xvar].values, self.output[yvar].values)
        return xx, yy, xlabel, ylabel, (ydim, xdim)

    def _grid(self, name: str, dimorder) -> np.ndarray:
        """
        A 2-D output array laid out to match the meshgrid from _resolve_axes.
        """
        return self.output[name].transpose(*dimorder).values

    def plot(
        self,
        show: bool = True,
        savefig: str = "",
        names=None,
        ax=None,
        legend: bool = True,
        infobox: bool = True,
        title=None,
    ):
        """
        Plots the output data. If variable names is specified, only
        plots those variables. Otherwise, refers to the plotsettings
        file.

        Passing an existing ax draws into it instead of making a new figure,
        which is how POPCON_scan tiles a grid of these. The legend and the
        info box are drawn outside the axes, so they are worth turning off
        when tiling.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plotsettings.figsize)
            owns_figure = True
        else:
            fig = ax.get_figure()
            owns_figure = False

        xx, yy, xlabel, ylabel, dimorder = self._resolve_axes()
        Paux = self._grid("Paux", dimorder)
        mask = np.logical_or(np.isnan(Paux), Paux >= UNPHYSICAL_PLOT_CUTOFF)
        if mask.all():
            raise ValueError(
                "No point in this scan has a physical solution, so there is nothing "
                "to plot. Re-run with verbosity > 0 to see why, and try a smaller "
                "impurity fraction or a different density/temperature range."
            )
        if self.plotsettings.fill_invalid:
            ax.contourf(
                xx,
                yy,
                np.ma.array(np.ones_like(xx), mask=np.logical_not(mask)),
                levels=[0, 2],
                colors="k",
                alpha=0.5,
            )
            if mask.any():
                ax.add_patch(
                    patches.Rectangle(
                        (0, 0), 0, 0, fc="k", alpha=0.5, label="No physical solution"
                    )
                )
        Qgrid = self._grid("Q", dimorder)
        if np.any(Qgrid > 1e4):
            maskburning = np.logical_not(np.logical_or(np.isnan(Qgrid), Qgrid >= 1e4))
            ax.contourf(
                xx,
                yy,
                np.ma.array(np.ones_like(xx), mask=maskburning),
                levels=[0, 2],
                colors="r",
                alpha=0.08,
            )
        if names is None:
            names = self.plotsettings.plotoptions.keys()
        for name in names:
            opdict = self.plotsettings.plotoptions[name]
            if opdict["plot"] == False:
                continue
            plotoptions = [
                opdict["color"],
                opdict["linewidth"],
                opdict["label"],
                opdict["fontsize"],
                opdict["fmt"],
            ]
            data = np.ma.array(self._grid(name, dimorder), mask=mask)
            if opdict["spacing"] == "lin":
                if opdict["scale"] == "minmax":
                    if self.settings.verbosity > 1:
                        print(
                            f"{name} min: {np.min(data)}, max: {np.max(data)}, levels: {opdict['levels']}"
                        )
                    levels = np.linspace(
                        1.01 * np.min(data), 0.99 * np.max(data), opdict["levels"]
                    )
                    if levels[-1] < levels[0]:
                        levels = levels[::-1]
                elif opdict["scale"] == "specified":
                    levels = np.linspace(opdict["min"], opdict["max"], opdict["levels"])
                else:
                    raise ValueError(
                        f"Invalid scale for {name}. Change scale in plotsettings."
                    )
            elif opdict["spacing"] == "log":
                if opdict["scale"] == "minmax":
                    levels = np.logspace(
                        np.log10(1.01 * np.min(data)),
                        np.log10(0.99 * np.max(data)),
                        opdict["levels"],
                    )
                elif opdict["scale"] == "specified":
                    levels = np.logspace(
                        np.log10(opdict["min"]),
                        np.log10(opdict["max"]),
                        opdict["levels"],
                    )
                else:
                    raise ValueError(
                        f"Invalid scale for {name}. Change scale in plotsettings."
                    )
            elif opdict["spacing"] == "manual":
                levels = np.asarray(opdict["manuallevels"], dtype=np.float64)
            else:
                raise ValueError(
                    f"Invalid spacing for {name}. Change spacing in plotsettings."
                )

            if self.settings.verbosity > 0:
                print(f"Plotting {name} with levels {levels} and options {plotoptions}")
            self.plot_contours(opdict["plot"], ax, data, xx, yy, levels, *plotoptions)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        p = self.algorithms

        # 1 = D-D, 2 = D-T, 3 = D-He3
        fueldict = {1: "D-D", 2: "D-T", 3: "D-He3"}

        if legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        if infobox:
            # Zeff is a function of temperature, so it has to come off the
            # temperature axis whichever of x and y that happens to be
            Tav = np.average(self.output.T_i_avg.values)
            infoboxtext = f"$I_p$ = {p.Ip:.2f}\n$B_0$ = {p.B0:.2f}\nR = {p.R:.2f}\na = {p.a:.2f}\n$\\kappa$ = {p.kappa:.2f}\n$\\delta$ = {p.delta:.2f}\n$M_i$ = {p.M_i:.2f}\nti/te = {p.tipeak_over_tepeak:.2f}\nfuel = {fueldict[p.fuel]}\n<Zeff>={p.Zeff(Tav):.2f}"
            ax.text(
                x=np.max(xx) + (np.max(xx) - np.min(xx)) / 64,
                y=np.min(yy),
                s=infoboxtext,
                bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8),
            )
        if owns_figure:
            fig.tight_layout()
        if savefig != "":
            # not plt.savefig: with a caller-supplied ax, the current figure
            # is not necessarily the one being drawn into
            fig.savefig(savefig)
        if show and owns_figure:
            plt.show()

        return fig, ax

    def plot_contours(
        self,
        plotbool: bool,
        ax,
        data: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        levels,
        color,
        linewidth,
        label: str,
        fontsize: int,
        fmt: str,
    ):
        if plotbool:
            contour = ax.contour(
                xx, yy, data, levels=levels, colors=color, linewidths=linewidth
            )
            rect = patches.Rectangle((0, 0), 0, 0, fc=color, label=label)
            ax.add_patch(rect)
            ax.clabel(contour, inline=1, fmt=fmt, fontsize=fontsize)
        pass

    def custom_plot(
        self,
        fig,
        ax,
        data,
        levels,
        color="k",
        linewidth=1.0,
        label: str = "custom",
        fontsize: int = 11,
        fmt: str = "%1.2f",
    ):
        xx, yy, xlabel, ylabel, dimorder = self._resolve_axes()
        Paux = self._grid("Paux", dimorder)
        mask = np.logical_or(np.isnan(Paux), Paux >= UNPHYSICAL_PLOT_CUTOFF)
        self.plot_contours(
            True,
            ax,
            np.ma.array(data, mask=mask),
            xx,
            yy,
            levels,
            *[color, linewidth, label, fontsize, fmt],
        )

        return fig, ax

    # -------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------

    def write_output(
        self,
        name: str = "",
        archive: bool = True,
        overwrite: bool = False,
        directory: str = None,
    ) -> None:
        """
        Saves the output results to a directory. If archive is True,
        archives the directory as a zip file. If overwrite is True,
        overwrites the directory/zip if it already exists. The directory
        parameter allows specifying the storage location.
        """
        name, outputsdir, savedir = _prepare_output_dir(
            name, directory, overwrite, self.settings.name
        )

        shutil.copyfile(self.settingsfile, savedir.joinpath("settings.yaml"))
        shutil.copyfile(self.plotsettingsfile, savedir.joinpath("plotsettings.yaml"))
        shutil.copyfile(self.scalinglawfile, savedir.joinpath("scalinglaws.yaml"))

        self.output.to_netcdf(savedir.joinpath("arrays.nc"))

        if self.settings.gfilename != "":
            shutil.copyfile(
                self.settings.gfilename,
                savedir.joinpath(self.settings.gfilename.split(str(os.sep))[-1]),
            )
        if self.settings.profsfilename != "":
            shutil.copyfile(
                self.settings.profsfilename,
                savedir.joinpath(self.settings.profsfilename.split(str(os.sep))[-1]),
            )

        self.plot(show=False, savefig=str(savedir.joinpath("POPCON_plot.pdf")))
        plt.close("all")

        print(f"Wrote output to {_finalize_output(savedir, outputsdir, name, archive)}")
        return

    def read_output(self, name: str, directory: str = None) -> None:
        """
        Restores the POPCON object based on the output directory.
        """
        if directory is None:
            outputsdir = pathlib.Path.cwd().joinpath("OpenPOPCON_outputs")
        else:
            outputsdir = pathlib.Path(directory)

        if name.endswith(".zip"):
            outputsdir.joinpath(name[:-4]).mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(
                outputsdir.joinpath(name), outputsdir.joinpath(name[:-4])
            )
            name = name[:-4]
        namepath = outputsdir.joinpath(name)
        self.settingsfile = str(namepath.joinpath("settings.yaml"))
        self.plotsettingsfile = str(namepath.joinpath("plotsettings.yaml"))
        self.scalinglawfile = str(namepath.joinpath("scalinglaws.yaml"))
        self.settings = POPCON_settings(self.settingsfile)
        self.plotsettings = POPCON_plotsettings(self.plotsettingsfile)
        self.__get_scaling_laws(self.scalinglawfile)

        ncpath = namepath.joinpath("arrays.nc")
        jsonpath = namepath.joinpath("arrays.json")
        if ncpath.is_file():
            self.output = xr.load_dataset(ncpath)
        elif jsonpath.is_file():
            # archives written before the netCDF switch
            with open(jsonpath, "r") as f:
                data = json.load(f)
            self.output = build_dataset(
                {k: np.array(v, dtype=np.float64) for k, v in data.items()}
            )
        else:
            raise FileNotFoundError(f"No arrays.nc or arrays.json in {namepath}.")
        if self.settings.gfilename != "":
            self.settings.gfilename = str(
                namepath.joinpath(self.settings.gfilename.split(str(os.sep))[-1])
            )
        if self.settings.profsfilename != "":
            self.settings.profsfilename = str(
                namepath.joinpath(self.settings.profsfilename.split(str(os.sep))[-1])
            )
        self.run_POPCON(setuponly=True)
        pass

    # -------------------------------------------------------------------
    # Setup functions
    # -------------------------------------------------------------------

    def __get_scaling_laws(self, scalinglawfile: str) -> None:
        if scalinglawfile.endswith(".yaml") or scalinglawfile.endswith(".yml"):
            data = yaml.safe_load(open(scalinglawfile, "r"))
        else:
            raise ValueError("Filename must end with .yaml or .yml")
        assert "H89" in data.keys()
        assert "H98y2" in data.keys()
        assert "H_NT23" in data.keys()
        self.scalinglaws = data

    def __setup_params(self) -> None:

        self.algorithms = POPCON_algorithms()
        self.algorithms.R = self.settings.R
        self.algorithms.a = self.settings.a
        self.algorithms.kappa = self.settings.kappa
        self.algorithms.delta = self.settings.delta
        self.algorithms.B0 = self.settings.B0
        self.algorithms.Ip = self.settings.Ip
        self.algorithms.M_i = self.settings.M_i
        self.algorithms.tipeak_over_tepeak = self.settings.tipeak_over_tepeak
        self.algorithms.fuel = self.settings.fuel
        self.algorithms.impurityfractions = self.settings.impurityfractions
        self.algorithms.verbosity = self.settings.verbosity
        self.algorithms.resistivity_alg = RESISTIVITY_MODELS[
            self.settings.resistivity_model.lower()
        ]

    def __get_profiles(self) -> None:
        if self.settings.profsfilename == "":
            if self.settings.ne_offset == 0:
                self.settings.ne_offset = 1e-6
            if self.settings.ni_offset == 0:
                self.settings.ni_offset = 1e-6
            if self.settings.Ti_offset == 0:
                self.settings.Ti_offset = 1e-6
            if self.settings.Te_offset == 0:
                self.settings.Te_offset = 1e-6

            self.algorithms._set_alpha_and_offset(
                self.settings.ne_alpha1,
                self.settings.ne_alpha2,
                self.settings.ne_offset,
                1,
            )
            self.algorithms._set_alpha_and_offset(
                self.settings.ni_alpha1,
                self.settings.ni_alpha2,
                self.settings.ni_offset,
                2,
            )
            self.algorithms._set_alpha_and_offset(
                self.settings.Ti_alpha1,
                self.settings.Ti_alpha2,
                self.settings.Ti_offset,
                3,
            )
            self.algorithms._set_alpha_and_offset(
                self.settings.Te_alpha1,
                self.settings.Te_alpha2,
                self.settings.Te_offset,
                4,
            )
        else:
            profstable = read_profsfile(self.settings.profsfilename)
            ne = np.asarray(profstable["n_e"])
            ne = ne / ne[0]
            ni = np.asarray(profstable["n_i"])
            ni = ni / ni[0]
            Ti = np.asarray(profstable["T_i"])
            Ti = Ti / Ti[0]
            Te = np.asarray(profstable["T_e"])
            Te = Te / Te[0]
            rho = np.asarray(profstable["rho"])
            ne_sqrtpsin = np.interp(np.linspace(0.001, 1, self.settings.nr), rho, ne)
            ni_sqrtpsin = np.interp(np.linspace(0.001, 1, self.settings.nr), rho, ni)
            Ti_sqrtpsin = np.interp(np.linspace(0.001, 1, self.settings.nr), rho, Ti)
            Te_sqrtpsin = np.interp(np.linspace(0.001, 1, self.settings.nr), rho, Te)
            self.algorithms._addextprof(ne_sqrtpsin, 1)
            self.algorithms._addextprof(ni_sqrtpsin, 2)
            self.algorithms._addextprof(Ti_sqrtpsin, 3)
            self.algorithms._addextprof(Te_sqrtpsin, 4)

        pass

    def __get_geometry(self) -> None:
        if self.settings.gfilename == "":
            rho = np.linspace(0.001, 1, self.settings.nr)
            if self.settings.j_offset == 0:
                self.settings.j_offset = 1e-6
            self.algorithms._addextprof(rho, -2)
            self.algorithms._set_alpha_and_offset(
                self.settings.j_alpha1,
                self.settings.j_alpha2,
                self.settings.j_offset,
                0,
            )
            self.algorithms.Itot = self.settings.Ip

        else:
            geo = _gfile_geometry(self.settings.gfilename, self.settings.nr)
            sqrtpsin = geo["sqrtpsin"]
            volgrid = geo["volgrid"]
            agrid = geo["agrid"]
            qr = geo["qr"]
            Ipint = geo["Ipint"]
            Jrms_norm = geo["Jrms_norm"]
            ftrapped_profile = geo["ftrapped_profile"]
            geq_a = geo["geq_a"]
            geq_R = geo["geq_R"]
            geq_z0 = geo["geq_z0"]
            geq_kappa = geo["geq_kappa"]
            geq_delta = geo["geq_delta"]

            if self.settings.verbosity > 1:
                print("gEQDSK geometry:")
                print(f"Minor radius: {geq_a}")
                print(f"Major radius: {geq_R}")
                print(f"Elongation: {geq_kappa}")
                print(f"Triangularity: {geq_delta}")
                print(f"z0: {geq_z0}")
                print("gEQDSK Ip:", Ipint)

            if self.settings.verbosity > 1:
                print("Len of sqrtpsin:", len(sqrtpsin))
                print("Len of volgrid:", len(volgrid))
                print("Len of Jrms_norm:", len(Jrms_norm))
                print("Len of qr:", len(qr))
                print("Len of agrid:", len(agrid))
                print("Len of ftrapped_profile:", len(ftrapped_profile))

            if np.abs(geq_a / self.settings.a - 1) > 0.1:
                print(
                    f"Warning: gEQDSK minor radius ({geq_a} m) differs significantly from settings a ({self.settings.a} m). Defaulting to gEQDSK value."
                )
                self.settings.a = geq_a
                self.algorithms.a = geq_a
            if np.abs(geq_R / self.settings.R - 1) > 0.1:
                print(
                    f"Warning: gEQDSK major radius ({geq_R} m) differs significantly from settings R ({self.settings.R} m). Defaulting to gEQDSK value."
                )
                self.settings.R = geq_R
                self.algorithms.R = geq_R
            if np.abs(geq_kappa / self.settings.kappa - 1) > 0.1:
                print(
                    f"Warning: gEQDSK elongation ({geq_kappa}) differs significantly from settings kappa ({self.settings.kappa}). Defaulting to gEQDSK value."
                )
                self.settings.kappa = geq_kappa
                self.algorithms.kappa = geq_kappa
            if np.abs(geq_delta / self.settings.delta - 1) > 0.1:
                print(
                    f"Warning: gEQDSK triangularity ({geq_delta}) differs significantly from settings delta ({self.settings.delta}). Defaulting to gEQDSK value."
                )
                self.settings.delta = geq_delta
                self.algorithms.delta = geq_delta
            if np.abs(Ipint / self.settings.Ip - 1) > 0.1:
                print(
                    f"Warning: gEQDSK Ip ({Ipint}) differs significantly from settings Ip ({self.settings.Ip}). Defaulting to gEQDSK value."
                )
                self.settings.Ip = Ipint
                self.algorithms.Ip = Ipint

            self.algorithms._addextprof(sqrtpsin, -2)
            self.algorithms._addextprof(volgrid, -1)
            self.algorithms._addextprof(Jrms_norm, 0)
            self.algorithms.Itot = Ipint
            self.algorithms._addextprof(qr, 5)
            self.algorithms._addextprof(agrid, -3)
            self.algorithms._addextprof(ftrapped_profile, 6)
        pass

    def __check_settings(self) -> None:
        """
        Checks the settings for consistency. Collects everything that is wrong
        and raises once, so that a settings file with several mistakes in it
        reports all of them rather than one per run. Anything suspicious but
        legal is printed as a warning instead.
        """
        if not hasattr(self, "settings"):
            return
        s = self.settings
        bad = []
        warn = []

        # names that must match something the code knows about
        if s.scalinglaw not in self.scalinglaws:
            bad.append(
                f"scalinglaw '{s.scalinglaw}' is not defined. "
                f"Available: {', '.join(sorted(self.scalinglaws))}."
            )
        elif self.scalinglaws[s.scalinglaw].get("Pheat_alpha") == -1:
            # the closed-form solver evaluates (W_tot/K)**(1/(1+Pheat_alpha)),
            # which is singular at Pheat_alpha = -1
            bad.append(
                f"scalinglaw '{s.scalinglaw}' has Pheat_alpha = -1, which makes "
                f"the closed-form power balance singular. Use a different exponent."
            )
        if s.resistivity_model.lower() not in RESISTIVITY_MODELS:
            bad.append(
                f"resistivity_model '{s.resistivity_model}' is not recognized. "
                f"Available: {', '.join(sorted(RESISTIVITY_MODELS))}."
            )

        # scan limits
        if s.Tmin_keV >= s.Tmax_keV:
            bad.append(
                f"Tmin_keV ({s.Tmin_keV}) must be less than Tmax_keV ({s.Tmax_keV})."
            )
        if s.nmin_frac >= s.nmax_frac:
            bad.append(
                f"nmin_frac ({s.nmin_frac}) must be less than nmax_frac ({s.nmax_frac})."
            )
        if s.Tmin_keV <= 0:
            bad.append(f"Tmin_keV must be positive, got {s.Tmin_keV}.")
        if s.nmin_frac <= 0:
            bad.append(f"nmin_frac must be positive, got {s.nmin_frac}.")

        # grid resolution
        for key in ("Nn", "NTi", "nr"):
            if getattr(s, key) < 2:
                bad.append(f"{key} must be at least 2, got {getattr(s, key)}.")

        # machine geometry
        for key in ("R", "a", "kappa", "tipeak_over_tepeak", "H_fac"):
            if getattr(s, key) <= 0:
                bad.append(f"{key} must be positive, got {getattr(s, key)}.")
        if s.a >= s.R:
            bad.append(
                f"minor radius a ({s.a} m) must be smaller than major radius R ({s.R} m)."
            )
        if abs(s.delta) >= 1:
            bad.append(
                f"delta (triangularity) must be between -1 and 1, got {s.delta}."
            )
        if s.B0 == 0:
            bad.append("B_0 must be nonzero.")
        if s.Ip == 0:
            bad.append("I_P must be nonzero.")

        # composition
        impsum = float(np.sum(s.impurityfractions))
        if not 0 <= impsum < 1:
            shown = ", ".join(f"{v:g}" for v in s.impurityfractions)
            bad.append(
                f"impurityfractions must sum to at least 0 and less than 1, "
                f"got {impsum:.4g} from [{shown}]."
            )
        if s.fuel not in (1, 2):
            bad.append(f"fuel must be 1 (D-D) or 2 (D-T), got {s.fuel}.")

        # legal, but worth saying out loud
        keys = getattr(s, "rawkeys", set())
        if ("Zeff_target" in keys) != ("impurity" in keys):
            warn.append(
                "Zeff_target and impurity must both be given to set an impurity "
                "fraction from Z_eff; specifying only one has no effect."
            )
        for key in sorted(keys & set(DEPRECATED_SETTINGS_KEYS)):
            warn.append(
                f"{key} is no longer used, because {DEPRECATED_SETTINGS_KEYS[key]}. "
                f"You can delete it from the settings file."
            )
        unknown = keys - KNOWN_SETTINGS_KEYS - set(DEPRECATED_SETTINGS_KEYS)
        if unknown:
            warn.append(
                f"unrecognized settings, which are ignored: {', '.join(sorted(unknown))}."
            )

        for w in warn:
            print(f"Warning: {w}")
        if bad:
            where = getattr(s, "settingsfile", "the settings file")
            raise ValueError(
                f"Found {len(bad)} problem(s) in {where}:\n  - " + "\n  - ".join(bad)
            )

    def update_plotsettings(self, plotsettingsfile=None):
        if plotsettingsfile is not None:
            self.plotsettings = POPCON_plotsettings(plotsettingsfile)
            self.plotsettingsfile = plotsettingsfile
        else:
            self.plotsettings = POPCON_plotsettings(self.plotsettingsfile)
        pass


@nb.njit(parallel=True, cache=True)
def solve_nT_par(state, nn: int, nT: int, n_e_20: np.ndarray, T_i_keV: np.ndarray):
    """
    Compiling the solver allows for parallelization, as each
    point in the grid can be solved independently.
    """

    Paux = np.empty((nn, nT), dtype=np.float64)
    flags = np.zeros((nn, nT), dtype=np.int64)

    for i in nb.prange(nn):
        for j in nb.prange(nT):
            Paux[i, j], flags[i, j] = P_aux_impfrac(state, n_e_20[i], T_i_keV[j])

    return Paux, flags


@nb.njit(parallel=False, cache=True)
def solve_nT(state, nn: int, nT: int, n_e_20: np.ndarray, T_i_keV: np.ndarray):
    """
    Compiling the solver allows for parallelization, as each
    point in the grid can be solved independently.
    """

    Paux = np.empty((nn, nT), dtype=np.float64)
    flags = np.zeros((nn, nT), dtype=np.int64)

    for i in nb.prange(nn):
        for j in nb.prange(nT):
            Paux[i, j], flags[i, j] = P_aux_impfrac(state, n_e_20[i], T_i_keV[j])

    return Paux, flags


@nb.njit(parallel=True, cache=True)
def populate_outputs_par(state, n_e_20_max, T_i_max, T_e_max, Paux, Nn, NTi):
    """
    Compiling this function is handy as it allows for
    parallelization of the most computationally expensive
    part of the code.
    """
    rho = state.sqrtpsin
    line_average_fac = np.average(get_profile(state, rho, 1))
    n_i_20_max = np.empty((Nn, NTi), dtype=np.float64)
    n_i_20_avg = np.empty((Nn, NTi), dtype=np.float64)
    Pfusion = np.empty((Nn, NTi), dtype=np.float64)
    Pfusionheating = np.empty((Nn, NTi), dtype=np.float64)
    Pohmic = np.empty((Nn, NTi), dtype=np.float64)
    Pbrems = np.empty((Nn, NTi), dtype=np.float64)
    Psynch = np.empty((Nn, NTi), dtype=np.float64)
    Pimprad = np.empty((Nn, NTi), dtype=np.float64)
    Prad = np.empty((Nn, NTi), dtype=np.float64)
    Pheat = np.empty((Nn, NTi), dtype=np.float64)
    Wtot = np.empty((Nn, NTi), dtype=np.float64)
    tauE = np.empty((Nn, NTi), dtype=np.float64)
    Pconf = np.empty((Nn, NTi), dtype=np.float64)
    Ploss = np.empty((Nn, NTi), dtype=np.float64)
    Pdd = np.empty((Nn, NTi), dtype=np.float64)
    Pdt = np.empty((Nn, NTi), dtype=np.float64)
    Palpha = np.empty((Nn, NTi), dtype=np.float64)
    Psol = np.empty((Nn, NTi), dtype=np.float64)
    f_rad = np.empty((Nn, NTi), dtype=np.float64)
    Q = np.empty((Nn, NTi), dtype=np.float64)
    H89 = np.empty((Nn, NTi), dtype=np.float64)
    H98 = np.empty((Nn, NTi), dtype=np.float64)
    vloop = np.empty((Nn, NTi), dtype=np.float64)
    betaN = np.empty((Nn, NTi), dtype=np.float64)
    for i in nb.prange(Nn):
        for j in nb.prange(NTi):
            dil = plasma_dilution(state, T_e_max[j])
            n_i_20_max[i, j] = n_e_20_max[i] * dil
            n_i_20_avg[i, j] = (
                n_i_20_max[i, j]
                * volume_integral(state, rho, get_profile(state, rho, 2))
                / state.V
            )
            Pfusion[i, j] = volume_integral(
                state, rho, _P_fusion(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pfusionheating[i, j] = volume_integral(
                state, rho, _P_fusion_heating(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pohmic[i, j] = volume_integral(
                state, rho, _P_OH_prof(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Pbrems[i, j] = volume_integral(
                state, rho, _P_brem_rad(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Psynch[i, j] = volume_integral(
                state, rho, _P_synch(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Pimprad[i, j] = volume_integral(
                state, rho, _P_impurity_rad(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Prad[i, j] = Pbrems[i, j] + Pimprad[i, j] + Psynch[i, j]
            Pheat[i, j] = (
                Pfusionheating[i, j]
                + Pohmic[i, j]
                + Paux[i, j]
                - Pbrems[i, j]
                - Psynch[i, j]
            )
            Wtot[i, j] = volume_integral(
                state, rho, _W_tot_prof(state, rho, T_i_max[j], n_e_20_max[i])
            )
            tauE[i, j] = tauE_scalinglaw(
                state, Pheat[i, j], n_e_20_max[i] * line_average_fac
            )
            Pconf[i, j] = Wtot[i, j] / tauE[i, j]
            # Ploss is all power leaving the plasma. Pconf is the transport loss the
            # scaling law predicts; brems and synch were removed before it (see the
            # solver's Pheat), so add them back for the total. Psol = Ploss - Prad
            # then leaves Pconf - Pimprad, the power reaching the divertor.
            Ploss[i, j] = Pconf[i, j] + Pbrems[i, j] + Psynch[i, j]
            Pdd[i, j] = volume_integral(
                state, rho, _P_DDnHe3_prof(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pdd[i, j] += volume_integral(
                state, rho, _P_DDpT_prof(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pdt[i, j] = volume_integral(
                state, rho, _P_DTnHe4_prof(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Palpha[i, j] = Pdt[i, j] * 3.52e3 / (3.52e3 + 14.06e3)
            Psol[i, j] = Ploss[i, j] - Prad[i, j]
            f_rad[i, j] = Prad[i, j] / Ploss[i, j]
            Q[i, j] = Q_fusion(state, T_i_max[j], n_e_20_max[i], Paux[i, j])
            H89[i, j] = tauE[i, j] / tauE_H89(
                state, Pheat[i, j], n_e_20_max[i] * line_average_fac
            )
            H98[i, j] = tauE[i, j] / tauE_H98(
                state, Pheat[i, j], n_e_20_max[i] * line_average_fac
            )
            vloop[i, j] = Vloop(state, T_e_max[j], n_e_20_max[i])
            betaN[i, j] = 100 * BetaN(state, T_i_max[j], n_e_20_max[i])  # in percent
    return ComputedOutputs(
        n_i_20_max,
        n_i_20_avg,
        Pfusion,
        Pfusionheating,
        Pohmic,
        Pbrems,
        Psynch,
        Pimprad,
        Prad,
        Pheat,
        Wtot,
        tauE,
        Pconf,
        Ploss,
        Pdd,
        Pdt,
        Palpha,
        Psol,
        f_rad,
        Q,
        H89,
        H98,
        vloop,
        betaN,
    )


@nb.njit(parallel=False, cache=True)
def populate_outputs(state, n_e_20_max, T_i_max, T_e_max, Paux, Nn, NTi):
    """
    Compiling this function is handy as it allows for
    parallelization of the most computationally expensive
    part of the code.
    """
    rho = state.sqrtpsin
    line_average_fac = np.average(get_profile(state, rho, 1))
    n_i_20_max = np.empty((Nn, NTi), dtype=np.float64)
    n_i_20_avg = np.empty((Nn, NTi), dtype=np.float64)
    Pfusion = np.empty((Nn, NTi), dtype=np.float64)
    Pfusionheating = np.empty((Nn, NTi), dtype=np.float64)
    Pohmic = np.empty((Nn, NTi), dtype=np.float64)
    Pbrems = np.empty((Nn, NTi), dtype=np.float64)
    Psynch = np.empty((Nn, NTi), dtype=np.float64)
    Pimprad = np.empty((Nn, NTi), dtype=np.float64)
    Prad = np.empty((Nn, NTi), dtype=np.float64)
    Pheat = np.empty((Nn, NTi), dtype=np.float64)
    Wtot = np.empty((Nn, NTi), dtype=np.float64)
    tauE = np.empty((Nn, NTi), dtype=np.float64)
    Pconf = np.empty((Nn, NTi), dtype=np.float64)
    Ploss = np.empty((Nn, NTi), dtype=np.float64)
    Pdd = np.empty((Nn, NTi), dtype=np.float64)
    Pdt = np.empty((Nn, NTi), dtype=np.float64)
    Palpha = np.empty((Nn, NTi), dtype=np.float64)
    Psol = np.empty((Nn, NTi), dtype=np.float64)
    f_rad = np.empty((Nn, NTi), dtype=np.float64)
    Q = np.empty((Nn, NTi), dtype=np.float64)
    H89 = np.empty((Nn, NTi), dtype=np.float64)
    H98 = np.empty((Nn, NTi), dtype=np.float64)
    vloop = np.empty((Nn, NTi), dtype=np.float64)
    betaN = np.empty((Nn, NTi), dtype=np.float64)
    for i in nb.prange(Nn):
        for j in nb.prange(NTi):
            dil = plasma_dilution(state, T_e_max[j])
            n_i_20_max[i, j] = n_e_20_max[i] * dil
            n_i_20_avg[i, j] = (
                n_i_20_max[i, j]
                * volume_integral(state, rho, get_profile(state, rho, 2))
                / state.V
            )
            Pfusion[i, j] = volume_integral(
                state, rho, _P_fusion(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pfusionheating[i, j] = volume_integral(
                state, rho, _P_fusion_heating(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pohmic[i, j] = volume_integral(
                state, rho, _P_OH_prof(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Pbrems[i, j] = volume_integral(
                state, rho, _P_brem_rad(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Psynch[i, j] = volume_integral(
                state, rho, _P_synch(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Pimprad[i, j] = volume_integral(
                state, rho, _P_impurity_rad(state, rho, T_e_max[j], n_e_20_max[i])
            )
            Prad[i, j] = Pbrems[i, j] + Pimprad[i, j] + Psynch[i, j]
            Pheat[i, j] = (
                Pfusionheating[i, j]
                + Pohmic[i, j]
                + Paux[i, j]
                - Pbrems[i, j]
                - Psynch[i, j]
            )
            Wtot[i, j] = volume_integral(
                state, rho, _W_tot_prof(state, rho, T_i_max[j], n_e_20_max[i])
            )
            tauE[i, j] = tauE_scalinglaw(
                state, Pheat[i, j], n_e_20_max[i] * line_average_fac
            )
            Pconf[i, j] = Wtot[i, j] / tauE[i, j]
            # Ploss is all power leaving the plasma. Pconf is the transport loss the
            # scaling law predicts; brems and synch were removed before it (see the
            # solver's Pheat), so add them back for the total. Psol = Ploss - Prad
            # then leaves Pconf - Pimprad, the power reaching the divertor.
            Ploss[i, j] = Pconf[i, j] + Pbrems[i, j] + Psynch[i, j]
            Pdd[i, j] = volume_integral(
                state, rho, _P_DDnHe3_prof(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pdd[i, j] += volume_integral(
                state, rho, _P_DDpT_prof(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Pdt[i, j] = volume_integral(
                state, rho, _P_DTnHe4_prof(state, rho, T_i_max[j], n_i_20_max[i, j])
            )
            Palpha[i, j] = Pdt[i, j] * 3.52e3 / (3.52e3 + 14.06e3)
            Psol[i, j] = Ploss[i, j] - Prad[i, j]
            f_rad[i, j] = Prad[i, j] / Ploss[i, j]
            Q[i, j] = Q_fusion(state, T_i_max[j], n_e_20_max[i], Paux[i, j])
            H89[i, j] = tauE[i, j] / tauE_H89(
                state, Pheat[i, j], n_e_20_max[i] * line_average_fac
            )
            H98[i, j] = tauE[i, j] / tauE_H98(
                state, Pheat[i, j], n_e_20_max[i] * line_average_fac
            )
            vloop[i, j] = Vloop(state, T_e_max[j], n_e_20_max[i])
            betaN[i, j] = 100 * BetaN(state, T_i_max[j], n_e_20_max[i])  # in percent
    return ComputedOutputs(
        n_i_20_max,
        n_i_20_avg,
        Pfusion,
        Pfusionheating,
        Pohmic,
        Pbrems,
        Psynch,
        Pimprad,
        Prad,
        Pheat,
        Wtot,
        tauE,
        Pconf,
        Ploss,
        Pdd,
        Pdt,
        Palpha,
        Psol,
        f_rad,
        Q,
        H89,
        H98,
        vloop,
        betaN,
    )


class ScanAxis:
    """
    One scanned parameter and the values it takes.
    """

    def __init__(self, parameter: str, values) -> None:
        self.parameter = str(parameter)
        self.values = list(values)
        # never a bare 'n' or 'T': Dataset.T is transpose
        self.dim = "scan_" + re.sub(r"\W", "_", self.parameter)

    @property
    def label(self) -> str:
        return self.parameter

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"ScanAxis({self.parameter!r}, {self.values!r})"


def _parse_axis_spec(spec, which):
    """
    Accepts the several ways a scan axis can be written and returns a
    ScanAxis. Handles ('I_P', [...]), {'parameter':..., 'values': [...]} and
    {'parameter':..., 'min':..., 'max':..., 'N':...}.
    """
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        parameter, rest = spec
        if isinstance(rest, dict):
            return _parse_axis_spec({"parameter": parameter, **rest}, which)
        return ScanAxis(parameter, np.asarray(rest, dtype=float).tolist())

    if not isinstance(spec, dict):
        raise ValueError(
            f"{which}: could not read the scan specification {spec!r}. Give "
            f"either ('I_P', [6.0, 8.0, 10.0]) or "
            f"{{'parameter': 'I_P', 'min': 6.0, 'max': 10.0, 'N': 3}}."
        )

    spec = dict(spec)
    parameter = spec.pop("parameter", None)
    if parameter is None:
        raise ValueError(f"{which}: the scan specification needs a 'parameter'.")

    if "values" in spec:
        return ScanAxis(parameter, list(spec["values"]))

    missing = [k for k in ("min", "max", "N") if k not in spec]
    if missing:
        raise ValueError(
            f"{which}: scanning '{parameter}' needs either 'values', or all of "
            f"'min', 'max' and 'N'. Missing: {', '.join(missing)}."
        )
    # N is validated in _check_scan; guard only against linspace throwing here
    N = int(spec["N"])
    if N < 1:
        return ScanAxis(parameter, [])
    return ScanAxis(
        parameter, np.linspace(float(spec["min"]), float(spec["max"]), N).tolist()
    )


def _fmt_value(v):
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)


class POPCON_scan:
    """
    Class POPCON_scan

    Runs a POPCON at every combination of two scanned machine parameters and
    plots the results as a grid, so a design space can be looked at in one
    figure. Holds one fully-formed POPCON per cell rather than subclassing
    POPCON, because every cell has its own settings, geometry and output.

        sc = op.POPCON_scan(settingsfile='POPCON_input_example.yaml',
                            plotsettingsfile='plotsettings.yml',
                            scan={'rows': ('I_P', {'min': 7, 'max': 12, 'N': 3}),
                                  'cols': ('B_0', [9.0, 10.5, 12.0])})
        sc.run_scan()
        sc.plot()
        sc.plot_metric('Q')

    The scan can equally be written as a `scan:` block in the settings file.
    Parameters are named as they appear in the settings file, and the whole
    settings derivation is re-run for each cell, so scanning R on a file that
    specifies qstar correctly re-derives Ip.
    """

    def __init__(
        self,
        settingsfile=None,
        plotsettingsfile=None,
        scalinglawfile=None,
        scan=None,
    ) -> None:
        self.settingsfile = os.path.abspath(settingsfile)
        self.plotsettingsfile = plotsettingsfile
        self.scalinglawfile = scalinglawfile
        self.base_settings = POPCON_settings(settingsfile)

        spec = scan if scan is not None else self.base_settings.scan
        if not spec:
            raise ValueError(
                "No scan specified. Give scan={'rows': ..., 'cols': ...} or add "
                "a 'scan:' block to the settings file."
            )
        self.row, self.col = self._read_spec(spec)
        self._check_scan()

        self.cells = []
        self.cell_overrides = []
        self._output = None
        self._build_cells()

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------

    def _read_spec(self, spec):
        if not isinstance(spec, dict):
            raise ValueError(
                "The scan specification must be a mapping with 'rows' and "
                f"'cols' keys, got {type(spec).__name__}."
            )
        unknown = set(spec) - {"rows", "cols"}
        if unknown:
            raise ValueError(
                f"Unknown key(s) in the scan specification: "
                f"{', '.join(sorted(unknown))}. Expected 'rows' and 'cols'."
            )
        row = _parse_axis_spec(spec["rows"], "rows") if "rows" in spec else None
        col = _parse_axis_spec(spec["cols"], "cols") if "cols" in spec else None
        return row, col

    def _check_scan(self) -> None:
        """
        Reports everything wrong with the scan specification at once, in the
        same style as POPCON.__check_settings.
        """
        bad, warn = [], []
        s = self.base_settings

        if self.row is None or self.col is None:
            bad.append(
                "a scan needs two parameters, given as 'rows' and 'cols'. "
                "To vary just one, give the other a single value."
            )
        if self.row is not None and self.col is not None:
            if self.row.parameter == self.col.parameter:
                bad.append(
                    f"rows and cols both scan '{self.row.parameter}'; they must "
                    f"be different parameters."
                )

        for axis, which in ((self.row, "rows"), (self.col, "cols")):
            if axis is None:
                continue
            p = axis.parameter
            if p in UNSCANNABLE_REASONS:
                bad.append(
                    f"{which}: '{p}' cannot be scanned, because {UNSCANNABLE_REASONS[p]}."
                )
            elif p not in SCANNABLE_SETTINGS_KEYS:
                bad.append(
                    f"{which}: '{p}' is not a scannable setting. Available: "
                    f"{', '.join(sorted(SCANNABLE_SETTINGS_KEYS))}."
                )
            if len(axis.values) < 1:
                bad.append(f"{which}: '{p}' has no values to scan.")
            if len(set(map(repr, axis.values))) != len(axis.values):
                warn.append(
                    f"{which}: '{p}' has repeated values, so some cells will be identical."
                )
            # the gEQDSK trap: __get_geometry overrides these from the
            # equilibrium, and always takes the ohmic current from it
            if s.gfilename != "" and p in GEQDSK_OWNED_KEYS:
                bad.append(
                    f"{which}: '{p}' cannot be scanned while gfilename is set "
                    f"({os.path.basename(s.gfilename)}). The geometry is read from "
                    f"the gEQDSK, which overrides R, a, kappa, delta and Ip whenever "
                    f"they differ from the settings by more than 10%, and always "
                    f"takes the ohmic current from the equilibrium. Some cells would "
                    f"silently be identical and others not. Scan B_0 or H_fac "
                    f"instead, or clear gfilename to use parabolic profiles."
                )

        for w in warn:
            print(f"Warning: {w}")
        if bad:
            raise ValueError(
                f"Found {len(bad)} problem(s) in the scan specification in "
                f"{self.settingsfile}:\n  - " + "\n  - ".join(bad)
            )

    def _build_cells(self) -> None:
        """
        Builds every cell's POPCON up front, so that a settings problem at one
        corner of the scan is reported before anything is solved.
        """
        problems = []
        self.cells = [[None] * len(self.col) for _ in range(len(self.row))]
        self.cell_overrides = [[None] * len(self.col) for _ in range(len(self.row))]

        for i, vi in enumerate(self.row.values):
            for j, vj in enumerate(self.col.values):
                overrides = {self.row.parameter: vi, self.col.parameter: vj}
                self.cell_overrides[i][j] = overrides
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        settings = self.base_settings.with_overrides(**overrides)
                        cell = POPCON(
                            plotsettingsfile=self.plotsettingsfile,
                            scalinglawfile=self.scalinglawfile,
                            settings=settings,
                        )
                    self.cells[i][j] = cell
                except (ValueError, KeyError) as e:
                    problems.append(
                        f"cell ({i}, {j}) with {self.row.parameter}="
                        f"{_fmt_value(vi)}, {self.col.parameter}={_fmt_value(vj)}: {e}"
                    )

        if problems:
            raise ValueError(
                f"{len(problems)} of {len(self.row) * len(self.col)} scan cells have "
                f"invalid settings:\n  - " + "\n  - ".join(problems)
            )

    # -------------------------------------------------------------------
    # Solving
    # -------------------------------------------------------------------

    @property
    def shape(self):
        return (len(self.row), len(self.col))

    def run_scan(self, progress: bool = True, quiet: bool = None) -> None:
        """
        Solves every cell. Cells run one at a time: settings.parallel already
        threads the solve across the whole n,T grid, so a second layer of
        parallelism would only oversubscribe the cores.
        """
        if quiet is None:
            # a scan reports its own per-cell line, so the cells' own output is
            # suppressed unless the settings file asks for real verbosity
            quiet = self.base_settings.verbosity < 2

        nrow, ncol = self.shape
        total = nrow * ncol
        gridpoints = total * self.base_settings.Nn * self.base_settings.NTi
        if gridpoints > 5e5:
            print(
                f"Warning: this scan is {nrow}x{ncol} cells of "
                f"{self.base_settings.Nn}x{self.base_settings.NTi} points "
                f"({gridpoints:.0f} total). Consider smaller Nn/NTi."
            )

        for i in range(nrow):
            for j in range(ncol):
                cell = self.cells[i][j]
                label = (
                    f"{self.row.parameter}={_fmt_value(self.row.values[i])}, "
                    f"{self.col.parameter}={_fmt_value(self.col.values[j])}"
                )
                start = datetime.datetime.now()
                if quiet:
                    # read() and __report_invalid print unconditionally; across
                    # a whole scan that is a wall of text
                    with contextlib.redirect_stdout(io.StringIO()):
                        cell.run_POPCON()
                else:
                    cell.run_POPCON()
                elapsed = (datetime.datetime.now() - start).total_seconds()
                if progress:
                    valid = int(
                        np.count_nonzero(
                            cell.output.Paux.values < UNPHYSICAL_PLOT_CUTOFF
                        )
                    )
                    npts = cell.output.Paux.size
                    print(
                        f"  [{i * ncol + j + 1}/{total}] {label}  "
                        f"{valid}/{npts} points solved  ({elapsed:.1f} s)"
                    )

        self._output = None

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------

    @property
    def datas(self):
        """Every cell's output Dataset, row-major."""
        return [c.output for row in self.cells for c in row if hasattr(c, "output")]

    @property
    def algorithms_list(self):
        return [
            c.algorithms for row in self.cells for c in row if hasattr(c, "algorithms")
        ]

    @property
    def scanvariables(self):
        return {
            self.row.parameter: np.asarray(self.row.values),
            self.col.parameter: np.asarray(self.col.values),
        }

    @property
    def output(self):
        """
        Every cell's results in one Dataset, with the two scan parameters as
        extra dimensions. Built on first use.
        """
        if self._output is None:
            self._output = self._combine()
        return self._output

    def _combine(self):
        missing = [
            (i, j)
            for i, row in enumerate(self.cells)
            for j, c in enumerate(row)
            if c is None or not hasattr(c, "output")
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} cell(s) have no output: {missing}. Call run_scan() first."
            )

        grid = [[c.output for c in row] for row in self.cells]
        # coords='all' is deliberate. n_e_20_max and friends genuinely differ
        # between cells whenever I_P or a is scanned (n_G = Ip/(pi a^2)), and
        # the default 'different' would make the result's shape depend on which
        # parameter happened to be scanned. They are non-index coordinates, so
        # concatenation is positional and nothing is silently dropped
        ds = xr.combine_nested(
            grid,
            concat_dim=[self.row.dim, self.col.dim],
            coords="all",
            data_vars="all",
            join="exact",
            combine_attrs="drop_conflicts",
        )
        ds = ds.assign_coords(
            {
                self.row.dim: np.asarray(self.row.values),
                self.col.dim: np.asarray(self.col.values),
            }
        )
        # combine_nested leaves the two new dimensions in its own order, which
        # is not the (rows, cols) the cells are indexed by. Pin it, so that
        # positional use of the arrays lines up with cells[i][j]
        ds = ds.transpose(self.row.dim, self.col.dim, DIM_N, DIM_T)
        ds[self.row.dim].attrs["parameter"] = self.row.parameter
        ds[self.col.dim].attrs["parameter"] = self.col.parameter
        ds.attrs["scan_parameters"] = json.dumps(
            [self.row.parameter, self.col.parameter]
        )
        return ds

    def metric(self, name: str = "Q", reduce: str = "max"):
        """
        One number per cell, reduced over the n,T grid with the unphysical
        points left out. 'Q'/'max' answers "how far does this machine get".
        """
        field = getattr(self.output, name)
        valid = self.output.Paux < UNPHYSICAL_PLOT_CUTOFF
        reduced = getattr(field.where(valid), reduce)(dim=[DIM_N, DIM_T])
        # (rows, cols), so that .values lines up with cells[i][j] and with
        # the tick labels plot_metric puts on the axes
        return reduced.transpose(self.row.dim, self.col.dim)

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    def _shared_axes(self):
        """
        Whether the cells' axes actually mean the same thing. Sharing y is
        only right for the Greenwald fraction: an absolute density axis moves
        with n_G = Ip/(pi a^2) and would be misleading if shared.
        """
        scanned = {self.row.parameter, self.col.parameter}
        yax = self.cells[0][0].plotsettings.yax
        sharey = yax == "nG" and not (scanned & {"nmin_frac", "nmax_frac"})
        sharex = not (scanned & {"Tmin_keV", "Tmax_keV", "tipeak_over_tepeak"})
        return sharex, sharey

    def _harmonize_levels(self, names=None) -> None:
        """
        Puts every panel on the same contour levels. Without this, 'minmax'
        scaling picks levels from each panel's own range and no two panels in
        the figure are comparable. Each cell owns its plotsettings, so this
        does not leak between them.
        """
        cells = [c for row in self.cells for c in row if hasattr(c, "output")]
        if not cells:
            return
        if names is None:
            names = list(cells[0].plotsettings.plotoptions.keys())

        for name in names:
            opts = cells[0].plotsettings.plotoptions[name]
            if not opts["plot"] or opts["spacing"] == "manual":
                continue
            lo, hi = np.inf, -np.inf
            for c in cells:
                if name not in c.output:
                    continue
                valid = c.output.Paux < UNPHYSICAL_PLOT_CUTOFF
                data = c.output[name].where(valid).values
                if np.all(np.isnan(data)):
                    continue
                lo = min(lo, float(np.nanmin(data)))
                hi = max(hi, float(np.nanmax(data)))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            if opts["spacing"] == "log" and lo <= 0:
                lo = hi * 1e-4
            for c in cells:
                o = c.plotsettings.plotoptions[name]
                o["scale"] = "specified"
                o["min"], o["max"] = lo, hi

    def plot(
        self,
        show: bool = True,
        savefig: str = "",
        names=None,
        figsize=None,
        panel_size=(3.4, 2.8),
        sharex=None,
        sharey=None,
        harmonize_levels: bool = True,
        infobox: bool = False,
    ):
        """
        The scan as a grid of POPCONs, one per cell.
        """
        nrow, ncol = self.shape
        auto_x, auto_y = self._shared_axes()
        sharex = auto_x if sharex is None else sharex
        sharey = auto_y if sharey is None else sharey
        if figsize is None:
            figsize = (panel_size[0] * ncol + 2.4, panel_size[1] * nrow + 1.0)

        if harmonize_levels:
            self._harmonize_levels(names)

        fig, axs = plt.subplots(
            nrow, ncol, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey
        )

        for i in range(nrow):
            for j in range(ncol):
                ax = axs[i][j]
                cell = self.cells[i][j]
                try:
                    # plot() narrates its contour levels at verbosity > 0,
                    # which across a whole grid is just noise
                    with contextlib.redirect_stdout(io.StringIO()):
                        cell.plot(
                            ax=ax,
                            show=False,
                            names=names,
                            legend=False,
                            infobox=infobox,
                        )
                except ValueError as e:
                    # one hopeless corner cell must not take the figure with it
                    ax.text(
                        0.5,
                        0.5,
                        "no physical\nsolution",
                        ha="center",
                        va="center",
                        color="0.4",
                        transform=ax.transAxes,
                    )
                    if self.base_settings.verbosity > 0:
                        print(f"  cell ({i}, {j}) not plotted: {e}")
                if i == 0:
                    ax.set_title(
                        f"{self.col.label} = {_fmt_value(self.col.values[j])}",
                        fontsize=11,
                    )
                if j == ncol - 1:
                    twin = ax.twinx()
                    twin.set_yticks([])
                    twin.set_ylabel(
                        f"{self.row.label} = {_fmt_value(self.row.values[i])}",
                        fontsize=11,
                    )
                if sharex and i < nrow - 1:
                    ax.set_xlabel("")
                if sharey and j > 0:
                    ax.set_ylabel("")

        handles, labels = [], []
        for i in range(nrow):
            for j in range(ncol):
                h, lbl = axs[i][j].get_legend_handles_labels()
                if len(h) > len(handles):
                    handles, labels = h, lbl
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

        fig.suptitle(
            f"{self.base_settings.name}: {self.row.parameter} vs {self.col.parameter}",
            fontsize=13,
        )
        fig.tight_layout()
        if savefig != "":
            fig.savefig(savefig, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axs

    def plot_metric(
        self,
        name: str = "Q",
        reduce: str = "max",
        show: bool = True,
        savefig: str = "",
        ax=None,
        cmap: str = "viridis",
        fmt: str = "{:.3g}",
    ):
        """
        One scalar per cell as a heatmap, so the trend across the scan reads
        at a glance.
        """
        z = self.metric(name, reduce).values
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(1.4 * self.shape[1] + 3, 1.1 * self.shape[0] + 2.5)
            )
            owns_figure = True
        else:
            fig, owns_figure = ax.get_figure(), False

        im = ax.imshow(z, cmap=cmap, origin="lower", aspect="auto")
        ax.set_xticks(range(self.shape[1]))
        ax.set_xticklabels([_fmt_value(v) for v in self.col.values])
        ax.set_yticks(range(self.shape[0]))
        ax.set_yticklabels([_fmt_value(v) for v in self.row.values])
        ax.set_xlabel(self.col.parameter)
        ax.set_ylabel(self.row.parameter)
        ax.set_title(f"{reduce} {name} over the n,T grid")

        finite = z[np.isfinite(z)]
        mid = (finite.max() + finite.min()) / 2 if finite.size else 0.0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if not np.isfinite(z[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    fmt.format(z[i, j]),
                    ha="center",
                    va="center",
                    color="w" if z[i, j] < mid else "k",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, label=f"{reduce} {name}")

        if owns_figure:
            fig.tight_layout()
        if savefig != "":
            fig.savefig(savefig, bbox_inches="tight")
        if show and owns_figure:
            plt.show()
        return fig, ax

    # -------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------

    def write_output(
        self,
        name: str = "",
        archive: bool = True,
        overwrite: bool = False,
        directory: str = None,
    ) -> None:
        """
        Saves the whole scan: the base settings, the scan specification, the
        combined arrays and the grid plot.
        """
        name, outputsdir, savedir = _prepare_output_dir(
            name, directory, overwrite, self.base_settings.name + "_scan"
        )

        shutil.copyfile(self.settingsfile, savedir.joinpath("settings.yaml"))
        cell0 = self.cells[0][0]
        shutil.copyfile(cell0.plotsettingsfile, savedir.joinpath("plotsettings.yaml"))
        shutil.copyfile(cell0.scalinglawfile, savedir.joinpath("scalinglaws.yaml"))

        # per-cell provenance goes here rather than into the netCDF attributes:
        # combine_attrs='drop_conflicts' discards the differing settings JSON,
        # and a few dozen KB of attributes would make ncdump unusable anyway
        with open(savedir.joinpath("scan.json"), "w") as f:
            json.dump(
                {
                    "rows": {
                        "parameter": self.row.parameter,
                        "values": self.row.values,
                    },
                    "cols": {
                        "parameter": self.col.parameter,
                        "values": self.col.values,
                    },
                    "overrides": self.cell_overrides,
                },
                f,
                indent=1,
            )

        for axis in (self.row, self.col):
            if not all(isinstance(v, (int, float, np.floating)) for v in axis.values):
                raise ValueError(
                    f"Cannot write a scan over '{axis.parameter}' to netCDF: its "
                    f"values are not numeric, and the default netCDF backend "
                    f"cannot store string coordinates. Install "
                    f"openpopcon[netcdf4] or scan a numeric parameter."
                )
        self.output.to_netcdf(savedir.joinpath("arrays.nc"))

        if self.base_settings.gfilename != "":
            shutil.copyfile(
                self.base_settings.gfilename,
                savedir.joinpath(os.path.basename(self.base_settings.gfilename)),
            )
        if self.base_settings.profsfilename != "":
            shutil.copyfile(
                self.base_settings.profsfilename,
                savedir.joinpath(os.path.basename(self.base_settings.profsfilename)),
            )

        self.plot(show=False, savefig=str(savedir.joinpath("POPCON_scan_plot.pdf")))
        plt.close("all")

        print(f"Wrote output to {_finalize_output(savedir, outputsdir, name, archive)}")

    @classmethod
    def read_output(cls, name: str, directory: str = None):
        """
        Reads back a scan written by write_output. Returns a POPCON_scan whose
        cells are set up but not re-solved; each cell's output is sliced out
        of the saved arrays.
        """
        if directory is None:
            outputsdir = pathlib.Path.cwd().joinpath("OpenPOPCON_outputs")
        else:
            outputsdir = pathlib.Path(directory)

        if name.endswith(".zip"):
            readdir = outputsdir.joinpath(name[:-4])
            shutil.unpack_archive(outputsdir.joinpath(name), readdir, "zip")
        else:
            readdir = outputsdir.joinpath(name)

        with open(readdir.joinpath("scan.json"), "r") as f:
            spec = json.load(f)

        sc = cls(
            settingsfile=str(readdir.joinpath("settings.yaml")),
            plotsettingsfile=str(readdir.joinpath("plotsettings.yaml")),
            scalinglawfile=str(readdir.joinpath("scalinglaws.yaml")),
            scan={
                "rows": (spec["rows"]["parameter"], spec["rows"]["values"]),
                "cols": (spec["cols"]["parameter"], spec["cols"]["values"]),
            },
        )

        ds = xr.load_dataset(readdir.joinpath("arrays.nc"))
        sc._output = ds
        for i in range(sc.shape[0]):
            for j in range(sc.shape[1]):
                cell = sc.cells[i][j]
                if cell.settings.gfilename != "":
                    cell.settings.gfilename = str(
                        readdir.joinpath(os.path.basename(cell.settings.gfilename))
                    )
                if cell.settings.profsfilename != "":
                    cell.settings.profsfilename = str(
                        readdir.joinpath(os.path.basename(cell.settings.profsfilename))
                    )
                with contextlib.redirect_stdout(io.StringIO()):
                    cell.run_POPCON(setuponly=True)
                cell.output = ds.isel({sc.row.dim: i, sc.col.dim: j}).drop_vars(
                    [sc.row.dim, sc.col.dim]
                )
        return sc
