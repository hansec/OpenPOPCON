# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 2024

OpenPOPCON v2.0
This is the Columbia University fork of the OpenPOPCON project developed
for MIT cours 22.63. This project is a refactor of the contributions
made to the original project in the development of MANTA. Contributors
to the original project are listed in the README.md file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.constants as const
import scipy.integrate as integrate
import xarray as xr
import yaml
from typing import Callable
from openpopcon_util import *
import numba as nb


@nb.experimental.jitclass(spec = [
            ('R', nb.float64), 
            ('a', nb.float64),
            ('kappa', nb.float64),
            ('B0', nb.float64),
            ('Ip', nb.float64),
            ('q_a', nb.float64),
            ('H', nb.float64),
            ('M_i', nb.float64),
            ('fHe', nb.float64),
            ('f_LH', nb.float64),
            ('volgrid', nb.float64[:]),
            ('sqrtpsin', nb.float64[:]),
            ('impurityfractions', nb.float64[:]),
            ('extprof', nb.boolean),
            ('profsdefined', nb.boolean),
            ('alpha1', nb.float64),
            ('alpha2', nb.float64),
            ('offset', nb.float64),
            ('extprofr', nb.float64[:]),
            ('extprofvals', nb.float64[:])
          ]) # type: ignore
class POPCON_params:
    """
    Physical parameters for the POPCON.
    """
    def __init__(self) -> None:
        self.R: float = 2.7
        "Major radius [m]"
        self.a: float
        "Minor radius [m]"
        self.kappa: float
        "Plasma elongation []"
        self.B0: float
        "Magnetic field at the magnetic axis [T]"
        self.Ip: float
        "Plasma current [MA]"
        self.q_a: float
        "Edge safety factor []"
        self.H: float
        "Assumed H factor relative to chosen scaling law []"
        self.M_i: float
        "Ion mass [AMU]"
        self.fHe: float
        "Ash fraction []"
        self.f_LH: float
        "Target LH fraction f_LH = P_sol / P_tot"
        self.volgrid = np.empty(0,dtype=np.float64)
        "Volume grid for flux surfaces"
        self.sqrtpsin = np.empty(0, dtype=np.float64)
        "Square root of poloidal flux ~ radial coordinate"
        # 0 = He, 1 = Ne, 2 = Ar, 3 = Kr, 4 = Xe, 5 = W
        self.impurityfractions = np.empty(6, dtype=np.float64)
        "Respective impurity fractions"
        self.extprof: bool
        self.profsdefined: bool = False
        self.alpha1: float
        self.alpha2: float
        self.offset: float
        self.extprofr = np.empty(0,dtype=np.float64)
        self.extprofvals = np.empty(0,dtype=np.float64)
        pass
    
    def _addextprof(self, extprofr, extprofvals):
        if not self.profsdefined:
            self.extprof = True
            self.profsdefined = True
            self.extprofr = extprofr
            self.extprofvals = extprofvals
        else:
            raise SyntaxError("Profile already defined. Create a new object.")
        
    def _set_alpha_and_offset(self, alpha1, alpha2, offset):
        self.extprof = False
        self.profsdefined = True
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.offset = offset
    
    def get_prof_val(self, r, v0):
        if self.extprof:
            return v0*np.interp(r,self.extprofr,self.extprofvals)
        else:
            return (v0-self.offset)*(1-r**self.alpha1)**self.alpha2+self.offset
    
    def etaext(self) -> float:
        return get_eta(self.R)


class POPCON_settings:
    """
    Settings for the POPCON.
    """
    Nn: int
    "Resolution in density"
    NTi: int
    "Resolution in temperature"
    scalinglaw: str
    "Which scaling law to use"
    nmax_frac: float
    "Greenwald fraction for nmax"
    nmin_frac: float
    "Minimum density fraction"
    Tmax_keV: float
    "Max temperature in keV"
    Tmin_keV: float
    "Min temperature in keV"
    maxit: int
    "Max iterations"
    accel: float
    "Overrelaxation parameter"
    verbosity: int
    """
    Verbosity level:
    0 - No output
    1 - Minimal output
    2 - Full output
    3 - Debug output
    4 - Print all matrices
    """
    # Additional functionality
    profiles: bool
    "Whether to use solved profiles."
    geom: bool
    "Whether to use gEQDSK equilibrium for flux surface volumes."
    eqdsk_f: str
    "Optional eqdsk file for geometry."
    profiles_f: str

    def __init__(self,
                 filename: str,
                 ) -> None:
        self.read(filename)
        pass

    def read(self, filename: str) -> None:
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError('Filename must end with .yaml or .yml')

        try:
            self.Nn = int(data['Nn'])
            self.NTi = int(data['NTi'])
            self.scalinglaw = data['scalinglaw']
            self.nmax_frac = float(data['nmax_frac'])
            self.nmin_frac = float(data['nmin_frac'])
            self.Tmax_keV = float(data['Tmax_keV'])
            self.Tmin_keV = float(data['Tmin_keV'])
            self.maxit = int(data['maxit'])
            self.accel = float(data['accel'])
            self.verbosity = int(data['verbosity'])
            self.profiles = bool(data['profiles'])
            self.geom = bool(data['geom'])
        except KeyError as e:
            raise KeyError(f'Key {e} not found in {filename}')
        if self.geom:
            self.eqdsk_f = data['eqdsk']
        if self.profiles:
            self.profiles_f = data['profiles']
        pass


@nb.experimental.jitclass
class POPCON_data:
    """
    Output data class for the POPCON.
    """
    def __init__(self) -> None:
        pass

class POPCON_plotsettings:
    def __init__(self,
                 filename: str,
                 ) -> None:
        pass


class POPCON:
    """
    The Primary class for the OpenPOPCON project.

    TODO: Write Documentation
    """

    def __init__(self,
                 params: POPCON_params,
                 settings: POPCON_settings,
                 plotsettings: POPCON_plotsettings) -> None:

        self.params = params
        self.settings = settings
        self.plotsettings = plotsettings

        self.__get_profiles()
        self.__get_geometry()
        self.__check_settings()

        self.volfunc: Callable = self.__get_volfunc()

        pass

    def save_file(self, filename: str) -> None:
        pass

    def load_file(self, filename: str) -> None:
        pass

    def __get_profiles(self) -> None:
        pass

    def __get_geometry(self) -> None:
        if self.settings.geom:
            gfile = read_eqdsk(self.settings.eqdsk_f)
            psin, volgrid = get_fluxvolumes(gfile)
            self.params.volgrid = volgrid
            self.params.sqrtpsin = np.sqrt(psin)

        pass

    def __get_volfunc(self) -> Callable:
        if self.settings.geom:
            return lambda psi: None
        else:
            return lambda psi: None

    def __check_settings(self) -> None:
        pass

    def __get_taue(self) -> None:
        pass
