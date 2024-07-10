# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 2024

OpenPOPCON v2.0
This is the Columbia University fork of the OpenPOPCON project developed
for MIT cours 22.63. This project is a refactor of the contributions
made to the original project in the development of MANTA. Contributors
to the original project are listed in the README.md file.
"""

import numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.constants as const, scipy.integrate as integrate
import xarray as xr
import yaml
from typing import Callable
from openpopcon_util import *

class POPCON_params:
    """
    Physical parameters for the POPCON.
    """
    R: float
    "Major radius [m]"
    a: float
    "Minor radius [m]"
    kappa: float
    "Plasma elongation []"
    B0: float
    "Magnetic field at the magnetic axis [T]"
    Ip: float
    "Plasma current [MA]"
    q_a: float
    "Edge safety factor []"
    H: float
    "Assumed H factor relative to chosen scaling law []"
    M_i: float
    "Ion mass [AMU]"
    fHe: float
    "Ash fraction []"
    impurities: set[tuple[str, float]]
    "Set of impurities and their fractions"
    f_LH: float
    "Target LH fraction f_LH = P_sol / P_tot"
    volgrid: np.ndarray
    "Volume grid for flux surfaces"
    sqrtpsin: np.ndarray
    "Square root of poloidal flux ~ radial coordinate"

    def __init__(self,
                 filename:str,
                 machinename:str = 'NEWDEVICE',
                 ) -> None:
        self.machinename = machinename
        self.read(filename)
        pass

    def read(self, filename:str) -> None:
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError('Filename must end with .yaml or .yml')
        
        try:
            self.R = float(data['R'])
            self.a = float(data['a'])
            self.kappa = float(data['kappa'])
            self.B0 = float(data['B0'])
            self.Ip = float(data['Ip'])
            self.q_a = float(data['q_a'])
            self.H = float(data['H'])
            self.M_i = float(data['M_i'])
            self.fHe = float(data['fHe'])
            self.impurities = set(data['impurities'])
            self.f_LH = float(data['f_LH'])
        except KeyError as e:
            raise KeyError(f'Key {e} not found in {filename}')
        pass

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
                 filename:str,
                 ) -> None:
        self.read(filename)
        pass

    def read(self, filename:str) -> None:
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

class POPCON_plotsettings:
    def __init__(self,
                 filename:str,
                 ) -> None:
        pass

class POPCON:
    """
    The Primary class for the OpenPOPCON project.

    TODO: Write Documentation
    """
    def __init__(self, 
                 params:POPCON_params, 
                 settings:POPCON_settings,
                 plotsettings:POPCON_plotsettings) -> None:        

        self.params = params
        self.settings = settings
        self.plotsettings = plotsettings
        
        self.__get_profiles()
        self.__get_geometry()
        self.__check_settings()

        self.volfunc: Callable = self.__get_volfunc()

        pass

    def save_file(self, filename:str) -> None:
        pass

    def load_file(self, filename:str) -> None:
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