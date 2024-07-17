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

# core of calculations
# jit compiled
@nb.experimental.jitclass(spec = [
            ('R', nb.float64), 
            ('a', nb.float64),
            ('kappa', nb.float64),
            ('B0', nb.float64),
            ('Ip', nb.float64),
            ('q_a', nb.float64),
            ('H', nb.float64),
            ('M_i', nb.float64),
            ('f_LH', nb.float64),
            ('volgrid', nb.float64[:]),
            ('sqrtpsin', nb.float64[:]),
            ('impurityfractions', nb.float64[:]),
            ('imcharges', nb.float64[:]),
            ('extprof_geoms', nb.boolean),
            ('geomsdefined', nb.boolean),
            ('extprof_imps', nb.boolean),
            ('impsdefined', nb.boolean),
            ('j_alpha1', nb.float64),
            ('j_alpha2', nb.float64),
            ('j_offset', nb.float64),
            ('ne_alpha1', nb.float64),
            ('ne_alpha2', nb.float64),
            ('ne_offset', nb.float64),
            ('ni_alpha1', nb.float64),
            ('ni_alpha2', nb.float64),
            ('ni_offset', nb.float64),
            ('Ti_alpha1', nb.float64),
            ('Ti_alpha2', nb.float64),
            ('Ti_offset', nb.float64),
            ('Te_alpha1', nb.float64),
            ('Te_alpha2', nb.float64),
            ('Te_offset', nb.float64),
            ('extprofr', nb.float64[:]),
            ('extprof_j', nb.float64[:]),
            ('extprof_ne', nb.float64[:]),
            ('extprof_ni', nb.float64[:]),
            ('extprof_Te', nb.float64[:]),
            ('extprof_Ti', nb.float64[:]),
            ('extprof_impfracs', nb.float64[:]),
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
        self.f_LH: float
        "Target LH fraction f_LH = P_sol / P_tot"
        self.volgrid = np.empty(0,dtype=np.float64)
        "Volume grid for flux surfaces"
        self.sqrtpsin = np.empty(0, dtype=np.float64)
        "Square root of poloidal flux ~ radial coordinate"
        # 0 = He, 1 = Ne, 2 = Ar, 3 = Kr, 4 = Xe, 5 = W
        self.impurityfractions = np.empty(6, dtype=np.float64)
        "Respective impurity fractions"
        self.imcharges = np.empty(6, dtype=np.float64)
        "Respective impurity charges"

        self.extprof_geoms: bool
        self.geomsdefined: bool = False

        self.extprof_imps: bool
        self.impsdefined: bool = False

        self.j_alpha1: float
        self.j_alpha2: float
        self.j_offset: float

        self.ne_alpha1: float
        self.ne_alpha2: float
        self.ne_offset: float

        self.ni_alpha1: float
        self.ni_alpha2: float
        self.ni_offset: float

        self.Ti_alpha1: float
        self.Ti_alpha2: float
        self.Ti_offset: float

        self.Te_alpha1: float
        self.Te_alpha2: float
        self.Te_offset: float

        self.extprofr = np.empty(0,dtype=np.float64)
        self.extprof_j = np.empty(0,dtype=np.float64)
        self.extprof_ne = np.empty(0,dtype=np.float64)
        self.extprof_ni = np.empty(0,dtype=np.float64)
        self.extprof_Te = np.empty(0,dtype=np.float64)
        self.extprof_Ti = np.empty(0,dtype=np.float64)

        # PLACEHOLDER. TODO: Add impurity profiles
        self.extprof_impfracs = np.empty(0,dtype=np.float64)

        # TODO: figure these out
        self.imcharges[0] = 2
        self.imcharges[1] = -1
        self.imcharges[2] = -1
        self.imcharges[3] = -1
        self.imcharges[4] = -1
        self.imcharges[5] = -1

        pass
    
    #-------------------------------------------------------------------
    # Setup functions
    #-------------------------------------------------------------------

    def _addextprof(self, extprofr, extprofvals):
        if not self.profsdefined:
            self.extprof = True
            self.profsdefined = True
            self.extprofr = extprofr
            self.extprofvals = extprofvals
        else:
            raise SyntaxError("Profile already defined. Create a new object.")
        
    def _define_volgrid(self, volgrid, sqrtpsin):
        self.volgrid = volgrid
        self.sqrtpsin = sqrtpsin
        
    def _set_alpha_and_offset(self, alpha1, alpha2, offset):
        self.extprof = False
        self.profsdefined = True
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.offset = offset

    #-------------------------------------------------------------------
    # Properties
    # TODO: Impurity profiles
    #-------------------------------------------------------------------

    # BS from Martin H Mode Scaling (Martin et al J. Phys 2008)
    # Update from kikuchi?
    @property
    def bs_factor(self):
        return self.B0**(0.8)*(2.*np.pi*self.R * 2*np.pi*self.a * np.sqrt((self.kappa**2+1)/2))**(0.94)
    
    @property
    def plasma_dilution(self):
        # n_e = n20 / dilution
        return 1/(1 + np.sum(self.impurityfractions*self.imcharges))

    # Z_eff, effective ion charge
    @property
    def Zeff(self):
        # Zeff = sum ( n_i * Z_i^2 ) / n_e
        # n_i = n20 * species fraction
        # n_e = n20 / dilution
        # n20 cancels

        # Hydrogen isotopes + impurities
        return ( (1-np.sum(self.impurityfractions)) + np.sum(self.impurityfractions*self.imcharges**2))*self.plasma_dilution

    # n_GR, Greenwald density in 10^20/m^3
    @property
    def n_GR(self):
        return self.Ip/(np.pi*self.a**2)
    
    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    
    # Maybe do two of these classes for the two different profile types
    # to skip this if-then-else?
    def get_prof_val(self, r, v0):
        if self.extprof:
            return v0*np.interp(r,self.extprofr,self.extprofvals)
        else:
            return (v0-self.offset)*(1-r**self.alpha1)**self.alpha2+self.offset
    
    def get_tauE(self, H, a_m, a_IP, a_R,
                 a_a, a_kappa, a_BT, a_P, a_n20, P, n20):
        
        # Calculate the energy confinement time
        # H = H factor
        # a_* = scaling law powers

        return H*self.M_i**(a_m)*self.Ip**(a_IP)*self.R**(a_R)\
            *self.a**(a_a)*self.kappa**(a_kappa)*self.B0**(a_BT)\
                *P**(a_P)*n20**(a_n20)

    # plasma volume
    # TODO: Add calculation from equilibrium
    def get_enclosed_vol(self, rho):
        # RHO IS NORMALIZED, TODO: CHANGE OTHER FUNCTIONS TO REFLECT THIS
        return 2.*np.pi**2*(rho*self.a)**2*self.R*self.kappa

    # volume of drho element, where rho=0 at magnetic axis and rho=1 at separatrix
    # TODO: Add calculation from equilibrium
    def get_dvolfac(self, rho):# -> Any:
                # RHO IS NORMALIZED, TODO: CHANGE OTHER FUNCTIONS TO REFLECT THIS

        return 4*np.pi**2*(rho*self.a)**2*self.R*self.kappa    

    # coefficient for Spitzer conductivity, necessary to obtain ohmic power
    def get_Cspitz(self, volavgcurr:bool):
        Fz    = (1+1.198*self.Zeff + 0.222*self.Zeff**2)/(1+2.966*self.Zeff + 0.753*self.Zeff**2)
        eta1  = 1.03e-4*self.Zeff*Fz
        j0avg = self.Ip/(np.pi*self.a**2*self.kappa)*1.0e6
        if (volavgcurr == True):
            Cspitz = eta1*self.q_a*j0avg**2
        else:
            Cspitz = eta1
        Cspitz /= 1.6e-16*1.0e20 #unit conversion to keV 10^20 m^-3
        return Cspitz

# NOT jit compiled
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

# jit compiled
@nb.experimental.jitclass
class POPCON_data:
    """
    Output data class for the POPCON.
    """
    def __init__(self) -> None:
        pass

# NOT jit compiled
class POPCON_plotsettings:
    def __init__(self,
                 filename: str,
                 ) -> None:
        pass

# NOT jit compiled
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

