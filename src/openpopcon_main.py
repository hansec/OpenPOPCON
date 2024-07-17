# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 2024

OpenPOPCON v2.0
This is the Columbia University fork of the OpenPOPCON project developed
for MIT course 22.63. This project is a refactor of the contributions
made to the original project in the development of MANTA. Contributors
to the original project are listed in the README.md file.
"""

# from re import S
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import scipy.constants as const
# import scipy.integrate as integrate
# import xarray as xr
import yaml
from typing import Callable
from openpopcon_util import *
import numba as nb
import ionization_and_radiation as Zeff_lib


# core of calculations
# jit compiled
@nb.experimental.jitclass(spec = [
            # ('R', nb.float64), 
            # ('a', nb.float64),
            # ('kappa', nb.float64),
            # ('B0', nb.float64),
            # ('Ip', nb.float64),
            # ('q_a', nb.float64),
            # ('H', nb.float64),
            # ('M_i', nb.float64),
            # ('f_LH', nb.float64),
            # ('volgrid', nb.float64[:]),
            # ('sqrtpsin', nb.float64[:]),
            # ('impurityfractions', nb.float64[:]),
            # ('imcharges', nb.float64[:]),
            # ('extprof_geoms', nb.boolean),
            # ('geomsdefined', nb.boolean),
            # ('extprof_imps', nb.boolean),
            # ('impsdefined', nb.boolean),
            # ('j_alpha1', nb.float64),
            # ('j_alpha2', nb.float64),
            # ('j_offset', nb.float64),
            # ('ne_alpha1', nb.float64),
            # ('ne_alpha2', nb.float64),
            # ('ne_offset', nb.float64),
            # ('ni_alpha1', nb.float64),
            # ('ni_alpha2', nb.float64),
            # ('ni_offset', nb.float64),
            # ('Ti_alpha1', nb.float64),
            # ('Ti_alpha2', nb.float64),
            # ('Ti_offset', nb.float64),
            # ('Te_alpha1', nb.float64),
            # ('Te_alpha2', nb.float64),
            # ('Te_offset', nb.float64),
            # ('extprofr', nb.float64[:]),
            # ('extprof_j', nb.float64[:]),
            # ('extprof_ne', nb.float64[:]),
            # ('extprof_ni', nb.float64[:]),
            # ('extprof_Te', nb.float64[:]),
            # ('extprof_Ti', nb.float64[:]),
            # ('extprof_impfracs', nb.float64[:]),
            # TODO: Refresh this
          ]) # type: ignore
class POPCON_params:
    """
    Physical parameters for the POPCON.
    """
    def __init__(self) -> None:
        self.R: float = 2.7
        self.a: float
        self.kappa: float
        self.B0: float
        self.Ip: float
        self.q_a: float
        self.H: float
        self.M_i: float
        self.f_LH: float
        self.nipeak_over_nepeak: float
        self.tipeak_over_tepeak: float


        self.sqrtpsin = np.empty(0, dtype=np.float64)
        self.volgrid = np.empty(0,dtype=np.float64)
        # 0 = He, 1 = Ne, 2 = Ar, 3 = Kr, 4 = Xe, 5 = W
        self.impurityfractions = np.empty(6, dtype=np.float64)
        self.imcharges = np.empty(6, dtype=np.float64)
        self.extprof_geoms: bool
        self.geomsdefined: bool = False
        self.rdefined: bool = False
        self.volgriddefined: bool = False

        self.extprof_imps: bool
        self.impsdefined: bool = False

        self._jdefined: bool = False
        self._nedefined: bool = False
        self._nidefined: bool = False
        self._Tidefined: bool = False
        self._Tedefined: bool = False
        self._qdefined: bool = False
        self._bmaxdefined: bool = False
        self._bavgdefined: bool = False

        self._jextprof: bool
        self._neextprof: bool
        self._niextprof: bool
        self._Tiextprof: bool
        self._Teextprof: bool

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

        self.extprofr   = np.empty(0,dtype=np.float64)
        self.extprof_j  = np.empty(0,dtype=np.float64)
        self.extprof_ne = np.empty(0,dtype=np.float64)
        self.extprof_ni = np.empty(0,dtype=np.float64)
        self.extprof_Te = np.empty(0,dtype=np.float64)
        self.extprof_Ti = np.empty(0,dtype=np.float64)
        self.extprof_q  = np.empty(0,dtype=np.float64)
        self.bmaxprof   = np.empty(0,dtype=np.float64)
        self.bavgprof   = np.empty(0,dtype=np.float64)

        # PLACEHOLDER. TODO: Add impurity profiles
        self.extprof_impfracs = np.empty(0,dtype=np.float64)

        # TODO: figure these out; sauter?
        self.imcharges[0] = 2
        self.imcharges[1] = -1
        self.imcharges[2] = -1
        self.imcharges[3] = -1
        self.imcharges[4] = -1
        self.imcharges[5] = -1

        pass

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

    # n_GR, Greenwald density in 10^20/m^3
    @property
    def n_GR(self):
        return self.Ip/(np.pi*self.a**2)
    
    #-------------------------------------------------------------------
    # Profiles and integration
    #-------------------------------------------------------------------

    # Maybe do two of these classes for the two different profile types
    # to skip this if-then-else?
    def get_extprof(self, rho, profid:int):
        """
        -2: Sqrt(psin) for geometry
        -1: Volume grid
        0: J profile
        1: n_e profile
        2: n_i profile
        3: Ti profile
        4: Te profile
        5: q profile
        6: bmax profile
        7: bavg profile
        """
        if profid == -2:
            if not self.rdefined:
                raise ValueError("Geometry profiles not defined.")
            return np.interp(rho, self.sqrtpsin, self.volgrid)
        elif profid == -1:
            if not self.volgriddefined:
                raise ValueError("Volume grid not defined.")
            return np.interp(rho, self.sqrtpsin, self.volgrid)
        elif profid == 0:
            if not self._jdefined:
                raise ValueError("J profile not defined.")
            return np.interp(rho, self.extprofr, self.extprof_j)
        elif profid == 1:
            if not self._nedefined:
                raise ValueError("n_e profile not defined.")
            return np.interp(rho, self.extprofr, self.extprof_ne)
        elif profid == 2:
            if not self._nidefined:
                raise ValueError("n_i profile not defined.")
            return np.interp(rho, self.extprofr, self.extprof_ni)
        elif profid == 3:
            if not self._Tidefined:
                raise ValueError("Ti profile not defined.")
            return np.interp(rho, self.extprofr, self.extprof_Ti)
        elif profid == 4:
            if not self._Tedefined:
                raise ValueError("Te profile not defined.")
            return np.interp(rho, self.extprofr, self.extprof_Te)
        elif profid == 5:
            if not self._qdefined:
                raise ValueError("q profile not defined.")
            return np.interp(rho, self.extprofr, self.extprof_q)
        elif profid == 6:
            if not self._bmaxdefined:
                raise ValueError("bmax profile not defined.")
            return np.interp(rho, self.extprofr, self.bmaxprof)
        elif profid == 7:
            if not self._bavgdefined:
                raise ValueError("bavg profile not defined.")
            return np.interp(rho, self.extprofr, self.bavgprof)
        else:
            raise ValueError("Invalid profile ID.")
    
    def volume_integral(self, rho, profile):
        # Integrates functions of rho dV-like
        V_interp = np.interp(rho, self.sqrtpsin, self.volgrid)
        return np.trapz(profile, V_interp)
    
    #-------------------------------------------------------------------
    # Physical Quantities
    #-------------------------------------------------------------------
    
    # Z_eff, effective ion charge
    def Zeff(self, T0:float):
        # Zeff = sum ( n_i * Z_i^2 ) / n_e
        # n_i = n20 * species fraction
        # n_e = n20 / dilution
        # n20 cancels

        # Hydrogen isotopes + impurities
        return ( (1-np.sum(self.impurityfractions)) + np.sum(self.impurityfractions*Zeff_lib.get_Zeffs(T0)**2))*self.plasma_dilution

    def get_tauE(self, H, a_m, a_IP, a_R,
                 a_a, a_kappa, a_BT, a_P, a_n20, P, n20):
        
        # Calculate the energy confinement time
        # H = H factor
        # a_* = scaling law powers

        return H*self.M_i**(a_m)*self.Ip**(a_IP)*self.R**(a_R)\
            *self.a**(a_a)*self.kappa**(a_kappa)*self.B0**(a_BT)\
                *P**(a_P)*n20**(a_n20)

    # # plasma volume
    # # TODO: Add calculation from equilibrium
    # def get_enclosed_vol(self, rho):
    #     # RHO IS NORMALIZED, TODO: CHANGE OTHER FUNCTIONS TO REFLECT THIS
    #     return 2.*np.pi**2*(rho*self.a)**2*self.R*self.kappa

    # # volume of drho element, where rho=0 at magnetic axis and rho=1 at separatrix
    # # TODO: Add calculation from equilibrium
    # def get_dvolfac(self, rho):# -> Any:
    #             # RHO IS NORMALIZED, TODO: CHANGE OTHER FUNCTIONS TO REFLECT THIS
    #     return 4*np.pi**2*(rho*self.a)**2*self.R*self.kappa    

    # coefficient for Spitzer conductivity, necessary to obtain ohmic power
    def get_Cspitz(self, volavgcurr:bool, T0):
        Fz    = (1+1.198*self.Zeff(T0) + 0.222*self.Zeff(T0)**2)/(1+2.966*self.Zeff(T0) + 0.753*self.Zeff(T0)**2)
        eta1  = 1.03e-4*self.Zeff(T0)*Fz
        j0avg = self.Ip/(np.pi*self.a**2*self.kappa)*1.0e6
        if (volavgcurr == True):
            # TODO: Change this to use volgrid for averaging
            Cspitz = eta1*self.q_a*j0avg**2
        else:
            Cspitz = eta1
        Cspitz /= 1.6e-16*1.0e20 #unit conversion to keV 10^20 m^-3
        return Cspitz
    
    def get_eta_NC(self, rho, T0, n20):
        # Calculate the neoclassical resistivity
        # Equations 16-17 from [1] Jardin et al. 1993

        T_e_r = T0*self.get_extprof(rho, 4)/self.tipeak_over_tepeak
        n_e_r = n20*self.get_extprof(rho, 1)/self.nipeak_over_nepeak
        q = self.get_extprof(rho, 5)
        Zeff = self.Zeff(T_e_r)
        Lambda = max(5, (T_e_r*1e3 / np.sqrt(n_e_r*1e20))*np.exp(17.1))
        eta_C = 1.03e-4 * np.log(Lambda) * T_e_r**(-3/2)

        Lambda_E = 3.4/Zeff * (1.13 + Zeff) / (2.67 + Zeff)
        C_R = 0.56/Zeff * (3.0 - Zeff) / (3.0 + Zeff)
        xi = 0.58 + 0.2*Zeff
        invaspect = self.a/self.R
        f_t = np.sqrt(2*rho*invaspect) # TODO: Replace with Jardin formula

        nu_star_e = 1/10.2e16 * self.R * q * n_e_r * Lambda / (f_t * invaspect * T_e_r**2)

        eta_C_eta_NC_ratio = Lambda_E * ( 1 - f_t/( 1 + xi * nu_star_e ) )*( 1 - C_R*f_t/( 1 + xi * nu_star_e ) )

        eta_NC = eta_C / eta_C_eta_NC_ratio
        return eta_NC
    
    #-------------------------------------------------------------------
    # Power profiles
    #-------------------------------------------------------------------

    def _W_tot_prof(self, rho, T0, n20):
        n_i_r = n20*self.get_extprof(rho, 2)*1e20
        T_i_r = T0*self.get_extprof(rho, 3)
        # W_density = 3/2 * n_i * T_i
        # = 3/2 * (1e20 * n_i_r) ( 1.60218e-22 * T_i_r (keV) ) (MJ/m^3)
        return 3/2 * 1.60218e-22 * n_i_r * T_i_r
    
    def _P_DD_prof(self, rho, T0, n20):
        n_i_r = n20*self.get_extprof(rho, 2)*1e20
        T_i_r = T0*self.get_extprof(rho, 3)
        # TODO: Complete
        return 
    
    #-------------------------------------------------------------------
    # Setup functions
    #-------------------------------------------------------------------

    def _addextprof(self, extprofvals, profid):
        if profid == -2:
            if self.rdefined:
                raise ValueError("Geometry profiles already defined.")
            self.sqrtpsin = extprofvals
            self.rdefined = True
        elif profid == -1:
            if self.volgriddefined:
                raise ValueError("Volume grid already defined.")
            self.volgrid = extprofvals
            self.volgriddefined = True
        elif profid == 0:
            if self._jdefined:
                raise ValueError("J profile already defined.")
            self.extprof_j = extprofvals
            self._jdefined = True
            self._jextprof = True
        elif profid == 1:
            if self._nedefined:
                raise ValueError("n_e profile already defined.")
            self.extprof_ne = extprofvals
            self._nedefined = True
            self._neextprof = True
        elif profid == 2:
            if self._nidefined:
                raise ValueError("n_i profile already defined.")
            self.extprof_ni = extprofvals
            self._nidefined = True
            self._niextprof = True
        elif profid == 3:
            if self._Tidefined:
                raise ValueError("Ti profile already defined.")
            self.extprof_Ti = extprofvals
            self._Tidefined = True
            self._Tiextprof = True
        elif profid == 4:
            if self._Tedefined:
                raise ValueError("Te profile already defined.")
            self.extprof_Te = extprofvals
            self._Tedefined = True
            self._Teextprof = True
        elif profid == 5:
            if self._qdefined:
                raise ValueError("q profile already defined.")
            self.extprof_q = extprofvals
            self._qdefined = True
        elif profid == 6:
            if self._bmaxdefined:
                raise ValueError("bmax profile already defined.")
            self.bmaxprof = extprofvals
            self._bmaxdefined = True
        elif profid == 7:
            if self._bavgdefined:
                raise ValueError("bavg profile already defined.")
            self.bavgprof = extprofvals
            self._bavgdefined = True
        else:
            raise ValueError("Invalid profile ID.")
        
    def _set_alpha_and_offset(self, alpha1, alpha2, offset, profid:int):
        if profid == 0:
            if self._jdefined:
                raise ValueError("J profile already defined.")
            self.j_alpha1 = alpha1
            self.j_alpha2 = alpha2
            self.j_offset = offset
            self._jdefined = True
        elif profid == 1:
            if self._nedefined:
                raise ValueError("n_e profile already defined.")
            self.ne_alpha1 = alpha1
            self.ne_alpha2 = alpha2
            self.ne_offset = offset
            self._nedefined = True
        elif profid == 2:
            if self._nidefined:
                raise ValueError("n_i profile already defined.")
            self.ni_alpha1 = alpha1
            self.ni_alpha2 = alpha2
            self.ni_offset = offset
            self._nidefined = True
        elif profid == 3:
            if self._Tidefined:
                raise ValueError("Ti profile already defined.")
            self.Ti_alpha1 = alpha1
            self.Ti_alpha2 = alpha2
            self.Ti_offset = offset
            self._Tidefined = True
        elif profid == 4:
            if self._Tedefined:
                raise ValueError("Te profile already defined.")
            self.Te_alpha1 = alpha1
            self.Te_alpha2 = alpha2
            self.Te_offset = offset
            self._Tedefined = True
        else:
            raise ValueError("Invalid profile ID.")
    
    def _addextprof_imps(self, extprofvals):
        # TODO: implement this
        pass

    def _setup_profs(self):
        # Populate all missing regions
        # TODO: implement this
        pass

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

if __name__ == '__main__':
    print('Hello, World!')
    pass