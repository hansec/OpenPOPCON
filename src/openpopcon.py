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

"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import json
from .lib.openpopcon_util import *
import numba as nb
from .lib import phys_lib as phys
import shutil
import datetime
from .lib import neat_json_encoder as nj

DEFAULT_PLOTSETTINGS = get_POPCON_homedir(['resources','default_plotsettings.yml'])
DEFAULT_SCALINGLAWS = get_POPCON_homedir(['resources','scalinglaws.yml'])

# core of calculations
# jit compiled
@nb.experimental.jitclass(spec = [
            ('iota_23', nb.float64),  # if you plan to pass in iota(2/3)
            ('iota_alpha', nb.float64),
            ('R', nb.float64), 
            ('a', nb.float64),
            ('kappa', nb.float64),
            ('delta', nb.float64),
            ('B0', nb.float64),
            ('Ip', nb.float64),
            ('Itot', nb.float64),
            ('H', nb.float64),
            ('M_i', nb.float64),
            ('tipeak_over_tepeak', nb.float64),
            ('fuel', nb.int64),
            ('sqrtpsin', nb.float64[:]),
            ('volgrid', nb.float64[:]),
            ('agrid', nb.float64[:]),
            ('nr', nb.int64),
            ('impurityfractions', nb.float64[:]),
            ('geomsdefined', nb.boolean),
            ('rdefined', nb.boolean),
            ('device_type', nb.int16),  # 0 = tokamak, 1 = stellarator
            ('volgriddefined', nb.boolean),
            ('agriddefined', nb.boolean),
            ('extprof_imps', nb.boolean),
            ('impsdefined', nb.boolean),
            ('_jdefined', nb.boolean),
            ('_nedefined', nb.boolean),
            ('_nidefined', nb.boolean),
            ('_Tidefined', nb.boolean),
            ('_Tedefined', nb.boolean),
            ('_qdefined', nb.boolean),
            ('_ftrappeddefined', nb.boolean),
            ('_extradefined', nb.boolean),
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
            ('j_prof', nb.float64[:]),
            ('ne_prof', nb.float64[:]),
            ('ni_prof', nb.float64[:]),
            ('Te_prof', nb.float64[:]),
            ('Ti_prof', nb.float64[:]),
            ('q_prof', nb.float64[:]),
            ('ftrapped_prof', nb.float64[:]),
            ('extraprof', nb.float64[:]),
            ('extprof_impfracs', nb.float64[:]),
            ('H_fac', nb.float64),
            ('scaling_const', nb.float64),
            ('M_i_alpha', nb.float64),
            ('Ip_alpha', nb.float64),
            ('R_alpha', nb.float64),
            ('a_alpha', nb.float64),
            ('kappa_alpha', nb.float64),
            ('B0_alpha', nb.float64),
            ('Pheat_alpha', nb.float64),
            ('n20_alpha', nb.float64),
            ('resistivity_alg', nb.int16),
            ('verbosity', nb.int64),
          ]) # type: ignore
class POPCON_algorithms:
    """
    Class POPCON_algorithms

    This class is the mathematical backbone of the OpenPOPCON code. It
    is compiled with numba to provide fast calculations for scans. It is
    best used in Jupyter notebooks or scripts where the user wants to
    conduct a scan, as it is compiled at runtime; this means that the
    first time the code executes, it will take a bit longer to compile,
    but the subsequent runs will be much faster than raw Python code.

    See __init__ for a list of parameters and their descriptions.

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

        #---------------------------------------------------------------
        # Machine parameters
        #---------------------------------------------------------------

        self.R: float                      # Major radius in meters
        self.a: float                      # Minor radius in meters
        self.kappa: float                  # Elongation
        self.delta: float                  # Triangularity
        self.B0: float                     # Magnetic field at axis in Tesla
        self.Ip: float                     # Plasma current in MA
        self.Itot: float                   # Total current in MA
        self.M_i: float                    # Ion mass in amu
        self.tipeak_over_tepeak: float     # Ti/Te peak ratio
        #*** 1 = D-D, 2 = D-T, 3 = D-He3 ***
        self.fuel: int                     # Fuel cycle


        #---------------------------------------------------------------
        # Geometry/profile parameters
        #---------------------------------------------------------------
        self.sqrtpsin = np.empty(0, dtype=np.float64)           # sqrt(psin) corresponding to profiles
        self.volgrid = np.empty(0,dtype=np.float64)             # Flux surface volumes
        self.agrid = np.empty(0,dtype=np.float64)               # Flux surface surface areas
        self.nr: int                                            # Number of radial points
        # 0 = He, 1 = Ne, 2 = Ar, 3 = Kr, 4 = Xe, 5 = W
        self.impurityfractions = np.empty(6, dtype=np.float64)  # Impurity fractions for each impurity, relative to ion density
        self.geomsdefined: bool = False                         # Whether geometry profiles are defined
        self.rdefined: bool = False                             # Whether radial grid is defined
        self.volgriddefined: bool = False                       # Whether volume grid is defined
        self.agriddefined: bool = False                         # Whether area grid is defined

        # Impurity profiles are not implemented yet
        self.extprof_imps: bool = False                         # Whether impurity profiles are defined (not implemented)
        self.impsdefined: bool = False                          # Whether impurity fractions are defined

        self._jdefined: bool = False                            # Whether J profile is defined
        self._nedefined: bool = False                           # Whether ne profile is defined
        self._nidefined: bool = False                           # Whether ni profile is defined
        self._Tidefined: bool = False                           # Whether Ti profile is defined
        self._Tedefined: bool = False                           # Whether Te profile is defined
        self._qdefined: bool = False                            # Whether q profile is defined
        self._ftrappeddefined: bool = False                         # Whether b_max profile is defined (not implemented)
        self._extradefined: bool = False                         # Whether b_avg profile is defined (not implemented)

        #---------------------------------------------------------------
        # Parameters for parabolic profiles case
        #---------------------------------------------------------------
        """
        Normalized parabolic profiles (or polynomial-like) are defined:
        f(rho) = (1 - offset) * (1 + rho^alpha1)^alpha2 + offset
        """

        self.iota_23: float = 1.0

        self.j_alpha1: float = 2.
        self.j_alpha2: float = 3.
        self.j_offset: float = 0.

        self.ne_alpha1: float = 2.
        self.ne_alpha2: float = 1.5
        self.ne_offset: float = 0.3

        self.ni_alpha1: float = 2.
        self.ni_alpha2: float = 1.5
        self.ni_offset: float = 0.3

        self.Ti_alpha1: float = 2.
        self.Ti_alpha2: float = 1.5
        self.Ti_offset: float = 0.

        self.Te_alpha1: float = 2.
        self.Te_alpha2: float = 1.5
        self.Te_offset: float = 0.

        #---------------------------------------------------------------
        # Profiles
        #---------------------------------------------------------------
        """
        All profiles are defined as arrays of the same length as 
        sqrtpsin, and **normalized**! 
        """
        self.j_prof  = np.empty(0,dtype=np.float64)             # Current density profile
        self.ne_prof = np.empty(0,dtype=np.float64)             # Electron density profile
        self.ni_prof = np.empty(0,dtype=np.float64)             # Ion density profile
        self.Te_prof = np.empty(0,dtype=np.float64)             # Electron temperature profile
        self.Ti_prof = np.empty(0,dtype=np.float64)             # Ion temperature profile
        self.q_prof  = np.empty(0,dtype=np.float64)             # Safety factor profile
        self.ftrapped_prof   = np.empty(0,dtype=np.float64)          # Maximum B field on flux surface (not implemented)
        self.extraprof   = np.empty(0,dtype=np.float64)          # Average B field on flux surface (not implemented)
        self.extprof_impfracs = np.empty(0,dtype=np.float64)    # Impurity fractions profile (not implemented)

        #---------------------------------------------------------------
        # Scaling law parameters
        #---------------------------------------------------------------

        self.H_fac: float = 1.0                                 # H (scaling law enhancement) factor
        self.scaling_const: float = 0.145                       # Scaling law coefficient
        self.M_i_alpha: float = 0.19                            # Ion mass scaling law exponent
        self.Ip_alpha: float = 0.93                             # Plasma current scaling law exponent 
        self.R_alpha: float = 1.39                              # Major radius scaling law exponent
        self.a_alpha: float = 0.58                              # Minor radius scaling law exponent
        self.kappa_alpha: float = 0.78                          # Elongation scaling law exponent
        self.B0_alpha: float = 0.15                             # Toroidal field scaling law exponent
        self.Pheat_alpha: float = -0.69                         # Power degradation scaling law exponent
        self.n20_alpha: float = 0.41                            # Greenwald density scaling law exponent
        self.iota_alpha: float = 0.41                           # Iota scaling law exponent (stellarator only)
        
        #---------------------------------------------------------------
        # Etc
        #---------------------------------------------------------------
        self.resistivity_alg: int = 0                           # Resistivity algorithm. 0 = Jardin, 1 = Paz-Soldan, 2 = local maximum
        self.verbosity: int = 0                                 # Verbosity level. 0 = silent, 1 = normal, 2 = debug, 3 = print all matrices

        pass

    #-------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------

    @property
    def n_GR(self):
        """
        Greenwald density in 10^20/m^3
        """
        return self.Ip/(np.pi*self.a**2)
    
    @property
    def V(self):
        """
        Plasma volume (in last closed flux surface) in m^3
        """
        return self.volume_integral(self.sqrtpsin, np.ones_like(self.sqrtpsin))
    
    @property
    def A(self):
        """
        Plasma surface area (in last closed flux surface) in m^2
        """
        return self.agrid[-1]
    #-------------------------------------------------------------------
    # Profiles and integration
    #-------------------------------------------------------------------

    def get_profile(self, rho:npt.NDArray[np.float64], profid:int) -> npt.NDArray[np.float64]:
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
            if not self.agriddefined:
                raise ValueError("Area grid not defined.")
            return np.interp(rho, self.sqrtpsin, self.agrid)
        elif profid == -2:
            if not self.rdefined:
                raise ValueError("Geometry profiles not defined.")
            return np.interp(rho, self.sqrtpsin, self.sqrtpsin)
        elif profid == -1:
            if not self.volgriddefined:
                raise ValueError("Volume grid not defined.")
            return np.interp(rho, self.sqrtpsin, self.volgrid)
        elif profid == 0:
            if not self._jdefined:
                raise ValueError("J profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.j_prof)
        elif profid == 1:
            if not self._nedefined:
                raise ValueError("n_e profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.ne_prof)
        elif profid == 2:
            if not self._nidefined:
                raise ValueError("n_i profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.ni_prof)
        elif profid == 3:
            if not self._Tidefined:
                raise ValueError("Ti profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.Ti_prof)
        elif profid == 4:
            if not self._Tedefined:
                raise ValueError("Te profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.Te_prof)
        elif profid == 5:
            if not self._qdefined:
                raise ValueError("q profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.q_prof)
        elif profid == 6:
            if not self._ftrappeddefined:
                raise ValueError("ftrapped profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.ftrapped_prof)
        elif profid == 7:
            if not self._extradefined:
                raise ValueError("extra profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extraprof)
        else:
            raise ValueError("Invalid profile ID.")
    
    def volume_integral(self, rho, func) -> float:
        r"""
        Integrates a function of rho dV-like:

        $\int_0^1 func(\rho) \frac{dV}{d\rho} d\rho$
        """
        # NOTE: profile must be an array of the same length as rho.
        # Integrates functions of rho dV-like
        V_interp = np.interp(rho, self.sqrtpsin, self.volgrid)
        return np.trapz(func, V_interp)
    
    #-------------------------------------------------------------------
    # Physical Quantities
    #-------------------------------------------------------------------
    
    def Zeff(self, T_e_keV) -> float:
        """
        Effective ion charge.

        Zeff = sum ( n_i * Z_i^2 ) / n_e
        """
        # n_i = n20 * species fraction
        # n_e = n20 / dilution
        # n20 cancels

        # Hydrogen isotopes + impurities
        return ( (1-np.sum(self.impurityfractions)) + np.sum(self.impurityfractions*phys.get_Zeffs(T_e_keV)**2))*self.plasma_dilution(T_e_keV)
    
    def plasma_dilution(self, T_e_keV) -> float:
        """
        Plasma dilution factor; number of ions per electron. Always <=1.
        """
        dil = 1/(1 + np.sum(self.impurityfractions*phys.get_Zeffs(T_e_keV)))
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
    
    def eta_NC(self, rho, T_e_keV:float, n_e_20:float):
        """
        Neoclassical resistivity in Ohm-m.

        Equations 16-17 from [1] Jardin et al. 1993, or equation 6 from [8] Paz-Soldan et al. 2016
        """

        if np.any(rho <= 0):
            raise ValueError("Invalid rho value. Neoclassical resistivity not defined at rho=0.")
        T_e_r = T_e_keV*self.get_profile(rho, 4)
        n_e_r = 1e20*n_e_20*self.get_profile(rho, 1)
        q = self.get_profile(rho, 5)
        Zeffprof = np.empty_like(T_e_r)
        for i in np.arange(T_e_r.shape[0]):
            Zeffprof[i] = self.Zeff(T_e_r[i])
        # logLambda = 17.1-np.log(np.sqrt(n_e_r)/(T_e_r*1e3)) # From Jardin
        logLambda = np.empty_like(Zeffprof)
        where = T_e_r*1e3>10*Zeffprof**2
        elsewhere = np.logical_not(where)
        logLambda[where] = 24 - np.log(np.sqrt(n_e_r)/(T_e_r*1e3))[where] # Plasma Formulary
        logLambda[elsewhere] = 23 - np.log(np.sqrt(n_e_r)*Zeffprof/(T_e_r*1e3)**(3/2))[elsewhere] # Plasma Formulary
        logLambda[logLambda<5] = 5
        eta_C = 1.03e-4 * logLambda * (T_e_r*1e3)**(-3/2)

        Lambda_E = 3.4/Zeffprof * (1.13 + Zeffprof) / (2.67 + Zeffprof)
        C_R = 0.56/Zeffprof * (3.0 - Zeffprof) / (3.0 + Zeffprof)
        xi = 0.58 + 0.2*Zeffprof
        invaspect = self.a/self.R
    
        if not self._ftrappeddefined:
            f_t = np.sqrt(2*rho*invaspect)
            nu_star_e = 1/10.2e16 * self.R * q * n_e_r * np.exp(logLambda) / (f_t * invaspect * (T_e_r*1e3)**2)
            eta_C_eta_NC_ratio = Lambda_E * ( 1 - f_t/( 1 + xi * nu_star_e ) )*( 1 - C_R*f_t/( 1 + xi * nu_star_e ) )
        else:
            f_t_prof = self.get_profile(rho, 6)
            nu_star_e = 1/10.2e16 * self.R * q * n_e_r * np.exp(logLambda) / (f_t_prof * invaspect * (T_e_r*1e3)**2)
            eta_C_eta_NC_ratio = Lambda_E * ( 1 - f_t_prof/( 1 + xi * nu_star_e ) )*( 1 - C_R*f_t_prof/( 1 + xi * nu_star_e ) )


        eta_NC = eta_C / eta_C_eta_NC_ratio

        if self.resistivity_alg == 0:
            return eta_NC

        f_CL = (1+1.2*Zeffprof + 0.22*Zeffprof**2)/(1+3*Zeffprof + 0.75*Zeffprof**2)
        
        f_NC = 1/ ( 1- np.sqrt( 2*invaspect / (1 + invaspect) ) )

        eta_NC_2 = eta_C * f_CL * f_NC

        if self.resistivity_alg == 1:
            return eta_NC_2

        if self.resistivity_alg == 2:
            eta_NC_max = np.empty_like(eta_NC)

            for i in range(len(eta_NC)):
                eta_NC_max[i] = max(eta_NC[i], eta_NC_2[i])

            return eta_NC_max
        
        else:
            raise ValueError("Invalid resistivity algorithm. Must be 0, 1, or 2.")

    
    def Vloop(self, T_e_keV, n_e_20) -> float:
        """
        Loop Voltage in Volts.

        V_loop = P_ohmic / Ip
        """
        P_OH = self.volume_integral(self.sqrtpsin, self._P_OH_prof(self.sqrtpsin, T_e_keV, n_e_20))
        return P_OH/self.Ip

        
    
    def BetaN(self, T_i_keV, n_e_20) -> float:
        """
        Normalized beta, beta*a*B0/Ip.

        beta = 2 mu0 <P> / (B^2)
        <P> = int P dV / V (average pressure)
        beta_N = beta a B0 / (Ip)
        """
        P_avg = 1e6*self.volume_integral(self.sqrtpsin, self._W_tot_prof(self.sqrtpsin, T_i_keV, n_e_20))/self.V
        beta =  2*(4e-7*np.pi)*P_avg/(self.B0**2)
        return beta*self.a*self.B0/self.Ip
    
    def tauE_scalinglaw(self, Pheat, n_e_20) -> float:
        """
        User-chosen confinement time scaling law
        """
        tauE = self.H_fac * self.scaling_const
        if self.device_type == 1: 
            tauE *= self.iota_23**self.iota_alpha
        else:
            tauE *= self.M_i**self.M_i_alpha
            tauE *= self.Ip**self.Ip_alpha
            tauE *= self.kappa**self.kappa_alpha
        tauE *= self.R**self.R_alpha
        tauE *= self.a**self.a_alpha
        tauE *= self.B0**self.B0_alpha
        tauE *= Pheat**self.Pheat_alpha
        tauE *= n_e_20**self.n20_alpha
        return tauE
    
    def tauE_H98(self, Pheat, n_e_20) -> float:
        """
        H98y2 scaling law for comparison
        """
        tauE = 0.145
        #M_i**(0.19)*Ip**(0.93)*R**(1.39)*a**(0.58)*kappa**(0.78)*B0**(0.15)*Pheat**(-0.69)*n20**(0.41)
        tauE *= self.M_i**0.19
        tauE *= self.Ip**0.93
        tauE *= self.R**1.39
        tauE *= self.a**0.58
        tauE *= self.kappa**0.78
        tauE *= self.B0**0.15
        tauE *= Pheat**(-0.69)
        tauE *= n_e_20**0.41
        return tauE
    
    def tauE_H89(self, Pheat, n_e_20) -> float:
        """
        H89 scaling law for comparison
        """
        #0.048*M_i**(0.5)*Ip**(0.85)*R**(1.2)*a**(0.3)*kappa**(0.5)*B0**(0.2)*Pheat**(-0.5)*n20**(0.1)
        tauE = 0.048
        tauE *= self.M_i**0.5
        tauE *= self.Ip**0.85
        tauE *= self.R**1.2
        tauE *= self.a**0.3
        tauE *= self.kappa**0.5
        tauE *= self.B0**0.2
        tauE *= Pheat**(-0.5)
        tauE *= n_e_20**0.1
        return tauE
    
    

    #-------------------------------------------------------------------
    # Power profiles
    #-------------------------------------------------------------------

    def _W_tot_prof(self, rho:npt.NDArray[np.float64], T_i_keV:float, n_e_20:float):
        """
        Plasma energy per cubic meter; also pressure.
        """
        n_e_r = 1e20*n_e_20*self.get_profile(rho, 2)
        T_e_r = T_i_keV*self.get_profile(rho, 4)/self.tipeak_over_tepeak
        dil = self.plasma_dilution(T_i_keV/self.tipeak_over_tepeak)
        n_i_r = 1e20*n_e_20*self.get_profile(rho, 2)*dil
        T_i_r = T_i_keV*self.get_profile(rho, 3)
        # W_density = 3/2 * n_i * T_i
        # = 3/2 * (n_i_r) ( 1.60218e-22 * T_i_r (keV) ) (MJ/m^3)
        return 3/2 * 1.60218e-22 * (n_i_r * T_i_r + n_e_r * T_e_r)
    
    def _P_DDpT_prof(self, rho:npt.NDArray[np.float64], T_i_keV:float, n_i_20:float):
        """
        D(d,p)T power per cubic meter
        """
        if self.fuel == 1:
            dfrac = 1
        elif self.fuel == 2:
            dfrac = 0.5
        else:
            raise ValueError("Invalid fuel cycle.")
        
        n_i_r = 1e20*n_i_20*self.get_profile(rho, 2)
        T_i_r = T_i_keV*self.get_profile(rho, 3)

        # reaction frequency f = n_D/sqrt(2) * n_D/sqrt(2) * <sigma v> (1/s)
        # 1.60218e-22 is the conversion factor from keV to MJ
        # <sigma v> is in cm^3/s so we need to convert to m^3/s, hence 1e-6
        # P_DD_density = 1.60218e-22 * f(DD->pT) * (T_tritium + T_proton)  (MW/m^3)

        f_ddpt = np.power( ( dfrac*n_i_r / (np.sqrt(2)) ), 2) * phys.get_reactivity(T_i_r,2) * 1e-6
        return 1.60218e-22 * f_ddpt * (1.01e3 + 3.02e3)
    
    def _P_DDnHe3_prof(self, rho:npt.NDArray[np.float64], T_i_keV:float, n_i_20:float):
        """
        D(d,n)He3 power per cubic meter
        """
        if self.fuel == 1:
            dfrac = 1-np.sum(self.impurityfractions)
        elif self.fuel == 2:
            dfrac = 0.5-np.sum(self.impurityfractions)/2
        else:
            raise ValueError("Invalid fuel cycle.")
        
        n_i_r = 1e20*n_i_20*self.get_profile(rho, 2)
        T_i_r = T_i_keV*self.get_profile(rho, 3)

        # See _P_DDpT_prof for explanation.
        # Note, for heating, only the He3 heats the plasma. For heating purposes,
        # multiply by 0.82e3/(2.45e3 + 0.82e3).

        f_ddnhe3 = np.power( (  dfrac*n_i_r / (np.sqrt(2)) ), 2) * phys.get_reactivity(T_i_r,3) * 1e-6
        return 1.60218e-22 * f_ddnhe3 * (2.45e3 + 0.82e3)
    
    def _P_DTnHe4_prof(self, rho:npt.NDArray[np.float64], T_i_keV:float, n_i_20:float):
        """
        T(d,n)He4 power density in MW/m^3
        """
        if self.fuel == 1:
            dfrac = 1-np.sum(self.impurityfractions)
            tfrac = 0
        elif self.fuel == 2:
            dfrac = 0.5-np.sum(self.impurityfractions)/2
            tfrac = 0.5-np.sum(self.impurityfractions)/2
        else:
            raise ValueError("Invalid fuel cycle.")
        
        n_i_r = 1e20*n_i_20*self.get_profile(rho, 2)
        T_i_r = T_i_keV*self.get_profile(rho, 3)

        # See _P_DDpT_prof for explanation.
        # Note, for heating, only the He4 / alpha heats the plasma. For heating purposes,
        # multiply by 3.52e3/(3.52e3 + 14.06e3).

        f_dtnhe4 = dfrac*(n_i_r) * tfrac*(n_i_r) * phys.get_reactivity(T_i_r,1) * 1e-6
        return 1.60218e-22 * f_dtnhe4 * (3.52e3 + 14.06e3)
    
    def _P_fusion_heating(self, rho:npt.NDArray[np.float64], T_i_keV:float, n_i_20:float):
        """
        D-D and D-T heating power density in MW/m^3
        """
        return  self._P_DDpT_prof(rho,T_i_keV,n_i_20) + \
              self._P_DDnHe3_prof(rho,T_i_keV,n_i_20)*(0.82e3/(2.45e3+0.82e3))+\
              self._P_DTnHe4_prof(rho,T_i_keV,n_i_20)*(3.52e3/(3.52e3+14.06e3))
    
    def _P_fusion(self, rho:npt.NDArray[np.float64], T_i_keV:float, n_i_20:float):
        """
        Fusion power density in MW/m^3
        """
        return self._P_DDpT_prof(rho, T_i_keV, n_i_20) + \
                self._P_DDnHe3_prof(rho,T_i_keV,n_i_20)+\
                self._P_DTnHe4_prof(rho,T_i_keV,n_i_20)
    
    def _P_brem_rad(self, rho:npt.NDArray[np.float64], T_e_keV:float, n_e_20:float):
        """
        Radiative power density in MW/m^3; See formulary.
        """

        T_e_r = T_e_keV*self.get_profile(rho, 4)
        n_e_r = 1e20*n_e_20*self.get_profile(rho, 1)

        total_Zeff = np.empty(T_e_r.shape[0],dtype=np.float64)
        for i in np.arange(T_e_r.shape[0]):
            total_Zeff[i] = self.Zeff(T_e_r[i])
        G = 1.1 # Gaunt factor
        # P_brem = 5.35e-3*1e-6*G*total_Zeff*(1e-20*n_e_r)**2*T_e_r**0.5
        # From Plasma Formulary
        P_brem = G*1e-6*np.sqrt(1000*T_e_r)*total_Zeff*(n_e_r/7.69e18)**2
        return P_brem
    
    def _P_synch(self, rho:npt.NDArray[np.float64], T_e_keV:float, n_e_20:float):
        """
        Synchrotron radiation power density in MW/m^3; see Zohm 2019.
        """
        T_e_r = T_e_keV*self.get_profile(rho, 4)
        n_e_r = n_e_20*self.get_profile(rho, 1)
        P_synch = 1.32e-7*(self.B0*T_e_r)**2.5 * np.sqrt(n_e_r/self.a) * (1 + 18*self.a/(self.R*np.sqrt(T_e_r)))

        return P_synch
    
    def _P_impurity_rad(self, rho:npt.NDArray[np.float64], T_e_keV:float, n_e_20:float):
        """
        Radiative power density from impurities in MW/m^3; see Zohm 2019.
        """
        T_e_r = T_e_keV*self.get_profile(rho, 4)
        n_e_r = 1e20*n_e_20*self.get_profile(rho, 1)
        T_e_r[T_e_r < 0.05] = 0.05
        Lz = np.empty((T_e_r.shape[0],6),dtype=np.float64)
        for i in np.arange(T_e_r.shape[0]):
            Lz[i,:] = phys.get_rads(T_e_r[i])
        
        P_line = np.sum(1e-6*(Lz.T*(n_e_r)**2).T*self.impurityfractions,axis=1)
        return P_line
    
    def _P_rad(self, rho:npt.NDArray[np.float64], T_e_keV:float, n_e_20:float):
        """
        Total radiative power density in MW/m^3.
        """
        return self._P_brem_rad(rho, T_e_keV, n_e_20) + self._P_impurity_rad(rho, T_e_keV, n_e_20) + self._P_synch(rho, T_e_keV, n_e_20)
    
    def _P_OH_prof(self, rho:npt.NDArray[np.float64], T_e_keV:float, n_e_20:float) -> npt.NDArray[np.float64]:
        """
        Ohmic power density in MW/m^3
        """

        eta_NC = self.eta_NC(rho, T_e_keV, n_e_20)
        J = self.Itot*1e6*self.get_profile(rho, 0)
        return 1e-6*eta_NC*J**2

    def Q_fusion(self, T_i_keV:float, n_e_20:float, Paux:float) -> float:
        """
        Physical fusion gain factor.
        """
        T_e_keV = T_i_keV/self.tipeak_over_tepeak
        dil = self.plasma_dilution(T_e_keV)
        n_i_20 = n_e_20*dil
        P_fusion = self.volume_integral(self.sqrtpsin, self._P_fusion(self.sqrtpsin, T_i_keV, n_i_20))
        P_OH = self.volume_integral(self.sqrtpsin, self._P_OH_prof(self.sqrtpsin, T_e_keV, n_e_20))
        if Paux == 0:
            return 9999999.
        else:
            return P_fusion/(Paux + P_OH)
    
    #-------------------------------------------------------------------
    # Power Balance Relaxation Solvers
    #-------------------------------------------------------------------

    def P_aux_relax_impfrac(self, n_e_20, T_i_keV, accel=1., err=1e-5, max_iters=1000):
        """
        Relaxation solver for holding impfrac constant.
        """
        
        P_aux_iter = 100. # Don't want to undershoot as we can end up with P~0 and this can create a NaN! So we choose a high starting point.
        T_e_keV = T_i_keV/self.tipeak_over_tepeak
        dil = self.plasma_dilution(T_e_keV)
        n_i_20 = n_e_20*dil
        line_average_fac = np.average(self.get_profile(self.sqrtpsin, 1))
        dPaux: float
        P_fusion_heating_iter = self.volume_integral(self.sqrtpsin, self._P_fusion_heating(self.sqrtpsin, T_i_keV, n_i_20))
        P_ohmic_heating_iter = self.volume_integral(self.sqrtpsin, self._P_OH_prof(self.sqrtpsin, T_e_keV, n_e_20))
        W_tot_iter = self.volume_integral(self.sqrtpsin, self._W_tot_prof(self.sqrtpsin, T_i_keV, n_e_20))
        P_brem_iter = self.volume_integral(self.sqrtpsin, self._P_brem_rad(self.sqrtpsin, T_e_keV, n_e_20))
        P_synch_iter = self.volume_integral(self.sqrtpsin, self._P_synch(self.sqrtpsin, T_e_keV, n_e_20))
        P_imp_iter = self.volume_integral(self.sqrtpsin, self._P_impurity_rad(self.sqrtpsin, T_e_keV, n_e_20))
        for ii in np.arange(max_iters):
            # Power in
            P_totalheating_iter = P_aux_iter + P_ohmic_heating_iter + P_fusion_heating_iter - P_brem_iter - P_synch_iter

            # Power out
            tauE_iter = self.tauE_scalinglaw(P_totalheating_iter, n_e_20*line_average_fac)
            P_confinement_loss_iter = W_tot_iter/tauE_iter

            # Power balance
            dPaux = P_confinement_loss_iter - P_totalheating_iter

            # Relaxation

            # Prevent negative aux power
            if -dPaux*accel > P_aux_iter:
                P_aux_iter -= 0.9*P_aux_iter
            else:
                P_aux_iter += accel*dPaux

            # Prevent negative total heating power
            if P_aux_iter < -(P_ohmic_heating_iter + P_fusion_heating_iter - P_brem_iter):
                P_aux_iter = P_brem_iter

            # Check for convergence
            if P_aux_iter < 0.001 and dPaux <= 0:
                break
            elif P_aux_iter < 1.0:
                if np.abs(dPaux/1.0) < err:
                    break
            else:
                if np.abs(dPaux/P_aux_iter) < err:
                    break
            
            if ii == max_iters-1:
                if self.verbosity > 0:
                    print(f"Power balance relaxation solver found",P_aux_iter,"but did not converge in {max_iters} iterations in state n20=", n_e_20, "T=" , T_i_keV , "keV.")
        
        # Warn for P_SOL < 0: This is when the impurity radiation power exceeds the total W_tot/tauE.
        # Impurity power is assumed to displace what would otherwise be conductive, turbulent, or instability-driven losses.
        # This means that it radiates power away that would otherwise cross the separatrix and strike the divertor.
        if P_imp_iter > P_confinement_loss_iter:
            if self.verbosity > 0:
                print(f"Warning: Impurity radiation exceeds confinement loss. State n20=", n_e_20, "T=" , T_i_keV , "is not physical.")
            P_aux_iter = 99999.
        
        if np.isnan(P_aux_iter):
            if self.verbosity > 0:
                print("NaN detected in P_aux_relax_impfrac. State n20=", n_e_20, "T=" , T_i_keV , "keV.")
            P_aux_iter = 99999.

        return P_aux_iter
    #-------------------------------------------------------------------
    # Setup functions
    #-------------------------------------------------------------------

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
                raise ValueError("extra profile already defined." + \
                                 " This is unused.")
            self.extraprof = extprofvals
            self._extradefined = True
        else:
            raise ValueError("Invalid profile ID.")
        
    def _set_alpha_and_offset(self, alpha1, alpha2, offset, profid:int):
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
            w07 = 1.
            Lp = 2*np.pi*self.a*(1+0.55*(self.kappa - 1))*(1+0.08*self.delta**2)*(1+0.2*(w07-1))
            epsilon = rho*self.a/self.R
            self.agrid = 2*np.pi*self.R*(1-0.32*self.delta*epsilon)*Lp # Sauter eqs 36
        if not self.volgriddefined:
            epsilon = rho*self.a/self.R
            w07 = 1.0
            S_phi = np.pi*((rho*self.a)**2)*self.kappa*(1+0.52*(w07-1))
            self.volgrid = 2*np.pi*self.R*(1-0.25*self.delta*epsilon)*S_phi # Sauter eqs 36
            self.volgriddefined = True
        if not self._jdefined:
            self.j_prof = (1-self.j_offset)*(1-rho**self.j_alpha1)**self.j_alpha2 + self.j_offset
            self._jdefined = True
        if not self._nedefined:
            self.ne_prof = (1-self.ne_offset)*(1-rho**self.ne_alpha1)**self.ne_alpha2 + self.ne_offset
            self._nedefined = True
        if not self._nidefined:
            self.ni_prof = (1-self.ni_offset)*(1-rho**self.ni_alpha1)**self.ni_alpha2 + self.ni_offset
            self._nidefined = True
        if not self._Tidefined:
            self.Ti_prof = (1-self.Ti_offset)*(1-rho**self.Ti_alpha1)**self.Ti_alpha2 + self.Ti_offset
            self._Tidefined = True
        if not self._Tedefined:
            self.Te_prof = (1-self.Te_offset)*(1-rho**self.Te_alpha1)**self.Te_alpha2 + self.Te_offset
            self._Tedefined = True
        if not self._qdefined:
            # self.q_a = 2*np.pi*self.a**2*self.B0*(self.kappa**2+1)/(2*self.R*(4e-7*np.pi)*self.Ip*1e6)
            self.q_prof = 2*np.pi*(self.a*rho)**2*self.B0*(self.kappa**2+1)/(2*self.R*(4e-7*np.pi)*self.Ip*1e6)
            self._qdefined = True
        if not self._ftrappeddefined:
            self.ftrapped_prof = np.sqrt((self.B0*self.R/(self.R-rho*self.a))**2 + ((4e-7*np.pi)*self.Ip*1e6/(2*np.pi*self.a*rho))**2)
            self._ftrappeddefined = True
        if not self._extradefined:
            self.extraprof = np.sqrt((self.B0)**2 + ((4e-7*np.pi)*self.Ip*1e6/(2*np.pi*self.a*rho))**2)
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
    def __init__(self,
                 filename: str,
                 ) -> None:
        self.read(filename)
        pass

    def read(self, filename: str) -> None:
        """
        Reads a YAML file and sets the settings.
        """
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError('Filename must end with .yaml or .yml')

        try:
            #-----------------------------------------------------------
            # Params
            #-----------------------------------------------------------
            self.name = str(data['name'])
            self.R = float(data['R'])
            self.a = float(data['a'])
            self.kappa = float(data['kappa'])
            self.delta = float(data['delta'])
            # Ip optional
            # B0 optional
            if 'B_0' in data:
                self.B0 = float(data['B_0'])
            elif 'B_coil' in data and 'wall_thickness' in data:
                self.B0 = phys.get_B0(data['B_coil'], data['wall_thickness'], self.R, self.a)
                print(f"B0 = {self.B0} calculated from B_coil and wall_thickness.")
            else:
                raise KeyError('B0 or B_coil and wall_thickness not found in settings file.')
            
            if 'I_P' in data:
                self.Ip = float(data['I_P'])
            elif 'qstar' in data:
                self.Ip = phys.get_Ip(data['qstar'], self.R, self.a, self.B0, self.kappa)
                print(f"Ip = {self.Ip} calculated from qstar.")
            else:
                raise KeyError('I_P or qstar not found in settings file.')
            self.tipeak_over_tepeak = float(data['tipeak_over_tepeak'])
            self.fuel = int(data['fuel'])
            if self.fuel == 3:
                raise ValueError("D-He3 fuel cycle not implemented yet. Please use D-D or D-T.")
            self.device_type = str(safe_get(data, 'device_type', 'tokamak')).lower()
            if self.device_type not in ['tokamak', 'stellarator']:
                raise ValueError("device_type must be 'tokamak' or 'stellarator'")
            self.impurityfractions = np.array(data['impurityfractions'], dtype=np.float64)
            if 'Zeff_target' in data and 'impurity' in data:
                impurity = int(data['impurity'])
                self.impurityfractions[impurity] = phys.get_impurity_fraction(data['Zeff_target'], self.impurityfractions[0], impurity, (float(data['Tmax_keV'])-float(data['Tmin_keV']))/2+float(data['Tmin_keV']))
                print(f"Impurity fractions = {self.impurityfractions} calculated from Zeff_target.")

            if self.fuel == 1:
                f_D = 1-np.sum(self.impurityfractions)
                f_T = 0
            elif self.fuel == 2:
                f_D = 0.5-np.sum(self.impurityfractions)/2
                f_T = 0.5-np.sum(self.impurityfractions)/2
            else:
                raise ValueError("Invalid fuel cycle.")
            

            self.M_i = f_D * 2.014 + f_T * 3.016 + np.dot(self.impurityfractions, np.array([4.002, 20.18, 39.948, 83.80, 131.29, 183.84]))

            self.scalinglaw = str(data['scalinglaw'])
            self.H_fac = float(data['H_fac'])
            self.nr = int(data['nr'])

            self.gfilename = str(safe_get(data,'gfilename',''))
            self.profsfilename = str(safe_get(data,'profsfilename',''))

            self.j_alpha1 = float(safe_get(data,'j_alpha1',1))
            self.j_alpha2 = float(safe_get(data,'j_alpha2',2))
            self.j_offset = float(safe_get(data,'j_offset',0))

            self.ne_alpha1 = float(safe_get(data,'ne_alpha1',2))
            self.ne_alpha2 = float(safe_get(data,'ne_alpha2',1.5))
            self.ne_offset = float(safe_get(data,'ne_offset',0))

            self.ni_alpha1 = float(safe_get(data,'ni_alpha1',2))
            self.ni_alpha2 = float(safe_get(data,'ni_alpha2',1.5))
            self.ni_offset = float(safe_get(data,'ni_offset',0))

            self.Ti_alpha1 = float(safe_get(data,'Ti_alpha1',2))
            self.Ti_alpha2 = float(safe_get(data,'Ti_alpha2',1.5))
            self.Ti_offset = float(safe_get(data,'Ti_offset',0))

            self.Te_alpha1 = float(safe_get(data,'Te_alpha1',2))
            self.Te_alpha2 = float(safe_get(data,'Te_alpha2',1.5))
            self.Te_offset = float(safe_get(data,'Te_offset',0))

            #-----------------------------------------------------------
            # Settings
            #-----------------------------------------------------------
            self.Nn = int(data['Nn'])
            self.NTi = int(data['NTi'])
            self.nmax_frac = float(data['nmax_frac'])
            self.nmin_frac = float(data['nmin_frac'])
            self.Tmax_keV = float(data['Tmax_keV'])
            self.Tmin_keV = float(data['Tmin_keV'])
            self.resistivity_model = str(safe_get(data,'resistivity_model','jardin')).lower()
            self.maxit = int(data['maxit'])
            self.accel = float(data['accel'])
            self.err = float(data['err'])
            self.verbosity = int(data['verbosity'])
            self.parallel = bool(data['parallel'])
        except KeyError as e:
            raise KeyError(f'Key {e} not found in {filename}')


POPCON_data_spec = [
    ('n_G_frac', nb.float64[:]),
    ('n_e_20_max', nb.float64[:]),
    ('n_e_20_avg', nb.float64[:]),
    ('n_i_20_max', nb.float64[:,:]),
    ('n_i_20_avg', nb.float64[:,:]),
    ('T_i_max', nb.float64[:]),
    ('T_i_avg', nb.float64[:]),
    ('T_e_max', nb.float64[:]),
    ('T_e_avg', nb.float64[:]),
    ('Paux', nb.float64[:,:]),
    ('Pfusion', nb.float64[:,:]),
    ('Pfusionheating', nb.float64[:,:]),
    ('Pohmic', nb.float64[:,:]),
    ('Pbrems', nb.float64[:,:]),
    ('Psynch', nb.float64[:,:]),
    ('Pimprad', nb.float64[:,:]),
    ('Prad', nb.float64[:,:]),
    ('Pheat', nb.float64[:,:]),
    ('Wtot', nb.float64[:,:]),
    ('tauE', nb.float64[:,:]),
    ('Pconf', nb.float64[:,:]),
    ('Ploss', nb.float64[:,:]),
    ('Pdd', nb.float64[:,:]),
    ('Pdt', nb.float64[:,:]),
    ('Palpha', nb.float64[:,:]),
    ('Psol', nb.float64[:,:]),
    ('f_rad', nb.float64[:,:]),
    ('Q', nb.float64[:,:]),
    ('H89', nb.float64[:,:]),
    ('H98', nb.float64[:,:]),
    ('vloop', nb.float64[:,:]),
    ('betaN', nb.float64[:,:])
]
# jit compiled
@nb.experimental.jitclass(spec = POPCON_data_spec) # type: ignore
class POPCON_data:
    """
    Class POPCON_data

    Output data class for the POPCON. Contains all output arrays.
    """
    def __init__(self) -> None:
        self.n_G_frac: np.ndarray
        self.n_e_20_max: np.ndarray
        self.n_e_20_avg: np.ndarray
        self.n_i_20_max: np.ndarray
        self.n_i_20_avg: np.ndarray
        self.T_i_max: np.ndarray
        self.T_i_avg: np.ndarray
        self.T_e_max: np.ndarray
        self.T_e_avg: np.ndarray
        self.Paux: np.ndarray
        self.Pfusion: np.ndarray
        self.Pfusionheating: np.ndarray
        self.Pohmic: np.ndarray
        self.Pbrems: np.ndarray
        self.Psynch: np.ndarray
        self.Pimprad: np.ndarray
        self.Prad: np.ndarray
        self.Pheat: np.ndarray
        self.Wtot: np.ndarray
        self.tauE: np.ndarray
        self.Pconf: np.ndarray
        self.Ploss: np.ndarray
        self.Pdd: np.ndarray
        self.Pdt: np.ndarray
        self.Palpha: np.ndarray
        self.Psol: np.ndarray
        self.f_rad: np.ndarray
        self.Q: np.ndarray
        self.H89: np.ndarray
        self.H98: np.ndarray
        self.vloop: np.ndarray
        self.betaN: np.ndarray
        pass

# NOT jit compiled
class POPCON_plotsettings:
    """
    Class POPCON_plotsettings

    This class reads a YAML file and stores the settings for a POPCON plot.
    Uses safe_get where appropriate to allow for missing keys in an outdated
    settings file. When applicable, sets to the default value.
    """
    def __init__(self,
                 filename: str,
                 ) -> None:
        
        self.plotoptions: dict
        self.figsize: tuple = (8,6)
        self.yax: str = 'nG'
        self.xax: str = 'T_i_av'
        self.fill_invalid: bool = True
        self.read(filename)

        pass

    def read(self, filename: str) -> None:
        """
        Reads a YAML file and sets the settings.
        """
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            data = yaml.safe_load(open(filename, 'r'))
        else:
            raise ValueError('Filename must end with .yaml or .yml')
        
        defaults = yaml.safe_load(open(get_POPCON_homedir(['resources','default_plotsettings.yml']), 'r'))

        self.plotoptions = {}
        for key in defaults['plotoptions']:
            self.plotoptions[key] = safe_get(data['plotoptions'], key, defaults['plotoptions'][key])
        self.figsize = tuple(safe_get(data, 'figsize', defaults['figsize']))
        self.yax = safe_get(data, 'yax', defaults['yax'])
        self.xax = safe_get(data, 'xax', defaults['xax'])
        self.fill_invalid = safe_get(data, 'fill_invalid', defaults['fill_invalid'])
        
        pass


# NOT jit compiled
class POPCON:
    """
    The Primary class for the OpenPOPCON project.

    This class is the primary interface for the OpenPOPCON project. It
    contains the settings, parameters, and output data for a POPCON run.


    """

    def __init__(self, settingsfile = None, plotsettingsfile = None, scalinglawfile = None) -> None:
        self.algorithms: POPCON_algorithms
        self.settings: POPCON_settings
        self.plotsettings: POPCON_plotsettings
        self.output: POPCON_data

        if settingsfile is not None:
            self.settings = POPCON_settings(settingsfile)
            self.settingsfile = settingsfile
        else:
            pass

        if plotsettingsfile is None:
            plotsettingsfile = DEFAULT_PLOTSETTINGS

        self.plotsettings = POPCON_plotsettings(plotsettingsfile)
        self.plotsettingsfile = plotsettingsfile

        if scalinglawfile is None:
            scalinglawfile = DEFAULT_SCALINGLAWS

        self.__get_scaling_laws(scalinglawfile)
        self.scalinglawfile = scalinglawfile
        
        self.__check_settings()
        compile_test = POPCON_algorithms()
        try:
            compile_test.P_aux_relax_impfrac(1,1,1,1,1)
        except:
            pass

        pass
    
    #-------------------------------------------------------------------
    # Solving
    #-------------------------------------------------------------------

    def run_POPCON(self, setuponly=False) -> None:
        """
        Wrapper function that sets up and solves the power balance 
        equations at each point in the grid.
        """
        self.__setup_params()
        self.__get_geometry()
        self.__get_profiles()

        self.algorithms._setup_profs()
        scalinglaw = self.settings.scalinglaw
        if self.settings.device_type == "stellarator":
            self.algorithms.device_type = 1
        elif self.settings.device_type == "tokamak":
            self.algorithms.device_type = 0
        else:
            raise ValueError(f"Invalid device_type: {self.settings.device_type}")
        slparam = self.scalinglaws[scalinglaw]
        self.algorithms.H_fac = self.settings.H_fac
        self.algorithms.scaling_const = slparam['scaling_const']
        self.algorithms.M_i_alpha = slparam['M_i_alpha']
        self.algorithms.Ip_alpha = slparam['Ip_alpha']
        self.algorithms.R_alpha = slparam['R_alpha']
        self.algorithms.a_alpha = slparam['a_alpha']
        self.algorithms.kappa_alpha = slparam['kappa_alpha']
        self.algorithms.B0_alpha = slparam['B0_alpha']
        self.algorithms.Pheat_alpha = slparam['Pheat_alpha']
        self.algorithms.n20_alpha = slparam['n20_alpha']
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
        n_e_avg_fac= params.volume_integral(rho, params.get_profile(rho,1))/params.V
        T_i_avg_fac = params.volume_integral(rho, params.get_profile(rho,3))/params.V
        T_e_avg_fac = params.volume_integral(rho, params.get_profile(rho,4))/params.V
        n_e_20 = np.linspace(self.settings.nmin_frac*n_G/n_e_avg_fac, self.settings.nmax_frac*n_G/n_e_avg_fac, self.settings.Nn)
        T_i_keV = np.linspace(self.settings.Tmin_keV/T_i_avg_fac, self.settings.Tmax_keV/T_i_avg_fac, self.settings.NTi)
        T_e_keV = T_i_keV/self.settings.tipeak_over_tepeak
        
        self.output = POPCON_data()
        self.output.n_G_frac = np.linspace(self.settings.nmin_frac, self.settings.nmax_frac, self.settings.Nn)
        self.output.n_e_20_max = n_e_20
        self.output.n_e_20_avg = n_e_20 * n_e_avg_fac
        self.output.T_i_max = T_i_keV
        self.output.T_i_avg = T_i_keV * T_i_avg_fac
        self.output.T_e_max = T_e_keV
        self.output.T_e_avg = T_e_keV * T_e_avg_fac
        
        if self.settings.verbosity > 0:
            print("Solving power balance equations")

        if self.settings.parallel:
            Paux = solve_nT_par(params, self.settings.Nn, self.settings.NTi,
                                    n_e_20, T_i_keV, self.settings.accel, 
                                    self.settings.err, self.settings.maxit,
                                    self.settings.verbosity)
        else:
            Paux = solve_nT(params, self.settings.Nn, self.settings.NTi,
                                    n_e_20, T_i_keV, self.settings.accel, 
                                    self.settings.err, self.settings.maxit,
                                    self.settings.verbosity)


        if self.settings.verbosity > 0:
            print("Power balance solutions found. Populating output arrays.")
        self.output.Paux = Paux
        self.output.n_i_20_avg = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.n_i_20_max = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pfusion = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pfusionheating = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pohmic = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pbrems = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Psynch = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pimprad = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Prad = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pheat = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Wtot = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.tauE = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pconf = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Ploss = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pdd = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Pdt = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Palpha = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Psol = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.f_rad = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.Q = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.H89 = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.H98 = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.vloop = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        self.output.betaN = -1*np.ones((self.settings.Nn,self.settings.NTi),dtype=np.float64)
        if self.settings.parallel:
            self.output = populate_outputs_par(params, self.output, self.settings.Nn, self.settings.NTi, self.settings.verbosity)
        else:
            self.output = populate_outputs(params, self.output, self.settings.Nn, self.settings.NTi, self.settings.verbosity)

        pass

    def single_point(self, n_G_frac:float, Ti_av:float, plot:bool=True, show:bool=True) -> None:
        """
        Solves power balance at a single n, T point. Prints the results
        plots the profiles of the solution if plot=True.
        """
        
        try: self.algorithms
        except AttributeError:
            self.run_POPCON(setuponly=True)

        n_G = self.algorithms.n_GR
        rho = self.algorithms.sqrtpsin
        n_e_avg_fac = self.algorithms.volume_integral(rho, self.algorithms.get_profile(rho, 1))/self.algorithms.V
        n_e_20 = n_G_frac*n_G/n_e_avg_fac
        T_i_avg_fac = self.algorithms.volume_integral(rho, self.algorithms.get_profile(rho, 3))/self.algorithms.V
        n_i_avg_fac = self.algorithms.volume_integral(rho, self.algorithms.get_profile(rho, 2))/self.algorithms.V
        T_i_keV = Ti_av/T_i_avg_fac
        T_e_keV = T_i_keV/self.algorithms.tipeak_over_tepeak
        dil = self.algorithms.plasma_dilution(T_e_keV)
        n_i_20 = n_e_20*dil
        line_avg_fac = np.average(self.algorithms.get_profile(rho, 1))

        Paux = self.algorithms.P_aux_relax_impfrac(n_e_20,T_i_keV,self.settings.accel,self.settings.err,self.settings.maxit)
        
        Pfusion = self.algorithms.volume_integral(rho,self.algorithms._P_fusion(rho, T_i_keV, n_i_20))
        Pfusion_heating = self.algorithms.volume_integral(rho,self.algorithms._P_fusion_heating(rho, T_i_keV, n_i_20))
        Pohmic = self.algorithms.volume_integral(rho,self.algorithms._P_OH_prof(rho, T_e_keV, n_e_20))
        Pbrems = self.algorithms.volume_integral(rho,self.algorithms._P_brem_rad(rho, T_e_keV, n_e_20))
        Psynch = self.algorithms.volume_integral(rho,self.algorithms._P_synch(rho, T_e_keV, n_e_20))
        Pimprad = self.algorithms.volume_integral(rho,self.algorithms._P_impurity_rad(rho, T_e_keV, n_e_20))
        Prad = self.algorithms.volume_integral(rho,self.algorithms._P_rad(rho, T_e_keV, n_e_20))
        Pheat = Pfusion_heating + Pohmic + Paux - Pbrems
        Palpha = self.algorithms.volume_integral(rho,self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20))*3.52e3/(3.52e3 + 14.06e3)
        Pdd = self.algorithms.volume_integral(rho,self.algorithms._P_DDnHe3_prof(rho, T_i_keV, n_i_20))
        Pdd += self.algorithms.volume_integral(rho,self.algorithms._P_DDpT_prof(rho, T_i_keV, n_i_20))
        Pdt = self.algorithms.volume_integral(rho,self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20))
        tauE = self.algorithms.tauE_scalinglaw(Pheat, n_e_20*line_avg_fac)
        Wtot = self.algorithms.volume_integral(rho,self.algorithms._W_tot_prof(rho,T_i_keV,n_e_20))
        Pconf = Wtot/tauE
        Ploss = Pconf + Psynch + Pbrems
        f_rad = Prad/Ploss
        Q = self.algorithms.Q_fusion(T_i_keV, n_e_20, Paux)
        H89 = tauE/self.algorithms.tauE_H89(Pheat,n_e_20*line_avg_fac)
        H98 = tauE/self.algorithms.tauE_H98(Pheat,n_e_20*line_avg_fac)
        vloop = self.algorithms.Vloop(T_e_keV, n_e_20)
        betaN = 100*self.algorithms.BetaN(T_i_keV, n_e_20) # in percent
        pstring = \
f"""
Params:
n_i_average = {n_i_avg_fac*n_i_20} x 10^20 m^-3
n_e_average = {n_G_frac*n_G:1.3f} x 10^20 m^-3
n_G = {n_G:1.3f} x 10^20 m^-3
n_i_axis = {n_i_20:1.3f} x 10^20 m^-3
n_e_axis = {n_e_20:1.3f} x 10^20 m^-3
Ti_average = {Ti_av:.1f} keV
Ti_axis = {T_i_keV:.1f} keV
Solution:
P_aux = {Paux:.2f} MW
P_fusion = {Pfusion:.2f} MW
P_SOL = {Ploss - Prad:.2f} MW
P_load = {(Ploss-Prad)/self.algorithms.A:.3f} MW/m^2
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
            Pfusion_heating_prof = self.algorithms._P_fusion_heating(rho, T_i_keV, n_i_20)
            Pohmic_prof = self.algorithms._P_OH_prof(rho, T_e_keV, n_e_20)
            Prad_prof = self.algorithms._P_rad(rho, T_e_keV, n_e_20)
            Pheat_prof = Pfusion_heating_prof + Pohmic_prof + Paux/self.algorithms.V
            Palpha_prof = self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20)*3.52e3/(3.52e3 + 14.06e3)
            Pdt_prof = self.algorithms._P_DTnHe4_prof(rho, T_i_keV, n_i_20)
            niprof = n_i_20*self.algorithms.get_profile(rho, 2)
            neprof = n_e_20*self.algorithms.get_profile(rho, 1)
            Tiprof = T_i_keV*self.algorithms.get_profile(rho, 3)
            Teprof = T_e_keV*self.algorithms.get_profile(rho, 4)
            qprof = self.algorithms.get_profile(rho, 5)

            # Density and temperatures
            # 
            fig, ax1 = plt.subplots(figsize=(4,3))
            ax2 = ax1.twinx()
            ax1.plot(rho, niprof, 'b-', label=r'$n_i$')
            ax1.plot(rho, neprof, 'b--', label=r'$n_e$')
            ax2.plot(rho, Tiprof, 'r-', label=r'$T_i$')
            ax2.plot(rho, Teprof, 'r--', label=r'$T_e$')
            ax1.set_xlabel(r'$\sqrt{\psi_N}$')
            ax1.set_ylabel(r'$n$ ($m^{-3}$)', color='b')
            ax2.set_ylabel(r'$T$ (keV)', color='r')
            ax1.legend(loc='lower left')
            ax2.legend(loc='upper right')
            if show:
                plt.show()
            
            # Power profiles
            fig, ax = plt.subplots(figsize=(8,6))
            # V_enclosed = self.algorithms.get_profile(rho, -1)
            radius = rho*self.algorithms.a
            ax.plot(radius, Pfusion_prof, 'k-', label=r'$P_{\text{fusion}}$')
            ax.plot(radius, Pohmic_prof, 'r-', label=r'$P_{\text{ohmic}}$')
            ax.plot(radius, Prad_prof, '-',color='purple', label=r'$P_{\text{rad}}$')
            ax.plot(radius, Pheat_prof, 'b-', label=r'$P_{\text{heat}}$')
            ax.plot(radius, Palpha_prof, 'g-', label=r'$P_{\alpha}$')
            ax.set_xlabel(r'$\rho$ ($m$)')
            ax.set_ylabel(r'dP/dV (MW/$m^3$)')
            # ax.set_yscale('log')
            ax.legend(loc='lower left')
            if show:
                plt.show()
        pass

    #-------------------------------------------------------------------
    # Plotting
    #-------------------------------------------------------------------

    def plot(self, show:bool=True, savefig:str='', names=None):
        """
        Plots the output data. If variable names is specified, only
        plots those variables. Otherwise, refers to the plotsettings
        file.
        """
        figsize = self.plotsettings.figsize
        fig, ax = plt.subplots(figsize=figsize)
        if self.plotsettings.xax == 'T_i_av':
            xx = self.output.T_i_avg
        elif self.plotsettings.xax == 'T_i_ax':
            xx = self.output.T_i_max
        elif self.plotsettings.xax == 'T_e_av':
            xx = self.output.T_e_avg
        elif self.plotsettings.xax == 'T_e_ax':
            xx = self.output.T_e_max
        else:
            raise ValueError("Invalid x-axis. Change xax in plotsettings.")
        
        if self.plotsettings.yax == 'n20_av':
            yy = self.output.n_e_20_avg
        elif self.plotsettings.yax == 'n20_ax':
            yy = self.output.n_e_20_max
        elif self.plotsettings.yax == 'nG':
            yy = self.output.n_G_frac
        else:
            raise ValueError("Invalid y-axis. Change yax in plotsettings.")
        xx, yy = np.meshgrid(xx,yy)
        mask = np.logical_or(np.isnan(self.output.Paux),self.output.Paux >=99998.)
        if self.plotsettings.fill_invalid:
            ax.contourf(xx,yy,np.ma.array(np.ones_like(xx),mask=np.logical_not(mask)),levels=[0,2],colors='k',alpha=0.5)
        if np.any(self.output.Q > 1e4):
            maskburning = np.logical_not(np.logical_or(np.isnan(self.output.Q),self.output.Q >=1e4))
            ax.contourf(xx,yy,np.ma.array(np.ones_like(xx),mask=maskburning),levels=[0,2],colors='r',alpha=0.08)
        if names is None:
            names = self.plotsettings.plotoptions.keys()
        for name in names:
            mask = np.logical_or(np.isnan(self.output.Paux),self.output.Paux >=99998.)
            opdict = self.plotsettings.plotoptions[name]
            if opdict['plot'] == False:
                continue
            plotoptions = [opdict['color'],opdict['linewidth'],opdict['label'],opdict['fontsize'],opdict['fmt']]
            data = getattr(self.output,name)
            data = np.ma.array(data,mask=mask)
            if opdict['spacing'] == 'lin':
                if opdict['scale'] == 'minmax':
                    if self.settings.verbosity > 1:
                        print(f"{name} min: {np.min(data)}, max: {np.max(data)}, levels: {opdict['levels']}")
                    levels = np.linspace(1.01*np.min(data),0.99*np.max(data),opdict['levels'])
                    if levels[-1] < levels[0]:
                        levels = levels[::-1]
                elif opdict['scale'] == 'specified':
                    levels = np.linspace(opdict['min'],opdict['max'],opdict['levels'])
                else:
                    raise ValueError(f"Invalid scale for {name}. Change scale in plotsettings.")
            elif opdict['spacing'] == 'log':
                if opdict['scale'] == 'minmax':
                    levels = np.logspace(np.log10(1.01*np.min(data)),np.log10(0.99*np.max(data)),opdict['levels'])
                elif opdict['scale'] == 'specified':
                    levels = np.logspace(np.log10(opdict['min']),np.log10(opdict['max']),opdict['levels'])
                else:
                    raise ValueError(f"Invalid scale for {name}. Change scale in plotsettings.")
            elif opdict['spacing'] == 'manual':
                levels = np.asarray(opdict['manuallevels'],dtype=np.float64)
            else:
                raise ValueError(f"Invalid spacing for {name}. Change spacing in plotsettings.")
            
            if self.settings.verbosity > 0:
                print(f"Plotting {name} with levels {levels} and options {plotoptions}")
            self.plot_contours(opdict['plot'], ax, data, xx, yy, levels, *plotoptions)
        
        if self.plotsettings.xax == 'T_i_av':
            ax.set_xlabel(r'$\langle T_i\rangle$ (keV)')
        elif self.plotsettings.xax == 'T_i_ax':
            ax.set_xlabel(r'$T_i$ (keV, On-axis)')
        elif self.plotsettings.xax == 'T_e_av':
            ax.set_xlabel(r'$\langle T_e\rangle$ (keV)')
        elif self.plotsettings.xax == 'T_e_ax':
            ax.set_xlabel(r'$T_e$ (keV, On-axis)')
        else:
            pass
        
        if self.plotsettings.yax == 'n20_av':
            ax.set_ylabel(r'$\langle n_{20}\rangle$ ($10^{20} m^{-3}$)')
        elif self.plotsettings.yax == 'n20_ax':
            ax.set_ylabel(r'$n_{20}(0)$')
        elif self.plotsettings.yax == 'nG':
            ax.set_ylabel(r'$\langle n\rangle /n_G$')
        else:
            pass
        p = self.algorithms

        # 1 = D-D, 2 = D-T, 3 = D-He3
        fueldict = {1:'D-D', 2:'D-T', 3:'D-He3'}

        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        infoboxtext = f"$I_p$ = {p.Ip:.2f}\n$B_0$ = {p.B0:.2f}\nR = {p.R:.2f}\na = {p.a:.2f}\n$\\kappa$ = {p.kappa:.2f}\n$\\delta$ = {p.delta:.2f}\n$M_i$ = {p.M_i:.2f}\nti/te = {p.tipeak_over_tepeak:.2f}\nfuel = {fueldict[p.fuel]}\n<Zeff>={p.Zeff(np.average(xx)):.2f}"
        ax.text(x=np.max(xx)+(np.max(xx)-np.min(xx))/64,y=np.min(yy),s=infoboxtext, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
        fig.tight_layout()
        if savefig != '':
            plt.savefig(savefig)
        if show:
            plt.show()

        return fig, ax
    
    def plot_contours(self, plotbool:bool, ax, data:np.ndarray, xx:np.ndarray, yy:np.ndarray, levels, color, linewidth, label:str, fontsize:int, fmt:str):
        if plotbool:
            contour = ax.contour(xx, yy, data, levels=levels, colors=color, linewidths=linewidth)
            rect = patches.Rectangle((0,0),0,0,fc = color,label = label)
            ax.add_patch(rect)
            ax.clabel(contour,inline=1,fmt=fmt,fontsize=fontsize)
        pass
    
    def custom_plot(self, fig, ax, data, levels, color='k', linewidth=1., label:str='custom', fontsize:int=11, fmt:str='%1.2f'):
        if self.plotsettings.xax == 'T_i_av':
            xx = self.output.T_i_avg
        elif self.plotsettings.xax == 'T_i_ax':
            xx = self.output.T_i_max
        elif self.plotsettings.xax == 'T_e_av':
            xx = self.output.T_e_avg
        elif self.plotsettings.xax == 'T_e_ax':
            xx = self.output.T_e_max
        else:
            raise ValueError("Invalid x-axis. Change xax in plotsettings.")
        
        if self.plotsettings.yax == 'n20_av':
            yy = self.output.n_e_20_avg
        elif self.plotsettings.yax == 'n20_ax':
            yy = self.output.n_e_20_max
        elif self.plotsettings.yax == 'nG':
            yy = self.output.n_G_frac
        else:
            raise ValueError("Invalid y-axis. Change yax in plotsettings.")
        xx, yy = np.meshgrid(xx,yy)
        mask = np.logical_or(np.isnan(self.output.Paux),self.output.Paux >=99998.)
        self.plot_contours(True, ax, np.ma.array(data,mask=mask), xx, yy, levels, *[color,linewidth,label,fontsize,fmt])

        return fig, ax

    #-------------------------------------------------------------------
    # File I/O
    #-------------------------------------------------------------------

    def write_output(self, name:str='', archive:bool=True, overwrite:bool=False, directory:str=None) -> None:
        """
        Saves the output results to a directory. If archive is True,
        archives the directory as a zip file. If overwrite is True,
        overwrites the directory/zip if it already exists. The directory
        parameter allows specifying the storage location.
        """
        if name == '':
            name = self.settings.name + '_' + datetime.datetime.now().strftime(r"%Y-%m-%d_%H:%M:%S")

        if directory is None:
            outputsdir = pathlib.Path(__file__).resolve().parent.parent.joinpath('outputs')
        else:
            outputsdir = pathlib.Path(directory)

        direxists = outputsdir.joinpath(name).exists()
        zipexists = outputsdir.joinpath(name + '.zip').exists()
        if not (direxists or zipexists):
            outputsdir.joinpath(name).mkdir()
        elif overwrite:
                if zipexists:
                    outputsdir.joinpath(name + '.zip').unlink()
                if direxists:
                    shutil.rmtree(outputsdir.joinpath(name))
                outputsdir.joinpath(name).mkdir()
        else:
            raise ValueError(f"{'Archive'*zipexists}{' and '*zipexists*direxists}{'Directory'*direxists} already exist{'s'*(not (direxists and zipexists))}. Set overwrite=True to overwrite.")
        
        savedir = outputsdir.joinpath(name)

        shutil.copyfile(self.settingsfile, savedir.joinpath('settings.yaml'))
        shutil.copyfile(self.plotsettingsfile, savedir.joinpath('plotsettings.yaml'))
        shutil.copyfile(self.scalinglawfile, savedir.joinpath('scalinglaws.yaml'))
        
        writedata = {}
        keys = POPCON_data_spec
        for key in keys:
            writedata[key[0]] = getattr(self.output,key[0])
        
        with open(savedir.joinpath('arrays.json'), 'w') as f:
            json.dump(writedata, f, cls=nj.CompactJSONEncoder)

        if self.settings.gfilename != '':
            shutil.copyfile(self.settings.gfilename, savedir.joinpath(self.settings.gfilename.split(str(os.sep))[-1]))
        if self.settings.profsfilename != '':
            shutil.copyfile(self.settings.profsfilename, savedir.joinpath(self.settings.profsfilename.split(str(os.sep))[-1]))

        self.plot(show=False, savefig=str(savedir.joinpath('POPCON_plot.pdf')))
        plt.close('all')

        if archive:
            shutil.make_archive(name, 'zip', savedir)
            shutil.rmtree(savedir)
            shutil.move(name+'.zip', outputsdir.joinpath(name+'.zip'))

    def read_output(self, name: str, directory: str = None) -> None:
        """
        Restores the POPCON object based on the output directory.
        """
        if directory is None:
            outputsdir = pathlib.Path(__file__).resolve().parent.parent.joinpath('outputs')
        else:
            outputsdir = pathlib.Path(directory)

        if name.endswith('.zip'):
            outputsdir.joinpath(name[:-4]).mkdir()
            shutil.unpack_archive(outputsdir.joinpath(name), outputsdir.joinpath(name[:-4]))
            name = name[:-4]
        namepath = outputsdir.joinpath(name)
        self.settingsfile = str(namepath.joinpath('settings.yaml'))
        self.plotsettingsfile = str(namepath.joinpath('plotsettings.yaml'))
        self.scalinglawfile = str(namepath.joinpath('scalinglaws.yaml'))
        self.settings = POPCON_settings(self.settingsfile)
        self.plotsettings = POPCON_plotsettings(self.plotsettingsfile)
        self.__get_scaling_laws(self.scalinglawfile)
        
        self.output = POPCON_data()
        with open(namepath.joinpath('arrays.json'), 'r') as f:
            data = json.load(f)
            for key in data.keys():
                setattr(self.output, key, np.array(data[key], dtype=np.float64))
        if self.settings.gfilename != '':
            self.settings.gfilename = str(namepath.joinpath(self.settings.gfilename.split(str(os.sep))[-1]))
        if self.settings.profsfilename != '':
            self.settings.profsfilename = str(namepath.joinpath(self.settings.profsfilename.split(str(os.sep))[-1]))
        self.run_POPCON(setuponly=True)
        pass

    #-------------------------------------------------------------------
    # Setup functions
    #-------------------------------------------------------------------

    def __get_scaling_laws(self, scalinglawfile: str) -> None:
        if scalinglawfile.endswith('.yaml') or scalinglawfile.endswith('.yml'):
            data = yaml.safe_load(open(scalinglawfile, 'r'))
        else:
            raise ValueError('Filename must end with .yaml or .yml')
        
        device_type = getattr(self.settings, 'device_type', 'tokamak').lower()
        
        if device_type == "tokamak":
            assert 'H89' in data.keys()
            assert 'H98y2' in data.keys()
            assert 'H_NT23' in data.keys()
        elif device_type == "stellarator":
            assert 'ISSO' in data.keys()
        else:
            raise ValueError(f"Unknown device_type '{device_type}', must be 'tokamak' or 'stellarator'")
        self.scalinglaws = data
    

    def __setup_params(self) -> None:

        res_dict = {"jardin": 0, "paz-soldan": 1, "maximum":2, "max":2}

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
        self.algorithms.resistivity_alg = res_dict[self.settings.resistivity_model.lower()]
        self.algorithms.device_type = 1 if self.settings.device_type == "stellarator" else 0

    def __get_profiles(self) -> None:
        if self.settings.profsfilename == '':
            if self.settings.ne_offset == 0:
                self.settings.ne_offset = 1e-6
            if self.settings.ni_offset == 0:
                self.settings.ni_offset = 1e-6
            if self.settings.Ti_offset == 0:
                self.settings.Ti_offset = 1e-6
            if self.settings.Te_offset == 0:
                self.settings.Te_offset = 1e-6

            self.algorithms._set_alpha_and_offset(self.settings.ne_alpha1, self.settings.ne_alpha2, self.settings.ne_offset, 1)
            self.algorithms._set_alpha_and_offset(self.settings.ni_alpha1, self.settings.ni_alpha2, self.settings.ni_offset, 2)
            self.algorithms._set_alpha_and_offset(self.settings.Ti_alpha1, self.settings.Ti_alpha2, self.settings.Ti_offset, 3)
            self.algorithms._set_alpha_and_offset(self.settings.Te_alpha1, self.settings.Te_alpha2, self.settings.Te_offset, 4)
        else:
            profstable = read_profsfile(self.settings.profsfilename)
            ne = np.asarray(profstable['n_e'])
            ne = ne/ne[0]
            ni = np.asarray(profstable['n_i'])
            ni = ni/ni[0]
            Ti = np.asarray(profstable['T_i'])
            Ti = Ti/Ti[0]
            Te = np.asarray(profstable['T_e'])
            Te = Te/Te[0]
            rho = np.asarray(profstable['rho'])
            ne_sqrtpsin = np.interp(np.linspace(0.001,1,self.settings.nr),rho,ne)
            ni_sqrtpsin = np.interp(np.linspace(0.001,1,self.settings.nr),rho,ni)
            Ti_sqrtpsin = np.interp(np.linspace(0.001,1,self.settings.nr),rho,Ti)
            Te_sqrtpsin = np.interp(np.linspace(0.001,1,self.settings.nr),rho,Te)
            self.algorithms._addextprof(ne_sqrtpsin,1)
            self.algorithms._addextprof(ni_sqrtpsin,2)
            self.algorithms._addextprof(Ti_sqrtpsin,3)
            self.algorithms._addextprof(Te_sqrtpsin,4)

        pass

    def __get_geometry(self) -> None:
        if self.settings.gfilename == '':
            rho = np.sqrt(np.linspace(0.001,1,self.settings.nr))
            if self.settings.j_offset == 0:
                self.settings.j_offset = 1e-6
            self.algorithms._addextprof(rho,-2)
            self.algorithms._set_alpha_and_offset(self.settings.j_alpha1, self.settings.j_alpha2, self.settings.j_offset, 0)
            self.algorithms.Itot = self.settings.Ip
            
        else:
            gfile = read_eqdsk(self.settings.gfilename)
            psin, volgrid, agrid, fs = get_fluxvolumes(gfile, self.settings.nr)
            sqrtpsin = np.sqrt(np.linspace(0.001,0.98,self.settings.nr))
            volgrid = np.interp(sqrtpsin,np.sqrt(psin),volgrid)
            _, jrms, jtoravg, cross_sec_areas = get_current_density(gfile, self.settings.nr)
            qpsi = np.asarray(gfile['qpsi'])
            psiq = np.linspace(0,1,qpsi.shape[0])
            qr = np.interp(sqrtpsin,np.sqrt(psiq),qpsi)



            Ipint = np.abs(np.trapz(y=jtoravg, x=cross_sec_areas))
            Jrmsint = np.abs(np.trapz(y=jrms, x=cross_sec_areas))
            
            Jrms_norm = jrms/Jrmsint


            lcfs = fs[-1]
            geq_a = (np.max(lcfs[:,0])-np.min(lcfs[:,0]))/2
            geq_R = (np.max(lcfs[:,0]) - geq_a)
            geq_z0 = (np.max(lcfs[:,1]) + np.min(lcfs[:,1]))/2
            geq_kappa = np.abs(np.max(lcfs[:,1]) - np.min(lcfs[:,1]))/(2*geq_a)
            geq_Rtop = lcfs[np.argmax(lcfs[:,1]),0]
            geq_Rbot = lcfs[np.argmin(lcfs[:,1]),0]
            geq_delta = ((geq_R-geq_Rtop)/geq_a + (geq_R-geq_Rtop)/geq_a)/2



            if self.settings.verbosity > 1:
                print("gEQDSK geometry:")
                print(f"Minor radius: {geq_a}")
                print(f"Major radius: {geq_R}")
                print(f"Elongation: {geq_kappa}")
                print(f"Triangularity: {geq_delta}")
                print(f"z0: {geq_z0}")
                print("gEQDSK Ip:",Ipint)


            
            psin, ftrapped_profile = get_trapped_particle_fraction(gfile)
            ftrapped_profile = np.interp(sqrtpsin,np.sqrt(psin),ftrapped_profile)

            if self.settings.verbosity > 1:
                print("Len of psin:",len(psin))
                print("Len of sqrtpsin:",len(sqrtpsin))
                print("Len of volgrid:",len(volgrid))
                print("Len of jrms:",len(jrms))
                print("Len of qr:",len(qr))
                print("Len of agrid:",len(agrid))
                print("Len of ftrapped_profile:",len(ftrapped_profile))

            print(np.shape(ftrapped_profile))

            if np.abs(geq_a/self.settings.a - 1) > 0.1:
                print(f"Warning: gEQDSK minor radius ({geq_a} m) differs significantly from settings a ({self.settings.a} m). Defaulting to gEQDSK value.")
                self.settings.a = geq_a
                self.algorithms.a = geq_a
            if np.abs(geq_R/self.settings.R - 1) > 0.1:
                print(f"Warning: gEQDSK major radius ({geq_R} m) differs significantly from settings R ({self.settings.R} m). Defaulting to gEQDSK value.")
                self.settings.R = geq_R
                self.algorithms.R = geq_R
            if np.abs(geq_kappa/self.settings.kappa - 1) > 0.1:
                print(f"Warning: gEQDSK elongation ({geq_kappa}) differs significantly from settings kappa ({self.settings.kappa}). Defaulting to gEQDSK value.")
                self.settings.kappa = geq_kappa
                self.algorithms.kappa = geq_kappa
            if np.abs(geq_delta/self.settings.delta - 1) > 0.1:
                print(f"Warning: gEQDSK triangularity ({geq_delta}) differs significantly from settings delta ({self.settings.delta}). Defaulting to gEQDSK value.")
                self.settings.delta
                self.algorithms.delta = geq_delta
            if np.abs(Ipint/self.settings.Ip - 1) > 0.1:
                print(f"Warning: gEQDSK Ip ({Ipint}) differs significantly from settings Ip ({self.settings.Ip}). Defaulting to gEQDSK value.")
                self.settings.Ip = Ipint
                self.algorithms.Ip = Ipint
                
            self.algorithms._addextprof(sqrtpsin,-2)
            self.algorithms._addextprof(volgrid,-1)
            self.algorithms._addextprof(Jrms_norm,0)
            self.algorithms.Itot = Ipint
            self.algorithms._addextprof(qr,5)
            self.algorithms._addextprof(agrid,-3)
            self.algorithms._addextprof(ftrapped_profile,6)
        pass

    def __check_settings(self) -> None:
        """
        Checks the settings for consistency.
        """
        pass
    
    def update_plotsettings(self, plotsettingsfile = None):
        if plotsettingsfile is not None:
            self.plotsettings = POPCON_plotsettings(plotsettingsfile)
            self.plotsettingsfile = plotsettingsfile
        else:
            self.plotsettings = POPCON_plotsettings(self.plotsettingsfile)
        pass

# @conditional_njit(enable=self.settings.parallel,parallel=self.settings.parallel)
@nb.njit(parallel=True)
def solve_nT_par(params:POPCON_algorithms, nn:int, nT:int,
                n_e_20:np.ndarray, T_i_keV:np.ndarray, 
                accel:float=1.2, err:float=1e-5,
                maxit:int=1000, verbosity=1):
    """
    Compiling the solver allows for parallelization, as each
    point in the grid can be solved independently.
    """

    Paux = np.empty((nn,nT),dtype=np.float64)
    
    for i in nb.prange(nn):
        for j in nb.prange(nT):
            if verbosity > 1:
                print("Running n20=",n_e_20[i],", T=",T_i_keV[j]," keV")

            Paux[i,j] = params.P_aux_relax_impfrac(n_e_20[i],
                                                    T_i_keV[j], 
                                                    accel, 
                                                    err,
                                                    maxit)
            
    return Paux

@nb.njit(parallel=False)
def solve_nT(params:POPCON_algorithms, nn:int, nT:int,
                n_e_20:np.ndarray, T_i_keV:np.ndarray, 
                accel:float=1.2, err:float=1e-5,
                maxit:int=1000, verbosity=1):
    """
    Compiling the solver allows for parallelization, as each
    point in the grid can be solved independently.
    """

    Paux = np.empty((nn,nT),dtype=np.float64)
    
    for i in nb.prange(nn):
        for j in nb.prange(nT):
            if verbosity > 1:
                print("Running n20=",n_e_20[i],", T=",T_i_keV[j]," keV")

            Paux[i,j] = params.P_aux_relax_impfrac(n_e_20[i],
                                                    T_i_keV[j], 
                                                    accel, 
                                                    err,
                                                    maxit)
            
    return Paux


@nb.njit(parallel=True)
def populate_outputs_par(params:POPCON_algorithms, result:POPCON_data, Nn, NTi, verbosity=1):
    """
    Compiling this function is handy as it allows for 
    parallelization of the most computationally expensive
    part of the code.
    """
    rho = params.sqrtpsin
    line_average_fac = np.average(params.get_profile(rho, 1))
    for i in nb.prange(Nn):
        for j in nb.prange(NTi):
            if verbosity > 1:
                print("Populating n20=",result.n_e_20_max[i],", T=",result.T_i_max[j]," keV")
            dil = params.plasma_dilution(result.T_e_max[j])
            result.n_i_20_max[i,j] = result.n_e_20_max[i]*dil
            result.n_i_20_avg[i,j] = result.n_i_20_max[i,j]*params.volume_integral(rho,params.get_profile(rho, 2))/params.V
            result.Pfusion[i,j] = params.volume_integral(rho,params._P_fusion(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pfusionheating[i,j] = params.volume_integral(rho,params._P_fusion_heating(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pohmic[i,j] = params.volume_integral(rho,params._P_OH_prof(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Pbrems[i,j] = params.volume_integral(rho,params._P_brem_rad(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Psynch[i,j] = params.volume_integral(rho,params._P_synch(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Pimprad[i,j] = params.volume_integral(rho,params._P_impurity_rad(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Prad[i,j] = params.volume_integral(rho,params._P_rad(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Pheat[i,j] = result.Pfusionheating[i,j] + result.Pohmic[i,j] + result.Paux[i,j] - result.Pbrems[i,j]
            result.Wtot[i,j] = params.volume_integral(rho,params._W_tot_prof(rho,result.T_i_max[j],result.n_e_20_max[i]))
            result.tauE[i,j] = params.tauE_scalinglaw(result.Pheat[i,j], result.n_e_20_max[i]*line_average_fac)
            result.Pconf[i,j] = result.Wtot[i,j]/result.tauE[i,j]
            result.Ploss[i,j] = result.Pconf[i,j]
            result.Pdd[i,j] = params.volume_integral(rho,params._P_DDnHe3_prof(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pdd[i,j] += params.volume_integral(rho,params._P_DDpT_prof(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pdt[i,j] = params.volume_integral(rho,params._P_DTnHe4_prof(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Palpha[i,j] = result.Pdt[i,j] * 3.52e3/(3.52e3 + 14.06e3)
            result.Psol[i,j] = result.Ploss[i,j] - result.Prad[i,j]
            result.f_rad[i,j] = result.Prad[i,j] / result.Ploss[i,j]
            result.Q[i,j] = params.Q_fusion(result.T_i_max[j], result.n_e_20_max[i], result.Paux[i,j])
            result.H89[i,j] = result.tauE[i,j]/params.tauE_H89(result.Pheat[i,j], result.n_e_20_max[i]*line_average_fac)
            result.H98[i,j] = result.tauE[i,j]/params.tauE_H98(result.Pheat[i,j], result.n_e_20_max[i]*line_average_fac)
            result.vloop[i,j] = params.Vloop(result.T_e_max[j], result.n_e_20_max[i])
            result.betaN[i,j] = 100*params.BetaN(result.T_i_max[j], result.n_e_20_max[i]) # in percent
    return result

@nb.njit(parallel=False)
def populate_outputs(params:POPCON_algorithms, result:POPCON_data, Nn, NTi, verbosity=1):
    """
    Compiling this function is handy as it allows for 
    parallelization of the most computationally expensive
    part of the code.
    """
    rho = params.sqrtpsin
    line_average_fac = np.average(params.get_profile(rho, 1))
    for i in nb.prange(Nn):
        for j in nb.prange(NTi):
            if verbosity > 1:
                print("Populating n20=",result.n_e_20_max[i],", T=",result.T_i_max[j]," keV")
            dil = params.plasma_dilution(result.T_e_max[j])
            result.n_i_20_max[i,j] = result.n_e_20_max[i]*dil
            result.n_i_20_avg[i,j] = result.n_i_20_max[i,j]*params.volume_integral(rho,params.get_profile(rho, 2))/params.V
            result.Pfusion[i,j] = params.volume_integral(rho,params._P_fusion(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pfusionheating[i,j] = params.volume_integral(rho,params._P_fusion_heating(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pohmic[i,j] = params.volume_integral(rho,params._P_OH_prof(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Pbrems[i,j] = params.volume_integral(rho,params._P_brem_rad(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Psynch[i,j] = params.volume_integral(rho,params._P_synch(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Pimprad[i,j] = params.volume_integral(rho,params._P_impurity_rad(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Prad[i,j] = params.volume_integral(rho,params._P_rad(rho, result.T_e_max[j], result.n_e_20_max[i]))
            result.Pheat[i,j] = result.Pfusionheating[i,j] + result.Pohmic[i,j] + result.Paux[i,j] - result.Pbrems[i,j]
            result.Wtot[i,j] = params.volume_integral(rho,params._W_tot_prof(rho,result.T_i_max[j],result.n_e_20_max[i]))
            result.tauE[i,j] = params.tauE_scalinglaw(result.Pheat[i,j], result.n_e_20_max[i]*line_average_fac)
            result.Pconf[i,j] = result.Wtot[i,j]/result.tauE[i,j]
            result.Ploss[i,j] = result.Pconf[i,j]
            result.Pdd[i,j] = params.volume_integral(rho,params._P_DDnHe3_prof(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pdd[i,j] += params.volume_integral(rho,params._P_DDpT_prof(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Pdt[i,j] = params.volume_integral(rho,params._P_DTnHe4_prof(rho, result.T_i_max[j], result.n_i_20_max[i,j]))
            result.Palpha[i,j] = result.Pdt[i,j] * 3.52e3/(3.52e3 + 14.06e3)
            result.Psol[i,j] = result.Ploss[i,j] - result.Prad[i,j]
            result.f_rad[i,j] = result.Prad[i,j] / result.Ploss[i,j]
            result.Q[i,j] = params.Q_fusion(result.T_i_max[j], result.n_e_20_max[i], result.Paux[i,j])
            result.H89[i,j] = result.tauE[i,j]/params.tauE_H89(result.Pheat[i,j], result.n_e_20_max[i]*line_average_fac)
            result.H98[i,j] = result.tauE[i,j]/params.tauE_H98(result.Pheat[i,j], result.n_e_20_max[i]*line_average_fac)
            result.vloop[i,j] = params.Vloop(result.T_e_max[j], result.n_e_20_max[i])
            result.betaN[i,j] = 100*params.BetaN(result.T_i_max[j], result.n_e_20_max[i]) # in percent
    return result

class POPCON_scan(POPCON):
    """
    Class POPCON_scan

    Placeholder class for running scans. Inherits from POPCON.
    """
    def __init__(self) -> None:
        self.datas: list[POPCON_data]
        self.algorithms_list: list[POPCON_algorithms]
        self.settings: POPCON_settings
        self.plotsettings: list[POPCON_plotsettings]
        self.scalinglaws: dict
        self.scanvariables: dict[str, np.ndarray]
        pass
