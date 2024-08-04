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
import scipy.constants as const
# import scipy.integrate as integrate
# import xarray as xr
import yaml
from typing import Callable
from openpopcon_util import *
import numba as nb
import phys_lib as phys


# core of calculations
# jit compiled
@nb.experimental.jitclass(spec = [
            ('R', nb.float64), 
            ('a', nb.float64),
            ('kappa', nb.float64),
            ('delta', nb.float64),
            ('B0', nb.float64),
            ('Ip', nb.float64),
            ('q_a', nb.float64),
            ('H', nb.float64),
            ('M_i', nb.float64),
            ('f_LH', nb.float64),
            # ('nipeak_over_nepeak', nb.float64),
            ('tipeak_over_tepeak', nb.float64),
            ('fuel', nb.int64),
            ('sqrtpsin', nb.float64[:]),
            ('volgrid', nb.float64[:]),
            ('nr', nb.int64),
            ('impurityfractions', nb.float64[:]),
            ('extprof_geoms', nb.boolean),
            ('geomsdefined', nb.boolean),
            ('rdefined', nb.boolean),
            ('volgriddefined', nb.boolean),
            ('extprof_imps', nb.boolean),
            ('impsdefined', nb.boolean),
            ('_jdefined', nb.boolean),
            ('_nedefined', nb.boolean),
            ('_nidefined', nb.boolean),
            ('_Tidefined', nb.boolean),
            ('_Tedefined', nb.boolean),
            ('_qdefined', nb.boolean),
            ('_bmaxdefined', nb.boolean),
            ('_bavgdefined', nb.boolean),
            ('_jextprof', nb.boolean),
            ('_neextprof', nb.boolean),
            ('_niextprof', nb.boolean),
            ('_Tiextprof', nb.boolean),
            ('_Teextprof', nb.boolean),
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
            ('extprof_j', nb.float64[:]),
            ('extprof_ne', nb.float64[:]),
            ('extprof_ni', nb.float64[:]),
            ('extprof_Te', nb.float64[:]),
            ('extprof_Ti', nb.float64[:]),
            ('extprof_q', nb.float64[:]),
            ('bmaxprof', nb.float64[:]),
            ('bavgprof', nb.float64[:]),
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
            ('n20_alpha', nb.float64)
          ]) # type: ignore
class POPCON_params:
    """
    Physical parameters for the POPCON.
    """
    def __init__(self) -> None:

        #---------------------------------------------------------------
        # Machine parameters
        #---------------------------------------------------------------

        self.R: float
        self.a: float
        self.kappa: float
        self.delta: float
        self.B0: float
        self.Ip: float
        self.q_a: float
        self.M_i: float
        self.f_LH: float
        # self.nipeak_over_nepeak: float
        self.tipeak_over_tepeak: float
        # 1 = D-D, 2 = D-T, 3 = D-He3
        self.fuel: int


        #---------------------------------------------------------------
        # Geometry/profile parameters
        #---------------------------------------------------------------
        self.sqrtpsin = np.empty(0, dtype=np.float64)
        self.volgrid = np.empty(0,dtype=np.float64)
        self.nr: int
        # 0 = He, 1 = Ne, 2 = Ar, 3 = Kr, 4 = Xe, 5 = W
        self.impurityfractions = np.empty(6, dtype=np.float64)
        # self.imcharges = np.empty(6, dtype=np.float64)
        self.extprof_geoms: bool
        self.geomsdefined: bool = False
        self.rdefined: bool = False
        self.volgriddefined: bool = False

        # Impurity profiles are not implemented yet
        self.extprof_imps: bool = False
        self.impsdefined: bool = False

        self._jdefined: bool = False
        self._nedefined: bool = False
        self._nidefined: bool = False
        self._Tidefined: bool = False
        self._Tedefined: bool = False
        self._qdefined: bool = False
        self._bmaxdefined: bool = False
        self._bavgdefined: bool = False

        #---------------------------------------------------------------
        # Whether to use external or parabolic profiles of each type
        #---------------------------------------------------------------

        self._jextprof: bool
        self._neextprof: bool
        self._niextprof: bool
        self._Tiextprof: bool
        self._Teextprof: bool

        #---------------------------------------------------------------
        # Parameters for parabolic profiles case
        #---------------------------------------------------------------

        self.j_alpha1: float = 1.
        self.j_alpha2: float = 2.
        self.j_offset: float = 0.

        self.ne_alpha1: float = 1.
        self.ne_alpha2: float = 2.
        self.ne_offset: float = 0.

        self.ni_alpha1: float = 1.
        self.ni_alpha2: float = 2.
        self.ni_offset: float = 0.

        self.Ti_alpha1: float = 1.
        self.Ti_alpha2: float = 2.
        self.Ti_offset: float = 0.

        self.Te_alpha1: float = 1.
        self.Te_alpha2: float = 2.
        self.Te_offset: float = 0.

        #---------------------------------------------------------------
        # External profiles
        #---------------------------------------------------------------

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

        #---------------------------------------------------------------
        # Scaling law parameters
        #---------------------------------------------------------------

        self.H_fac: float
        self.scaling_const: float
        self.M_i_alpha: float
        self.Ip_alpha: float
        self.R_alpha: float
        self.a_alpha: float
        self.kappa_alpha: float
        self.B0_alpha: float
        self.Pheat_alpha: float
        self.n20_alpha: float

        pass

    #-------------------------------------------------------------------
    # Properties
    # TODO: Impurity profiles
    #-------------------------------------------------------------------

    # BS from Martin H Mode Scaling (Martin et al J. Phys 2008) TODO: Cite
    # Update from kikuchi?
    @property
    def bs_factor(self):
        return self.B0**(0.8)*(2.*np.pi*self.R * 2*np.pi*self.a * np.sqrt((self.kappa**2+1)/2))**(0.94)
    
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
            return np.interp(rho, self.sqrtpsin, self.sqrtpsin)
        elif profid == -1:
            if not self.volgriddefined:
                raise ValueError("Volume grid not defined.")
            return np.interp(rho, self.sqrtpsin, self.volgrid)
        elif profid == 0:
            if not self._jdefined:
                raise ValueError("J profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extprof_j)
        elif profid == 1:
            if not self._nedefined:
                raise ValueError("n_e profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extprof_ne)
        elif profid == 2:
            if not self._nidefined:
                raise ValueError("n_i profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extprof_ni)
        elif profid == 3:
            if not self._Tidefined:
                raise ValueError("Ti profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extprof_Ti)
        elif profid == 4:
            if not self._Tedefined:
                raise ValueError("Te profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extprof_Te)
        elif profid == 5:
            if not self._qdefined:
                raise ValueError("q profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.extprof_q)
        elif profid == 6:
            if not self._bmaxdefined:
                raise ValueError("bmax profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.bmaxprof)
        elif profid == 7:
            if not self._bavgdefined:
                raise ValueError("bavg profile not defined.")
            return np.interp(rho, self.sqrtpsin, self.bavgprof)
        else:
            raise ValueError("Invalid profile ID.")
    
    def volume_integral(self, rho, profile):
        # NOTE: profile must be an array of the same length as rho.
        # Integrates functions of rho dV-like
        V_interp = np.interp(rho, self.sqrtpsin, self.volgrid)
        return np.trapz(profile, V_interp)
    
    #-------------------------------------------------------------------
    # Physical Quantities
    #-------------------------------------------------------------------
    
    # Z_eff, effective ion charge
    def Zeff(self, T_e:float):
        # Zeff = sum ( n_i * Z_i^2 ) / n_e
        # n_i = n20 * species fraction
        # n_e = n20 / dilution
        # n20 cancels

        # Hydrogen isotopes + impurities
        return ( (1-np.sum(self.impurityfractions)) + np.sum(self.impurityfractions*phys.get_Zeffs(T_e)**2))*self.get_plasma_dilution(T_e)

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

    # def get_average_plasma_dilution(self, T0):
    #     # n_e = n20 / dilution
    #     T_e_r = T0*self.get_extprof(self.sqrtpsin, 4)/self.tipeak_over_tepeak
    #     dil = np.empty_like(T_e_r)
    #     for i in np.arange(T_e_r.shape[0]):
    #         dil[i] = 1/(1 + np.sum(self.impurityfractions*phys.get_Zeffs(T_e_r[i])))

    #     return self.volume_integral(self.sqrtpsin, dil)/self.volume_integral(self.sqrtpsin, np.ones_like(dil))
    
    def get_plasma_dilution(self, T_e):
        dil = 1/(1 + np.sum(self.impurityfractions*phys.get_Zeffs(T_e)))
        return dil

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
        # TODO: Enforce rho > 0 to avoid division by zero
        if np.any(rho <= 0):
            raise ValueError("Invalid rho value. Neoclassical resistivity not defined at rho=0.")
        T_e_r = T0*self.get_extprof(rho, 4)/self.tipeak_over_tepeak
        n_i_r = 1e20*n20*self.get_extprof(rho, 2)
        dil = np.empty_like(n_i_r)
        for i in np.arange(T_e_r.shape[0]):
            dil[i] = self.get_plasma_dilution(T_e_r[i])
        n_e_r = 1e20*n20*self.get_extprof(rho, 1)/dil
        q = self.get_extprof(rho, 5)
        Zeffprof = np.empty_like(T_e_r)
        for i in np.arange(T_e_r.shape[0]):
            Zeffprof[i] = self.Zeff(T_e_r[i])
        logLambda = 17.1-np.log(np.sqrt(n_e_r)/(T_e_r*1e3))
        eta_C = 1.03e-4 * logLambda * T_e_r**(-3/2)

        Lambda_E = 3.4/Zeffprof * (1.13 + Zeffprof) / (2.67 + Zeffprof)
        C_R = 0.56/Zeffprof * (3.0 - Zeffprof) / (3.0 + Zeffprof)
        xi = 0.58 + 0.2*Zeffprof
        invaspect = self.a/self.R
        f_t = np.sqrt(2*rho*invaspect) # TODO: Replace with Jardin formula

        nu_star_e = 1/10.2e16 * self.R * q * n_e_r * np.exp(logLambda) / (f_t * invaspect * T_e_r**2)

        eta_C_eta_NC_ratio = Lambda_E * ( 1 - f_t/( 1 + xi * nu_star_e ) )*( 1 - C_R*f_t/( 1 + xi * nu_star_e ) )

        eta_NC = eta_C / eta_C_eta_NC_ratio
        return eta_NC
    
    def tauE_scalinglaw(self, P, n20):
        tauE = self.H_fac*self.scaling_const
        tauE *= self.M_i**self.M_i_alpha
        tauE *= self.Ip**self.Ip_alpha
        tauE *= self.R**self.R_alpha
        tauE *= self.a**self.a_alpha
        tauE *= self.kappa**self.kappa_alpha
        tauE *= self.B0**self.B0_alpha
        tauE *= P**self.Pheat_alpha
        tauE *= n20**self.n20_alpha
        return tauE
    
    def tauE_H98(self, P, n20):
        # H98y2 scaling law
        tauE = 0.145
        #M_i**(0.19)*Ip**(0.93)*R**(1.39)*a**(0.58)*kappa**(0.78)*B0**(0.15)*Pheat**(-0.69)*n20**(0.41)
        tauE *= self.M_i**0.19
        tauE *= self.Ip**0.93
        tauE *= self.R**1.39
        tauE *= self.a**0.58
        tauE *= self.kappa**0.78
        tauE *= self.B0**0.15
        tauE *= P**(-0.69)
        tauE *= n20**0.41
        return tauE
    
    def tauE_H89(self, P, n20):
        # H89 scaling law
        #0.048*M_i**(0.5)*Ip**(0.85)*R**(1.2)*a**(0.3)*kappa**(0.5)*B0**(0.2)*Pheat**(-0.5)*n20**(0.1)
        tauE = 0.048
        tauE *= self.M_i**0.5
        tauE *= self.Ip**0.85
        tauE *= self.R**1.2
        tauE *= self.a**0.3
        tauE *= self.kappa**0.5
        tauE *= self.B0**0.2
        tauE *= P**(-0.5)
        tauE *= n20**0.1
        return tauE

    #-------------------------------------------------------------------
    # Power profiles
    #-------------------------------------------------------------------

    def _W_tot_prof(self, rho, T0, n20):
        """
        Plasma energy per cubic meter
        """
        n_i_r = 1e20*n20*self.get_extprof(rho, 2)
        T_i_r = T0*self.get_extprof(rho, 3)
        # W_density = 3/2 * n_i * T_i
        # = 3/2 * (1e20 * n_i_r) ( 1.60218e-22 * T_i_r (keV) ) (MJ/m^3)
        return 3/2 * 1.60218e-22 * n_i_r * T_i_r
    
    def _P_DDpT_prof(self, rho, T0, n20):
        """
        D(d,p)T power per cubic meter
        """
        n_i_r = 1e20*n20*self.get_extprof(rho, 2)
        T_i_r = T0*self.get_extprof(rho, 3)

        # reaction frequency f = n_D/sqrt(2) * n_D/sqrt(2) * <sigma v> (MW/m^3)
        # 1.60218e-22 is the conversion factor from keV to MJ
        # <sigma v> is in cm^3/s so we need to convert to m^3/s, hence 1e-6
        # P_DD_density = 1.60218e-22 * f(DD->pT) * (T_tritium + T_proton)  (MW/m^3)

        f_ddpt = np.power( ( n_i_r / (2*np.sqrt(2)) ), 2) * phys.get_reactivity(T_i_r,2) * 1e-6
        return 1.60218e-22 * f_ddpt * (1.01e3 + 3.02e3)
    
    def _P_DDnHe3_prof(self, rho, T0, n20):
        """
        D(d,n)He3 power per cubic meter
        """
        n_i_r = 1e20*n20*self.get_extprof(rho, 2)
        T_i_r = T0*self.get_extprof(rho, 3)

        # See _P_DDpT_prof for explanation.
        # Note, for heating, only the He3 heats the plasma. For heating purposes,
        # multiply by 0.82e3/(2.45e3 + 0.82e3).

        f_ddnhe3 = np.power( (  n_i_r / (2*np.sqrt(2)) ), 2) * phys.get_reactivity(T_i_r,3) * 1e-6
        return 1.60218e-22 * f_ddnhe3 * (2.45e3 + 0.82e3)
    
    def _P_DTnHe4_prof(self, rho, T0, n20):
        """
        T(d,n)He4 power per cubic meter
        """
        n_i_r = 1e20*n20*self.get_extprof(rho, 2)
        T_i_r = T0*self.get_extprof(rho, 3)

        # See _P_DDpT_prof for explanation.
        # Note, for heating, only the He4 / alpha heats the plasma. For heating purposes,
        # multiply by 3.52e3/(3.52e3 + 14.06e3).

        f_dtnhe4 = (n_i_r)/2 * (n_i_r)/2 * phys.get_reactivity(T_i_r,1) * 1e-6
        return 1.60218e-22 * f_dtnhe4 * (3.52e3 + 14.06e3)
    
    def _P_fusion_heating(self, rho, T0, n20):
        """
        D-D and D-T heating power per cubic meter
        """
        return self._P_DDpT_prof(rho, T0, n20) + \
              self._P_DDnHe3_prof(rho,T0,n20)*(0.82e3/(2.45e3+0.82e3))+\
              self._P_DTnHe4_prof(rho,T0,n20)*(3.52e3/(3.52e3+14.06e3))
    
    def _P_rad(self, rho, T0, n20):
        """
        Radiative power per cubic meter. Zohm 2019.
        """

        T_i_r = T0*self.get_extprof(rho, 3)
        T_e_r = T0/self.tipeak_over_tepeak*self.get_extprof(rho, 4)
        n_i_r = 1e20*n20*self.get_extprof(rho, 2)
        dil = np.empty_like(n_i_r)
        for i in np.arange(n_i_r.shape[0]):
            dil[i] = self.get_plasma_dilution(T_i_r[i])
        n_e_r = 1e20*n20*self.get_extprof(rho, 1)/dil

        Lz = np.empty((T_e_r.shape[0],6),dtype=np.float64)
        for i in np.arange(T_e_r.shape[0]):
            Lz[i,:] = phys.get_rads(T_e_r[i])

        total_Zeff = np.empty(T_e_r.shape[0],dtype=np.float64)
        for i in np.arange(T_e_r.shape[0]):
            total_Zeff[i] = self.Zeff(T_e_r[i])
        G = 1.1 # Gaunt factor
        # P_brem = 5.35e-3*1e-6*G*total_Zeff*(1e-20*n_e_r)**2*T_e_r**0.5
        # From Plasma Formulary
        P_brem = G*1e-6*np.sqrt(1000*T_e_r)*total_Zeff*(n_e_r/7.69e18)**2
        P_line = np.sum(1e-6*(Lz.T*(n_e_r)**2).T*self.impurityfractions,axis=1)
        P_synchotron = 1.32e-7 * (self.B0 * T_i_r)**(2.5) * (n_i_r/1e20/self.a)**0.5 * (1+18*self.a/(self.R*T_i_r**0.5))
        return P_brem + P_line + P_synchotron
    
    def _P_OH_avg(self, n20, T0):# -> Any: #ohmic heating power density calculation, assuming constant current TODO: Rewrite
        T_e_keV = T0/self.tipeak_over_tepeak
        loglam = 24-np.log(np.sqrt(n20*1.0e20))/(T_e_keV*1e3)
        Pd = self.get_Cspitz(True, T_e_keV) / (T_e_keV*1e3)**(1.5)*0.016*loglam/15.0
        #loglam = 24-np.log(np.sqrt(self._nfun(rho,n20)*1.0e20)/(self._Tfun(rho,Te)*1e3))
        #Pd = self._Cspitzfun(rho,Te,impfrac) / (self._Tfun(rho,Te)*1e3)**(1.5)*0.016*loglam/15.0
        return(Pd)
    
    def _P_OH_prof(self, rho, T0, n20):
        """
        Ohmic power per cubic meter
        """

        eta_NC = self.get_eta_NC(rho, T0, n20)
        J = self.Ip*self.get_extprof(rho, 0)
        return eta_NC*J**2

    #-------------------------------------------------------------------
    # Power Balance Relaxation Solvers
    #-------------------------------------------------------------------

    def P_aux_relax_impfrac(self, n20, Ti, accel=1., err=1e-5, max_iters=1000):
        """
        Relaxation solver for holding impfrac constant.
        """

        P_aux_iter = 0.0001
        dPaux: float
        for ii in np.arange(max_iters):
            # Power in
            P_fusion_heating_iter = self.volume_integral(self.sqrtpsin, self._P_fusion_heating(self.sqrtpsin, Ti, n20))
            P_ohmic_heating_iter = self.volume_integral(self.sqrtpsin, self._P_OH_prof(self.sqrtpsin, Ti, n20))
            P_totalheating_iter = P_aux_iter + P_ohmic_heating_iter + P_fusion_heating_iter
            # Power out
            P_rad_iter = self.volume_integral(self.sqrtpsin, self._P_rad(self.sqrtpsin, Ti, n20))
            W_tot_iter = self.volume_integral(self.sqrtpsin, self._W_tot_prof(self.sqrtpsin, Ti, n20))
            tauE_iter = self.tauE_scalinglaw(P_totalheating_iter, n20)
            P_confinement_loss_iter = W_tot_iter/tauE_iter
            P_totalloss_iter = P_rad_iter + P_confinement_loss_iter

            # Power balance
            dPaux = P_totalloss_iter - P_totalheating_iter
            P_aux_iter += accel*dPaux

            if P_aux_iter < 1.0:
                if np.abs(dPaux/1.0) < err:
                    break
            else:
                if np.abs(dPaux/P_aux_iter) < err:
                    break
            
            if ii == max_iters-1:
                raise Warning(f"Power balance relaxation solver did not converge in {max_iters} iterations.")

        return P_aux_iter, P_totalheating_iter, P_totalloss_iter
    #-------------------------------------------------------------------
    # Setup functions
    #-------------------------------------------------------------------

    def _addextprof(self, extprofvals, profid):
        if profid == -2:
            if self.rdefined:
                raise ValueError("Geometry profiles already defined.")
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
    
    def _addextprof_imps(self, extprofvals):
        # TODO: implement this
        pass

    def _setup_profs(self):
        rho = self.sqrtpsin
        if not self.rdefined:
            raise ValueError("Geometry profile not defined.")
        if not self.volgriddefined:
            epsilon = rho*self.a/self.R
            w07 = 1.0
            S_phi = np.pi*((rho*self.a)**2)*self.kappa*(1+0.52*(w07-1))
            self.volgrid = 2*np.pi*self.R*(1-0.25*self.delta*epsilon)*S_phi
            self.volgriddefined = True
        if not self._jdefined:
            self.extprof_j = (1-self.j_offset)*(1-rho**self.j_alpha1)**self.j_alpha2 + self.j_offset
            self._jdefined = True
            self._jextprof = False
        if not self._nedefined:
            self.extprof_ne = (1-self.ne_offset)*(1-rho**self.ne_alpha1)**self.ne_alpha2 + self.ne_offset
            self._nedefined = True
            self._neextprof = False
        if not self._nidefined:
            self.extprof_ni = (1-self.ni_offset)*(1-rho**self.ni_alpha1)**self.ni_alpha2 + self.ni_offset
            self._nidefined = True
            self._niextprof = False
        if not self._Tidefined:
            self.extprof_Ti = (1-self.Ti_offset)*(1-rho**self.Ti_alpha1)**self.Ti_alpha2 + self.Ti_offset
            self._Tidefined = True
            self._Tiextprof = False
        if not self._Tedefined:
            self.extprof_Te = (1-self.Te_offset)*(1-rho**self.Te_alpha1)**self.Te_alpha2 + self.Te_offset
            self._Tedefined = True
            self._Teextprof = False
        if not self._qdefined:
            self.q_a = 2*np.pi*self.a**2*self.B0*(self.kappa**2+1)/(2*self.R*(4e-7*np.pi)*self.Ip*1e6)
            self.extprof_q = 2*np.pi*(self.a*rho)**2*self.B0*(self.kappa**2+1)/(2*self.R*(4e-7*np.pi)*self.Ip*1e6)
            self._qdefined = True
        if not self._bmaxdefined:
            self.bmaxprof = np.sqrt((self.B0*self.R/(self.R-rho*self.a))**2 + ((4e-7*np.pi)*self.Ip*1e6/(2*np.pi*self.a*rho))**2)
            self._bmaxdefined = True
        if not self._bavgdefined:
            self.bavgprof = np.sqrt((self.B0)**2 + ((4e-7*np.pi)*self.Ip*1e6/(2*np.pi*self.a*rho))**2)
            self._bavgdefined = True
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