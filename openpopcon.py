# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:43:11 2020

OpenPOPCON
Sam Frank's POPCon Routine based on GENC POPCON Routine From Ian Hutchinson's
22.62 course. This is an object oriented routine meant to eventually be
integrated into a more advanced modeling framework developed for the 22.63
course.

Denpendencies:
    -sys
    -os
    -numpy
    -scipy

@author: sfrnk, rnies, trubin, nelsonand
"""

import sys
import os

sys.path.append("../")
sys.path.append(os.getcwd())

import numpy as np
import scipy.constants as cnst
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import Reactivity_CrossSection.FusionCrossSectionHaleBosch as halebosch
import Lrad_coefficients.get_Lrad as lrad
import Lrad_coefficients.get_Zeff as zeff_imp
sys.path.insert(0, './Tools/')
#from progress_bar import printProgressBar
import matplotlib.patches as patches

class POPCON():
    def __init__(self):
        print("\n---------------------------------------------------------------")
        print("--- Welcome to OpenPOPCON, a tokamak parameter scoping code ---")
        print("---------------------------------------------------------------\n")

        #A bunch of default input parameters based on V0 ARCH scoping
        self.machinename = 'ARCH'
        self.plotfilename = 'OpenPOPCON_{}.pdf'.format(self.machinename)
        self.datafilename = 'OpenPOPCON_{}.p'.format(self.machinename)

        #baseline 0D Params
        self.R       = 3.7     #Major Radius (m)
        self.a       = 1.14    #Minor Radius (m)
        self.kappa   = 1.75    #elongation
        self.B0      = 12.3    #B_toroidal on axis (T)
        self.Ip      = 17.4    #Plasma Current (MA)
        self.H       = 1.15    #H Factor
        self.M_i     = 2.5     #Ion Mass
        self.fHe     = 0.02    #Helium Ash Fraction
        self.impfrac = 0.00046  #xenon 0.00046 kr 0.0011
        self.impname = "Xenon"
        self.f_LH    = 0.9     #target f_LH = P_sol/P_LH
        self.fixed_quantity = "impfrac" #impfrac, f_LH, P_SOL
        self.constant_impurity = True # If true, assume single value of Zeff and impfrac in plasma. TODO: implement profiles
        self.Psol_target = 20
        self.imcharg = 42 #34  # Impurity charge when constant_impurity = True

        #not used
        self.eta_enh = 1.0     #resistivity enhancement
        self.fle     = 0.5     #fractional loss to electron channel
        self.Fe      = 1.0     #Fraction of P_RF to electrons (100% for LHCD)
        self.fnc     = 0.0     #neoclassical ion losses

        #profile parameters alpha1, alpha2, such that
        # f = (f_center-f_edge)*(1-rho**alpha1)**alpha2 + f_edge
        self.T_edge  = 0.1 #keV
        self.Talpha1 = 2
        self.Talpha2 = 1.5
        self.n_frac_edge = 0.4 # = n_edge/n_center
        self.nalpha1 = 2
        self.nalpha2 = 1.5
        self.jalpha1 = self.Talpha1
        self.jalpha2 = 1.5 * self.Talpha2

        #debug/algorithm settings
        self.override   = False #set to True to allow violation of stability conditions
        self.volavgcurr = False #set to True to assume constant current
        self.maxit      = 150   #maximal number of iterations for solver
        self.relax      = 0.9   #relaxation parameter in power balance solver
        self.nmin_frac  = 0.05  #min density, as a fraction of n_GW
        self.nmax_frac  = 1     #max density, as a fraction of n_GW
        self.timin      = 1     #min temperature, keV
        self.timax      = 30    #max temperature, keV
        self.matlen     = 40    #number of n, T points for which POPCON solver is run

        #plotting settings
        self.plot_Paux   = True
        self.plot_Pfus   = True
        self.plot_Q      = True
        self.plot_Prol   = True
        self.plot_Psol   = True
        self.plot_Pohm   = True
        self.plot_impf   = True
        self.plot_f_LH   = True
        self.plot_Vloop  = True
        self.plot_Pload_fus = True
        self.plot_Pload_rad = True
        self.plot_BetaN  = True

    #Externally interfacing routines
    #--------------------------------------------------------------------------
    #Add a front end gui interface here
    #def popcon_frontend(self):

    def make_popcons(self):
    #Externally called routine that holds master logic for reading inputs
    #and making POPCONS
        #self._read_inputs()

        self._process_inputs()
        self._print_parameters()
        self._initialise_arrays(self.matlen)

        #set up list of Ti and n for which POPCON will evaluate power balance
        n_norm  = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
        print("n_norm is {}".format(n_norm))
        n20list = np.linspace(self.nmin_frac*self.n_g/n_norm, self.nmax_frac*self.n_g/n_norm, self.matlen)
        Tilist  = np.linspace(self.timin,self.timax,self.matlen)

        print("\n### Solve power balance at each (n,T)")
        for i in np.arange(self.matlen): #T_i loop
            Ti        =Tilist[i]
            self.xv[i]=Ti
            for j in np.arange(self.matlen): #n_e loop
                n20       =n20list[j]
                self.yv[j]=n20

                #compute normalised Prad for our impurity to quickly obtain impurity fraction from Prad
                Prad_norm = self.dvolfac*integrate.quad(self._P_rad,0,1,args=(n20,Ti,1, self.B0, self.a, self.R),limit=200)[0]

                #call solver
                [self.auxp[i,j],self.plmat[i,j],self.pamat[i,j], self.prmat[i,j],self.qpmat[i,j], self.pohmat[i,j]] \
                     = self._auxpowIt(n20,Ti,Prad_norm)

                #calculate some derived quantites from power balance
                #lh threshold
                self.flh[i,j]= (self.plmat[i,j]-self.prmat[i,j]) / get_P_LH_threshold(n20, self.bs)

                #impurity fraction
                self.impfrc[i,j] = self.prmat[i,j] / Prad_norm

                #Loop voltage
                self.Vloop[i,j] = self._vloopfun(n20, Ti, self.impfrc[i,j])

                # Normalize beta [%]
                Volfactor = 0.31 # Assumed shaping factor for density/temperature profile
                self.BetaN[i,j] = 4 * (4*np.pi*1e-7) * (n20*1e20) * (Ti*1000*1.6021e-19) \
                    * self.a * 100 / self.B0 / self.Ip * Volfactor

                #set ignition curve
                if(self.qpmat[i,j]>=10000.0 or self.qpmat[i,j]<0.0):
                    self.qpmat[i,j]=10000

                #debug prints
                if(self.matlen < 4):
                    print('For n20= ', n20,', Ti= ',Ti)
                    print('P_aux= ', self.auxp[i,j], ' P_alpha= ', self.pamat[i,j], \
                          'P_rad= ', self.prmat[i,j], ' P_ohm = ',self.pohmat[i,j])

            #progress bar for overview
            #printProgressBar(i+1, self.matlen, prefix = 'Progress:', length = 50)

    def plot_popcons(self): # plot the popcons
         plt.rcParams.update({'font.size': 18})
         if self.matlen > 3:
             self.plot_contours(ifshow=True)
         else:
             self.plot_profiles(ifshow=True)

    def save_popconplot(self): # save the plot without plotting it
         plt.rcParams.update({'font.size': 18})
         if self.matlen > 3:
             self.plot_contours(ifshow=False)
         else:
             self.plot_profiles(ifshow=False)
         self.fig.savefig(self.plotfilename)

    def save_popcondata(self): # print select quantities to the console
        data = {}
        n_norm = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
        xx, yy = np.meshgrid(self.xv, self.yv/self.n_g*n_norm)
        data['xx'] = xx
        data['yy'] = yy
        data['Paux'] = self.auxp
        data['Pfus'] = self.pamat*5.03
        data['Q'] = self.qpmat
        data['Prad_Ploss'] = self.prmat/self.plmat
        data['Psol'] = self.prmat-self.plmat
        data['Pohm'] = self.pohmat
        data['fLH'] = self.flh
        data['impfrac'] = self.impfrc
        data['Vloop'] = self.Vloop
        data['BetaN'] = self.BetaN
        data['n20'] = np.linspace(self.nmin_frac*self.n_g/n_norm, self.nmax_frac*self.n_g/n_norm, self.matlen)

        try:
            import cPickle as pickle
        except ImportError:  # Python 3.x
            import pickle

        with open(self.datafilename, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_contours(self, ifshow=True): # plot popcon contours
         n_norm = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
         xx, yy = np.meshgrid(self.xv, self.yv/self.n_g*n_norm)

         self.fig, ax = plt.subplots(figsize=(16,9))

         Pauxlevels = [0,1,3,5,7,10,15,20,25]
         Pfuslevels = [30,50,100,300,500,1000,2000,3000,4000,5000,7000,10000]
         Qlevels    = [1,10,100]
         Prad_oloss_levels = [0.1, 0.5, 0.9, 0.99]
         Psollevels = [10, 20, 30, 40]
         Pohmlevels = [1]
         flh_levels = [0.1, 0.2, 0.5]
         Vlooplevels = [1e-2, 1.3e-2, 1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2, 7e-2, 9e-2, 1.2e-1, 1.5e-1]
         Pload_fus_levels = ""
         Pload_rad_levels = ""
         BetaN_levels = [1,2,3]

         Igshade=plt.contourf(xx,yy,np.transpose(self.qpmat),     levels=[6000,10000],colors='r',alpha=0.25)

         plot_single_contour(self.plot_Paux,ax,   "$P_\mathrm{aux}$"       , xx,yy,self.auxp,      Pauxlevels,                   'r'              ,1.0,fmt='%1.0f')
         plot_single_contour(self.plot_Pfus,ax,   "$P_\mathrm{fus}$"       , xx,yy,(self.pamat*5.03),Pfuslevels,                 'k'              ,fmt='%1.0f')
         plot_single_contour(self.plot_Q   ,ax,   "Q"          , xx,yy,self.qpmat,     Qlevels   ,                   'lime'           ,fmt='%1.0f')
         plot_single_contour(self.plot_Prol,ax,   "$P_\mathrm{rad}/P_{loss}$" , xx,yy,self.prmat/self.plmat,Prad_oloss_levels,      'c'              ,1.0,fmt='%1.2f')
         plot_single_contour(self.plot_Psol,ax,   "$P_\mathrm{sol}$"       , xx,yy,self.prmat-self.plmat,Psollevels,             'darkviolet'     ,1.0,fmt='%1.0f')
         plot_single_contour(self.plot_Pohm,ax,   "$P_\mathrm{ohm}$"       , xx,yy,self.pohmat,Pohmlevels,                       'orange'         ,1.0,fmt='%1.0f')
         plot_single_contour(self.plot_f_LH,ax,   "$f_\mathrm{LH}$"       , xx,yy,self.flh,flh_levels,                          'forestgreen'    ,1.0,fmt='%1.2f')
         plot_single_contour(self.plot_impf,ax,   "$n_\mathrm{imp}/n_e$"    , xx,yy,self.impfrc,"",                               'magenta'        ,1.0,fmt='%0.1e')
         plot_single_contour(self.plot_Vloop, ax, "$V_\mathrm{loop}$"      , xx, yy, self.Vloop,Vlooplevels,                     'mediumvioletred', 1.0, fmt = '%0.2e')
         plot_single_contour(self.plot_BetaN, ax, "$\u03B2_\mathrm{N}$"      , xx, yy, self.BetaN,BetaN_levels,                     'darkslategrey', 1.0, fmt = '%0.2e')

         # Plot power loads (MW/m^2) on wall.
         # For area, use as first estimate plasma surface area
         area = 2*np.pi*self.a * np.sqrt((1+self.kappa**2)/2) * 2*np.pi*self.R
         plot_single_contour(self.plot_Pload_fus, ax, "$P_\mathrm{n}/A$"  , xx, yy, self.pamat*4/area, Pload_fus_levels, 'peru', 1.0, fmt = '%0.2e')
         plot_single_contour(self.plot_Pload_rad, ax, "$P_\mathrm{rad}/A$", xx, yy, self.prmat/area, Pload_rad_levels, 'dodgerblue', 1.0, fmt = '%0.2e')


         #plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
         ax.set_ylabel(r"$\bar{n} / n_{GR}$")
         ax.set_xlabel('$T^i_\mathrm{keV}$')
         ax.legend(bbox_to_anchor=(1, 1), loc='upper left',fontsize = 12)
         plt.tight_layout()
         if ifshow:
             plt.show()

    def plot_profiles(self, ifshow=True): # plot 1D profiles
        self.fig = plt.figure(figsize=(12,9))
        for i_temp in np.arange(self.matlen):
             T0 = self.xv[i_temp]
             for i_dens in np.arange(self.matlen):
                  n20 = self.yv[i_dens]

                  ax1 = plt.subplot(self.matlen, self.matlen, i_dens*self.matlen+i_temp+1)
                  #ax1.set_xticklabels([])
                  #ax1.set_yticklabels([])
                  plt.subplots_adjust(wspace=None, hspace=None)

                  plt.title(r"$n_0/n_{GR} =$ %.1f, $T^0_\mathrm{keV} =$ %.0f" % (n20/get_n_GR(self.Ip, self.a), T0), pad=-100)

                  rho = np.linspace(1e-10, 1, 100)
                  Prad = np.empty(len(rho))
                  Pfus = np.empty(len(rho))
                  Pohm = np.empty(len(rho))
                  for i_rho in np.arange(len(rho)):
                      Prad[i_rho] = self._P_rad(rho[i_rho], n20, T0, self.impfrc[i_temp, i_dens], self.B0, self.a, self.R)/rho[i_rho]
                      Pfus[i_rho] = self._P_DD(rho[i_rho], n20, T0, self.impfrc[i_temp, i_dens])/rho[i_rho] + self._P_alpha(rho[i_rho], n20, T0, self.impfrc[i_temp, i_dens])/rho[i_rho]
                      Pohm[i_rho] = self._P_OH(rho[i_rho], n20, T0, self.impfrc[i_temp,i_dens])/rho[i_rho]

                  plt.plot(rho, Prad, label="$P_\mathrm{rad}$")
                  plt.plot(rho, Pfus, label="$P_\mathrm{fus}$")
                  plt.plot(rho, Pohm, label="$P_\mathrm{ohm}$")
                  #plt.semilogy(rho, Prad, label="$P_\mathrm{rad}$")
                  #plt.semilogy(rho, Pfus, label="$P_\mathrm{fus}$")
                  #plt.semilogy(rho, Pohm, label="$P_\mathrm{ohm}$")

                  plt.grid()
                  plt.xlim([0,1])
#                  plt.ylim(ymin=Pfus[int(0.8*len(rho))])

                  if i_temp == 0:
                      plt.ylabel(r"$P$ in MW/m$^3$")
                  if i_dens == self.matlen-1:
                      plt.xlabel(r"$\rho$")
                  if i_dens == 0 and i_temp == 0:
                      plt.legend()

        if ifshow:
            plt.show()

    #Internal Routines
    #--------------------------------------------------------------------------
    #def _read_inputs(self):
    #front end to read inputs goes here

    def _process_inputs(self):
    #takes things that have been input and calculates some derived quantities
    #with them including:
    # - Plasma Volume (vol)
    # - Plasma Dilution (dil)
    # - Z Effective (Zeff)
    # - Greenwald Limit (n_g)
    # - Spitzer Resistivity Coefficient (Cspitz)
    # - B0*Surface Area (bs) with exponential scalings

        #plasma volume
        self.vol = get_plasma_vol(self.a, self.R, self.kappa)

	#volume of drho element for integration
        self.dvolfac = get_plasma_dvolfac(self.a, self.R, self.kappa)

        #plasma dilution
        self.dil = get_plasma_dilution(self.impfrac, self.imcharg, self.fHe)

        #Z effective, effective ion charge
        self.Zeff = get_Zeff(self.impfrac, self.imcharg, self.fHe, self.dil)

        #n_g, Greenwald density (10^20 m^-3)
        self.n_g = get_n_GR(self.Ip, self.a)

        #q_a edge safety factor
        self.q_a = get_q_a(self.a, self.B0, self.kappa, self.R, self.Ip)
        if (self.q_a <= 2.0 and self.override == False):
            sys.exit('q_a too low. External kink unstable')

        #spitzer resistivity coef for ohmic heating power
        self.Cspitz = get_Cspitz(self.Zeff, self.Ip, self.q_a, self.a, self.kappa, self.volavgcurr)

        #BS from Martin H Mode Scaling (Martin et al J. Phys 2008)
        self.bs = get_bs_factor(self.B0, self.R, self.a, self.kappa)

        # H89_base = H89/n20**(0.1) for convenience as n20 varies in POPCON
        self.H89_base = get_ITER_H89_base(self.H,self.M_i,self.Ip,self.R,self.a,self.kappa,self.B0)

        # Peak current density, from integral(j) = Ip
        self.j0 = self.Ip*1e6 / (2*np.pi*self.a**2*self.kappa) / integrate.quad(self._ijfun,0,1,args=(1))[0]


    #Iterative solvers for power balance.
    #--------------------------------------------------------------------------
    def _auxpowIt(self,n20,Ti,Prad_norm):
    #Iterate to get the aux power needed for power balance

        #Intialize local variables
        #relaxt  = 0.7
        accel   = 1.7
        pderror = 1e-5

        #Alpha power density in units of 10^20keV
        Ptalpha = self.dvolfac*integrate.quad(self._P_alpha,0,1,args=(n20,Ti,self.impfrac))[0]
        PtDD = self.dvolfac*integrate.quad(self._P_DD,0,1,args=(n20,Ti,self.impfrac))[0]
        #Total thermal energy in plasma
        Wtot = self.dvolfac*integrate.quad(self._Wtot,0,1,args=(Ti,n20,self.impfrac))[0]

        #Volume averaged Ptoh
        if (self.volavgcurr == True):
            navg = self.dvolfac*integrate.quad(self._infun,0,1,args=(n20))[0]/self.vol
            Tavg = self.dvolfac*integrate.quad(self._iTfun,0,1,args=(Ti))[0]/self.vol
            Ptoh = self.vol*self._P_OH_avg(navg,Tavg)
        else:
            Ptoh = self.dvolfac*integrate.quad(self._P_OH,0,1,args=(n20,Ti,self.impfrac))[0]

        # iterative solve for equilibrium solution to power balance
        Ph = 0.0
        for i in np.arange(self.maxit):
            #Total heating power
            Pheat = Ptoh+Ph+Ptalpha+PtDD

            #Radiated power
            if (self.fixed_quantity == "impfrac"):
                Prad = self.dvolfac*integrate.quad(self._P_rad,0,1,args=(n20,Ti,self.impfrac, self.B0, self.a, self.R))[0]

            elif (self.fixed_quantity == "f_LH" or self.fixed_quantity == "Psol"):
                if self.fixed_quantity == "f_LH":
                    # Set Prad to satisfy f_LH target specified, but keep > 0.
                    Prad = max(0, Pheat - self.f_LH * get_P_LH_threshold(n20, self.bs))

                elif self.fixed_quantity == "Psol":
                    # Set Prad to bring Psol < Psol_target
                    Prad = max(0, Pheat - self.Psol_target)

                # Infer impurity fraction necessary to get this Prad
                self.impfrac = Prad / Prad_norm
                # Update dilution and Zeff with the new impurity fraction
                self.dil = get_plasma_dilution(self.impfrac, self.imcharg, self.fHe)
                self.Zeff = get_Zeff(self.impfrac, self.imcharg, self.fHe, self.dil)

            else:
                sys.exit("Choose fixed quantity for radiation: impfrac, f_LH, Psol.")

            #Recalculate powers to account for changed dilution, Zeff, ...
            #TODO: CAN DO THIS MORE EFFECTIVELY BY MULTIPLYING PREVIOUS VALUE BY FACTOR TAKING DIL. & Zeff INTO ACCOUNT?
            Ptalpha = self.dvolfac*integrate.quad(self._P_alpha,0,1,args=(n20,Ti,self.impfrac))[0]
            PtDD = self.dvolfac*integrate.quad(self._P_DD,0,1,args=(n20,Ti,self.impfrac))[0]
            Wtot = self.dvolfac*integrate.quad(self._Wtot,0,1,args=(Ti,n20,self.impfrac))[0]
            if (self.volavgcurr == True):
                Ptoh = self.vol*self._P_OH_avg(navg,Tavg)
            else:
                Ptoh = self.dvolfac*integrate.quad(self._P_OH,0,1,args=(n20,Ti,self.impfrac))[0]

            #Total power
            Ptot  = Pheat-Prad

            #calculate confinement time and W/tau_E
            tauE  = self.H89_base*n20**(0.1)*Pheat**(-0.5)
            WoTau = Wtot/tauE

            # Adjust auxiliary power to get towards power balance W/tau_E = P_heat
            dPh = WoTau-Pheat

            #ensure self consistency in power balance
            if(Pheat+dPh-Prad < 0):
                dPh = Prad-Pheat

            # overrelaxation
            reld = abs(dPh/max(Ph,1.0))
            Ph   = Ph + ((reld*self.relax+0.2*accel)/(reld+0.2))*dPh

            # check for convergence
            if(abs(dPh)/max(abs(Ph),1.) <= pderror):
                break


        Paux   = Ph
        if(i+1 == self.maxit):
            print("Auxpow did not converge. Ph = %.3f, Ti = %.3f, n = %.3f" % (Paux, Ti, n20))

        Ploss  = Ptalpha+Paux+Ptoh + PtDD
        Q      = (5.03*Ptalpha+1.505*PtDD)/(Paux+Ptoh)
        outarr = [Paux,Ploss,Ptalpha+PtDD,Prad,Q,Ptoh]
        return(outarr)



    #Power Calculations, in units of MW/m^3
    #--------------------------------------------------------------------------
    def _P_OH(self,rho,n20,Te,impfrac): #ohmic heating power density calculation, assuming constant current
        Pd = rho * self._etafun(rho,n20,Te,impfrac) * self._jfun(rho,self.j0)**2 * 1e-6
        return(Pd)

    def _P_OH_avg(self,n20,Te): #ohmic heating power density calculation, assuming constant current
        loglam = 24-np.log(np.sqrt(n20)*1.0e20)/(Te*1e3)
        Pd = self.Cspitz / (Te*1e3)**(1.5)*0.016*loglam/15.0
        #loglam = 24-np.log(np.sqrt(self._nfun(rho,n20)*1.0e20)/(self._Tfun(rho,Te)*1e3))
        #Pd = self._Cspitzfun(rho,Te,impfrac) / (self._Tfun(rho,Te)*1e3)**(1.5)*0.016*loglam/15.0
        return(Pd)

    def _P_rad(self,rho,n20,Te,impfrac, B0, a, R):
#        P_brems = rho * 0.016 * (self._nfun(rho,n20))**2 * np.sqrt(self._Tfun(rho,Te))*0.331*self.Zeff

        P_line = rho * lrad.get_Lrad(self.impname, self._Tfun(rho,Te))/1e6 * (self._nfun(rho, n20)*1e20)**2 * impfrac

        # Calculates Cyclotron radiation from Zohm 2019, [MW/m^3]
        # Note: Assumes 80% wall reflectivity
        P_cyl = rho * 1.32e-7 * (B0 * self._Tfun(rho,Te))**(2.5) * (self._nfun(rho, n20)/a)**(0.5) \
            * (1+18*a/(R*self._Tfun(rho,Te)**0.5))

        P = P_line + P_cyl
        return(P)

    def _P_alpha(self,rho,n20,Ti,impfrac):
        #P = rho*0.016*3.5e3*1e20*self._dtrtcf(self._Tfun(rho,Ti)) \
        #    *(self._nfun(rho,n20)*self.dil)**2/4

        P = rho*0.016*3.5e3*1e20 * halebosch.reactivity('T(d,n)He4',self._Tfun(rho,Ti))*1e-6 \
            *(self._nfun(rho,n20)*self.dil)**2/4
            #*(self._nfun(rho,n20)*self._dilfun(rho,Ti,impfrac))**2/4
        return(P)

    def _P_DD(self,rho,n20,Ti,impfrac):
        P = rho*0.016*1e20 * ((1.01e3+3.02e3)*halebosch.reactivity('D(d,p)T',self._Tfun(rho,Ti))+0.82e3*halebosch.reactivity('D(d,n)He3',self._Tfun(rho,Ti)))*1e-6 \
            *(self._nfun(rho,n20)*self.dil)**2/8
        return(P)

    def _Wtot(self,rho,T0,n0,impfrac):
        # NOTE: dilution can have large effects on POPCON
        Wtot = rho*0.016*1.5*self._nfun(rho,n0)*(1+self.dil)*self._Tfun(rho,T0)
        #Wtot = rho*0.016*1.5*self._nfun(rho,n0)*(1+self._dilfun(rho,T0,impfrac))*self._Tfun(rho,T0)
        return(Wtot)

    #Simple functions for physical parameter calculations
    #--------------------------------------------------------------------------
#   def _dtrtcf(self,Ti):
   #DT reaction rate coefficient calculation from Hively, Nucl. Fus 17,873,1977
#       r  =  0.30366647
#       a1 = -20.779964
#       a2 = -39.629382
#       a3 = -0.066251351
#       a4 =  3.0934551e-4
#       result = np.exp(a1/Ti**r+a2+Ti*(a3+Ti*a4))
#       return(result)

    def _fionalp(self,Te):
    #calculates fraction of alpha energy to ions at electron temp Te
        Ei = 3500
        Tfac = 31.90
        twosqrt3 = 1.1547
        atsq3 = -0.52360

        x=Ei/(Tfac*Te)
        result = (-0.333*np.log((1.+np.sqrt(x))**2/(1.-np.sqrt(x)+x)) \
                  +twosqrt3*(np.arctan((np.sqrt(x)-0.5)*twosqrt3)-atsq3))/x
        return(result)

    #profile functions and differential elements of profile functions for vol.
    #integrals
    #--------------------------------------------------------------------------
    def _Tfun(self,rho,T0):
        result = (T0-self.T_edge)*(1-rho**self.Talpha1)**self.Talpha2 + self.T_edge
        return(result)

    def _iTfun(self,rho,T0):
        return(rho*self._Tfun(rho,T0))

    def _nfun(self,rho,n0):
        result = n0 * ( (1-self.n_frac_edge)*(1-rho**self.nalpha1)**self.nalpha2 + self.n_frac_edge)
        return(result)

    def _infun(self,rho,n0):
        return(rho*self._nfun(rho,n0))

    def _jfun(self,rho,j0):
        return j0 * (1-rho**self.jalpha1)**self.jalpha2

    def _ijfun(self,rho,j0):
        return(rho*self._jfun(rho,j0))

    def _imchargfun(self,rho,T0):
        return zeff_imp.get_Zeff(self.impname, self._Tfun(rho,T0))

    def _dilfun(self,rho,T0,impfrac):
        return get_plasma_dilution(self.impfrac, self._imchargfun(rho,T0), self.fHe)

    def _zefffun(self,rho,T0,impfrac):
        return get_Zeff(impfrac, self._imchargfun(rho,T0), self.fHe, self.dil)

    # Neoclassical resistivity from Jardin et al. 1993, in Ohm*m
    def _etafun(self,rho,n20,T0,impfrac):
        Zeff = self._zefffun(rho,T0,impfrac)
        Lambda = max(5, (self._Tfun(rho,T0)*1e3 / np.sqrt(self._nfun(rho,n20)*1e20)) *np.exp(17.1))
        Lambda_E = 3.40/Zeff * (1.13+Zeff)/(2.67+Zeff)
        C_R = 0.56/Zeff * (3.0-Zeff)/(3.0+Zeff)
        xi  = 0.58 + 0.20*Zeff
        # inverse aspect ratio
        epsilon = self.a/self.R
        # trapped particle fraction
        f_T = np.sqrt(2*rho*epsilon)           # TODO: correct for elongation? get something out of Jardin's integral formula?
        # safety factor
        q = rho * get_q_a(self.a, self.B0, self.kappa, self.R, self.Ip) # TODO: more accurate safety factor?
        # electron collisionality
        nu_star_e = 1./(10.2e16) * self.R * q * (self._nfun(rho,n20)*1e20) * Lambda / (f_T * epsilon * (self._Tfun(rho,T0)*1e3)**2)

        factor_NC = Lambda_E * (1.-f_T/(1.+xi*nu_star_e))*(1.-C_R*f_T/(1.+xi*nu_star_e))

        eta_C = 1.03e-4 * np.log(Lambda) * (self._Tfun(rho,T0)*1e3)**(-3/2)

        return eta_C / factor_NC



    def _initialise_arrays(self,matlen):
        #allocate solution arrays (<= dead giveaway Sam Frank writes too much FORTRAN)
        self.auxp   = np.empty([matlen,matlen])   #auxiliary power (RF, NBI, ...)
        self.pamat  = np.empty([matlen,matlen])   #alpha power
        self.prmat  = np.empty([matlen,matlen])   #radiated power
        self.tratio = np.empty([matlen,matlen])   #TODO: for future use, Ti != Te
        self.plmat  = np.empty([matlen,matlen])   #power loss, i.e. alpha + auxiliary powers
        self.qpmat  = np.empty([matlen,matlen])   #Q=P_fus/(P_aux+P_oh)
        self.pohmat = np.empty([matlen,matlen])   #Ohmic power
        self.beta   = np.empty([matlen,matlen])   #TODO: for future use, compute 2*mu0*p/B^2
        self.flh    = np.empty([matlen,matlen])   #LH fraction, i.e. (P_loss - P_rad) / P_{LH threshold}, varies if impfrac or P_SOL is set
        self.impfrc = np.empty([matlen,matlen])   #impurity fraction, varies e.g. if target f_LH or P_SOL is set.
        self.Vloop  = np.empty([matlen,matlen])   #Vloop
        self.BetaN  = np.empty([matlen,matlen])   #Normalized Beta

        self.xv     = np.empty(matlen)            #array to save T_i      vals for plotting
        self.yv     = np.empty(matlen)            #array to save <n>/n_GR vals for plotting

    def _vspecific(self, rho, n20, Te, impfrac):
        return self._etafun(rho,n20,Te,impfrac) * self._jfun(rho,self.j0)*rho

    def _vloopfun(self, n20, Te, impfrac):
        return self.dvolfac*integrate.quad(self._vspecific,0,1,args=(n20,Te,impfrac),limit=200)[0]/get_plasma_area(self.a,self.kappa)


    #output functions
    #--------------------------------------------------------------------------
    def _print_parameters(self):
        print("### POPCON parameters:")
        print('\n'.join("  %-15s: %s" % item for item in vars(self).items()))

#----------------------------------------------------------
# Compendium of useful functions

# power at which plasma transitions from L- to H-mode, from Martin H Mode Scaling (Martin et al J. Phys 2008)
def get_P_LH_threshold(n20, bs):
    return 0.049*n20**(0.72)*bs

# plasma volume
def get_plasma_vol(a, R, kappa):
    return 2.*np.pi**2*a**2*R*kappa

# plasma cross section area
def get_plasma_area(a, kappa):
    return np.pi*a**2*kappa

# volume of drho element, where rho=0 at magnetic axis and rho=1 at separatrix
def get_plasma_dvolfac(a, R, kappa):
    return 4*np.pi**2*a**2*R*kappa

# plasma dilution, TODO: arbitrary impurity cocktail
def get_plasma_dilution(impfrac, imcharg, fHe):
    return 1/(1 + impfrac*imcharg + 2*fHe)

#----------------------------------------------------------
# Compendium of useful functions

# power at which plasma transitions from L- to H-mode, from Martin H Mode Scaling (Martin et al J. Phys 2008)
def get_P_LH_threshold(n20, bs):
    return 0.049*n20**(0.72)*bs

# plasma volume
def get_plasma_vol(a, R, kappa):
    return 2.*np.pi**2*a**2*R*kappa

# volume of drho element, where rho=0 at magnetic axis and rho=1 at separatrix
def get_plasma_dvolfac(a, R, kappa):
    return 4*np.pi**2*a**2*R*kappa

# plasma dilution, TODO: arbitrary impurity cocktail
def get_plasma_dilution(impfrac, imcharg, fHe):
    return 1/(1 + impfrac*imcharg + 2*fHe)

# Z_eff, effective ion charge, TODO: arbitrary impurity cocktail
def get_Zeff(impfrac, imcharg, fHe, dil):
    return ( (1-impfrac-fHe) + impfrac*imcharg**2 + 4*fHe)*dil

# n_GR, Greenwald density in 10^20/m^3
def get_n_GR(Ip, a):
    return Ip/(np.pi*a**2)

# q_a, edge safety factor
# TODO: could eventually take this from an eqdsk calculation
def get_q_a(a, B0, kappa, R, Ip):
    return 2*np.pi*a**2*B0*(kappa**2+1)/(2*R*cnst.mu_0*Ip*10**6)

# coefficient for Spitzer conductivity, necessary to obtain ohmic power
def get_Cspitz(Zeff, Ip, q_a, a, kappa, volavgcurr):
    Fz    = (1+1.198*Zeff + 0.222*Zeff**2)/(1+2.966*Zeff + 0.753*Zeff**2)
    eta1  = 1.03e-4*Zeff*Fz
    j0avg = Ip/(np.pi*a**2*kappa)*1.0e6
    if (volavgcurr == True):
        Cspitz = eta1*q_a*j0avg**2
    else:
        Cspitz = eta1
    Cspitz /= 1.6e-16*1.0e20 #unit conversion to keV 10^20 m^-3
    return Cspitz

#BS from Martin H Mode Scaling (Martin et al J. Phys 2008)
def get_bs_factor(B0, R, a, kappa):
    return B0**(0.8)*(2.*np.pi*R * 2*np.pi*a * np.sqrt((kappa**2+1)/2))**(0.94)


# H89/n20**(0.1), for convenience as n20 varies in the POPCON
def get_ITER_H89_base(H_fac,M_i,Ip,R,a,kappa,B0):
    return 0.048*H_fac*M_i**(0.5)*Ip**(0.85)*R**(1.2)*a**(0.3)*kappa**(0.5)*B0**(0.2)

# Function to plot a contour
def plot_single_contour(plot_bool,ax,name,xx,yy,quantity,levels,color,linewidth=1.5,fmt='%1.3f',fontsize=10):
     if plot_bool:
          if levels == "":
              contour=ax.contour(xx,yy,np.transpose(quantity),colors=color,linewidths=linewidth)
              rect = patches.Rectangle((0,0),0,0,fc = color,label = name)
          else:
              contour=ax.contour(xx,yy,np.transpose(quantity),levels=levels,colors=color,linewidths=linewidth)
              rect = patches.Rectangle((0,0),0,0,fc = color,label = name)
          ax.add_patch(rect)
          ax.clabel(contour,inline=1,fmt=fmt,fontsize=fontsize)
