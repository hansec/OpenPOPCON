# -*- coding: utf-8 -*-
"""
Created on Fri Oct 02 16:52:13 2020

Scans opencopcom over user-defined parameters, saving the output figures in a
folder specified by <outdir>. This script is all user-editable. All calculations
are done externally in openpopcon.py

Denpendencies:
    -sys
    -numpy
    -scipy
    -os
    -openpopcon.py

@author: sfrnk, rnies, trubin, nelsonand
"""

import sys
import numpy as np
import os
#os.chdir('../../')
sys.path.append(os.getcwd())
sys.path.append('../../')
import Open_POPCON.openpopcon as pc

# Initialise POPCON
mypops = pc.POPCON()

# Adjust input parameters that won't be scanned
mypops.a        = 1.15      # minor radius (m)
mypops.kappa    = 2         # elongation
mypops.B0       = 12.3      # B_toroidal on axis (T)
mypops.imcharg  = 54        # Impurity charge
#mypops.R        = 4.0       # major radius (m)
mypops.f_LH     = 0.2

mypops.timin    = 2         # min temperature, keV
mypops.timax    = 40        # max temperature, keV
mypops.matlen   = 40        # number of n, T points for which POPCON solver is run

mypops.Psol_target = 20
mypops.volavgcurr = True
mypops.fixed_quantity = "Psol"

# plotting controls
mypops.plot_Paux      = True
mypops.plot_Pfus      = True
mypops.plot_Q         = True
mypops.plot_Prol      = True
mypops.plot_Psol      = True
mypops.plot_Pohm      = True
mypops.plot_impf      = True
mypops.plot_f_LH      = True
mypops.plot_Vloop     = True
mypops.plot_Pload_fus = True
mypops.plot_Pload_rad = True

# Adjust input parameters that will be scanned and run openpopcon
for Ip in [18]:
    for H in [1.0, 1.5]:
        for R in [4.0]:
            mypops.Ip       = Ip        # plasma current
            mypops.H        = H         # H Factor
            mypops.R        = R         # major radius

            # Assign where to save the figures to
            if os.path.isdir(os.getcwd() + '/my_popcon_scans/')==False:
                os.mkdir(os.getcwd() + '/my_popcon_scans/')
            savedir = os.getcwd() + '/my_popcon_scans/'
            # prefered naming convention:
            # [activity]_[year]_[month]_[day]_[identifier].pdf
            # Ex: 'archpcscan_02_10_2020_Ip{}.R{}.pdf'.format(Ip,R)
            plotfilename = 'apcs_2020_10_13_Ip{}-H{}-R{}'.format(Ip,H,R)
            mypops.plotfilename = savedir + plotfilename + '.pdf'
            mypops.datafilename = savedir + plotfilename + '.p'

            # Run
            mypops.make_popcons()

            # Save plot
            mypops.save_popconplot()

            # Save metadata
            mypops.save_popcondata()

            # mypops._process_inputs()
            # mypops._print_parameters()
