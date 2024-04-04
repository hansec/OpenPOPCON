import sys
import numpy as np
import os

sys.path.append("../../")
sys.path.append(os.getcwd()) 

import openpopcon as pc

# Initialise POPCON
mypops = pc.POPCON()

mypops.machinename = 'negD_reactor'
mypops.R = 9
mypops.a = 3
mypops.Ip = 21
mypops.B0 = 5.86
mypops.kappa = 2
mypops.BetaN = 2.1
mypops.H = 1.12
mypops.fHe = 0.05
mypops.impfrac = 0.002


# Adjust input parameters
mypops.matlen = 50
mypops.imcharg = 54
mypops.timin = 2
mypops.volavgcurr = True

#mypops.fixed_quantity = "impfrac"

#mypops.fixed_quantity = "f_LH"
mypops.f_LH = 0.2

mypops.fixed_quantity = "Psol"
mypops.Psol_target = 20

mypops.plot_Psol  = not (mypops.fixed_quantity == "Psol")
mypops.plot_impf  = not (mypops.fixed_quantity == "impfrac")
mypops.plot_f_LH  = not (mypops.fixed_quantity == "f_LH")

# Run
mypops.make_popcons()

# Plot
mypops.plot_popcons()

print(mypops.qpmat)
