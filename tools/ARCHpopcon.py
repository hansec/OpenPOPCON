import sys
import numpy as np
import os

sys.path.append(os.getcwd())
sys.path.append("../../")
import Open_POPCON.openpopcon as pc

# Initialise POPCON
mypops = pc.POPCON()

mypops.matlen = 75
mypops.impname = "Krypton"
mypops.imcharg = 34
mypops.timin = 2
mypops.Ip = 14
mypops.R = 4.2
mypops.B0 = 11.5
mypops.a = 1.2
mypops.kappa = 1.8
mypops.Talpha1 = 2.0
mypops.Talpha2 = 1.5
mypops.H       = 1.3
mypops.n_frac_edge = 0.35 # = n_edge/n_center
mypops.nmax_frac = 1.4 # fraction of nG to scan too
mypops.nalpha1 = 2.0
mypops.nalpha2 = 1.1
mypops.volavgcurr = True
mypops.fixed_quantity = "Psol"
mypops.Psol_target = 80

mypops.matlen   = 40

if os.path.isdir(os.getcwd() + '/my_popcon_scans/')==False:
    os.mkdir(os.getcwd() + '/my_popcon_scans/')
savedir = os.getcwd() + '/my_popcon_scans/'
# prefered naming convention:
# [activity]_[year]_[month]_[day]_[identifier].pdf
# Ex: 'archpcscan_02_10_2020_Ip{}.R{}.pdf'.format(Ip,R)
plotfilename = 'ARCHpcbase'
mypops.plotfilename = savedir + plotfilename + '.pdf'
mypops.datafilename = savedir + plotfilename + '.p'

# Run
mypops.make_popcons()

# Save plot
mypops.save_popconplot()

# Save metadata
mypops.save_popcondata()
