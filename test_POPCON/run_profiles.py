import sys
import numpy as np

sys.path.append("../../")

import Open_POPCON.openpopcon as pc

# Initialise POPCON
mypops = pc.POPCON()

# Adjust input parameters
mypops.matlen = 1
mypops.timin = 18
mypops.timax = 20
mypops.nmin_frac = 0.5
mypops.nmax_frac = 0.6

mypops.Ip = 17.4 

#mypops.fixed_quantity = "impfrac"

#mypops.fixed_quantity = "f_LH"

mypops.fixed_quantity = "Psol"
mypops.plot_Psol = True
mypops.Psol_target = 20


# Run
mypops.make_popcons()

# Plot
mypops.plot_popcons()
