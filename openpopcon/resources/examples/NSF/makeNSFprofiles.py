# Define profiles.csv from parameters alpha1, alpha2, such that
# f = (f_center-f_edge)*(1-rho**alpha1)**alpha2 + f_edge

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_frac_edge = 0.4 # = n_edge/n_0
n0 = 4.48e20 # m^-3
n_edge = n_frac_edge * n0
T0 = 5e3 #eV
T_edge = 0.1e3 #eV

alpha1_T = 1.79252435
alpha1_n = 2.57629833
alpha2_T = 1.97246825
alpha2_n = 3.6968161

tite = 1

N = 201
rho = np.linspace(0,1,N)
n_e = (n0-n_edge)*(1-rho**alpha1_n)**alpha2_n + n_edge
n_i = n_e
T_e = (T0-T_edge)*(1-rho**alpha1_T)**alpha2_T + T_edge
T_i = T_e * tite

if False:
    plt.figure()
    plt.plot(rho,T_e)
    plt.show()

data = {'n_e':n_e,'n_i':n_i,'T_i':T_i,'T_e':T_e,'rho':rho}
df = pd.DataFrame(data)
df.to_csv("./resources/examples/NSF/pNSF.csv", index=False)
