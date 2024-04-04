import numpy as np
import matplotlib.pyplot as plt
import get_Zeff

# Settings
#species = "Helium"
#ymin = 0
#ymax = 4

#species = "Neon"
#ymin = 0
#ymax = 11

#species = "Argon"
#ymin = 0
#ymax = 20

#species = "Krypton"
#ymin = 0
#ymax = 36

species = "Xenon"
ymin = 0
ymax = 54


# Temperature
T_min_exp = -1
T_max_exp = 2
Nr_elems_array = 1000
T_array = np.logspace(T_min_exp, T_max_exp, Nr_elems_array)

# Get Zeff
Zeff_array = np.zeros(Nr_elems_array)
for i in np.arange(Nr_elems_array):
	Zeff_array[i] = get_Zeff.get_Zeff(species, T_array[i])


# Plot
plt.semilogx(T_array, Zeff_array)
plt.xlim([10**T_min_exp, 10**T_max_exp])
plt.ylim([ymin, ymax])
plt.xlabel(r"$T_e$ [keV]")
plt.ylabel(r"$Z_{imp, eff}$")
plt.grid()
plt.show()
