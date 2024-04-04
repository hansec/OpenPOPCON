import numpy as np
import matplotlib.pyplot as plt
import get_Lrad

# Settings
#species = "Helium"
#ymin = 10**(-36)
#ymax = 10**(-34)
#
#species = "Neon"
#ymin = 10**(-34)
#ymax = 10**(-32)
#
#species = "Argon"
#ymin = 10**(-33)
#ymax = 10**(-31)
#
#species = "Krypton"
#ymin = 3*10**(-33)
#ymax = 3*10**(-31)

species = "Xenon"
ymin = 10**(-32)
ymax = 10**(-30)


# Temperature
T_min_exp = -1
T_max_exp = 2
Nr_elems_array = 1000
T_array = np.logspace(T_min_exp, T_max_exp, Nr_elems_array)

# Get Lrad
Lrad_array = np.zeros(Nr_elems_array)
for i in np.arange(Nr_elems_array):
	Lrad_array[i] = get_Lrad.get_Lrad(species, T_array[i])


# Plot
plt.loglog(T_array, Lrad_array)
plt.xlim([10**T_min_exp, 10**T_max_exp])
plt.ylim([ymin, ymax])
plt.xlabel(r"$T_e$ [keV]")
plt.ylabel(r"$L_z$ [Wm$^3$]")
plt.grid()
plt.show()
