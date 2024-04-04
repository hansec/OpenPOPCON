import numpy as np
import warnings

##############
# Function to obtain average charge state per impurity ion, from Mavrin 2018
# ... INPUT:  - species name [options: Helium, Neon, Argon, Krypton, Xenon]
#             - Electron temperature in keV [region of validity: 0.1 <= T_e_keV <= 100]
# ... OUTPUT: - Zeff
def get_Zeff(species, T_e_keV):
	# Formulas valid only for region of validity: 0.1 <= T_e_keV <= 100
	if T_e_keV < 0.1 or T_e_keV > 100:
		warnings.warn("Temperature is outside of range of validity of Zeff coefficients (0.1 - 100 keV).")

	[B0, B1, B2, B3, B4] = get_coeff_Zeff(species, T_e_keV)

	X = np.log10(T_e_keV)

	return B0 + B1*X + B2*X**2 + B3*X**3 + B4*X**4

##############
# Function to obtain coefficients in Zeff polynomial fit, from Mavrin 2018
# ... INPUT:  - species name
#             - Electron temperature in keV
# ... OUTPUT: - array with coefficients
def get_coeff_Zeff(species, T_e_keV):

	if species == "Helium":
		B0 =  2
		B1 =  0 
		B2 =  0 
		B3 =  0 
		B4 =  0 

	elif species == "Neon":
		if T_e_keV <= 0.5:
			B0 =  8.9737e0
			B1 = -1.3242e1
			B2 = -5.3631e1 
			B3 = -6.4696e1
			B4 = -2.5303e1
		elif T_e_keV > 0.5 and T_e_keV <= 2:
			B0 =  9.9532e0
			B1 =  2.1413e-1
			B2 = -8.0723e-1
			B3 =  3.6868e0
			B4 = -7.0678e0
		elif T_e_keV > 2:
			B0 = 1.0000e1
			B1 = 0
			B2 = 0
			B3 = 0
			B4 = 0

	elif species == "Argon":
		if T_e_keV <= 0.6:
			B0 =  1.3171e1
			B1 = -2.0781e1
			B2 = -4.3776e1
			B3 = -1.1595e1
			B4 =  6.8717e0
		elif T_e_keV > 0.6 and T_e_keV <= 3:
			B0 =  1.5986e1
			B1 =  1.1413e0
			B2 =  2.5023e0
			B3 =  1.8455e0
			B4 = -4.8830e-2
		elif T_e_keV > 3:
			B0 =  1.4948e1
			B1 =  7.9986e0
			B2 = -8.0048e0
			B3 =  3.5667e0
			B4 = -5.9213e-1

	elif species == "Krypton":
		if T_e_keV <= 0.447:
			B0 =  7.7040e1
			B1 =  3.0638e2
			B2 =  5.6890e2
			B3 =  4.6320e2
			B4 =  1.3630e2
		elif T_e_keV > 0.447 and T_e_keV <= 4.117:
			B0 =  2.4728e1
			B1 =  1.5186e0
			B2 =  1.5744e1
			B3 =  6.8446e1
			B4 = -1.0279e2
		elif T_e_keV > 4.117:
			B0 =  2.5368e1
			B1 =  2.3443e1
			B2 = -2.5703e1 
			B3 =  1.3215e1
			B4 = -2.4682e0

	elif species == "Xenon":
		if T_e_keV <= 0.3:
			B0 =  3.0532e2
			B1 =  1.3973e3
			B2 =  2.5189e3
			B3 =  1.9967e3
			B4 =  5.8178e2
		elif T_e_keV > 0.3 and T_e_keV <= 1.5:
			B0 =  3.2616e1
			B1 =  1.6271e1
			B2 = -4.8384e1
			B3 = -2.9061e1
			B4 =  8.6824e1
		elif T_e_keV > 1.5 and T_e_keV <= 8:
			B0 =  4.8066e1
			B1 = -1.7259e2
			B2 =  6.6739e2
			B3 = -9.0008e2
			B4 =  4.0756e2
		elif T_e_keV > 8:
			B0 = -5.7527e1
			B1 =  2.4056e2
			B2 = -1.9931e2
			B3 =  7.3261e1
			B4 = -1.0019e1
	
	else:
		raise ValueError("Species must be Helium, Neon, Argon, Krypton, Xenon.")

	return [B0, B1, B2, B3, B4]
