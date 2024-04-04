import numpy as np
import warnings

##############
# Function to obtain radiative cooling rate per free electron per ion, from Mavrin 2018
# ... INPUT:  - species name [options: Helium, Neon, Argon, Krypton, Xenon]
#             - Electron temperature in keV [region of validity: 0.1 <= T_e_keV <= 100]
# ... OUTPUT: - Lrad in units of W*m^3 [Lrad = Prad / (n_e * n_impurity)]
def get_Lrad(species, T_e_keV):
	# Formulas valid only for region of validity: 0.1 <= T_e_keV <= 100
	if T_e_keV < 0.1 or T_e_keV > 100:
		warnings.warn("Temperature is outside of range of validity of Lrad coefficients (0.1 - 100 keV).")

	[A0, A1, A2, A3, A4] = get_coeff_Lrad(species, T_e_keV)

	X = np.log10(T_e_keV)

	return 10**(A0 + A1*X + A2*X**2 + A3*X**3 + A4*X**4)

##############
# Function to obtain coefficients in Lrad polynomial fit, from Mavrin 2018
# ... INPUT:  - species name
#             - Electron temperature in keV
# ... OUTPUT: - array with coefficients
def get_coeff_Lrad(species, T_e_keV):

	if species == "Helium":
		A0 = -3.5551e1
		A1 =  3.1469e-1
		A2 =  1.0156e-1
		A3 = -9.3730e-2
		A4 =  2.5020e-2	

	elif species == "Neon":
		if T_e_keV <= 0.7:
			A0 = -3.3132e1
			A1 =  1.7309e0
			A2 =  1.5230e1
			A3 =  2.8939e1
			A4 =  1.5648e1	
		elif T_e_keV > 0.7 and T_e_keV <= 5:
			A0 = -3.3290e1
			A1 = -8.7750e-1
			A2 =  8.6842e-1
			A3 = -3.9544e-1
			A4 =  1.7244e-1
		elif T_e_keV > 5:
			A0 = -3.3410e1
			A1 = -4.5345e-1
			A2 =  2.9731e-1
			A3 =  4.3960e-2
			A4 = -2.6930e-2

	elif species == "Argon":
		if T_e_keV <= 0.6:
			A0 = -3.2155e1
			A1 =  6.5221e0
			A2 =  3.0769e1
			A3 =  3.9161e1
			A4 =  1.5353e1	
		elif T_e_keV > 0.6 and T_e_keV <= 3:
			A0 = -3.2530e1
			A1 =  5.4490e-1
			A2 =  1.5389e0
			A3 = -7.6887e0
			A4 =  4.9806e0
		elif T_e_keV > 3:
			A0 = -3.1853e1
			A1 = -1.6674e0
			A2 =  6.1339e-1
			A3 =  1.7480e-1
			A4 = -8.2260e-2

	elif species == "Krypton":
		if T_e_keV <= 0.447:
			A0 = -3.4512e1
			A1 = -2.1484e1
			A2 = -4.4723e1
			A3 = -4.0133e1
			A4 = -1.3564e1	
		elif T_e_keV > 0.447 and T_e_keV <= 2.364:
			A0 = -3.1399e1
			A1 = -5.0091e-1
			A2 =  1.9148e0
			A3 = -2.5865e0
			A4 = -5.2704e0
		elif T_e_keV > 2.364:
			A0 = -2.9954e1
			A1 = -6.3683e0
			A2 =  6.6831e0
			A3 = -2.9674e0
			A4 =  4.8356e-1

	elif species == "Xenon":
		if T_e_keV <= 0.5:
			A0 = -2.9303e1
			A1 =  1.4351e1
			A2 =  4.7081e1
			A3 =  5.9580e1
			A4 =  2.5615e1
		elif T_e_keV > 0.5 and T_e_keV <= 2.5:
			A0 = -3.1113e1
			A1 =  5.9339e-1
			A2 =  1.2808e0
			A3 = -1.1628e1
			A4 =  1.0748e1
		elif T_e_keV > 2.5 and T_e_keV <= 10:
			A0 = -2.5813e1
			A1 = -2.7526e1
			A2 =  4.8614e1
			A3 = -3.6885e1
			A4 =  1.0069e1
		elif T_e_keV > 10:
			A0 = -2.2138e1
			A1 = -2.2592e1
			A2 =  1.9619e1
			A3 = -7.5181e0
			A4 =  1.0858e0
	
	else:
		raise ValueError("Species must be Helium, Neon, Argon, Krypton, Xenon.")

	return [A0, A1, A2, A3, A4]
