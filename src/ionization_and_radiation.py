import numpy as np
import numba as nb

# Mavrin 2018.

@nb.njit
def get_Zeffs(T_e_keV:float):
	"""
	Function to obtain Zeff for each impurity, from Mavrin 2018.
	"""

	# Formulas valid only for region of validity: 0.1 <= T_e_keV <= 100
	if T_e_keV < 0.1 or T_e_keV > 100:
		print('Warning: T_e_keV out of range for Zeff calculation. Valid range is 0.1 <= T_e_keV <= 100')

	zeff_coeffs = get_coeffs_Zeff(T_e_keV)
	# 6x5 matrix of Zeff values for each impurity

	Tlog = np.log10(T_e_keV)

	poly = np.empty(5)
	for i in nb.prange(5):
		poly[i] = np.power(Tlog, i)

	return np.sum(zeff_coeffs*poly, axis=1)

@nb.njit
def get_coeffs_Zeff(T_e_keV):
	"""
	Function to obtain the coefficients for Zeff calculation, from Mavrin 2018.
	"""

	coeffs = np.empty((6, 5))

	coeffs[0,0] = 2.
	coeffs[1,0] = 0.
	coeffs[2,0] = 0.
	coeffs[3,0] = 0.
	coeffs[4,0] = 0.

	if T_e_keV < 0.5:
		coeffs[1,0] =  8.9737e0
		coeffs[1,1] = -1.3242e1
		coeffs[1,2] = -5.3631e1
		coeffs[1,3] = -6.4696e1
		coeffs[1,4] = -2.5303e1
	elif T_e_keV > 0.5 and T_e_keV <= 2:
		coeffs[1,0] =  9.9532e0
		coeffs[1,1] =  2.1413e-1
		coeffs[1,2] = -8.0723e-1
		coeffs[1,3] =  3.6868e0
		coeffs[1,4] = -7.0678e0
	elif T_e_keV > 2:
		coeffs[1,0] = 1.0000e1
		coeffs[1,1] = 0
		coeffs[1,2] = 0
		coeffs[1,3] = 0
		coeffs[1,4] = 0
	
	if T_e_keV < 0.6:
		coeffs[2,0] =  1.3171e1
		coeffs[2,1] = -2.0781e1
		coeffs[2,2] = -4.3776e1
		coeffs[2,3] = -1.1595e1
		coeffs[2,4] =  6.8717e0
	elif T_e_keV > 0.6 and T_e_keV <= 3:
		coeffs[2,0] =  1.5986e1
		coeffs[2,1] =  1.1413e0
		coeffs[2,2] =  2.5023e0
		coeffs[2,3] =  1.8455e0
		coeffs[2,4] = -4.8830e-2
	elif T_e_keV > 3:
		coeffs[2,0] =  1.4948e1
		coeffs[2,1] =  7.9986e0
		coeffs[2,2] = -8.0048e0
		coeffs[2,3] =  3.5667e0
		coeffs[2,4] = -5.9213e-1

	if T_e_keV < 0.447:
		coeffs[3,0]  =  7.7040e1
		coeffs[3,1]  =  3.0638e2
		coeffs[3,2]  =  5.6890e2
		coeffs[3,3]  =  4.6320e2
		coeffs[3,4]  =  1.3630e2
	elif T_e_keV > 0.447 and T_e_keV <= 4.117:
		coeffs[3,0]  =  2.4728e1
		coeffs[3,1]  =  1.5186e0
		coeffs[3,2]  =  1.5744e1
		coeffs[3,3]  =  6.8446e1
		coeffs[3,4]  = -1.0279e2
	elif T_e_keV > 4.117:
		coeffs[3,0]  =  2.5368e1
		coeffs[3,1]  =  2.3443e1
		coeffs[3,2]  = -2.5703e1 
		coeffs[3,3]  =  1.3215e1
		coeffs[3,4]  = -2.4682e0

	if T_e_keV <= 0.3:
		coeffs[4,0] =  3.0532e2
		coeffs[4,1] =  1.3973e3
		coeffs[4,2] =  2.5189e3
		coeffs[4,3] =  1.9967e3
		coeffs[4,4] =  5.8178e2
	elif T_e_keV > 0.3 and T_e_keV <= 1.5:
		coeffs[4,0] =  3.2616e1
		coeffs[4,1] =  1.6271e1
		coeffs[4,2] = -4.8384e1
		coeffs[4,3] = -2.9061e1
		coeffs[4,4] =  8.6824e1
	elif T_e_keV > 1.5 and T_e_keV <= 8:
		coeffs[4,0] =  4.8066e1
		coeffs[4,1] = -1.7259e2
		coeffs[4,2] =  6.6739e2
		coeffs[4,3] = -9.0008e2
		coeffs[4,4] =  4.0756e2
	elif T_e_keV > 8:
		coeffs[4,0] = -5.7527e1
		coeffs[4,1] =  2.4056e2
		coeffs[4,2] = -1.9931e2
		coeffs[4,3] =  7.3261e1
		coeffs[4,4] = -1.0019e1

	# Tungsten is not implemented yet. TODO: Find this. callen?
	coeffs[5,0] =  1.0000e0
	coeffs[5,1] =  0.0000e0
	coeffs[5,2] =  0.0000e0
	coeffs[5,3] =  0.0000e0
	coeffs[5,4] =  0.0000e0


	return coeffs