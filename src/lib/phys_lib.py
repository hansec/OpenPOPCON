import numpy as np
import numba as nb

"""
Created Thurs July 18 2024

Compiled library that calculates L_Z and Z_eff from the temperature
using polynomial fits from Mavrin 2018. Also contains cross sections from
H.-S. Bosch and G.M. Hale 1992.
TODO: Add D-He3 reaction to cross section / reactivity.
"""

@nb.njit
def get_Zeffs(T_e_keV:float):
	"""
	Function to obtain Zeff for each impurity, from Mavrin 2018.
	"""

	# Formulas valid only for region of validity: 0.1 <= T_e_keV <= 100
	# if T_e_keV < 0.1 or T_e_keV > 100:
	# 	print('Warning: T_e_keV =',T_e_keV,'out of range for Zeff calculation. Valid range is 0.1 <= T_e_keV <= 100')

	zeff_coeffs = get_coeffs_Zeff(T_e_keV)
	# 6x5 matrix of Zeff values for each impurity

	Tlog = np.log10(T_e_keV)

	poly = np.empty(5)
	for i in np.arange(5):
		poly[i] = np.power(Tlog, i)

	return np.sum(zeff_coeffs*poly, axis=1)

@nb.njit
def get_rads(T_e_keV:float):
	"""
	Function to get radiative power Lz per Zeff, from Mavrin 2018.
	"""
	# # Formulas valid only for region of validity: 0.1 <= T_e_keV <= 100
	# if T_e_keV < 0.1 or T_e_keV > 100:
	# 	print('Warning: T_e_keV =',T_e_keV,'out of range for Lz calculation. Valid range is 0.1 <= T_e_keV <= 100')

	Lz_coeffs = get_coeffs_Lz(T_e_keV)
	# 6x5 matrix of Lz values for each impurity
	Tlog = np.log10(T_e_keV)

	poly = np.empty(5)
	for i in np.arange(5):
		poly[i] = np.power(Tlog, i)

	return np.power(10,np.sum(Lz_coeffs*poly, axis=1))

@nb.njit
def get_coeffs_Zeff(T_e_keV):
	"""
	Function to obtain the coefficients for Zeff calculation, from Mavrin 2018.
	"""

	coeffs = np.empty((6, 5))

	coeffs[0,0] = 2.
	coeffs[0,1] = 0.
	coeffs[0,2] = 0.
	coeffs[0,3] = 0.
	coeffs[0,4] = 0.

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

	if T_e_keV < 1.5:
		coeffs[5,0] =  2.6703e1
		coeffs[5,1] =  1.6518e1
		coeffs[5,2] =  2.1027e1
		coeffs[5,3] =  3.4582e1
		coeffs[5,4] =  1.6823e1
	elif T_e_keV > 1.5 and T_e_keV <= 4:
		coeffs[5,0] =  3.6902e1
		coeffs[5,1] = -7.9611e1
		coeffs[5,2] =  2.5532e2
		coeffs[5,3] = -1.0577e1
		coeffs[5,4] = -2.5887e2
	elif T_e_keV > 4:
		coeffs[5,0] =  6.3795e1
		coeffs[5,1] = -1.0011e2
		coeffs[5,2] =  1.5985e2
		coeffs[5,3] = -8.4207e1
		coeffs[5,4] =  1.5119e1

	return coeffs

@nb.njit
def get_coeffs_Lz(T_e_keV):
	"""
	Function to obtain the coefficients for Lz calculation, from Mavrin 2018.
	"""

	coeffs = np.empty((6, 5))

	coeffs[0,0] = -3.5551e1
	coeffs[0,1] =  3.1469e-1
	coeffs[0,2] =  1.0156e-1
	coeffs[0,3] = -9.3730e-2
	coeffs[0,4] =  2.5020e-2	

	if T_e_keV <= 0.7:
		coeffs[1,0] = -3.3132e1
		coeffs[1,1] =  1.7309e0
		coeffs[1,2] =  1.5230e1
		coeffs[1,3] =  2.8939e1
		coeffs[1,4] =  1.5648e1

	elif T_e_keV > 0.7 and T_e_keV <= 5:
		coeffs[1,0] = -3.3290e1
		coeffs[1,1] = -8.7750e-1
		coeffs[1,2] =  8.6842e-1
		coeffs[1,3] = -3.9544e-1
		coeffs[1,4] =  1.7244e-1
		
	elif T_e_keV > 5:
		coeffs[1,0] = -3.3410e1
		coeffs[1,1] = -4.5345e-1
		coeffs[1,2] =  2.9731e-1
		coeffs[1,3] =  4.3960e-2
		coeffs[1,4] = -2.6930e-2

	if T_e_keV <= 0.6:
		coeffs[2,0] = -3.2155e1
		coeffs[2,1] =  6.5221e0
		coeffs[2,2] =  3.0769e1
		coeffs[2,3] =  3.9161e1
		coeffs[2,4] =  1.5353e1	
	elif T_e_keV > 0.6 and T_e_keV <= 3:
		coeffs[2,0] = -3.2530e1
		coeffs[2,1] =  5.4490e-1
		coeffs[2,2] =  1.5389e0
		coeffs[2,3] = -7.6887e0
		coeffs[2,4] =  4.9806e0
	elif T_e_keV > 3:
		coeffs[2,0] = -3.1853e1
		coeffs[2,1] = -1.6674e0
		coeffs[2,2] =  6.1339e-1
		coeffs[2,3] =  1.7480e-1
		coeffs[2,4] = -8.2260e-2

	if T_e_keV <= 0.447:
		coeffs[3,0] = -3.4512e1
		coeffs[3,1] = -2.1484e1
		coeffs[3,2] = -4.4723e1
		coeffs[3,3] = -4.0133e1
		coeffs[3,4] = -1.3564e1	
	elif T_e_keV > 0.447 and T_e_keV <= 2.364:
		coeffs[3,0] = -3.1399e1
		coeffs[3,1] = -5.0091e-1
		coeffs[3,2] =  1.9148e0
		coeffs[3,3] = -2.5865e0
		coeffs[3,4] = -5.2704e0
	elif T_e_keV > 2.364:
		coeffs[3,0] = -2.9954e1
		coeffs[3,1] = -6.3683e0
		coeffs[3,2] =  6.6831e0
		coeffs[3,3] = -2.9674e0
		coeffs[3,4] =  4.8356e-1

	if T_e_keV <= 0.5:
		coeffs[4,0] = -2.9303e1
		coeffs[4,1] =  1.4351e1
		coeffs[4,2] =  4.7081e1
		coeffs[4,3] =  5.9580e1
		coeffs[4,4] =  2.5615e1
	elif T_e_keV > 0.5 and T_e_keV <= 2.5:
		coeffs[4,0] = -3.1113e1
		coeffs[4,1] =  5.9339e-1
		coeffs[4,2] =  1.2808e0
		coeffs[4,3] = -1.1628e1
		coeffs[4,4] =  1.0748e1
	elif T_e_keV > 2.5 and T_e_keV <= 10:
		coeffs[4,0] = -2.5813e1
		coeffs[4,1] = -2.7526e1
		coeffs[4,2] =  4.8614e1
		coeffs[4,3] = -3.6885e1
		coeffs[4,4] =  1.0069e1
	elif T_e_keV > 10:
		coeffs[4,0] = -2.2138e1
		coeffs[4,1] = -2.2592e1
		coeffs[4,2] =  1.9619e1
		coeffs[4,3] = -7.5181e0
		coeffs[4,4] =  1.0858e0

	if T_e_keV <= 1.5:
		coeffs[5,0] = -3.0374e1
		coeffs[5,1] =  3.8304e-1
		coeffs[5,2] = -9.5126e-1
		coeffs[5,3] = -1.0311e0
		coeffs[5,4] = -1.0103e-1
	elif T_e_keV > 1.5 and T_e_keV <= 4:
		coeffs[5,0] = -3.0238e1
		coeffs[5,1] = -2.9308e0
		coeffs[5,2] =  2.2824e1
		coeffs[5,3] = -6.3303e1
		coeffs[5,4] =  5.1849e1
	elif T_e_keV > 4:
		coeffs[5,0] = -3.2153e1
		coeffs[5,1] =  5.2499e0
		coeffs[5,2] = -6.2740e0
		coeffs[5,3] =  2.6627e0
		coeffs[5,4] = -3.6759e-1

	return coeffs
	
@nb.njit
def get_sigma(T_i_keV, reaction_num):

	"""
	Reaction cross section in barns (1e-28 m^2), from Hale/Bosch 1992.

	reaction_num table:
		- 1: T(d,n)He4
		- 2: D(d,p)T
		- 3: D(d,n)He3
	"""

	#D-T
	if reaction_num == 1:
		A1 = 6.927e4
		A2 = 7.454e8
		A3 = 2.050e6
		A4 = 5.2002e4
		A5 = 0
		B1 = 6.38e1
		B2 = -9.95e-1
		B3 = 6.981e-5
		B4 = 1.728e-4		
		B_G = 34.3827

	#D-D -> p, T
	if reaction_num == 2:
		A1 = 5.5576e4
		A2 = 2.1054e2
		A3 = -3.2638e-2
		A4 = 1.4987e-6
		A5 = 1.8181e-10
		B1 = 0
		B2 = 0
		B3 = 0
		B4 = 0
		B_G = 31.3970

	#D-D -> n, He3
	if reaction_num == 3:
		A1 = 5.3701e4
		A2 = 3.3027e2
		A3 = -1.2706e-1
		A4 = 2.9327e-5
		A5 = -2.5151e-9
		B1 = 0
		B2 = 0
		B3 = 0
		B4 = 0	
		B_G = 31.3970

    #KeV/c^2
	S = lambda T: (A1 + T * (A2 + T * (A3 + T * (A4 + T * A5))))/(1+T * (B1 + T * (B2 + T * (B3 + T* B4))))
	return 0.001*S(T_i_keV) / (T_i_keV* np.exp(B_G / np.sqrt((T_i_keV))))

@nb.njit
def get_reactivity(T_i_keV, reaction_num):
	"""
	Reaction cross section in barns (1e-28 m^2), from Hale/Bosch 1992.
	See equations 12-14.

	reaction_num table:
		- 1: T(d,n)He4
		- 2: D(d,p)T
		- 3: D(d,n)He3
	"""

	#D-T
	if reaction_num == 1:
		C1 = 1.17302e-9
		C2 = 1.51361e-2
		C3 = 7.51886e-2
		C4 = 4.60643e-3
		C5 = 1.35000e-2
		C6 = -1.06750e-4
		C7 = 1.36600e-5
		
		mc2 = 1124656
		B_G = 34.3827

	#D-D -> p, T
	if reaction_num == 2:
		C1 = 5.65718e-12
		C2 = 3.41267e-3
		C3 = 1.99167e-3
		C4 = 0
		C5 = 1.05060e-5
		C6 = 0
		C7 = 0
		
		mc2 = 937814
		B_G = 31.3970

	#D-D -> n, He3
	if reaction_num == 3:
		C1 = 5.43360e-12
		C2 = 5.85778e-3
		C3 = 7.68222e-3
		C4 = 0
		C5 = -2.96400e-6
		C6 = 0
		C7 = 0
		
		mc2 = 937814
		B_G = 31.3970

	theta = T_i_keV / \
		( 1 - (	(T_i_keV*(C2+T_i_keV*(C4+T_i_keV*C6)))/ \
						   (1+T_i_keV*(C3+T_i_keV*(C5+T_i_keV*C7))) ) )
	xi = (B_G**2/(4 * theta))**(1/3)

	return C1*theta*np.sqrt( xi / (mc2* T_i_keV**3) )*np.exp(-3 * xi)


