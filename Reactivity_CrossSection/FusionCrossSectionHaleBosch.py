# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:26:46 2020

@author: trubin
"""
import numpy as np

def sigma(Reactant,E): 
    '''#E = energy in keV, Lab frame, sigmaDT in barns (1E-28 m^2) '''


#DT
    if Reactant=='T(d,n)He4':
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
        E_min = 0.5
        E_max = 530

     #   mu=dalton*masses['D']*masses['T']/(masses['D']+masses['T'])
     #   Ecm=lambda E: E*masses['T']/(masses['D']+masses['T'])

    if Reactant=='D(d,p)T':
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
        E_min = 0.5
        E_max = 5000
       
  ##      mu=dalton*masses['D']/2
    #    Ecm=lambda E: E/2

    if Reactant=='D(d,n)He3':
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
        E_min = 0.5
        E_max = 4900
        
  ##      mu=dalton*masses['D']/2
    #    Ecm=lambda E: E/2

#KeV/c^2
    S = lambda E: (A1 + E * (A2 + E * (A3 + E * (A4 + E * A5))))/(1+E * (B1 + E * (B2 + E * (B3 + E* B4))))

    
    


    return 0.001*S((E)) / ((E)* np.exp(B_G / np.sqrt((E))))
def reactivity(Reactant,T): 
 

    if Reactant=='T(d,n)He4':
        C1 = 1.17302e-9
        C2 = 1.51361e-2
        C3 = 7.51886e-2
        C4 = 4.60643e-3
        C5 = 1.35000e-2
        C6 = -1.06750e-4
        C7 = 1.36600e-5
        
        mc2 = 1124656
        B_G = 34.3827
        Ti_min = 0.2
        Ti_max = 100
     #   mu=dalton*masses['D']*masses['T']/(masses['D']+masses['T'])
     #   Ecm=lambda E: E*masses['T']/(masses['D']+masses['T'])

    if Reactant=='D(d,p)T':
        C1 = 5.65718e-12
        C2 = 3.41267e-3
        C3 = 1.99167e-3
        C4 = 0
        C5 = 1.05060e-5
        C6 = 0
        C7 = 0
        
        mc2 = 937814
        B_G = 31.3970
        Ti_min = 0.2
        Ti_max = 100
  ##      mu=dalton*masses['D']/2
    #    Ecm=lambda E: E/2

    if Reactant=='D(d,n)He3':
        C1 = 5.43360e-12
        C2 = 5.85778e-3
        C3 = 7.68222e-3
        C4 = 0
        C5 = -2.96400e-6
        C6 = 0
        C7 = 0
        
        mc2 = 937814
        B_G = 31.3970
        Ti_min = 0.2
        Ti_max = 100
  ##      mu=dalton*masses['D']/2
    #    Ecm=lambda E: E/2



    Theta = lambda T: T / (1-  (T * (C2 + T * (C4 + T * C6)))/(1 + T * (C3 + T * (C5 + T * C7))))
    xi = lambda T: (B_G**2/(4 * Theta(T)))**(1/3)
    
    return C1*Theta(T) * np.sqrt(xi(T) / (mc2* T**3)) * np.exp(-3 * xi(T))

    
