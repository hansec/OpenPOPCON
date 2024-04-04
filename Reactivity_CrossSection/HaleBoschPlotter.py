# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:43:27 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt

import FusionCrossSectionHaleBosch as HB


Egrid = np.geomspace(0.5, 550, 550)



plt.plot(Egrid,HB.sigma('T(d,n)He4',Egrid),label="DT-HB")
plt.plot(Egrid,HB.sigma('D(d,p)T',Egrid),label="D(d,p)T-HB")
plt.plot(Egrid,HB.sigma('D(d,n)He3',Egrid),label="D(d,n)He3-HB")




plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('E [keV]')
plt.ylabel('$\\sigma$ [barns]')

#plt.yscale('log')

Tgrid = np.geomspace(0.2,100,100)
plt.figure()
plt.plot(Tgrid,HB.reactivity('T(d,n)He4',Tgrid),label="DT-HB")
plt.plot(Tgrid,HB.reactivity('D(d,p)T',Tgrid),label="D(d,p)T-HB")
plt.plot(Tgrid,HB.reactivity('D(d,n)He3',Tgrid),label="D(d,n)He3-HB")
plt.yscale('log')
plt.xscale('log')
plt.xlabel('T_i [keV]')
plt.ylabel('$\\left<\\sigma v \\right> $ [cm^3/s]')
plt.legend()
plt.show()
