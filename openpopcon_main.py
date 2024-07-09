# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 2024

OpenPOPCON v2.0
This is the Columbia University fork of the OpenPOPCON project developed
for MIT cours 22.63. This project is a refactor of the contributions
made to the original project in the development of MANTA. Contributors
to the original project are listed in the README.md file.
"""

import numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.constants as const, scipy.integrate as integrate
import xarray as xr

class POPCON_params:
    def __init__(self, 
                 machinename:str = 'NEWDEVICE',
                 ) -> None:
        
        self.machinename = machinename
        pass


class POPCON:
    """
    The Primary class for the OpenPOPCON project.

    TODO: Write Documentation
    """
    def __init__(self, params:POPCON_params) -> None:        

        self.params = params
        
        pass