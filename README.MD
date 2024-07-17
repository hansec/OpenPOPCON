# OpenPOPCON v2.0 development version
# ===================================

OpenPOPCON is a tool for scoping Tokamak operation with 0d fitted scaling laws. This version has been refactored from the original developed for MIT 22.63, with major contributions from Sam Frank, Richard Nies, Tal Rubin, Oak Nelson, Matthew Pharr, Leonardo Corsaro, and many minor contributions from others. This code is intended for use in Columbia's Fusion Reactor Design course.

## Installation

We recommend the use of miniconda to manage the python environment. On a cluster, you may use `module load anaconda` to load anaconda. If you are running on your private machine, see [the anaconda project](https://docs.anaconda.com/miniconda/) for installation instructions.

To install the required packages, run `conda env create -f environment.yml` in the root directory of this repository. This will create a new environment called `openpopcon` with all the required packages. From here, you can activate the environment with `conda activate openpopcon`. 

## Bibliography

[1] H.-S. Bosch and G. M. Hale, *Improved Formulas for Fusion Cross-Sections and Thermal Reactivities,* Nucl. Fusion **32**, 611 (1992).
[2] S. C. Jardin, M. G. Bell, and N. Pomphrey, *TSC Simulation of Ohmic Discharges in TFTR,* Nucl. Fusion **33**, 371 (1993).
[3] A. A. Mavrin, *Improved Fits of Coronal Radiative Cooling Rates for High-Temperature Plasmas,* Radiation Effects and Defects in Solids **173**, 388 (2018).
[4] O. Sauter, *Geometric Formulas for System Codes Including the Effect of Negative Triangularity,* Fusion Engineering and Design **112**, 633 (2016).

## Numerical strategy

OpenPOPCON, unlike other POPCON codes, employs a relaxation algorithm. This means it can often take quite long to converge on an answer compared to a code that would solve the problem explicitly. However, we have taken an approach with [numba](https://numba.pydata.org/) to speed up the code. This means that the code is compiled to machine code and can be run much faster than a traditional python code. The code structure is as follows:

```python
class openPOPCON:
    """The general class the user interacts with."""
    openPOPCON_params: jitted 'class'
    openPOPCON_settings: 'class'
    openPOPCON_plot_settings: 'class'
    openPOPCON_results: 'class'

    def helper(self):
        "Populates openPOPCON_params"
    
    @njit(parallel=True)
    def solve(self):
        "calls openPOPCON_params.solve and creates a new openPOPCON_results algorithm"
        "prange allows for parallelization like an OMP for collapse(2)."
        for i in numba.prange(n_n):
            for j in numba.prange(n_T):
                self.openPOPCON_params.solve(n, T)

    def plot(self):
        "Plots, with the option to update plot settings
    def read(self):
        "Reads in a pickle with a previous solution"
    def write(self):
        "Writes out a pickle with the current solution"

@jitclass
class openPOPCON_params:
    """A compiled class that does all numerical calculations."""
    Parameters: floats, arrays

    def solve(self, n, T):
        "Solves the equations for a given n, T"
    
    def get_value(self):
        "Helper functions to get physical parameters"
    
class openPOPCON_results:

    openPOPCON_params: jitted 'class'
    openPOPCON_settings: 'class'

    
```