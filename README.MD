# OpenPOPCON v2.0
# ===================================

OpenPOPCON is a tool for scoping Tokamak design and operation with 0-D fitted scaling laws. This version has been refactored from the original developed for MIT 22.63, with major contributions from Sam Frank, Richard Nies, Tal Rubin, Oak Nelson, Matthew Pharr, Leonardo Corsaro, and many minor contributions from others. This code is intended for use in Columbia's Fusion Reactor Design course.

## Installation

We recommend the use of miniconda to manage the python environment. On a cluster, you may use `module load anaconda` to load anaconda. If you are running on your private machine, see [the anaconda project](https://docs.anaconda.com/miniconda/) for installation instructions.

To install the required packages, run `conda env create -f environment.yml` in the root directory of this repository. This will create a new environment called `openpopcon` with all the required packages. From here, you can activate the environment with `conda activate openpopcon`. 

## Usage

Example notebooks are found under `resources/examples`. However, generally, the steps are:

```python
# 1. Import the POPCON class
import openpopcon as op

# 2. Load the settings files; make sure to create these, see the examples for a template
pc = op.POPCON(settingsfile=settingsfile, plotsettingsfile=plotsettingsfile, scalinglawfile=scalinglawfile)

# 3. Run the code
pc.run_POPCON()

# 4. Plot the results
pc.plot()
```

## Bibliography

[1] H.-S. Bosch and G. M. Hale, *Improved Formulas for Fusion Cross-Sections and Thermal Reactivities,* Nucl. Fusion **32**, 611 (1992).

[2] S. C. Jardin, M. G. Bell, and N. Pomphrey, *TSC Simulation of Ohmic Discharges in TFTR,* Nucl. Fusion **33**, 371 (1993).

[3] A. A. Mavrin, *Improved Fits of Coronal Radiative Cooling Rates for High-Temperature Plasmas,* Radiation Effects and Defects in Solids **173**, 388 (2018).

[4] O. Sauter, *Geometric Formulas for System Codes Including the Effect of Negative Triangularity,* Fusion Engineering and Design **112**, 633 (2016).

[5] Y. R. Martin, T. Takizuka (and the ITPA CDBM H.-mode Threshold Database Working Group), *Power Requirement for Accessing the H-Mode in ITER*, J. Phys.: Conf. Ser. **123**, 012033 (2008).

[6] H. Zohm, *On the Use of High Magnetic Field in Reactor Grade Tokamaks*, J Fusion Energy **38**, 3 (2019).

[7] I. P. E. G. on Confinement, Transport, I. P. E. G. on C. Modelling, Database, and I. P. B. Editors, Chapter 2: Plasma confinement and transport, Nucl. Fusion 39, 2175 (1999).

[8] C. Paz-Soldan, R. J. L. Haye, D. Shiraki, R. J. Buttery, N. W. Eidietis, E. M. Hollmann, R. A. Moyer, J. E. Boom, I. T. Chapman, and J. E. T. Contributors, The non-thermal origin of the tokamak low-density stability limit, Nucl. Fusion 56, 056010 (2016).



## Numerical strategy

<img width="1668" alt="image" src="https://github.com/user-attachments/assets/a1234137-260f-4c5b-9124-c8312b3b9ef7" />

OpenPOPCON employs a relaxation algorithm instead of solving power balance directly. This means it can often take quite long to converge on an answer compared to a code that would solve the problem explicitly. However, we have taken an approach with [numba](https://numba.pydata.org/) to speed up the code. This means that the code is compiled to machine code and can be run much faster than a traditional python code. To make full use of this feature, run OpenPOPCON from a Jupyter notebook. Every time you restart the Jupyter notebook, the code will be recompiled. The code structure is as follows:

```python
class POPCON:
    """The general class the user interacts with."""
    self.algorithms: POPCON_algorithms: jitted 'class'
    self.settings: POPCON_settings: 'class'
    self.plotsettings: POPCON_plotsettings: 'class'
    self.output: POPCON_data: jitted 'class'
    
    @njit(parallel=True)
    def run_POPCON(self):
        """calls a function from self.algorithms and creates a new 
        openPOPCON_results object. numba.prange allows for 
        parallelization like an OMP for collapse(2)."""
        for i in numba.prange(n_n):
            for j in numba.prange(n_T):
                self.openPOPCON_params.solve(n, T)
    def single_point(self, n, T):
        "Runs a single n,T and plots the solution profiles"
    def plot(self):
        "Plots, with the option to update plot settings"
    def read_output(self):
        "Reads in a previous solution"
    def write_output(self):
        "Writes the current solution to a folder or zip archive"

@jitclass
class POPCON_algorithms:
    """A compiled class that does all numerical calculations."""

    def P_aux_relax_impfrac:
        "Solves the equations for a given n,T, holding impfrac const."
    
    def get_value(self):
        "Helper functions to get all physical parameters"

@jitclass    
class POPCON_data:
    """output arrays of physical parameters at power balance"""

    
```
