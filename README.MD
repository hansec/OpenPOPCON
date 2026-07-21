# OpenPOPCON v2.0
# ===================================

OpenPOPCON is a tool for scoping Tokamak design and operation with 0-D fitted scaling laws. This version has been refactored from the original developed for MIT 22.63, with major contributions from Sam Frank, Richard Nies, Tal Rubin, Oak Nelson, Matthew Pharr, Leonardo Corsaro, and many minor contributions from others. This code is intended for use in Columbia's Fusion Reactor Design course.

## Installation

**With pip.** This installs the package and its dependencies, and works from anywhere afterwards:

```bash
pip install git+https://github.com/hansec/OpenPOPCON.git
```

**From a clone, with uv.** If you have [uv](https://docs.astral.sh/uv/), nothing needs installing:

```bash
uv run example_run.py      # writes POPCON.png and single_point.png
uv run --extra notebooks jupyter lab
```

**From a clone, with pip.** If you would rather manage the environment yourself, install the package in editable mode from the root of the repository:

```bash
pip install -e .              # add [notebooks] for jupyterlab, [netcdf4] for the h5netcdf backend
```

This pulls in the dependencies declared in `pyproject.toml` and makes `import openpopcon` work from anywhere.

## Usage

`example_run.py` is a complete working example; run it first to check your installation.

Worked examples for several machines ship with the package. The installed copies are read only, so to get a set you can edit:

```bash
openpopcon examples ./my_scans
cd my_scans/MANTA
```

Each example directory has its own settings file, a notebook, and where applicable a gEQDSK and a profiles file.

```python
# 1. Import the POPCON class
import openpopcon as op

# 2. Load the settings files; make sure to create these, see the examples for a template.
#    scalinglawfile and plotsettingsfile both fall back to the copies shipped
#    with the package if you leave them out.
pc = op.POPCON(settingsfile=settingsfile, plotsettingsfile=plotsettingsfile)

# 3. Run the code
pc.run_POPCON()

# 4. Plot the results
pc.plot()

# 5. Or look at one operating point in detail, here 60% of the Greenwald
#    density at 9 keV volume-averaged Ti
pc.single_point(0.6, 9.0)
```

`gfilename` and `profsfilename` inside a settings file are resolved relative to that settings file, so an example directory works wherever you copy it. The paths you pass to `POPCON(...)` are resolved relative to wherever you are running.

Results written with `pc.write_output()` go to `./OpenPOPCON_outputs/`, or to whatever you pass as `directory=`.

If a settings file has a mistake in it, OpenPOPCON reports every problem it finds at once, rather than stopping at the first:

```
ValueError: Found 2 problem(s) in .../POPCON_input_example.yaml:
  - scalinglaw 'H98' is not defined. Available: H89, H98y2, H_NT23.
  - Tmin_keV (11.0) must be less than Tmax_keV (6.0).
```

Grid points where power balance has no physical solution are reported in aggregate after a run and masked out of the plot.

The trapped particle fraction that the neoclassical resistivity needs is taken from the gEQDSK when one is supplied. Without a gEQDSK it falls back to `f_t = sqrt(2*rho*a/R)`, which stays below 1 for the aspect ratios these examples use but would not for a spherical tokamak.

## Bibliography

[1] H.-S. Bosch and G. M. Hale, *Improved Formulas for Fusion Cross-Sections and Thermal Reactivities,* Nucl. Fusion **32**, 611 (1992).

[2] S. C. Jardin, M. G. Bell, and N. Pomphrey, *TSC Simulation of Ohmic Discharges in TFTR,* Nucl. Fusion **33**, 371 (1993).

[3] A. A. Mavrin, *Improved Fits of Coronal Radiative Cooling Rates for High-Temperature Plasmas,* Radiation Effects and Defects in Solids **173**, 388 (2018).

[4] O. Sauter, *Geometric Formulas for System Codes Including the Effect of Negative Triangularity,* Fusion Engineering and Design **112**, 633 (2016).

[5] Y. R. Martin, T. Takizuka (and the ITPA CDBM H.-mode Threshold Database Working Group), *Power Requirement for Accessing the H-Mode in ITER*, J. Phys.: Conf. Ser. **123**, 012033 (2008).

[6] H. Zohm, *On the Use of High Magnetic Field in Reactor Grade Tokamaks*, J Fusion Energy **38**, 3 (2019).

[7] I. P. E. G. on Confinement, Transport, I. P. E. G. on C. Modelling, Database, and I. P. B. Editors, Chapter 2: Plasma confinement and transport, Nucl. Fusion 39, 2175 (1999).

[8] C. Paz-Soldan, R. J. L. Haye, D. Shiraki, R. J. Buttery, N. W. Eidietis, E. M. Hollmann, R. A. Moyer, J. E. Boom, I. T. Chapman, and J. E. T. Contributors, The non-thermal origin of the tokamak low-density stability limit, Nucl. Fusion 56, 056010 (2016).

[9] Y. R. Lin-Liu and R. L. Miller, Upper and lower bounds of the effective trapped particle fraction in general tokamak equilibria, Physics of Plasmas 2, 1666 (1995).




## Numerical strategy

<img width="1668" alt="image" src="https://github.com/user-attachments/assets/a1234137-260f-4c5b-9124-c8312b3b9ef7" />

Power balance is solved in closed form. With the impurity fraction held fixed, every power integral at a given (n, T) is fixed, and the only self-consistency is that the confinement time is a power law in the heating power, `tauE = K * P_heat^alpha`. That makes the balance a single scalar equation with an exact root:

```
P_heat = (W_tot / K)^(1/(1+alpha))
P_aux  = P_heat - P_OH - P_fus_heat + P_brem + P_synch
```

so there is no iteration, no convergence tolerance, and no points that fail to converge. Points with no physical solution, where impurity radiation exceeds the confinement loss or the balance has no real root, are set to `P_aux = 99999`, counted in a summary after the run, and masked out of the plot.

The numerics are compiled to machine code with [numba](https://numba.pydata.org/). The compiled functions are cached to disk, so the compilation cost is paid the first time the code runs on a machine and not on every session; a second run in a fresh interpreter starts in a second or two. The code structure is as follows:

```python
# module-level, @nb.njit(cache=True), operating on an immutable State namedtuple
def get_profile(s, rho, profid)      # J, ne, ni, Ti, Te, q, ... by integer id
def volume_integral(s, rho, func)    # integrates dV-like
def P_aux_impfrac(s, n_e_20, T_i_keV)
    "Closed-form power balance. Returns (P_aux, flag)."

# module-level drivers; numba.prange parallelizes like an OMP for collapse(2)
def solve_nT[_par](state, ...)       # P_aux over the whole grid, then
def populate_outputs[_par](state, ...)  # every other diagnostic from it

class POPCON:
    """The general class the user interacts with."""
    self.algorithms: POPCON_algorithms   # sets up State, calls the functions above
    self.settings: POPCON_settings
    self.plotsettings: POPCON_plotsettings
    self.output: xarray.Dataset          # output arrays at power balance

    def run_POPCON(self):
        "Sets up and solves the whole n,T grid"
    def single_point(self, n, T):
        "Runs a single n,T and plots the solution profiles"
    def plot(self):
        "Plots, with the option to update plot settings"
    def read_output(self):
        "Reads in a previous solution"
    def write_output(self):
        "Writes the current solution to a folder or zip archive"
```
