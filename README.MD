# OpenPOPCON v2.2.0
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

Apart from the two NSF examples, which keep their original files, each example directory has the same layout: a `POPCON_input_example.yaml` settings file with a `scan:` block, a `plotsettings.yml` that lists only what differs from the defaults, a `<name>_ex.ipynb` notebook that runs a POPCON and then the scan, and where applicable a gEQDSK and a profiles file.

| Example | Machine | Notes |
| --- | --- | --- |
| `MANTA` | 4.55 m, 11 T, 10 MA, negative triangularity | gEQDSK + profiles file; scans `B_0` x `H_fac` |
| `SPARC` | 1.85 m, 12.2 T, 8.7 MA | parabolic profiles; scans `R_0` x `B_0` |
| `ITER` | 6.2 m, 5.3 T, 15 MA | 500 MW at Q = 10, parabolic profiles; scans `I_P` x `H_fac` |
| `CENTAUR` | 2.0 m, 10.9 T, 9.6 MA, negative triangularity | 40 MW breakeven, `H_NT23` scaling; scans `I_P` x `B_0` |
| `radiative_ARC` | 4.2 m, 11.5 T, 14 MA | radiative L-mode, `H89`; scans `Zeff_target` x `H_fac` |
| `NSF` | 0.8 m, 3 T, 1 MA, negative triangularity | gEQDSK + profiles file; older layout, no scan block |
| `NSF_NT_CMOD` | 0.8 m, 3 T, 1 MA | parabolic profiles; older layout, no scan block or notebook |

`ITER` and `CENTAUR` deliberately use parabolic profiles rather than a gEQDSK, which keeps their geometry free for `POPCON_scan` to vary. Each of the two carries a note at the bottom of its settings file on how closely the 0-D result tracks the published design point, and where it does not.

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

### Plot axes

Either plot axis takes any of seven labels, listed in `openpopcon.PLOT_AXES` and documented in the plotsettings files. They are not split into an x set and a y set: the only rule is that one has to be a density and the other a temperature, since those are the two grid dimensions.

```yaml
yax: "nG"      # nG, n20_av, n20_ax
xax: "T_i_av"  # T_i_av, T_i_ax, T_e_av, T_e_ax
```

Putting `nG` on `xax` and `T_i_av` on `yax` draws the same POPCON with the axes swapped.

### Scanning machine parameters

`POPCON_scan` runs a full POPCON at every combination of two machine parameters and tiles them into one figure, for scoping a design space rather than a single machine.

```python
sc = op.POPCON_scan(
    settingsfile=settingsfile,
    plotsettingsfile=plotsettingsfile,
    scan={"rows": ("I_P", {"min": 7.0, "max": 12.0, "N": 3}),
          "cols": ("B_0", [9.0, 10.5, 12.0])},
)
sc.run_scan()
sc.plot()                      # the N x M grid of POPCONs
sc.plot_metric("Q", "max")     # one number per cell, as a heatmap
sc.output                      # every cell in one xarray Dataset
```

The same scan can live in the settings file instead, which is what every example except the two NSF ones does:

```yaml
scan:
  rows: {parameter: I_P, min: 7.0, max: 12.0, N: 3}
  cols: {parameter: B_0, values: [9.0, 10.5, 12.0]}
```

Each axis takes either an explicit list of `values` or a `min`/`max`/`N` range, and a `scan=` argument overrides the settings file. Parameters are named as they appear in the settings file, and the whole settings derivation is re-run for each cell rather than a solved field being patched, so scanning `R` on a file that specifies `qstar` correctly re-derives `Ip`. `openpopcon.SCANNABLE_SETTINGS_KEYS` lists what can be varied.

The major radius can be given as either `R` or `R_0` (not both), matching `B_0`, and either name can be scanned whichever one the settings file uses.

Every panel is drawn on the same contour levels so the cells can be compared directly, and the y axis is shared only when it is the Greenwald fraction, since an absolute density axis moves with `n_G = I_p / (pi a^2)`.

#### One parameter at a time, relative changes, and what stays fixed

Give only `rows` (or only `cols`) to vary one parameter on its own. `scale` gives the values relative to the settings file instead of absolutely, and `hold` names what stays fixed while the parameter varies:

```python
# the field on its own, +-25%
sc = op.POPCON_scan(settingsfile=settingsfile, plotsettingsfile=plotsettingsfile,
                    scan={"rows": {"parameter": "B_0", "scale": [0.75, 1.0, 1.25]}})

# the major radius +-25%, three ways
{"rows": {"parameter": "R", "scale": [0.75, 1.0, 1.25], "hold": ["aspect_ratio", "I_P"]}}
{"rows": {"parameter": "R", "scale": [0.75, 1.0, 1.25], "hold": ["aspect_ratio", "qstar"]}}
{"rows": {"parameter": "R", "scale": [0.75, 1.0, 1.25], "hold": ["a", "qstar"]}}
```

| `hold` | effect in every cell |
| --- | --- |
| `aspect_ratio` | moves `a` with `R` (or `R` with `a`) so `R/a` stays at the base value |
| `a` | the minor radius stays put, which is what happens anyway |
| `I_P` | the current stays at the base value |
| `qstar` | `q*` stays at the base value, re-deriving `I_P` from each cell's `R`, `a`, `B_0` and `kappa` |
| `B_0` | the on-axis field stays at the base value, even if the file derives it from `B_coil` |

The same keys work in a settings file's `scan:` block. `hold` counts as setting those keys, so the other axis cannot also scan them. Scanning `Zeff_target` is refused unless the settings file also gives `impurity`: without it `Zeff_target` has no effect and every cell would come out the same.

`sc.operating_point(T_i_avg=8.0, n_G_frac=0.5)` returns one row per cell at the same point on the grid: the cell's `R`, `a`, `B_0`, `I_P` and `q*`, plus the fusion power, auxiliary power, Q and the rest. Give `n_e_20_avg` instead of `n_G_frac` to hold the density itself rather than the Greenwald fraction, which moves with `I_P / a^2`. At a fixed density and temperature the fusion power does not depend on the field or the confinement factor; those move the power needed to get there, so compare `Paux` and `Q` as well as `Pfusion`.

#### gEQDSK examples

Without further settings, a gEQDSK's own geometry and current win: `__get_geometry` takes `R`, `a`, `kappa`, `delta` and `I_P` from the equilibrium when they differ from the settings by more than 10%, and always takes the ohmic current from it. Scanning any of them is then refused rather than silently giving some identical cells.

`gfile_rescale: True`, which the `MANTA` example sets, uses the equilibrium for its shape and profiles only. Every flux surface is mapped onto the settings `R` and `a`, as `r -> R + (a/a_eq)(r - R_eq)` with `z` scaled the same way about the plasma mid-height, and the current profile carries the settings `I_P`. That makes `R`, `R_0`, `a`, `I_P` and `qstar` scannable; `kappa` and `delta` stay the equilibrium's. The flux surface volumes and areas are exact for the mapped surfaces. The q profile and trapped particle fraction are rescaled to leading order in the inverse aspect ratio, `q ~ a^2 B_0 / (R I_p)` and `f_t ~ sqrt(a/R)`; that is not a new equilibrium, so keep geometry scans near the file's own values.

`sc.write_output()` saves the base settings, the scan specification, the combined arrays and the grid plot; `op.POPCON_scan.read_output(name)` reads it back.

The trapped particle fraction that the neoclassical resistivity needs is taken from the gEQDSK when one is supplied. Without a gEQDSK it falls back to `f_t = sqrt(2*rho*a/R)`, which stays below 1 for the aspect ratios these examples use but would not for a spherical tokamak.

## Development

Run the tests with:

```bash
uv run --extra test pytest
```

The suite (in `tests/`) solves the MANTA example on a small grid and checks the results against a frozen golden file (`tests/data/golden_manta.json`), verifies physics/consistency invariants, checks that settings validation accepts every shipped example while rejecting bad inputs, covers every plot axis pairing including the swapped ones, and runs a small `POPCON_scan` end to end. GitHub Actions runs it on every commit to an open pull request. If a change intentionally alters the computed results, regenerate the golden file and commit it alongside the change:

```bash
uv run python tests/generate_golden.py
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

[9] Y. R. Lin-Liu and R. L. Miller, Upper and lower bounds of the effective trapped particle fraction in general tokamak equilibria, Physics of Plasmas 2, 1666 (1995).

[10] *Compact Experimental Negative TriAngUlarity Reactor (CENTAUR): A design study for a compact, affordable breakeven tokamak,* arXiv:2605.27549. Source of the `CENTAUR` example's machine parameters.




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

class POPCON_scan:
    """Scoping two machine parameters at once. Holds one POPCON per cell,
    each with its own settings, geometry and output, rather than mutating
    one; POPCON_settings.with_overrides re-runs the whole derivation chain
    for each, so nothing derived is left stale."""
    self.cells: list[list[POPCON]]       # [row][col]
    self.row, self.col: ScanAxis         # parameter name and its values
    self.output: xarray.Dataset          # every cell, scan dims first

    def run_scan(self):
        "Solves every cell, one at a time"
    def plot(self):
        "Tiles the cells into one figure on shared contour levels"
    def plot_metric(self, name, reduce):
        "One number per cell, as a heatmap"
    def metric(self, name, reduce):
        "The same numbers, reduced over the valid points only"
```
