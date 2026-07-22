"""
Minimal OpenPOPCON run, using the MANTA example that ships with the package.
From a clone of the repository:

    uv run example_run.py

or, once installed with pip, from anywhere:

    python example_run.py

To start changing the machine, get an editable copy of the examples with
`openpopcon examples ./my_scans` and point settingsfile at one of those.
"""

import os

import matplotlib.pyplot as plt

import openpopcon as op

manta = op.example_dir("MANTA")
settingsfile = os.path.join(manta, "POPCON_input_example.yaml")
plotsettingsfile = os.path.join(manta, "plotsettings.yml")

# scalinglawfile defaults to the copy shipped with the package
pc = op.POPCON(settingsfile=settingsfile, plotsettingsfile=plotsettingsfile)

# solve power balance over the whole density/temperature grid
pc.run_POPCON()

# the POPCON itself
fig, ax = pc.plot(show=False)
fig.savefig("POPCON.png", dpi=150, bbox_inches="tight")
print("Wrote POPCON.png")

# profiles and a power breakdown at one operating point,
# here 60% of the Greenwald density and 9 keV volume-averaged Ti
pc.single_point(0.6, 9.0, plot=True, show=False)
plt.savefig("single_point.png", dpi=150, bbox_inches="tight")
print("Wrote single_point.png")
