"""
Minimal OpenPOPCON run. From the repository root:

    uv run example_run.py

or, with the conda environment activated:

    python example_run.py

Copy resources/examples/MANTA/ somewhere and point settingsfile at your own
copy to start changing the machine.
"""
import matplotlib.pyplot as plt

from src import openpopcon as op

settingsfile = "./resources/examples/MANTA/POPCON_input_example.yaml"
plotsettingsfile = "./resources/examples/MANTA/plotsettings.yml"
scalinglawfile = "./resources/scalinglaws.yml"

pc = op.POPCON(settingsfile=settingsfile,
               plotsettingsfile=plotsettingsfile,
               scalinglawfile=scalinglawfile)

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
