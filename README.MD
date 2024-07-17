# OpenPOPCON v2.0 development version
# ===================================

OpenPOPCON is a tool for scoping Tokamak operation with 0d fitted scaling laws. This version has been refactored from the original developed for MIT 22.63, with major contributions from Sam Frank, Richard Nies, Tal Rubin, Oak Nelson, Matthew Pharr, Leonardo Corsaro, and many minor contributions from others. This code is intended for use in Columbia's Fusion Reactor Design course.

## Installation

We recommend the use of miniconda to manage the python environment. On a cluster, you may use `module load anaconda` to load anaconda. If you are running on your private machine, see [the anaconda project](https://docs.anaconda.com/miniconda/) for installation instructions.

To install the required packages, run `conda env create -f environment.yml` in the root directory of this repository. This will create a new environment called `openpopcon` with all the required packages. From here, you can activate the environment with `conda activate openpopcon`. 