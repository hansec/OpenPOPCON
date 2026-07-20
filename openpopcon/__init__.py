"""
OpenPOPCON: a 0-D tokamak scoping tool.

    import openpopcon as op
    pc = op.POPCON(settingsfile='POPCON_input_example.yaml')
    pc.run_POPCON()
    pc.plot()

Copies of the worked examples can be written somewhere editable with

    openpopcon examples ./my_scans

Names not listed here live in openpopcon.core, which is importable if you need
the physics functions or the State namedtuple directly.
"""

from .core import (
    POPCON,
    POPCON_scan,
    POPCON_settings,
    POPCON_plotsettings,
    POPCON_data,
    POPCON_algorithms,
    POPCON_data_spec,
)
from .lib.openpopcon_util import example_dir, list_examples

__version__ = "2.0.0"

__all__ = [
    'POPCON',
    'POPCON_scan',
    'POPCON_settings',
    'POPCON_plotsettings',
    'POPCON_data',
    'POPCON_algorithms',
    'POPCON_data_spec',
    'example_dir',
    'list_examples',
    '__version__',
]
