"""
Beat Detection - A package for detecting beats in music and adding metronome sounds.
"""

__version__ = "0.1.0"

from . import core
from . import utils
from . import cli

__all__ = ["core", "utils", "cli"]
