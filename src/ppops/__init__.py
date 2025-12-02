"""
PPOPS - a python package for Performance Prediction of Optical Particle
Spectrometers.

This package provides tools for calculating the Mie scattering of light
by spherical particles and calculating the collection efficiency of
optical particle spectrometers based on their geometry and optical
design.
"""

from importlib.metadata import version

__version__ = version("ppops")
__author__ = "Corey Pedersen, Sophie Abou-Rizk, and Andrew Chu"
__license__ = "MIT"
__copyright__ = "2025, Corey Pedersen, Sophie Abou-Rizk, and Andrew Chu"
__email__ = "coreyped@gmail.com"
__url__ = "https://github.com/c-pedersen/PPOPS"

from . import mie_modules
from . import geometry
from . import detector
from .OPS import OpticalParticleSpectrometer

__all__ = [
    "OpticalParticleSpectrometer",
    "mie_modules",
    "geometry",
    "detector",
]
