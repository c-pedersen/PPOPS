"""
detector.py
------------
Handles detector signal and noise calculations.

This module includes functions to compute the expected signal and noise
levels for a detector based on the hardware specifications.

Constants:
    - ELEMENTARY_CHARGE: Electron charge (C)
    - ANODE_RADIANT_SENSITIVITY: Anode radiant sensitivity (A/W)
    - DARK_CURRENT: Dark current noise (A)
    - BANDWIDTH: Bandwidth (Hz)
    - INPUT_CURRENT_NOISE: Preamplifier input current noise (A/√Hz)

Functions:
    - laser_power_density(laser_power, beam_major, beam_minor): Computes
      the laser power density at the aerosol stream.
    - estimate_signal_noise(truncated_csca, laser_power): Estimates
      signal and noise levels for a given truncated scattering cross
      section and laser power.

Citations:
    - Gao, R.S., et al. 2016. A light-weight, high-sensitivity particle
      spectrometer for PM2.5 aerosol measurements. Aerosol Science and
      Technology 50, 88-99.
      https://doi.org/10.1080/02786826.2015.1131809
    - Thor Labs. TIA60 Transimpedance Amplifier Datasheet. https://www.thorlabs.com/drawings/c627bb63fbc792f9-BE0A1D53-FE71-D4E9-369F32E2683F2FB6/TIA60-SpecSheet.pdf
    - Hamamatsu Photonics. H10720 Series Photomultiplier Tube Datasheet. https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/etd/H10720_H10721_TPMO1062E.pdf

"""

from __future__ import annotations
from typing import TYPE_CHECKING

from warnings import warn
import math

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .OPS import OpticalParticleSpectrometer

# Physical constants
ELEMENTARY_CHARGE = 1.602176634e-19  # C

# Detector specifications - Hamamatsu H10720-110 PMT
H10720_110_ANODE_RADIANT_SENSITIVITY = 2.2e5  # A/W
H10720_110_DARK_CURRENT = 1e-9  # typical dark current (A)

# Specifications from Gao et al. 2016
BANDWIDTH = 4e6  # bandwidth (Hz)

# Preamplifier specifications - Thor TIA60 PMT amplifier
TIA60_INPUT_CURRENT_NOISE = 4.8e-12  # A/sqrt(Hz)


def laser_power_density(
    laser_power: float,
    beam_major: float = 6,
    beam_minor: float = 0.054,
) -> float:
    """Return the laser power density at the aerosol stream.

    The inital beam dimensions are taken from Gao et al. 2016 and the
    beam waist at the aerosol stream is estimated below assuming a
    Gaussian beam profile. The laser power density is then calculated as
    the laser power divided by the beam area at the aerosol stream
    assuming a uniform power distribution.

    Horizontal beam waist at aerosol stream:
    spot_diameter = 4M² * λ * L / (π * w_initial) = 0.054 mm
    where λ = 405 nm, L = 75 mm, w_initial_horizontal = 1 mm, M² = 1.4
    (assumed).

    Vertical beam waist at aerosol stream:
    spot_diameter = 2M² * λ * L / (π * w_initial)
    where L = 25 mm, w_initial_vertical = 3 mm.
    DOF = 2 * π * (spot_diameter / 2)² / (M² * λ)
    beam_diameter = spot_diameter * sqrt(
        1 + (distance_from_lens - L)² / (DOF/2)²) = 6 mm

    Parameters
    ----------
    laser_power : float
        Laser power in mW.
    beam_major : float, optional
        Length of major axis of the oval laser beam at the aerosol
        stream in millimeters. Default is 6.
    beam_minor : float, optional
        Length of minor axis of the oval laser beam at the aerosol
        stream in millimeters. Default is 0.054.
    Returns
    -------
    float
        Optical power density (W/µm^2).

    References
    ----------
    Gao, R.S., et al. 2016. A light-weight, high-sensitivity particle
    spectrometer for PM2.5 aerosol measurements. Aerosol Science and
    Technology 50, 88-99. https://doi.org/10.1080/02786826.2015.1131809

    Laser spot size and beam waist calculator and formulas. Gentec.
    (n.d.).
    https://www.gentec-eo.com/laser-calculators/beam-waist-spot-size
    """

    if beam_major <= 0 or beam_minor <= 0:
        raise ValueError("Beam major and minor axes must be positive values.")
    if beam_major > 10 or beam_minor > 10 or beam_major < 1e-3 or beam_minor < 1e-3:
        warn(
            "Beam dimensions in millimeters seem unrealistic. "
            "Please verify the input values."
        )

    beam_area = math.pi * (beam_major / 2 * 1e3) * (beam_minor / 2 * 1e3)  # µm^2
    return laser_power * 1e-3 / beam_area  # W/µm^2


def estimate_signal_noise(
    ops: OpticalParticleSpectrometer,
    truncated_csca: float | NDArray[np.float64],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Return signal and noise estimates.

    Computes signal and noise estimates for a given truncated single
    scattering cross section and laser power. This function uses the
    maximum gain settings for the PMT and preamplifier. This is expected
    to be larger than the actual signal and noise levels in typical
    operation, but provides a useful upper bound.

    Parameters
    ----------
    ops : OpticalParticleSpectrometer
        Instance of the OpticalParticleSpectrometer class.
    truncated_csca : float or np.ndarray
        Truncated scattering cross section in units of µm².

    Returns
    -------
    float or np.ndarray
        Estimated signal current (A).
    float or np.ndarray
        Estimated noise (A).
    """

    signal_current = (
        truncated_csca  # µm²
        * laser_power_density(ops.laser_power)  # W/µm²
        * ops.anode_radiant_sensitivity  # A/W
        * ops.mirror_reflectivity  # unitless
    )  # A

    signal_noise = 2 * ELEMENTARY_CHARGE * signal_current  # C^2 s^-1
    dark_noise = 2 * ELEMENTARY_CHARGE * ops.dark_current  # C^2 s^-1
    preamp_noise = ops.input_current_noise**2  # C^2 s^-1
    total_noise = np.sqrt(
        (signal_noise + dark_noise + preamp_noise) * ops.bandwidth
    )  # A

    return signal_current, total_noise
