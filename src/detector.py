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
    - Thor Labs. TIA60 Transimpedance Amplifier Datasheet.
    https://www.thorlabs.com/drawings/c627bb63fbc792f9-BE0A1D53-FE71-D4E9-369F32E2683F2FB6/TIA60-SpecSheet.pdf
    - Hamamatsu Photonics. H10720 Series Photomultiplier Tube Datasheet.
    https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/etd/H10720_H10721_TPMO1062E.pdf

"""

from warnings import warn
import math

import numpy as np
from numpy.typing import NDArray

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
    beam_major: float = 3e-3,
    beam_minor: float = 1e-3,
) -> float:
    """
    Return the laser power density at the aerosol stream. The assumed
    beam geometry is an oval. Note: the beam dimensions are taken from
    Gao et al. 2016 at the laser assembly exit, not at the aerosol
    stream. This function assumes the beam does not diverge
    significantly over this optical path which may or may not be a valid
    assumption.

    Parameters
    ----------
    laser_power : float
        Laser power in mW.
    beam_major : float
        Length of major axis of the oval laser beam at the aerosol stream in meters.
    beam_minor : float
        Length of minor axis of the oval laser beam at the aerosol stream in meters.

    Returns
    -------
    float
        Optical power density (W/m^2).
    """

    if beam_major <= 0 or beam_minor <= 0:
        raise ValueError("Beam major and minor axes must be positive values.")
    if beam_major > 1e-2 or beam_minor > 1e-2 or beam_major < 1e-5 or beam_minor < 1e-5:
        warn(
            "Beam dimensions in meters seem unrealistic. "
            "Please verify the input values."
        )
    if laser_power < 0:
        raise ValueError("Laser power cannot be negative.")
    if laser_power > 1000 or laser_power <= 10:
        warn("Laser power in mW seems unrealistic. Please verify the input value.")

    beam_area = math.pi * (beam_major / 2 * 1e6) * (beam_minor / 2 * 1e6)  # µm^2

    return laser_power * 1e-3 / beam_area  # W/µm^2


def estimate_signal_noise(
    truncated_csca: float | NDArray[np.float64],
    laser_power: float,
    anode_radiant_sensitivity: float = H10720_110_ANODE_RADIANT_SENSITIVITY,
    dark_current: float = H10720_110_DARK_CURRENT,
    bandwidth: float = BANDWIDTH,
    input_current_noise: float = TIA60_INPUT_CURRENT_NOISE,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """
    Return signal and noise estimates for a given truncated single
    scattering cross section and laser power.

    Parameters
    ----------
    truncated_csca : float | NDArray[np.float64]
        Truncated scattering cross section in units of µm².
    laser_power : float
        Laser power in mW.
    anode_radiant_sensitivity : float
        Anode radiant sensitivity in A/W. Default is 2.2e5 A/W.
    dark_current : float
        Dark current noise in A. Default is 1e-9 A.
    bandwidth : float
        Bandwidth in Hz. Default is 4e6 Hz.
    input_current_noise : float
        Preamplifier input current noise in A/√Hz. Default is 4.8e-12 A/√Hz.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[float, float]
        Tuple containing:
        - signal_current: Estimated signal current (A).
        - noise: Estimated noise (A).
    """

    signal_current = (
        truncated_csca * laser_power_density(laser_power) * anode_radiant_sensitivity
    )  # A

    signal_noise = 2 * ELEMENTARY_CHARGE * signal_current  # C^2 s^-1
    dark_noise = 2 * ELEMENTARY_CHARGE * dark_current  # C^2 s^-1
    preamp_noise = input_current_noise**2  # C^2 s^-1
    total_noise = np.sqrt((signal_noise + dark_noise + preamp_noise) * bandwidth)  # A

    return signal_current, total_noise
