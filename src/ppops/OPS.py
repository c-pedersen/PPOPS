"""
OPS.py
---------
Defines the OpticalParticleSpectrometer class for simulating Mie
scattering intensity collected by the OPS instrument. The class includes
methods to compute the truncated scattering cross-section based on the
instrument's geometry and optical properties. The implementation uses
numerical integration over the instrument's angular acceptance,
leveraging Mie scattering calculations from mie_modules.py and geometric
transformations from geometry.py.

References:
    - Bohren & Huffman (1983), "Absorption and Scattering of Light by Small Particles"
    - C. Mätzler (2002), Mie scattering implementations
"""

import numpy as np
from miepython.core import S1_S2
from . import detector
from .mie_modules import mie_s12
from .geometry import ptz2r_sc


class OpticalParticleSpectrometer:
    """Class representing the OPS instrument for scattering simulations."""

    def __init__(
        self,
        wavelength: float = 0.405,
        h: float = 7.68 + 2.159,
        mirror_radius: float = 12.5,
        y0: float = 14.2290,
    ):
        """Initialize the OPS instrument parameters.

        Parameters
        ----------
        wavelength : float
            Wavelength of the incident light in micrometers.
        h : float
            Distance from the scattering region to the mirror edge in millimeters.
        mirror_radius : float
            Radius of the mirror in millimeters.
        y0 : float
            Center position of the mirror in millimeters.
        """

        self.wavelength = wavelength
        self.h = h
        self.mirror_radius = mirror_radius
        self.y0 = y0

    def truncated_scattering_cross_section(
        self,
        ior: complex,
        diameter: float,
    ) -> float:
        """
        Simulates OPS scattering and computed truncated cross-sections.

        This function performs a numerical integration of Mie-scattered
        intensity over the instrument's angular field of view, computing
        the total light collected by the OPS mirror.

        Parameters
        ----------
        ior : complex
            Complex refractive index of the particle.
        diameter : float
            Diameter of the particle in micrometers.


        Returns
        -------
        float
            Truncated scattering cross-section in square micrometers.
        """
        n_theta = 50  # Polar angle samples
        n_phi = 40  # Azimuthal angle samples

        # -------------------------------------------------------------------------
        # Derived Quantities
        # -------------------------------------------------------------------------
        theta_max = np.arctan(self.mirror_radius / self.h)
        theta_values = np.linspace(
            np.pi / 2 - theta_max, np.pi / 2 + theta_max, n_theta
        )
        size_parameter = np.pi / self.wavelength * diameter
        r_min = np.sqrt(self.mirror_radius**2 + self.h**2)

        # -------------------------------------------------------------------------
        # Integration Setup
        # -------------------------------------------------------------------------
        integrand = np.zeros((n_theta, n_phi))

        mp_s1s2 = S1_S2(m=ior, x = size_parameter, mu = np.cos(theta_values), norm = 'wiscombe')
        s1 = mp_s1s2[0]
        s2 = mp_s1s2[1]
        for j, theta in enumerate(theta_values):
            phi_max = np.arccos(np.clip(self.h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)), -1, 1))
            phi_values = np.linspace(-phi_max, phi_max, n_phi)

            for k, phi in enumerate(phi_values):
                _, _, _, _, ws, wp, _ = ptz2r_sc(phi, theta, self.h, self.mirror_radius, self.y0)
                integrand[j, k] = ws * np.abs(s1[j]) ** 2 + wp * np.abs(s2[j]) ** 2

        # -------------------------------------------------------------------------
        # Double Integration
        # -------------------------------------------------------------------------
        total_signal = 0.0
        for j, theta in enumerate(theta_values):
            phi_max = np.arccos(np.clip(self.h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)), -1, 1))
            phi_values = np.linspace(-phi_max, phi_max, n_phi)
            d_phi = phi_values[1] - phi_values[0]
            sum_phi = np.sum(integrand[j, :]) * d_phi
            total_signal += sum_phi * np.sin(theta)

        d_theta = theta_values[1] - theta_values[0]
        total_signal *= d_theta

        # -------------------------------------------------------------------------
        # Compute Cross Sections
        # -------------------------------------------------------------------------
        trunc_qsca = total_signal / size_parameter**2
        geometric_cross_section = np.pi * (diameter / 2)**2
        trunc_csca = trunc_qsca * geometric_cross_section

        return trunc_csca
    

    def estimate_signal(
        self,
        ior: complex,
        diameter: float,
        laser_power: float = 70,
    ) -> tuple[float, float]:
        """
        Estimate the signal amplitude from the scattered light incident 
        on the OPS photomultiplier tube (PMT).

        Parameters
        ----------
        ior : complex
            Complex refractive index of the particle.
        diameter : float
            Diameter of the particle in micrometers.
        laser_power : float
            Laser power in mW. Default is 70 mW.

        Returns
        -------
        tuple[float, float]
            Tuple containing:
            - signal : float units of Amperes (A)
            - noise : float units of Amperes (A)
        """

        trunc_csca = self.truncated_scattering_cross_section(ior, diameter)
        signal, noise = detector.estimate_signal_noise(trunc_csca, laser_power)
        
        return signal, noise
