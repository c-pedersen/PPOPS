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
from numpy.typing import ArrayLike
from miepython.core import S1_S2
from . import detector
from .geometry import ptz2r_sc
from .mirror import mirror_depth


class OpticalParticleSpectrometer:
    """Class representing the OPS instrument for scattering simulations."""

    def __init__(
        self,
        laser_wavelength: float = 0.405,
        laser_power: float = 70,
        laser_polarization: str = "horizontal",
        mirror_radius: float = 12.5,
        mirror_radius_of_curvature: float = 20.0,
        aerosol_mirror_separation: float = 14.2290,
    ):
        """Initialize the OPS instrument parameters.

        Parameters
        ----------
        laser_wavelength : float
            Wavelength of the incident light in micrometers.
        laser_power : float
            Laser power in milliwatts.
        laser_polarization : str
            Polarization state of the incident laser light. Options are
            'unpolarized', 'horizontal', or 'vertical'. Default is 
            'horizontal'.
        mirror_radius : float
            Radius of the spherical mirror in millimeters.
        mirror_radius_of_curvature : float
            Radius of curvature of the spherical mirror in millimeters.
        aerosol_mirror_separation : float
            Separation between the aerosol and the center of the mirror
            in millimeters.
        """

        self.laser_wavelength = laser_wavelength
        self.laser_power = laser_power
        self.laser_polarization = laser_polarization
        self.mirror_radius = mirror_radius
        self.mirror_radius_of_curvature = mirror_radius_of_curvature
        self.aerosol_mirror_separation = aerosol_mirror_separation
        self.y0 = aerosol_mirror_separation
        # Axial distance from aerosol stream to the top edge of mirror.
        self.h = aerosol_mirror_separation - mirror_depth(
            mirror_radius=mirror_radius, radius_of_curvature=mirror_radius_of_curvature
        )

    def truncated_scattering_cross_section(
        self,
        ior: complex,
        diameter: float,
        n_theta: int = 50,
        n_phi: int = 40,
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
        n_theta : int
            Number of polar angle samples for integration.
        n_phi : int
            Number of azimuthal angle samples for integration.

        Returns
        -------
        float
            Truncated scattering cross-section in square micrometers.
        """
        if not isinstance(n_theta, int) or n_theta <= 0:
            raise ValueError("n_theta must be a positive integer.")
        if not isinstance(n_phi, int) or n_phi <= 0:
            raise ValueError("n_phi must be a positive integer.")

        # -------------------------------------------------------------------------
        # Derived Quantities
        # -------------------------------------------------------------------------
        theta_max = np.arctan(self.mirror_radius / self.h)
        theta_values = np.linspace(
            np.pi / 2 - theta_max, np.pi / 2 + theta_max, n_theta
        )
        size_parameter = np.pi / self.laser_wavelength * diameter
        r_min = np.sqrt(self.mirror_radius**2 + self.h**2)

        # -------------------------------------------------------------------------
        # Integration Setup
        # -------------------------------------------------------------------------
        integrand = np.zeros((n_theta, n_phi))

        mp_s1s2 = S1_S2(m=ior, x=size_parameter, mu=np.cos(theta_values), norm="qsca")
        s1 = mp_s1s2[0]
        s2 = mp_s1s2[1]
        for j, theta in enumerate(theta_values):
            phi_max = np.arccos(
                np.clip(self.h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)), -1, 1)
            )
            phi_values = np.linspace(-phi_max, phi_max, n_phi)

            for k, phi in enumerate(phi_values):
                _, _, _, _, ws, wp, _ = ptz2r_sc(
                    phi=phi,
                    theta=theta,
                    mirror_radius=self.mirror_radius,
                    mirror_radius_of_curvature=self.mirror_radius_of_curvature,
                    y0=self.y0,
                    h=self.h,
                    laser_polarization=self.laser_polarization,
                )
                integrand[j, k] = ws * np.abs(s1[j]) ** 2 + wp * np.abs(s2[j]) ** 2

        # -------------------------------------------------------------------------
        # Double Integration
        # -------------------------------------------------------------------------
        total_signal = 0.0
        for j, theta in enumerate(theta_values):
            phi_max = np.arccos(
                np.clip(self.h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)), -1, 1)
            )
            phi_values = np.linspace(-phi_max, phi_max, n_phi)
            d_phi = phi_values[1] - phi_values[0]
            sum_phi = np.sum(integrand[j, :]) * d_phi
            total_signal += sum_phi * np.sin(theta)

        d_theta = theta_values[1] - theta_values[0]
        total_signal *= d_theta

        # -------------------------------------------------------------------------
        # Compute Cross Sections
        # -------------------------------------------------------------------------
        trunc_qsca = total_signal
        geometric_cross_section = np.pi * (diameter / 2) ** 2
        trunc_csca = trunc_qsca * geometric_cross_section

        return trunc_csca

    def estimate_signal_noise(
        self,
        ior: complex,
        diameters: float | ArrayLike,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Estimate the signal amplitude from the scattered light incident
        on the OPS photomultiplier tube (PMT).

        Parameters
        ----------
        ior : complex
            Complex refractive index of the particle.
        diameters : float | np.ndarray
            Diameter of the particle in micrometers.

        Returns
        -------
        tuple[float, float]
            Tuple containing:
            - signal : float units of Amperes (A)
            - noise : float units of Amperes (A)
        """
        try:
            diameters = np.asarray(diameters, dtype=float)
        except Exception as e:
            raise TypeError(
                "Diameters must be convertible to a numpy array of floats"
            ) from e

        trunc_csca = np.array([])
        for diameter in diameters:
            trunc_csca = np.append(
                trunc_csca,
                self.truncated_scattering_cross_section(ior, diameter),
            )

        signal, noise = detector.estimate_signal_noise(trunc_csca, self.laser_power)

        return signal, noise
