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
from numpy.typing import ArrayLike, NDArray
import scipy
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
        anode_radiant_sensitivity = detector.H10720_110_ANODE_RADIANT_SENSITIVITY,
        dark_current = detector.H10720_110_DARK_CURRENT,
        bandwidth = detector.BANDWIDTH,
        input_current_noise = detector.TIA60_INPUT_CURRENT_NOISE,
    ):
        """Initialize the OPS instrument parameters.

        Parameters
        ----------
        laser_wavelength : float
            Wavelength of the incident light in micrometers.
        laser_power : float, default 70
            Laser power in milliwatts.
        laser_polarization : str, default 'horizontal'
            Polarization state of the incident laser light. Options are
            'unpolarized', 'horizontal', or 'vertical'. Default is
            'horizontal'.
        mirror_radius : float, default 12.5
            Radius of the spherical mirror in millimeters.
        mirror_radius_of_curvature : float, default 20.0
            Radius of curvature of the spherical mirror in millimeters.
        aerosol_mirror_separation : float, default 14.2290
            Separation between the aerosol and the center of the mirror
            in millimeters.
        anode_radiant_sensitivity : float, default 2.2e5 (see detector.py)
            Anode radiant sensitivity of the detector in Amperes per 
            Watt.
        dark_current : float, default 1e-9 (see detector.py)
            Dark current of the detector in Amperes.
        bandwidth : float, default 4e6 (see detector.py)
            Bandwidth of the detector in Hertz.
        input_current_noise : float, default 4.8e-12 (see detector.py)
            Input current noise of the detector in Amperes per square 
            root Hertz.
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
        self.anode_radiant_sensitivity = anode_radiant_sensitivity
        self.dark_current = dark_current
        self.bandwidth = bandwidth
        self.input_current_noise = input_current_noise


    def truncated_scattering_cross_section(
        self,
        ior: complex,
        diameter: float,
        n_theta: int = 50,
        n_phi: int = 40,
    ) -> NDArray[np.floating]:
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
        np.ndarray
            Truncated scattering cross-section in square micrometers.
        """
        if not isinstance(n_theta, int) or n_theta <= 0:
            raise ValueError("n_theta must be a positive integer.")
        if not isinstance(n_phi, int) or n_phi <= 0:
            raise ValueError("n_phi must be a positive integer.")

        # Derived quantities
        theta_max = np.arctan(self.mirror_radius / self.h)
        theta_values = np.linspace(
            np.pi / 2 - theta_max, np.pi / 2 + theta_max, n_theta
        )
        size_parameter = np.pi / self.laser_wavelength * diameter
        r_min = np.sqrt(self.mirror_radius**2 + self.h**2)

        # Compute S1, S2
        mp_s1s2 = S1_S2(m=ior, x=size_parameter, mu=np.cos(theta_values), norm="qsca")
        s1_sq = np.abs(mp_s1s2[0]) ** 2
        s2_sq = np.abs(mp_s1s2[1]) ** 2

        # Build complete grid of (theta, phi) pairs
        # Preallocate numpy arrays
        n_points = n_theta * n_phi
        all_thetas = np.zeros(n_points)
        all_phis = np.zeros(n_points)
        theta_indices = np.zeros(n_points, dtype=int)
        phi_values_per_theta = []  # Store phi arrays for later

        idx = 0
        for j, theta in enumerate(theta_values):
            phi_max = np.arccos(
                np.clip(self.h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)), -1, 1)
            )
            phi_values = np.linspace(-phi_max, phi_max, n_phi)
            phi_values_per_theta.append(phi_values)

            # Fill arrays using slicing
            all_thetas[idx : idx + n_phi] = theta
            all_phis[idx : idx + n_phi] = phi_values
            theta_indices[idx : idx + n_phi] = j
            idx += n_phi

        # Single vectorized call for ALL geometry calculations
        _, _, _, _, ws, wp, _ = ptz2r_sc(
            ops=self,
            phi=all_phis,
            theta=all_thetas,
        )

        # Compute integrand for all points
        integrand = ws * s1_sq[theta_indices] + wp * s2_sq[theta_indices]

        # Reshape back to (n_theta, n_phi) grid
        integrand_grid = integrand.reshape(n_theta, n_phi)

        # Use Simpson's rule for both dimensions
        theta_integrand = np.zeros(n_theta)
        for j in range(n_theta):
            # Integrate over phi using Simpson's rule
            theta_integrand[j] = scipy.integrate.simpson(
                integrand_grid[j, :], x=phi_values_per_theta[j]
            ) * np.sin(theta_values[j])

        # Integrate over theta using Simpson's rule
        total_signal = scipy.integrate.simpson(theta_integrand, x=theta_values)

        geometric_cross_section = np.pi * (diameter / 2) ** 2
        truncated_csca = np.array(total_signal * geometric_cross_section, dtype=float)
        return truncated_csca

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
        signal : float or np.ndarray
            Estimated signal amplitude in units of Amperes (A).
        noise : float or np.ndarray
            Estimated noise amplitude in units of Amperes (A).
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

        signal, noise = detector.estimate_signal_noise(self, trunc_csca)

        return signal, noise
