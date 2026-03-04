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

from warnings import warn
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
        pmt_control_voltage: float = 0.619,
        dark_current=detector.H10720_110_DARK_CURRENT,
        bandwidth=detector.BANDWIDTH,
        input_current_noise=detector.TIA60_INPUT_CURRENT_NOISE,
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
        pmt_control_voltage : float, default 0.619
            Control voltage of the PMT in volts. Default is 0.619 V,
            which was measured on a Handix POPS. Adjust for different
            instruments.
        dark_current : float, default 1e-9 (see detector.py)
            Dark current of the detector in Amperes.
        bandwidth : float, default 4e6 (see detector.py)
            Bandwidth of the detector in Hertz.
        input_current_noise : float, default 4.8e-12 (see detector.py)
            Input current noise of the detector in Amperes per square
            root Hertz.
        """

        # Ensure inputs are valid
        if np.any(laser_wavelength <= 0):
            raise ValueError("All laser wavelengths must be positive.")
        if np.any(laser_power <= 0):
            raise ValueError("All laser powers must be positive.")
        if laser_polarization not in ["unpolarized", "horizontal", "vertical"]:
            raise ValueError(
                "laser_polarization must be 'unpolarized', 'horizontal', or 'vertical'."
            )
        if mirror_radius <= 0 or mirror_radius_of_curvature <= 0:
            raise ValueError("Mirror radius and radius of curvature must be positive.")
        if aerosol_mirror_separation <= 0:
            raise ValueError("Aerosol-mirror separation must be positive.")

        # Ensure variables are close to expected mangitude
        if np.any(laser_wavelength > 2) or np.any(laser_wavelength < 0.2):
            warn(
                "Laser wavelength outside of expected range. "
                "Verify laser wavelength is in micrometers."
            )
        if np.any(laser_power > 1000) or np.any(laser_power < 1):
            warn(
                "Laser power outside of expected range. "
                "Verify laser power is in milliwatts."
            )
        if mirror_radius > 100 or mirror_radius < 1:
            warn(
                "Mirror radius outside of expected range. "
                "Verify mirror radius is in millimeters."
            )
        if mirror_radius_of_curvature > 200 or mirror_radius_of_curvature < 5:
            warn(
                "Mirror radius of curvature outside of expected range. "
                "Verify radius of curvature is in millimeters."
            )
        if aerosol_mirror_separation > 100 or aerosol_mirror_separation < 5:
            warn(
                "Aerosol-mirror separation outside of expected range. "
                "Verify separation is in millimeters."
            )

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
        self.pmt_control_voltage = pmt_control_voltage
        self.dark_current = dark_current
        self.bandwidth = bandwidth
        self.input_current_noise = input_current_noise

        self.mirror_reflectivity = 0.80  # Edmund Optics #43-464 at 405 nm
        self.anode_radiant_sensitivity = detector.anode_radiant_sensitivity(
            self.pmt_control_voltage
        )

    def truncated_scattering_cross_section(
        self,
        ri: complex,
        diameter: float,
        n_theta: int = 51,
        n_phi: int = 41,
    ) -> NDArray[np.floating]:
        """
        Simulates OPS scattering and computed truncated cross-sections.

        This function performs a numerical integration of Mie-scattered
        intensity over the instrument's angular field of view, computing
        the total light collected by the OPS mirror.

        Parameters
        ----------
        ri : complex
            Complex refractive index of the particle.
        diameter : float
            Diameter of the particle in micrometers.
        n_theta : int, default 51
            Number of polar angle samples for integration. Should be an
            odd integer for Simpson's rule.
        n_phi : int, default 41
            Number of azimuthal angle samples for integration.Should be
            an odd integer for Simpson's rule.

        Returns
        -------
        np.ndarray
            Truncated scattering cross-section in square micrometers.
        """
        # Validate inputs
        if np.any(np.imag(ri) < 0) or np.any(np.real(ri) < 1):
            raise ValueError(
                "Imaginary part of refractive indices must be non-negative and real"
                "part must be >= 1."
            )
        if diameter <= 0:
            raise ValueError("Diameter must be a positive value.")
        if diameter < 0.001 or diameter > 100:
            warn(
                "Diameter outside of expected range. Verify diameter is in micrometers."
            )

        if not isinstance(n_theta, int) or n_theta <= 0:
            raise ValueError("n_theta must be a positive, odd integer.")
        if n_theta % 2 == 0:
            warn("n_theta should be an odd integer for Simpson's rule.")
        if not isinstance(n_phi, int) or n_phi <= 0:
            raise ValueError("n_phi must be a positive, odd integer.")
        if n_phi % 2 == 0:
            warn("n_phi should be an odd integer for Simpson's rule.")

        # Derived quantities
        theta_max = np.arctan(self.mirror_radius / self.h)
        theta_values = np.linspace(
            np.pi / 2 - theta_max, np.pi / 2 + theta_max, n_theta
        )
        size_parameter = np.pi / self.laser_wavelength * diameter
        r_min = np.sqrt(self.mirror_radius**2 + self.h**2)

        # Compute S1, S2
        mp_s1s2 = S1_S2(m=ri, x=size_parameter, mu=np.cos(theta_values), norm="qsca")
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
        ri: complex,
        diameters: float | ArrayLike,
        n_theta: int | None = None,
        n_phi: int | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Estimate the signal amplitude from the scattered light incident
        on the OPS photomultiplier tube (PMT).

        Parameters
        ----------
        ri : complex
            Complex refractive index of the particle.
        diameters : float | np.ndarray
            Diameter of the particle in micrometers.
        n_theta : int | None
            Number of polar angle samples for integration. If None, uses
            default value in truncated_scattering_cross_section.
        n_phi : int | None
            Number of azimuthal angle samples for integration. If None,
            uses default value in truncated_scattering_cross_section.

        Returns
        -------
        signal : float or np.ndarray
            Estimated signal amplitude in units of Amperes (A).
        noise : float or np.ndarray
            Estimated noise amplitude in units of Amperes (A).
        """
        try:
            if isinstance(diameters, (float, int)):
                diameters = np.array([diameters], dtype=float)
            else:
                diameters = np.asarray(diameters, dtype=float)
        except Exception as e:
            raise TypeError(
                "Diameters must be convertible to a numpy array of floats"
            ) from e

        kwargs = {}
        if n_theta is not None:
            kwargs["n_theta"] = n_theta
        if n_phi is not None:
            kwargs["n_phi"] = n_phi

        trunc_csca = []
        for diameter in diameters:
            trunc_csca.append(
                self.truncated_scattering_cross_section(ri, diameter, **kwargs)
            )
        trunc_csca = np.array(trunc_csca, dtype=float)

        signal, noise = detector.estimate_signal_noise(self, trunc_csca)

        return signal, noise


def digitize_signal(
    signal_current: float | NDArray[np.float64],
    max_voltage: float = 5,
    digitizer_bins: int = 65536,
    feedback_resistor: float = 2050,
) -> float | NDArray[np.float64]:
    """Convert signal current to digitizer bins.

    Parameters
    ----------
    signal_current : float or np.ndarray
        Signal current in Amperes (A).
    max_voltage : float, default 5
        Maximum voltage of the digitizer in Volts (V).
    digitizer_bins : int, default 65536
        Number of bins in the digitizer (16-bit digitizer).
    feedback_resistor : float, default 2050
        Feedback resistor value in Ohms (Ω) used in the transimpedance
        amplifier. Default is 2050 Ω, based on correspondence with
        Handix Scientific.

    Returns
    -------
    float or np.ndarray
        Signal amplitude in digitizer bins.
    """
    if np.any(signal_current < 0):
        raise ValueError("Signal current must be non-negative.")

    return signal_current * feedback_resistor * (digitizer_bins / max_voltage)
