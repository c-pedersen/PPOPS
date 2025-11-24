#!/usr/bin/env python3
"""
main.py
---------
Main driver for POPS scattering simulation and calibration.

This script integrates Mie scattering intensity over the POPS instrument’s
solid-angle acceptance, using helper functions from mie_modules.py and geometry.py.

References:
    - Bohren & Huffman (1983), "Absorption and Scattering of Light by Small Particles"
    - C. Mätzler (2002), Mie scattering implementations
"""

import numpy as np
from mie_modules import mie_s12
from geometry import ptz2r_sc


def main():
    """Run the POPS scattering simulation and compute truncated cross-sections.

    This function performs a numerical integration of Mie-scattered intensity
    over the instrument's angular field of view, computing the total light
    collected by the POPS mirror.

    Args:
        None

    Returns:
        None
    """
    # -------------------------------------------------------------------------
    # Instrument Parameters
    # -------------------------------------------------------------------------
    ior = 1.53 + 0j              # Complex refractive index of particle
    laser_wavelength = 0.405     # Wavelength [µm]
    particle_diameter = 0.5      # Diameter [µm]
    h = 7.68 + 2.159             # Distance from scattering region to mirror edge [mm]
    mirror_radius = 12.5         # Mirror radius [mm]
    y0 = 14.2290                 # Mirror center position [mm]

    n_theta = 50                 # Polar angle samples
    n_phi = 40                   # Azimuthal angle samples

    # -------------------------------------------------------------------------
    # Derived Quantities
    # -------------------------------------------------------------------------
    theta_max = np.arctan(mirror_radius / h)
    theta_values = np.linspace(np.pi / 2 - theta_max,
                               np.pi / 2 + theta_max,
                               n_theta)
    size_parameter = np.pi / laser_wavelength * particle_diameter
    r_min = np.sqrt(mirror_radius ** 2 + h ** 2)

    # -------------------------------------------------------------------------
    # Integration Setup
    # -------------------------------------------------------------------------
    integrand = np.zeros((n_theta, n_phi))
    s1 = np.zeros_like(theta_values, dtype=complex)
    s2 = np.zeros_like(theta_values, dtype=complex)

    for j, theta in enumerate(theta_values):
        s12 = mie_s12(ior, size_parameter, np.cos(theta))
        s1[j], s2[j] = s12[0], s12[1]

        phi_max = np.arccos(h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)))
        phi_values = np.linspace(-phi_max, phi_max, n_phi)

        for k, phi in enumerate(phi_values):
            _, _, _, _, ws, wp, _ = ptz2r_sc(phi, theta, y0)
            integrand[j, k] = ws * np.abs(s1[j])**2 + wp * np.abs(s2[j])**2

    # -------------------------------------------------------------------------
    # Double Integration
    # -------------------------------------------------------------------------
    total_signal = 0.0
    for j, theta in enumerate(theta_values):
        phi_max = np.arccos(h / (r_min * np.sqrt(1 - np.cos(theta) ** 2)))
        phi_values = np.linspace(-phi_max, phi_max, n_phi)
        d_phi = phi_values[1] - phi_values[0]
        sum_phi = np.sum(integrand[j, :]) * d_phi
        total_signal += sum_phi * np.sin(theta)

    d_theta = theta_values[1] - theta_values[0]
    total_signal *= d_theta

    # -------------------------------------------------------------------------
    # Compute Cross Sections
    # -------------------------------------------------------------------------
    trunc_qsca = total_signal / size_parameter ** 2
    geometric_cross_section = np.pi * particle_diameter ** 2
    trunc_csca = trunc_qsca * geometric_cross_section

    print("\n--- POPS Scattering Results ---")
    print(f"Truncated Scattering Efficiency (Qsca_trunc): {trunc_qsca:.6e}")
    print(f"Truncated Scattering Cross Section (Csca_trunc): {trunc_csca:.6e}")


if __name__ == "__main__":
    main()

