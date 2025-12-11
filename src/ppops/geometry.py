"""
geometry.py
------------
Handles POPS geometry and polarization calculations.

This module includes functions for computing geometric parameters related to
light scattering from the particle-laser interaction zone to the POPS mirror.
It also calculates polarization weighting factors (s- and p-components) for
the instrument’s optical collection efficiency.

Functions:
    ptz2r_sc(phi, theta, y0): Computes mirror intersection geometry, maximum
    azimuthal angle, and polarization weights.
"""

import numpy as np


def ptz2r_sc(
    phi: float,
    theta: float,
    mirror_radius: float,
    mirror_radius_of_curvature: float,
    y0: float,
    h: float,
):
    """
    Compute POPS mirror geometry and polarization weighting.

    This function calculates the intersection of a scattered light ray with
    the spherical POPS mirror and determines polarization weighting factors
    (s- and p-polarization) based on instrument geometry.

    Parameters:
    -----------
    phi : float
        Azimuthal scattering angle [radians].
    theta : float
        Polar scattering angle [radians].
    mirror_radius : float
        Radius of the POPS mirror [mm].
    mirror_radius_of_curvature : float
        Radius of curvature of the POPS mirror [mm].
    y0 : float
        Distance from mirror vertex to particle-laser interaction
        region along the mirror axis [mm].
    h : float
        Height of the particle-laser interaction region above
        the mirror vertex [mm].

    Returns:
    --------
    rp : float
        Positive intersection distance from the scattering
        region to the mirror surface [mm].
    rm : float
        Negative intersection distance (unphysical, retained
        for completeness).
    x : np.ndarray
        Cartesian coordinates of intersection vector.
    phi_max : float
        Maximum azimuthal half-angle collected by mirror.
    ws : float
        s-polarization weighting factor (perpendicular).
    wp : float
        p-polarization weighting factor (parallel).
    obf : float
        Obliquity factor (cosine of incidence angle).
    """
    # -------------------------------------------------------------------------
    # Geometric setup: ray direction and intersection with spherical mirror
    # -------------------------------------------------------------------------
    alpha = np.sin(phi)
    mu = np.cos(theta)

    # Coefficients for quadratic mirror-intersection equation
    ax = alpha * np.sqrt(1 - mu**2)
    ay = np.sqrt(1 - alpha**2) * np.sqrt(1 - mu**2)
    az = mu
    a = ax**2 + ay**2 + az**2
    b = 2 * y0 * ay
    c = y0**2 - mirror_radius_of_curvature**2

    # Quadratic solutions for intersection distance
    rp = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    rm = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # Compute maximum azimuthal collection angle
    r_min = np.sqrt(h**2 + mirror_radius**2)
    phi_max = np.arccos(np.clip(h / (r_min * np.sin(theta)), -1, 1))

    # Cartesian coordinates of intersection point
    x = rp * np.array(
        [
            np.sin(phi) * np.sqrt(1 - np.cos(theta) ** 2),
            np.cos(phi) * np.sqrt(1 - np.cos(theta) ** 2),
            np.cos(theta),
        ]
    )

    # -------------------------------------------------------------------------
    # Compute mirror orientation and obliquity factor
    # -------------------------------------------------------------------------
    x_norm = x / np.linalg.norm(x)
    s_norm = x - np.array([0, -y0, 0])
    s_norm = s_norm / np.linalg.norm(s_norm)

    # Obliquity factor: cosine between incident ray and mirror normal
    obf = np.dot(x_norm, s_norm)

    # -------------------------------------------------------------------------
    # Polarization weighting computation
    # -------------------------------------------------------------------------
    e0 = np.array([1, 0, 0])  # Laser electric field (polarization direction)
    k_i = np.array([0, 0, 1])  # Incident laser wave vector (+z direction)
    k_s = x_norm  # Scattered wave vector

    n_vec = np.cross(k_i, k_s)  # Normal to scattering plane
    n2 = np.dot(n_vec, n_vec)  # Magnitude squared

    # Handle degenerate case: forward or backward scattering (no defined plane)
    if n2 < 1e-12:
        ws, wp = 1.0, 0.0
    else:
        n_hat = n_vec / np.sqrt(n2)
        e_s = np.dot(e0, n_hat) * n_hat  # s-polarized (⊥ plane)
        e_p = e0 - e_s  # p-polarized (‖ plane)
        e0_mag_sq = np.dot(e0, e0)
        ws = np.dot(e_s, e_s) / e0_mag_sq
        wp = np.dot(e_p, e_p) / e0_mag_sq

    return rp, rm, x, phi_max, ws, wp, obf
