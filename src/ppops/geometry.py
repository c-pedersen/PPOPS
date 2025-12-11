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
    phi: np.ndarray,
    theta: np.ndarray,
    mirror_radius: float,
    mirror_radius_of_curvature: float,
    y0: float,
    h: float,
):
    """Compute POPS mirror geometry and polarization weighting.

    This function calculates the intersection of scattered light rays with
    the spherical POPS mirror and determines polarization weighting factors
    (s- and p-polarization) based on instrument geometry.

    Parameters
    ----------
    phi : np.ndarray
        Azimuthal scattering angles [radians]. Shape: (n,)
    theta : np.ndarray
        Polar scattering angles [radians]. Shape: (n,)
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

    Returns
    -------
    rp : np.ndarray
        Positive intersection distances [mm]. Shape: (n,)
    rm : np.ndarray
        Negative intersection distances (unphysical, retained
        for completeness). Shape: (n,)
    x : np.ndarray
        Cartesian coordinates of intersection vectors. Shape: (n, 3)
    phi_max : np.ndarray
        Maximum azimuthal half-angles. Shape: (n,)
    ws : np.ndarray
        s-polarization weighting factors (perpendicular). Shape: (n,)
    wp : np.ndarray
        p-polarization weighting factors (parallel). Shape: (n,)
    obf : np.ndarray
        Obliquity factors (cosine of incidence angle). Shape: (n,)
    """
    # Ensure inputs are arrays
    try:
        phi = np.asarray(phi)
        theta = np.asarray(theta)
    except Exception as e:
        raise TypeError("phi and theta must be convertible to numpy arrays") from e

    # Geometric setup
    alpha = np.sin(phi)
    mu = np.cos(theta)
    sin_theta = np.sin(theta)

    # Coefficients for quadratic mirror-intersection equation
    ax = alpha * sin_theta
    ay = np.sqrt(1 - alpha**2) * sin_theta
    az = mu
    a = ax**2 + ay**2 + az**2
    b = 2 * y0 * ay
    c = y0**2 - mirror_radius_of_curvature**2

    # Quadratic solutions
    discriminant = np.sqrt(b**2 - 4 * a * c)
    rp = (-b + discriminant) / (2 * a)
    rm = (-b - discriminant) / (2 * a)

    # Compute maximum azimuthal collection angle
    r_min = np.sqrt(h**2 + mirror_radius**2)
    phi_max = np.arccos(np.clip(h / (r_min * sin_theta), -1, 1))

    # Cartesian coordinates - shape (n, 3)
    x = np.stack(
        [
            rp * np.sin(phi) * sin_theta,
            rp * np.cos(phi) * sin_theta,
            rp * np.cos(theta),
        ],
        axis=-1,
    )

    # Compute mirror orientation and obliquity factor
    x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True)
    s_norm = x - np.array([0, -y0, 0])
    s_norm = s_norm / np.linalg.norm(s_norm, axis=-1, keepdims=True)

    # Obliquity factor: dot product along last axis
    obf = np.sum(x_norm * s_norm, axis=-1)

    # Polarization weighting computation
    e0 = np.array([1, 0, 0])  # Laser electric field
    _k_i = np.array([0, 0, 1])  # Incident laser wave vector
    k_s = x_norm  # Scattered wave vectors

    # Cross product: k_i x k_s for each vector
    # k_i is [0, 0, 1], so cross product simplifies
    n_vec = np.stack(
        [
            -k_s[:, 1],  # 0*k_s[2] - 1*k_s[1]
            k_s[:, 0],  # 1*k_s[0] - 0*k_s[2]
            np.zeros_like(k_s[:, 0]),  # 0*k_s[1] - 0*k_s[0]
        ],
        axis=-1,
    )

    n2 = np.sum(n_vec * n_vec, axis=-1)  # Magnitude squared

    # Handle degenerate case
    degenerate = n2 < 1e-12

    # Initialize outputs
    ws = np.ones_like(n2)
    wp = np.zeros_like(n2)

    # Non-degenerate cases
    non_degen = ~degenerate
    if np.any(non_degen):
        n_hat = n_vec[non_degen] / np.sqrt(n2[non_degen, np.newaxis])
        # e_s = dot(e0, n_hat) * n_hat
        dot_e0_nhat = np.sum(e0 * n_hat, axis=-1, keepdims=True)
        e_s = dot_e0_nhat * n_hat
        e_p = e0 - e_s

        e0_mag_sq = np.dot(e0, e0)
        ws[non_degen] = np.sum(e_s * e_s, axis=-1) / e0_mag_sq
        wp[non_degen] = np.sum(e_p * e_p, axis=-1) / e0_mag_sq

    return rp, rm, x, phi_max, ws, wp, obf
