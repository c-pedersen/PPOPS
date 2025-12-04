"""
mie_modules.py
---------------
Core Mie scattering calculations for spherical particles.

This module provides:
    - mie_ab: Mie coefficients a_n and b_n
    - mie_pt: angular functions π_n(u) and τ_n(u)
    - mie_s12: scattering amplitudes S1 and S2
"""

import numpy as np
from scipy.special import jv, yv


def mie_ab(m: complex, x: float):
    """
    Compute Mie coefficients a_n and b_n for a spherical particle.

    Parameters
    ----------
    m : complex
        Complex refractive index of the particle.
    x : float
        Size parameter defined as π * D / λ.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, n_max) containing:
        - a_n in row 0
        - b_n in row 1
    """
    # Mode truncation
    n_max = int(round(2.0 + x + 4.0 * x ** (1.0 / 3.0)))
    z = m * x
    n_mx = int(round(max(n_max, abs(z)) + 16.0))

    n = np.arange(1, n_max + 1)
    nu = n + 0.5

    # Riccati–Bessel functions ψ_n, χ_n, ξ_n
    factor = np.sqrt(0.5 * np.pi * x)

    psi = factor * jv(nu, x)
    psi_1 = factor * jv(nu - 1.0, x)

    chi = -factor * yv(nu, x)
    chi_1 = -factor * yv(nu - 1.0, x)

    xi = psi - 1j * chi
    xi_1 = psi_1 - 1j * chi_1

    # Logarithmic derivative D_n(z) via backward recurrence
    dn_x = np.zeros(n_mx + 1, dtype=complex)
    for j in range(n_mx, 1, -1):
        dn_x[j - 1] = j / z - 1.0 / (dn_x[j] + j / z)

    dn = dn_x[1:n_max + 1]

    # Coefficients
    da = dn / m + n / x
    db = m * dn + n / x

    a_n = (da * psi - psi_1) / (da * xi - xi_1)
    b_n = (db * psi - psi_1) / (db * xi - xi_1)

    return np.array([a_n, b_n])


def mie_pt(u: float, n_max: int):
    """
    Compute angular functions π_n(u) and τ_n(u) for Mie scattering.

    Parameters
    ----------
    u : float
        Cosine of the scattering angle (cos(theta)).
    n_max : int
        Maximum multipole order.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, n_max) containing π_n(u) (row 0) and τ_n(u) (row 1).
    """
    p = np.zeros(n_max)
    t = np.zeros(n_max)

    # Base cases
    p[0] = 1.0
    t[0] = u

    if n_max > 1:
        p[1] = 3.0 * u
        t[1] = 3.0 * (2.0 * u**2 - 1.0)

    # Recurrence for n ≥ 3
    for n1 in range(2, n_max):
        p[n1] = ((2 * n1 - 1) / (n1 - 1)) * u * p[n1 - 1] \
                 - (n1 / (n1 - 1)) * p[n1 - 2]

        t[n1] = n1 * u * p[n1] - (n1 + 1) * p[n1 - 1]

    return np.array([p, t])


def mie_s12(m: complex, x: float, u: float):
    """
    Compute scattering amplitude functions S1 and S2.

    Parameters
    ----------
    m : complex
        Complex refractive index of the particle.
    x : float
        Size parameter.
    u : float
        Cosine of scattering angle (cos(theta)).

    Returns
    -------
    numpy.ndarray
        Array [S1, S2] of complex scattering amplitudes.
    """
    n_max = int(round(2.0 + x + 4.0 * x ** (1.0 / 3.0)))

    a_n, b_n = mie_ab(m, x)
    pi_n, tau_n = mie_pt(u, n_max)

    n = np.arange(1, n_max + 1)
    norm = (2 * n + 1) / (n * (n + 1))

    pi_n = pi_n * norm
    tau_n = tau_n * norm

    s1 = np.dot(a_n, pi_n) + np.dot(b_n, tau_n)
    s2 = np.dot(a_n, tau_n) + np.dot(b_n, pi_n)

    return np.array([s1, s2])

