"""
mie_modules.py
---------------
This module implements Mie scattering physics functions used to compute
the scattered light intensity for spherical particles.

Functions
---------
mie_ab(m, x)
    Computes Mie coefficients a_n and b_n.
mie_pt(u, n_max)
    Computes angular functions π_n(u) and τ_n(u).
mie_s12(m, x, u)
    Computes Mie scattering amplitude functions S1 and S2.


References
----------
Bohren & Huffman (1983), "Absorption and Scattering of Light by Small Particles"
C. Mätzler (2002), "MATLAB Functions for Mie Scattering"
"""

import numpy as np
from scipy.special import jv, yv


def mie_ab(m: complex, x: float) -> np.ndarray:
    """
    Compute Mie coefficients a_n and b_n.

    Parameters
    ----------
    m : complex
        Complex refractive index of the particle (m = n + i*k).
    x : float
        Size parameter (π * D / λ).

    Returns
    -------
    np.ndarray
        Array of shape (2, n_max) containing a_n and b_n.
    """
    # ---------------------------------------------------------------------
    # Determine n_max according to Bohren & Huffman recommendation
    # ---------------------------------------------------------------------
    n_max = round(2 + x + 4 * x ** (1 / 3))
    n_mx = round(max(n_max, abs(m * x)) + 16)

    n = np.arange(1, n_max + 1)
    nu = n + 0.5  # cylindrical Bessel order (MATLAB uses n + 1/2)

    # ---------------------------------------------------------------------
    # Correct Riccati–Bessel functions (MATLAB-compatible)
    # ---------------------------------------------------------------------
    factor = np.sqrt(0.5 * np.pi * x)

    # ψ_n(x)
    psi_x = factor * jv(nu, x)
    psi_1x = factor * jv(nu - 1, x)

    # χ_n(x)
    chi_x = -factor * yv(nu, x)
    chi_1x = -factor * yv(nu - 1, x)

    # ξ_n(x)
    xi_x = psi_x - 1j * chi_x
    xi_1x = psi_1x - 1j * chi_1x

    # ---------------------------------------------------------------------
    # Logarithmic derivative D_n(z) via backward recurrence
    # ---------------------------------------------------------------------
    z = m * x
    dn_x = np.zeros(n_mx + 1, dtype=complex)

    for j in range(n_mx, 1, -1):
        dn_x[j - 1] = j / z - 1 / (dn_x[j] + j / z)

    dn = dn_x[1:n_max + 1]  # keep first n_max elements

    # ---------------------------------------------------------------------
    # Compute Mie coefficients a_n, b_n
    # ---------------------------------------------------------------------
    da = dn / m + n / x
    db = m * dn + n / x

    a_n = (da * psi_x - psi_1x) / (da * xi_x - xi_1x)
    b_n = (db * psi_x - psi_1x) / (db * xi_x - xi_1x)

    return np.array([a_n, b_n])


def mie_pt(u: float, n_max: int) -> np.ndarray:
    """
    Compute angular functions π_n(u) and τ_n(u) for Mie scattering.

    This function implements the exact recurrences given in
    Bohren & Huffman (1983) and Mätzler (2002). These angular
    functions determine the angular variation of the scattered
    field and are required to compute the scattering amplitude
    functions S1 and S2.


    Parameters
    ----------
    u : float
        Cosine of the scattering angle, u = cos(theta).
    n_max : int
        Maximum multipole order to compute.

    Returns
    -------
    np.ndarray
        Array of shape (2, n_max) containing:
        - π_n(u) in row 0
        - τ_n(u) in row 1
    """
    p = np.zeros(n_max)
    t = np.zeros(n_max)

    # Base cases
    p[0] = 1.0          # π_1
    t[0] = u            # τ_1

    if n_max > 1:
        p[1] = 3.0 * u  # π_2
        # Use identity cos(2θ) = 2u^2 - 1
        t[1] = 3.0 * (2.0 * u**2 - 1.0)

    # Recurrence for n >= 3
    for n1 in range(2, n_max):
        p[n1] = ((2 * n1 - 1) / (n1 - 1)) * u * p[n1 - 1] \
                 - (n1 / (n1 - 1)) * p[n1 - 2]

        t[n1] = n1 * u * p[n1] - (n1 + 1) * p[n1 - 1]

    return np.array([p, t])


def mie_s12(m: complex, x: float, u: float) -> np.ndarray:
    """
    Compute the complex scattering amplitude functions S1 and S2.

    Parameters
    ----------
    m : complex
        Complex refractive index of the particle.
    x : float
        Size parameter (π * D / λ).
    u : float
        Cosine of scattering angle (cos(theta)).

    Returns
    -------
    np.ndarray
        Array [S1, S2] of complex scattering amplitudes.
    """
    n_max = round(2 + x + 4 * x ** (1 / 3))

    ab = mie_ab(m, x)
    a_n, b_n = ab[0, :], ab[1, :]

    pi_n, tau_n = mie_pt(u, n_max)

    n = np.arange(1, n_max + 1)
    norm = (2 * n + 1) / (n * (n + 1))

    pi_n *= norm
    tau_n *= norm

    s1 = np.dot(a_n, pi_n) + np.dot(b_n, tau_n)
    s2 = np.dot(a_n, tau_n) + np.dot(b_n, pi_n)

    return np.array([s1, s2])

