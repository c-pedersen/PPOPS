"""
mie_modules.py
---------------
This module implements Mie scattering physics functions used to compute
the scattered light intensity for spherical particles.

Functions:
    mie_ab(m, x): Computes Mie coefficients a_n and b_n.
    mie_pt(u, n_max): Computes angular functions π_n(u) and τ_n(u).
    mie_s12(m, x, u): Computes Mie scattering amplitude functions S1 and S2.

References:
    - Bohren & Huffman (1983), "Absorption and Scattering of Light by Small Particles"
    - C. Mätzler (2002), Mie scattering implementation
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def mie_ab(m: complex, x: float) -> np.ndarray:
    """Compute Mie coefficients a_n and b_n.

    The coefficients describe the relative contributions of electric (a_n)
    and magnetic (b_n) multipole modes to light scattering by a sphere.

    Args:
        m (complex): Complex refractive index of the particle (m = n + i*k).
        x (float): Size parameter, defined as π * D / λ.

    Returns:
        np.ndarray: A 2xN array of complex coefficients [a_n, b_n].
    """
    z = m * x
    n_max = round(2 + x + 4 * x ** (1 / 3))
    n_mx = round(max(n_max, abs(z)) + 16)

    n = np.arange(1, n_max + 1)

    #  Riccati–Bessel functions:
    #   psi_n(x) = x * j_n(x)
    #   chi_n(x) = -x * y_n(x)
    psi_x = x * spherical_jn(n, x)
    chi_x = -x * spherical_yn(n, x)

    psi_1x = np.empty_like(psi_x)
    chi_1x = np.empty_like(chi_x)

    psi_1x[0] = np.sin(x)  # ψ_0(x)
    chi_1x[0] = np.cos(x)  # χ_0(x)

    psi_1x[1:] = x * spherical_jn(n[:-1], x)
    chi_1x[1:] = -x * spherical_yn(n[:-1], x)


    # Outgoing spherical Hankel functions
    xi_x = psi_x + 1j * chi_x
    xi_1x = psi_1x + 1j * chi_1x


    # Logarithmic derivative D_n(z) via backward recurrence
    dn_x = np.zeros(n_mx + 1, dtype=complex)

    for j in range(n_mx, 1, -1):
        dn_x[j - 1] = j / z - 1.0 / (dn_x[j] + j / z)

    dn = dn_x[1 : n_max + 1]


    # Mie coefficients
    da = dn / m + n / x
    db = m * dn + n / x

    a_n = (da * psi_x - psi_1x) / (da * xi_x - xi_1x)
    b_n = (db * psi_x - psi_1x) / (db * xi_x - xi_1x)

    return np.array([a_n, b_n])


def mie_pt(u: float, n_max: int) -> np.ndarray:
    """Compute angular functions π_n(u) and τ_n(u).

    These functions describe how scattered light intensity varies with
    scattering angle and are derived from Legendre polynomials.

    Args:
        u (float): Cosine of the scattering angle (u = cos(θ)).
        n_max (int): Maximum multipole order.

    Returns:
        np.ndarray: A 2×N array with π_n(u) and τ_n(u) values. hiiiii
    """
    p = np.zeros(n_max)
    t = np.zeros(n_max)

    # Initial dipole (n=1) and quadrupole (n=2) terms
    p[0] = 1
    t[0] = u
    p[1] = 3 * u
    t[1] = 6 * u * u - 3

    # Recurrence relation for higher-order terms
    for n1 in range(2, n_max):
        n = n1 + 1  # Physics order (n=3, 4, 5, ...)
        p[n1] = ((2 * n - 1) * u * p[n1 - 1] - n * p[n1 - 2]) / (n - 1)
        t[n1] = n * u * p[n1] - (n + 1) * p[n1 - 1]
    return np.array([p, t])


def mie_s12(m: complex, x: float, u: float) -> np.ndarray:
    """Compute Mie scattering amplitude functions S1 and S2.

    Combines the mode coefficients (a_n, b_n) and angular functions (π_n, τ_n)
    to determine how light is scattered for s- and p-polarizations at a
    specific scattering angle.

    Args:
        m (complex): Complex refractive index of the particle.
        x (float): Size parameter (π * D / λ).
        u (float): Cosine of the scattering angle (cos(θ)).

    Returns:
        np.ndarray: Complex scattering amplitudes [S1, S2].
    """
    n_max = round(2 + x + 4 * x ** (1 / 3))

    ab = mie_ab(m, x)
    a_n, b_n = ab[0, :], ab[1, :]

    pt = mie_pt(u, n_max)
    pi_n, tau_n = pt[0, :], pt[1, :]

    n = np.arange(1, n_max + 1)
    n2 = (2 * n + 1) / (n * (n + 1))
    pi_n *= n2
    tau_n *= n2

    s1 = np.dot(a_n, pi_n) + np.dot(b_n, tau_n)
    s2 = np.dot(a_n, tau_n) + np.dot(b_n, pi_n)
    return np.array([s1, s2])
