import os
import sys
import numpy as np
import miepython
import matplotlib.pyplot as plt
from mie_modules import mie_s12

# Workaround to resolve path issues/being unable to see src directory
# 1. Get the directory where this file (conftest.py) is located (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory of 'tests/' (which is the PPOPS project root)
project_root = os.path.join(current_dir, "..")

# 3. Add the PPOPS project root to the very start of the search path (sys.path)
# This allows Python to correctly resolve 'from src.geometry...'
sys.path.insert(0, os.path.abspath(project_root))

from src.ppops.OPS import OpticalParticleSpectrometer


def compare_s1s2_methods(
    ior: complex,
    diameter: float,
) -> None:
    """
    Compare S1 and S2 scattering amplitude functions computed using the
    miepython package and a custom Mie implementation.

    Parameters
    ----------
    ior : complex
        Complex refractive index of the particle.
    diameter : float
        Diameter of the particle in micrometers.

    Returns
    -------
        None
    """
    ops = OpticalParticleSpectrometer()

    # Determine theta values and size parameter
    n_theta = 100  # Polar angle samples
    theta_max = np.arctan(ops.mirror_radius / ops.h)
    theta_values = np.linspace(np.pi / 2 - theta_max, np.pi / 2 + theta_max, n_theta)
    size_parameter = np.pi / ops.wavelength * diameter

    # Compute S1 and S2 using custom Mie implementation
    s1 = np.zeros_like(theta_values, dtype=complex)
    s2 = np.zeros_like(theta_values, dtype=complex)
    for j, theta in enumerate(theta_values):
        s12 = mie_s12(ior, size_parameter, np.cos(theta))
        s1[j], s2[j] = s12[0], s12[1]

    # Compute S1 and S2 using miepython package
    miepython_s1, miepython_s2 = miepython.core.S1_S2(
        m=ior,
        x=size_parameter,
        mu=np.cos(theta_values),
        norm="wiscombe",
    )

    plt.figure(figsize=(6, 4))
    plt.plot(np.real(miepython_s1), np.imag(miepython_s1), label='MiePython', marker='o')
    plt.plot(np.real(s1), np.imag(s1), label='Custom Mie Modules', marker='x')
    plt.xlabel('Real part')
    plt.title(f'Comparison of S1 for IOR={ior} and Diameter={diameter} µm')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{project_root}/validation/s1_comparison.png', dpi=600)

    np.testing.assert_allclose(s1, miepython_s1, rtol=1e-3)
    np.testing.assert_allclose(s2, miepython_s2, rtol=1e-3)


if __name__ == "__main__":
    # Example test case
    compare_s1s2_methods(ior=1.4 + 0j, diameter=1.0)
    print("S1 and S2 functions match between custom implementation and miepython.")
