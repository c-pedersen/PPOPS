import numpy as np
import miepython
from .src.ppops.OPS import OpticalParticleSpectrometer
from mie_modules import mie_s12

ops = OpticalParticleSpectrometer()

def compare_tcsca_methods(
    ops: OpticalParticleSpectrometer,
    ior: complex,
    diameter: float,
) -> tuple[float, float]:
    """
    Compare S1 and S2 scattering amplitude functions computed using the 
    miepython package and a custom Mie implementation.

    Parameters
    ----------
    ops : OpticalParticleSpectrometer
        Instance of the OpticalParticleSpectrometer class.
    ior : complex
        Complex refractive index of the particle.
    diameter : float
        Diameter of the particle in micrometers.

    Returns
    -------

    """
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
        norm='wiscombe',
    )

    np.testing.assert_allclose(s1, miepython_s1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(s2, miepython_s2, rtol=1e-5, atol=1e-8)
