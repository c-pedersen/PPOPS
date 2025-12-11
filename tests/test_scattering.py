import pytest
import numpy as np
import sys
import os
import miepython

# Workaround to resolve path issues/being unable to see src directory
# 1. Get the directory where this file (conftest.py) is located (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory of 'tests/' (which is the PPOPS project root)
project_root = os.path.join(current_dir, "..")

# 3. Add the src directory to the very start of the search path (sys.path)
# This allows Python to correctly resolve 'from ppops.geometry...'
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, os.path.abspath(src_dir))

print("\n--- Current Python Search Paths (sys.path) ---")
for p in sys.path:
    print(p)
print("------------------------------------------")
# The error label below (E402) suppresses the ruff error that the module import is not at the top of the file
import ppops  # noqa: E402


@pytest.mark.slow
def test_Qsca_truncation():
    """
    Test that the truncated scattering cross section calculation
    in OPS matches the full scattering cross section within a small
    tolerance.
    """
    ops = ppops.OpticalParticleSpectrometer()
    ops.h = 0.000001  # Set a small height for the interaction region above the mirror vertex

    iors = np.array([1.33 + 1e-8j, 1.5 + 0.01j, 2.0 + 0.1j, 1.0 + 0.0j, 1.6 + 0.0j])
    diameters = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # in micrometers

    for ior in iors:
        for diameter in diameters:
            geometric_cross_section = np.pi * (diameter / 2) ** 2
            qsca = miepython.efficiencies(
                m=ior, d=diameter, lambda0=ops.laser_wavelength
            )[1]

            trunc_qsca = (
                ops.truncated_scattering_cross_section(
                    ior=ior, diameter=diameter, n_theta=150, n_phi=150
                )
                / geometric_cross_section
            )

            # Check that the truncated Qsca is approximately equal to the full Qsca
            assert qsca == pytest.approx(trunc_qsca * 2, rel=1e-2)
