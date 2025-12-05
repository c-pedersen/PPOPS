import pytest
import numpy as np
import sys
import os

# Workaround to resolve path issues/being unable to see src directory
# 1. Get the directory where this file (conftest.py) is located (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory of 'tests/' (which is the PPOPS project root)
project_root = os.path.join(current_dir, "..")

# 3. Add the PPOPS project root to the very start of the search path (sys.path)
# This allows Python to correctly resolve 'from src.geometry...'
sys.path.insert(0, os.path.abspath(project_root))

print(f"Added to path: {project_root}")
print("\n--- Current Python Search Paths (sys.path) ---")
for p in sys.path:
    print(p)
print("------------------------------------------")
from src.ppops.geometry import ptz2r_sc

# Tolerance for floating point comparisons
TOL = 1e-6


def test_basic_run_and_output_structure():
    """
    Very basic test to ensure the function runs with typical parameters
    and returns the expected output structure and types.
    """
    # Typical inputs: 90-degree scattering, center offset 0
    phi = np.pi / 4
    theta = np.pi / 2
    y0 = 0.0
    h = 7.68 + 2.159
    mirror_radius = 12.5

    result = ptz2r_sc(
        phi=phi,
        theta=theta,
        h=h,
        mirror_radius=mirror_radius,
        y0=y0,
    )

    # 1. Check if the correct number of results (7) is returned
    assert len(result) == 7, "Function should return a tuple of 7 elements."

    rp, rm, x, phi_max, ws, wp, obf = result

    # 2. Check the types of the main outputs
    assert isinstance(rp, (float, np.float64)), "rp should be a float."
    assert isinstance(x, np.ndarray), "x should be a numpy array."
    assert x.shape == (3,), "x (Cartesian coordinates) should be a 3-element array."
    assert isinstance(ws, (float, np.float64)), "ws should be a float."

    # 3. Check for obvious non-physical results (rp should be positive)
    assert rp > 0, "Positive intersection distance (rp) must be greater than zero."


def test_polarization_conservation():
    """
    Tests the fundamental physics sanity check:
    The s-polarization (ws) and p-polarization (wp) weights must sum to 1.0.
    """
    h = 7.68 + 2.159
    mirror_radius = 12.5

    # Test a few different realistic geometries
    test_cases = [
        (0.0, np.pi / 4, 0.0),  # Phi = 0 (Scattering in y-z plane)
        (np.pi / 2, np.pi / 2, 5.0),  # Phi = 90 deg (Scattering in x-z plane), y-offset
        (0.8, 1.2, -3.0),  # Arbitrary non-trivial case
    ]

    for phi, theta, y0 in test_cases:
        _, _, _, _, ws, wp, _ = ptz2r_sc(
            phi=phi,
            theta=theta,
            h=h,
            mirror_radius=mirror_radius,
            y0=y0,
        )

        # Check that ws + wp equals 1.0 within tolerance
        np.testing.assert_almost_equal(
            ws + wp,
            1.0,
            decimal=TOL,
            err_msg=f"Polarization weights (ws + wp) must sum to 1.0 for phi={phi}, theta={theta}, y0={y0}",
        )

        # Check bounds
        assert 0.0 <= ws <= 1.0
        assert 0.0 <= wp <= 1.0


def test_degenerate_forward_scattering():
    """
    Tests the degenerate scattering case (forward scatter, theta=0),
    where the scattering plane is undefined. The code handles this
    explicitly by setting ws=1.0 and wp=0.0.
    """
    phi = 0.5  # Arbitrary azimuthal angle
    theta = 0.0  # Forward scattering
    y0 = 0.0
    h = 7.68 + 2.159
    mirror_radius = 12.5

    _, _, _, _, ws, wp, _ = ptz2r_sc(
        phi=phi,
        theta=theta,
        h=h,
        mirror_radius=mirror_radius,
        y0=y0,
    )

    # In the degenerate case (n2 < 1e-12), the function should return:
    # ws = 1.0 and wp = 0.0
    np.testing.assert_almost_equal(
        ws, 1.0, decimal=TOL, err_msg="WS should be 1.0 in degenerate (theta=0) case."
    )
    np.testing.assert_almost_equal(
        wp, 0.0, decimal=TOL, err_msg="WP should be 0.0 in degenerate (theta=0) case."
    )


def test_pure_s_polarization():
    """
    Tests a specific case where the scattered light ray is entirely
    s-polarized relative to the incident laser field (E0 = [1, 0, 0]).

    When phi = 90 deg (pi/2) and theta = 90 deg (pi/2), the scattering
    vector k_s is [1, 0, 0] * rp, the same direction as the E-field.
    Wait, no.
    phi=pi/2, theta=pi/2 -> ax=1, ay=0, az=0. x=[rp, 0, 0].

    Laser E0 = [1, 0, 0]
    Scattered k_s = [1, 0, 0]
    Incident k_i = [0, 0, 1]

    The scattering plane normal n_vec = k_i x k_s = [0, 0, 1] x [1, 0, 0] = [0, 1, 0].
    n_hat = [0, 1, 0].

    e_s = dot(e0, n_hat) * n_hat = dot([1, 0, 0], [0, 1, 0]) * [0, 1, 0] = 0.

    This means ws = 0.0 and wp = 1.0. This is a pure p-polarization case.
    """
    phi = np.pi / 2
    theta = np.pi / 2
    y0 = 0.0
    h = 7.68 + 2.159
    mirror_radius = 12.5

    _, _, _, _, ws, wp, _ = ptz2r_sc(
        phi=phi,
        theta=theta,
        h=h,
        mirror_radius=mirror_radius,
        y0=y0,
    )

    np.testing.assert_almost_equal(
        ws,
        0.0,
        decimal=TOL,
        err_msg="WS should be 0.0 in the pure p-polarization case (phi=pi/2, theta=pi/2).",
    )
    np.testing.assert_almost_equal(
        wp,
        1.0,
        decimal=TOL,
        err_msg="WP should be 1.0 in the pure p-polarization case (phi=pi/2, theta=pi/2).",
    )
