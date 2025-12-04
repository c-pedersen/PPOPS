import pytest
import numpy as np
from src.geometry import ptz2r_sc

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

    result = ptz2r_sc(phi, theta, y0)
    
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
    # Test a few different realistic geometries
    test_cases = [
        (0.0, np.pi / 4, 0.0),      # Phi = 0 (Scattering in y-z plane)
        (np.pi / 2, np.pi / 2, 5.0), # Phi = 90 deg (Scattering in x-z plane), y-offset
        (0.8, 1.2, -3.0),           # Arbitrary non-trivial case
    ]

    for phi, theta, y0 in test_cases:
        _, _, _, _, ws, wp, _ = ptz2r_sc(phi, theta, y0)
        
        # Check that ws + wp equals 1.0 within tolerance
        np.testing.assert_almost_equal(ws + wp, 1.0, decimal=TOL,
                                       err_msg=f"Polarization weights (ws + wp) must sum to 1.0 for phi={phi}, theta={theta}, y0={y0}")
        
        # Check bounds
        assert 0.0 <= ws <= 1.0
        assert 0.0 <= wp <= 1.0