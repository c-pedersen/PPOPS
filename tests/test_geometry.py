import numpy as np
import sys
import os

# Workaround to resolve path issues/being unable to see src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, os.path.abspath(src_dir))

from ppops.geometry import ptz2r_sc  # noqa: E402
from ppops.OPS import OpticalParticleSpectrometer  # noqa: E402

# Tolerance for floating point comparisons
TOL = 1e-6


def test_basic_run_and_output_structure():
    """
    Very basic test to ensure the function runs with typical parameters
    and returns the expected output structure and types.
    """
    # Typical inputs: 90-degree scattering, center offset 0
    phi = np.array([np.pi / 4])
    theta = np.array([np.pi / 2])
    ops = OpticalParticleSpectrometer()

    result = ptz2r_sc(ops=ops, phi=phi,theta=theta)

    # 1. Check if the correct number of results (7) is returned
    assert len(result) == 7, "Function should return a tuple of 7 elements."

    rp, _, x, _, ws, wp, _ = result

    # 2. Check the types of the main outputs
    assert isinstance(rp, np.ndarray), "rp should be an array of floats."
    assert isinstance(x, np.ndarray), "x should be a numpy array."
    assert x.shape == (len(phi), 3), "x (Cartesian coordinates) should be a nx3 array."
    assert isinstance(ws, np.ndarray), "ws should be an array of floats."
    assert isinstance(wp, np.ndarray), "wp should be an array of floats."

    # 3. Check for obvious non-physical results (rp should be positive)
    assert np.all(rp > 0), (
        "Positive intersection distance (rp) must be greater than zero."
    )


def test_polarization_conservation():
    """
    Tests the fundamental physics sanity check:
    The s-polarization (ws) and p-polarization (wp) weights must sum to 1.0.
    """
    ops = OpticalParticleSpectrometer()

    # Test a few different geometries
    phi = np.array([0.0, np.pi / 4, 0.8])
    theta = np.array([np.pi / 4, np.pi / 2, 5.0])
    y0_values = [0.8, 1.2, -3.0]

    for y0 in y0_values:
        ops.y0 = y0
        _, _, _, _, ws, wp, _ = ptz2r_sc(ops=ops, phi=phi, theta=theta)

        # Check that ws + wp equals 1.0 within tolerance
        np.testing.assert_almost_equal(
            ws + wp,
            np.ones_like(ws),
            decimal=TOL,
            err_msg="Polarization weights (ws + wp) must sum to 1.0",
        )

        # Check bounds
        assert np.all((0.0 <= ws) & (ws <= 1.0))
        assert np.all((0.0 <= wp) & (wp <= 1.0))


def test_degenerate_forward_scattering():
    """
    Tests the degenerate scattering case (forward scatter, theta=0),
    where the scattering plane is undefined. The code handles this
    explicitly by setting ws=1.0 and wp=0.0.
    """
    phi = np.array([0.5])  # Arbitrary azimuthal angle
    theta = np.array([0.0])  # Forward scattering
    ops = OpticalParticleSpectrometer()

    _, _, _, _, ws, wp, _ = ptz2r_sc(ops=ops, phi=phi, theta=theta)

    # In the degenerate case (n2 < 1e-12), the function should return:
    # ws = 1.0 and wp = 0.0
    np.testing.assert_almost_equal(
        ws,
        np.ones_like(ws),
        decimal=TOL,
        err_msg="WS should be 1.0 in degenerate (theta=0) case.",
    )
    np.testing.assert_almost_equal(
        wp,
        np.zeros_like(wp),
        decimal=TOL,
        err_msg="WP should be 0.0 in degenerate (theta=0) case.",
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
    phi = np.array([np.pi / 2])
    theta = np.array([np.pi / 2])
    ops = OpticalParticleSpectrometer()

    _, _, _, _, ws, wp, _ = ptz2r_sc(ops=ops, phi=phi, theta=theta)

    np.testing.assert_almost_equal(
        ws,
        np.zeros_like(ws),
        decimal=TOL,
        err_msg="WS should be 0.0 in the pure p-polarization case (phi=pi/2, theta=pi/2).",
    )
    np.testing.assert_almost_equal(
        wp,
        np.ones_like(wp),
        decimal=TOL,
        err_msg="WP should be 1.0 in the pure p-polarization case (phi=pi/2, theta=pi/2).",
    )
