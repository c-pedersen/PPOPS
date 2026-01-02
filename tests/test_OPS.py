import pytest
import sys
import os

# Workaround to resolve path issues/being unable to see src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, os.path.abspath(src_dir))

import ppops  # noqa: E402


def test_OPS_laser_wavelength_inputs():
    """
    Test that the OPS class correctly handles different laser wavelength inputs.
    """
    # Test with custom wavelength
    custom_wavelength = 0.556241  # in micrometers
    ops_custom = ppops.OpticalParticleSpectrometer(laser_wavelength=custom_wavelength)
    assert ops_custom.laser_wavelength == custom_wavelength

    # Test with invalid wavelength (negative value)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(laser_wavelength=-0.532)

    # Test with wavelengths outside typical range
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(laser_wavelength=0.195)
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(laser_wavelength=100.0)

    # Test with invalid wavelength (zero value)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(laser_wavelength=0)


def test_OPS_laser_power_inputs():
    """
    Test that the OPS class correctly handles different laser power inputs.
    """
    # Test with custom laser power
    custom_power = 50  # in mW
    ops_custom = ppops.OpticalParticleSpectrometer(laser_power=custom_power)
    assert ops_custom.laser_power == custom_power

    # Test with invalid laser power (negative value)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(laser_power=-10)

    # Test with laser power outside typical range
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(laser_power=0.070)
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(laser_power=1200)

    # Test with zero laser power
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(laser_power=0)


def test_OPS_dimension_inputs():
    """
    Test that the OPS class correctly handles different dimension inputs.
    """
    # Test with custom dimensions
    mirror_radius = 23.245
    mirror_radius_of_curvature = 45.5
    aerosol_mirror_separation = 12.56

    ops_custom = ppops.OpticalParticleSpectrometer(
        mirror_radius=mirror_radius,
        mirror_radius_of_curvature=mirror_radius_of_curvature,
        aerosol_mirror_separation=aerosol_mirror_separation,
    )
    assert ops_custom.mirror_radius == mirror_radius
    assert ops_custom.mirror_radius_of_curvature == mirror_radius_of_curvature
    assert ops_custom.aerosol_mirror_separation == aerosol_mirror_separation

    # Test with invalid dimensions (negative values)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(mirror_radius=-10)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(mirror_radius_of_curvature=-20)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(aerosol_mirror_separation=-5)

    # Test with dimensions outside typical range
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(mirror_radius=500)
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(mirror_radius_of_curvature=1000)
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(aerosol_mirror_separation=1)
