import pytest
import sys
import os
import numpy as np

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


def test_OPS_pmt_control_voltage_inputs():
    """
    Test that the OPS class correctly handles different PMT control voltage inputs.
    """
    # Test with custom PMT control voltage
    custom_voltage = 0.8  # in volts
    ops_custom = ppops.OpticalParticleSpectrometer(pmt_control_voltage=custom_voltage)
    assert ops_custom.pmt_control_voltage == custom_voltage

    # Test with invalid PMT control voltage (negative value)
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(pmt_control_voltage=-0.5)

    # Test with PMT control voltage outside typical range
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(pmt_control_voltage=0.3)
    with pytest.warns(UserWarning):
        ppops.OpticalParticleSpectrometer(pmt_control_voltage=1.5)

    # Test with zero PMT control voltage
    with pytest.raises(ValueError):
        ppops.OpticalParticleSpectrometer(pmt_control_voltage=0)


def test_aerosol_parameters():
    """
    Test that the OPS class correctly handles different aerosol parameter inputs.
    """
    # Test refractive index inputs
    custom_ri = 1.5 + 0.01j
    ops_custom = ppops.OpticalParticleSpectrometer()
    qsca = ops_custom.truncated_scattering_cross_section(ri=custom_ri, diameter=1.0)
    assert isinstance(qsca, np.ndarray)

    # Test with invalid refractive index (negative real part)
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(ri=-1.5 + 0.01j, diameter=1.0)

    # Test with invalid refractive index (negative imaginary part)
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(ri=1.5 - 0.01j, diameter=1.0)

    # Test with invalid diameter (negative value)
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(ri=custom_ri, diameter=-1.0)

    # Test with invalid diameter (zero value)
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(ri=custom_ri, diameter=0)

    # Test large and small diameters
    with pytest.warns(UserWarning):
        ops_custom.truncated_scattering_cross_section(ri=custom_ri, diameter=200.0)
    with pytest.warns(UserWarning):
        ops_custom.truncated_scattering_cross_section(ri=custom_ri, diameter=0.0001)


def test_integration_parameters():
    """
    Test that the OPS class correctly handles different integration parameter inputs.
    """
    ops_custom = ppops.OpticalParticleSpectrometer()

    # Test with invalid n_theta and n_phi (negative values)
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(
            ri=1.5 + 0.01j, diameter=1.0, n_theta=-19, n_phi=9
        )
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(
            ri=1.5 + 0.01j, diameter=1.0, n_theta=33, n_phi=-21
        )

    # Test with zero n_theta and n_phi
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(
            ri=1.5 + 0.01j, diameter=1.0, n_theta=0, n_phi=11
        )
    with pytest.raises(ValueError):
        ops_custom.truncated_scattering_cross_section(
            ri=1.5 + 0.01j, diameter=1.0, n_theta=13, n_phi=0
        )

    # Test with even n_theta and n_phi
    with pytest.warns(UserWarning):
        ops_custom.truncated_scattering_cross_section(
            ri=1.5 + 0.01j, diameter=1.0, n_theta=14, n_phi=11
        )
    with pytest.warns(UserWarning):
        ops_custom.truncated_scattering_cross_section(
            ri=1.5 + 0.01j, diameter=1.0, n_theta=13, n_phi=18
        )


def test_signal_digitization():
    """
    Test that the signal and noise estimates are correctly computed for a given
    truncated scattering cross section and laser power.
    """
    ri = 1.5 + 0.01j
    diameter = 1.0  # in micrometers
    ops_custom = ppops.OpticalParticleSpectrometer()
    signal, noise = ops_custom.estimate_signal_noise(ri=ri, diameters=diameter)
    digitized_signal = ppops.digitize_signal(signal)

    assert isinstance(signal, np.ndarray)
    assert isinstance(noise, np.ndarray)
    assert np.all(signal >= 0)
    assert np.all(noise >= 0)
    assert np.all(digitized_signal >= 0)

    # Test with invalid signal value for digitization (negative value)
    with pytest.raises(ValueError):
        ppops.digitize_signal(-0.001)
