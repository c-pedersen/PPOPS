import pytest
import numpy as np
import sys
import os

# Workaround to resolve path issues/being unable to see src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, os.path.abspath(src_dir))

import ppops  # noqa: E402
from ppops.detector import (  # noqa: E402
    laser_power_density,
    estimate_signal_noise,
    ELEMENTARY_CHARGE,
    H10720_110_ANODE_RADIANT_SENSITIVITY,
    H10720_110_DARK_CURRENT,
    BANDWIDTH,
    TIA60_INPUT_CURRENT_NOISE,
)


def test_laser_power_density_zero_power():
    """Test that zero power results in zero density."""
    assert laser_power_density(0, 3e-3, 1e-3) == 0.0


def test_laser_power_density_negative_inputs():
    """Test that negative power or dimensions raise ValueErrors."""
    # Negative dimensions
    with pytest.raises(ValueError, match="Beam major and minor axes must be positive"):
        laser_power_density(100, -3e-3, 1e-3)

    with pytest.raises(ValueError, match="Beam major and minor axes must be positive"):
        laser_power_density(100, 3e-3, 0)


def test_laser_power_density_warnings_dimensions():
    """Test that unrealistic beam dimensions trigger warnings."""
    # Too large (> 1e-2)
    with pytest.warns(
        UserWarning, match="Beam dimensions in millimeters seem unrealistic"
    ):
        laser_power_density(100, 1000, 5)

    # Too small (< 1e-5)
    with pytest.warns(
        UserWarning, match="Beam dimensions in millimeters seem unrealistic"
    ):
        laser_power_density(100, 3, 1e-4)


def test_estimate_signal_noise_scalar():
    """
    Test signal and noise estimation with scalar inputs.
    Verifies the physics formulas are implemented correctly.
    """
    csca = 1.0  # um^2
    power = 100.0

    ops = ppops.OpticalParticleSpectrometer(laser_power=power)

    # Get density to manually calculate expectation
    # We use default beam dimensions from the function signature in detector.py
    density = laser_power_density(power)

    # Manual Signal Calculation
    expected_signal = (
        csca * density * H10720_110_ANODE_RADIANT_SENSITIVITY * ops.mirror_reflectivity
    )

    # Manual Noise Calculation
    signal_noise_sq = 2 * ELEMENTARY_CHARGE * expected_signal
    dark_noise_sq = 2 * ELEMENTARY_CHARGE * H10720_110_DARK_CURRENT
    preamp_noise_sq = TIA60_INPUT_CURRENT_NOISE**2

    expected_total_noise = np.sqrt(
        (signal_noise_sq + dark_noise_sq + preamp_noise_sq) * BANDWIDTH
    )

    signal, noise = estimate_signal_noise(ops, csca)

    assert signal == pytest.approx(expected_signal)
    assert noise == pytest.approx(expected_total_noise)


def test_estimate_signal_noise_vectorized():
    """
    Test that the function handles numpy arrays for csca correctly.
    """
    csca_array = np.array([1.0, 2.0, 3.0])
    power = 100.0

    ops = ppops.OpticalParticleSpectrometer(laser_power=power)

    signal, noise = estimate_signal_noise(ops, csca_array)

    # Check types
    assert isinstance(signal, np.ndarray)
    assert isinstance(noise, np.ndarray)

    # Check shape
    assert signal.shape == (3,)
    assert noise.shape == (3,)

    # Check logic (Signal should scale linearly with CSCA)
    assert signal[1] == pytest.approx(signal[0] * 2)

    # Noise should not scale exactly linearly due to the fixed noise floor (dark + preamp)
    # but strictly speaking, higher signal = higher noise
    assert noise[1] > noise[0]


def test_estimate_signal_noise_custom_parameters():
    """
    Test that passing custom hardware parameters overrides defaults.
    """
    csca = 1.0
    power = 100.0

    ops = ppops.OpticalParticleSpectrometer(laser_power=power)

    # Defaults
    sig_def, noise_def = estimate_signal_noise(ops, csca)

    # Custom: Zero Dark Current, Zero Preamp noise
    ops = ppops.OpticalParticleSpectrometer(
        laser_power=power,
        dark_current=0.0,
        input_current_noise=0.0,
    )
    sig_custom, noise_custom = estimate_signal_noise(ops, csca)

    # Signal should be identical
    assert sig_custom == pytest.approx(sig_def)

    # Noise should be lower due to zero dark current and preamp noise
    assert noise_custom < noise_def
