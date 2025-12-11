# ruff: noqa: E402

import pytest
import numpy as np
import math
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

# The error label at the start of the file (E402) suppresses the ruff error that the module import is not at the top of the file
from src.ppops.detector import (
    laser_power_density,
    estimate_signal_noise,
    ELEMENTARY_CHARGE,
    H10720_110_ANODE_RADIANT_SENSITIVITY,
    H10720_110_DARK_CURRENT,
    BANDWIDTH,
    TIA60_INPUT_CURRENT_NOISE,
)

# TESTS


def test_laser_power_density_calculation():
    """
    Test the basic calculation of laser power density.

    Logic check:
    Power = 100 mW -> 0.1 W
    Major = 2e-3 m -> Radius = 1e-3 m -> 1000 um
    Minor = 2e-3 m -> Radius = 1e-3 m -> 1000 um
    Area = pi * 1000 * 1000 (um^2)
    Density = 0.1 / Area
    """
    power_mw = 100.0
    major_m = 2e-3
    minor_m = 2e-3

    expected_area = math.pi * 1000.0 * 1000.0  # um^2
    expected_density = (power_mw * 1e-3) / expected_area

    result = laser_power_density(power_mw, major_m, minor_m)

    assert result == pytest.approx(expected_density)


def test_laser_power_density_zero_power():
    """Test that zero power results in zero density."""
    with pytest.warns(UserWarning, match="Laser power in mW seems unrealistic"):
        assert laser_power_density(0, 3e-3, 1e-3) == 0.0


def test_laser_power_density_negative_inputs():
    """Test that negative power or dimensions raise ValueErrors."""
    # Negative power
    with pytest.raises(ValueError, match="Laser power cannot be negative"):
        laser_power_density(-10, 3e-3, 1e-3)

    # Negative dimensions
    with pytest.raises(ValueError, match="Beam major and minor axes must be positive"):
        laser_power_density(100, -3e-3, 1e-3)

    with pytest.raises(ValueError, match="Beam major and minor axes must be positive"):
        laser_power_density(100, 3e-3, 0)


def test_laser_power_density_warnings_dimensions():
    """Test that unrealistic beam dimensions trigger warnings."""
    # Too large (> 1e-2)
    with pytest.warns(UserWarning, match="Beam dimensions in meters seem unrealistic"):
        laser_power_density(100, 2e-2, 1e-3)

    # Too small (< 1e-5)
    with pytest.warns(UserWarning, match="Beam dimensions in meters seem unrealistic"):
        laser_power_density(100, 3e-3, 1e-6)


def test_laser_power_density_warnings_power():
    """Test that unrealistic laser power triggers warnings."""
    # Too high (> 1000)
    with pytest.warns(UserWarning, match="Laser power in mW seems unrealistic"):
        laser_power_density(1500, 3e-3, 1e-3)

    # Too low (<= 10 but > 0)
    with pytest.warns(UserWarning, match="Laser power in mW seems unrealistic"):
        laser_power_density(5, 3e-3, 1e-3)


def test_estimate_signal_noise_scalar():
    """
    Test signal and noise estimation with scalar inputs.
    Verifies the physics formulas are implemented correctly.
    """
    csca = 1.0  # um^2
    power_mw = 100.0

    # Get density to manually calculate expectation
    # We use default beam dimensions from the function signature in detector.py
    # (3e-3, 1e-3)
    density = laser_power_density(power_mw, 3e-3, 1e-3)

    # Manual Signal Calculation
    expected_signal = csca * density * H10720_110_ANODE_RADIANT_SENSITIVITY

    # Manual Noise Calculation
    signal_noise_sq = 2 * ELEMENTARY_CHARGE * expected_signal
    dark_noise_sq = 2 * ELEMENTARY_CHARGE * H10720_110_DARK_CURRENT
    preamp_noise_sq = TIA60_INPUT_CURRENT_NOISE**2

    expected_total_noise = np.sqrt(
        (signal_noise_sq + dark_noise_sq + preamp_noise_sq) * BANDWIDTH
    )

    signal, noise = estimate_signal_noise(csca, power_mw)

    assert signal == pytest.approx(expected_signal)
    assert noise == pytest.approx(expected_total_noise)


def test_estimate_signal_noise_vectorized():
    """
    Test that the function handles numpy arrays for csca correctly.
    """
    csca_array = np.array([1.0, 2.0, 3.0])
    power_mw = 100.0

    signal, noise = estimate_signal_noise(csca_array, power_mw)

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

    # Defaults
    sig_def, noise_def = estimate_signal_noise(csca, power)

    # Custom: Zero Dark Current, Zero Preamp noise, Reduced
