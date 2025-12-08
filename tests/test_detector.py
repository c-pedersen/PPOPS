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

from src.ppops.detector import (
    laser_power_density,
    estimate_signal_noise,
    ELEMENTARY_CHARGE,
    H10720_110_ANODE_RADIANT_SENSITIVITY,
    H10720_110_DARK_CURRENT,
    BANDWIDTH,
    TIA60_INPUT_CURRENT_NOISE
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
    
    expected_area = math.pi * 1000.0 * 1000.0 # um^2
    expected_density = (power_mw * 1e-3) / expected_area
    
    result = laser_power_density(power_mw, major_m, minor_m)
    
    assert result == pytest.approx(expected_density)

def test_laser_power_density_zero_power():
    """Test that zero power results in zero density."""
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