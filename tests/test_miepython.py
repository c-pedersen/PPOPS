# ruff: noqa: E402

import sys
import os

# Workaround to resolve path issues/being unable to see src directory
# 1. Get the directory where this file (conftest.py) is located (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory of 'tests/' (which is the PPOPS project root)
project_root = os.path.join(current_dir, "..")

# 3. Add the src directory to the very start of the search path (sys.path)
# This allows Python to correctly resolve 'from ppops.geometry...'
val_dir = os.path.join(project_root, "validation")
sys.path.insert(0, os.path.abspath(val_dir))

print("\n--- Current Python Search Paths (sys.path) ---")
for p in sys.path:
    print(p)
print("------------------------------------------")

# The error label at the start of the file (E402) suppresses the ruff error that the module import is not at the top of the file
import validate_mie  # noqa: E402


def test_validate_mie():
    """
    Test the validate_mie function to ensure it runs without errors.
    """
    iors = [1.4, 1.5, 1.6]
    diameters = [0.5, 1.0, 1.5]

    for ior in iors:
        for diameter in diameters:
            validate_mie.compare_s1s2_methods(ior=ior, diameter=diameter)
