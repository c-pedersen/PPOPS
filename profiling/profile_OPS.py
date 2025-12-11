# ruff: noqa: E402

import cProfile
import pstats
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
project_path = os.path.abspath(project_root)
src_path = os.path.join(project_path, "src")
sys.path.insert(0, project_path)

print(f"Added to path: {project_root}")
print("\n--- Current Python Search Paths (sys.path) ---")
for p in sys.path:
    print(p)
print("------------------------------------------")

# The error label at the start of the file (E402) suppresses the ruff error that the module import is not at the top of the file
import ppops


def run_profiling():
    """Instantiates the class and runs the target function."""

    # 1. Instantiate the class
    ops = ppops.OpticalParticleSpectrometer()

    # 2. Call the function to be profiled
    ops.estimate_signal_noise(diameters=DIAMETERS, ior=IOR)


if __name__ == "__main__":
    # Ensure a command line argument is given and only contains regular characters
    if len(sys.argv) < 2:
        print("Error: Please provide a test name as a command line argument.")
        sys.exit(1)
    test_name = sys.argv[1]
    if not all(c.isalnum() or c == "_" for c in test_name):
        print("Error: Test name must only contain letters, numbers, or underscores.")
        sys.exit(1)

    # Output file for profiling results
    profile_output_file = f"OPS_profile_data_{test_name}.prof"

    # Inputs for OPS
    DIAMETERS = np.arange(0.1, 2, 0.01)  # ~190 diameters
    IOR = 1.6  # Simple real refractive index

    print(f"Starting profiling for {test_name} with {len(DIAMETERS)} diameters...")

    # Run the function under cProfile
    cProfile.run("run_profiling()", project_path + "/profiling/" + profile_output_file)

    # Print a text summary (pstats) for quick analysis
    p = pstats.Stats(project_path + "/profiling/" + profile_output_file)
    print("\n--- Top 10 Functions by Cumulative Time (cumtime) ---")

    # Sort by 'cumulative' time (time spent in function and all sub-functions)
    p.strip_dirs().sort_stats("cumulative").print_stats(10)

    print(f"\nProfile data saved to '{profile_output_file}'.")
    print(f"Run 'snakeviz profiling/{profile_output_file}' for visualization.")
