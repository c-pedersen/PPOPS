# PPOPS
Performance Prediction for Optical Particle Spectrometer

[![codecov](https://codecov.io/gh/c-pedersen/ppops/branch/main/graph/badge.svg?ts=20251106)](https://codecov.io/gh/c-pedersen/ppops)
[![CI](https://github.com/c-pedersen/PPOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/c-pedersen/PPOPS/actions/workflows/ci.yml)
# POPS Scattering Simulation and Calibration

This repository implements a Mie scattering model and POPS (Portable Optical Particle Spectrometer) geometry simulation to calculate the truncated scattering cross section — the portion of scattered light collected by the POPS mirror.

---

## Overview

This simulation integrates the Mie scattering intensity over the solid-angle acceptance of the POPS instrument, based on the geometry of its optical collection system.

The workflow includes:
1. Computing Mie coefficients (`a_n`, `b_n`) for electric and magnetic scattering modes.
2. Generating angular functions (`π_n`, `τ_n`) that describe angle-dependent intensity.
3. Combining them into scattering amplitudes (`S₁`, `S₂`) for both polarization states.
4. Integrating over the mirror’s polar and azimuthal acceptance using POPS geometry.
5. Returning the truncated scattering cross section and scattering efficiency.

---

## Repository Structure

POPS-Scattering/
│
├── src/
│   ├── main.py              # Main driver script (integration and results)
│   ├── mie_modules.py       # Physics functions: mie_ab, mie_pt, mie_s12
│   ├── geometry.py          # Instrument geometry and polarization handling
│   ├── __init__.py
│
├── data/                    # Example input data or results (optional)
├── notebooks/               # Jupyter demos or visualizations (optional)
├── tests/                   # Unit tests for validation (optional)
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
└── LICENSE

---

## Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/PPOPS.git
cd PPOPS
pip install -r requirements.txt

---

## Usage

Run the main simulation from the command line:

python src/main.py


You can modify parameters such as:
- Refractive index (`ior`)
- Laser wavelength
- Particle diameter
- Mirror geometry

inside `main.py`.

---

## References

XX
---

## Author Notes

This repository is intended as a transparent, well-documented implementation of Mie scattering and optical geometry for POPS calibration and analysis.  

---

## License

XX
