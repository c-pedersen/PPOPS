# PPOPS

Performance Prediction for Optical Particle Spectrometer

[![codecov](https://codecov.io/gh/c-pedersen/ppops/branch/main/graph/badge.svg?ts=20251106)](https://codecov.io/gh/c-pedersen/ppops)
[![CI](https://github.com/c-pedersen/PPOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/c-pedersen/PPOPS/actions/workflows/ci.yml)

## POPS Scattering Simulation and Calibration

This repository implements a Mie scattering model and Optical Particle
Spectrometer (OPS) geometry simulation to calculate the truncated
scattering cross section—the portion of scattered light collected by the
POPS mirror-and the resulting signal and noise current.

## Overview

This repository implements a Mie scattering model and Optical Particle
Spectrometer (OPS) geometry simulation to calculate the truncated
scattering cross section—the portion of scattered light collected by the
POPS mirror-and the resulting signal and noise current. To do this, the
package integrates the Mie scattering intensity over the solid-angle
acceptance of the OPS, based on the geometry of its optical collection
system.

The workflow includes:

1. Computing Mie coefficients (`a_n`, `b_n`) for electric and magnetic
scattering modes.
2. Generating angular functions (`π_n`, `τ_n`) that describe
angle-dependent intensity.
3. Combining them into scattering amplitudes (`S₁`, `S₂`) for both
polarization states. (Steps 1-3 are now implemented using miepython)
4. Integrating over the OPS collection mirror’s polar and azimuthal
acceptance to compute the truncated scattering cross section.
5. Convert the truncated scattering cross section to signal and noise
current based on detector specifications.

## Assumptions

This package make several assumptions:

1. Particles are perfectly spherical.
2. Scattering is elastic.
3. All scattered light incident on the collection mirror is detected.
4. The only noise sources are signal noise, dark noise, and preamp noise
(i.e., we neglect stray scattering).
5. The detector is shot noise limited.

## Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/PPOPS.git
cd PPOPS
pip install -r requirements.txt

## Usage

```Python
import ppops

ops = ppops.OpticalParticleSpectrometer()

ops.estimate_signal_noise(diameters = [0.1, 0.5, 1.0], ior = 1.5+0.001j)
```

## References

Gao, R.S., Telg, H., McLaughlin, R.J., Ciciora, S.J., Watts, L.A.,
Richardson, M.S., Schwarz, J.P., Perring, A.E., Thornberry, T.D.,
Rollins, A.W., Markovic, M.Z., Bates, T.S., Johnson, J.E., Fahey, D.W.,
2016. A light-weight, high-sensitivity particle spectrometer for PM2.5
2017. aerosol measurements. Aerosol Science and Technology 50, 88–99.
2018. https://doi.org/10.1080/02786826.2015.1131809

miepython: 10.5281/zenodo.7949263

## Author Notes

This repository is intended as a transparent, well-documented
implementation of Mie scattering and optical geometry for OPS
calibration and analysis.  

## License

``PPOPS`` is released under the MIT license.
