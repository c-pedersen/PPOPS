# PPOPS

Performance Prediction for Optical Particle Spectrometer

[![codecov](https://codecov.io/gh/c-pedersen/ppops/branch/main/graph/badge.svg?ts=20251106)](https://codecov.io/gh/c-pedersen/ppops)
[![CI](https://github.com/c-pedersen/PPOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/c-pedersen/PPOPS/actions/workflows/ci.yml)

## Overview

This repository simulates the performance of a simple Optical Particle
Spectrometer (OPS) using a physics-based model. The default optical
system is based on Gao et al., Aerosol Sci. Technol., 2016, which is
also what the POPS instrument sold by Handix Scientific is based on. The
PPOPS model is semi-validated in that PPOPS modeled signal was within
50% of the actual signal of a POPS unit from Handix Scientific (see
validation section of PPOPS website for more information).

The PPOPS model calculates the amount and direction of scattered light
by aerosols according to Mie theory. The scattered light is integrated
over the solid-angle acceptance of the collection mirror to calculate a
truncated scattering cross section (the effective scattering cross
section). This truncated scattering cross section is then used to
calculate the instrument signal and noise current.

To summarize, the PPOPS workflow includes,

1. Use miepython to calculate the scattering amplitudes (`S₁`, `S₂`)
according to Mie theory.
   1. Compute Mie coefficients (`a_n`, `b_n`) for electric and magnetic
scattering modes.
   1. Generate angular functions (`π_n`, `τ_n`) that describe
angle-dependent intensity.
1. Integrating over the OPS collection mirror’s polar and azimuthal
acceptance to compute the truncated scattering cross section.
1. Convert the truncated scattering cross section to signal and noise
current based on detector specifications.
1. Optionally, convert signal current to digitizer bins for direct
comparison with POPS data.

## Assumptions

This package makes several assumptions:

1. Particles are perfectly spherical.
2. Scattering is elastic.
3. All scattered light incident on the collection mirror is detected.
4. The only noise sources are signal noise, dark noise, and preamp noise
(i.e., we neglect stray scattering).
5. The detector is shot noise limited.

## Installation

PPOPS is available via PyPI

```Python
pip install ppops
```

## Usage

See PPOPS website for more information.

```Python
import ppops

ops = ppops.OpticalParticleSpectrometer()

ops.estimate_signal_noise(diameters = [0.1, 0.5, 1.0], ior = 1.5+0.001j)
```

## References

Gao, R.S., Telg, H., McLaughlin, R.J., Ciciora, S.J., Watts, L.A., Richardson, M.S., Schwarz, J.P., Perring, A.E., Thornberry, T.D., Rollins, A.W., Markovic, M.Z., Bates, T.S., Johnson, J.E., Fahey, D.W., 2016. A light-weight, high-sensitivity particle spectrometer for PM2.5 aerosol measurements. Aerosol Science and Technology 50, 88–99. <https://doi.org/10.1080/02786826.2015.1131809>

miepython: 10.5281/zenodo.7949263

## Author Notes

This repository is intended as a transparent, well-documented
implementation of Mie scattering and optical geometry for POPS
performance prediction and analysis.

## License

`PPOPS` is released under the MIT license.
