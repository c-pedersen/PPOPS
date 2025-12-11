# Physical Basis: Mie Theory
## `mie_modules.py` — Mie Scattering Utilities

This module implements numerical routines for computing **Mie scattering** by spherical particles.  
It provides functions for calculating:

- Mie coefficients \(a_n\) and \(b_n\)
- Angular functions \(\pi_n(u)\) and \(\tau_n(u)\)
- Scattering amplitude functions \(S_1(\theta)\) and \(S_2(\theta)\)

These quantities follow standard formulations from electromagnetic scattering theory and are commonly used in aerosol physics, atmospheric science, and optical particle characterization.

---

## Overview

### Functions Included

| Function | Description |
|---------|-------------|
| **`mie_ab(m, x)`** | Computes the complex Mie coefficients \(a_n\) and \(b_n\) describing electric and magnetic multipole contributions. |
| **`mie_pt(u, n_max)`** | Computes angular functions \(\pi_n(u)\) and \(\tau_n(u)\) derived from associated Legendre polynomials. |
| **`mie_s12(m, x, u)`** | Computes the scattering amplitude functions \(S_1\) and \(S_2\) for s- and p-polarized scattered light. |

These are the fundamental components needed to compute phase functions, scattering efficiencies, polarization ratios, and full intensity distributions from Mie theory.

---

## Dependencies

```python
import numpy as np
from scipy.special import spherical_jn, spherical_yn
```

---

## Function Documentation

---

### `mie_ab(m: complex, x: float) -> np.ndarray`

Compute the Mie coefficients \(a_n\) and \(b_n\), which describe how a spherical particle scatters light.

**Parameters:**

- **`m`** (*complex*)  
  Complex refractive index of the particle, \(m = n + ik\).

- **`x`** (*float*)  
  Size parameter, defined as  
  \[
  x = \frac{\pi D}{\lambda}
  \]

**Returns:**  
A 2×N NumPy array:

- `a_n` — electric multipole coefficients  
- `b_n` — magnetic multipole coefficients

**Key Implementation Notes:**

- Determines the required order \(n_\text{max}\) via the approximation:
  \[
  n_{\max} \approx 2 + x + 4x^{1/3}
  \]

- Uses Riccati–Bessel functions:
  \[
  \psi_n(x) = x j_n(x), \quad
  \chi_n(x) = -x y_n(x)
  \]

- Computes the logarithmic derivative \(D_n(z)\) using *backward recurrence*.

- Constructs the Mie coefficients from:
  \[
  a_n = \frac{(\frac{D_n}{m} + \frac{n}{x})\psi_n - \psi_{n-1}}
               {(\frac{D_n}{m} + \frac{n}{x})\xi_n - \xi_{n-1}}
  \]
  with an analogous expression for \(b_n\).

---

### `mie_pt(u: float, n_max: int) -> np.ndarray`

Compute angular functions \(\pi_n(u)\) and \(\tau_n(u)\), where:

- \(u = \cos\theta\)
- \(\theta\) is the scattering angle.

These functions capture the angular dependence of scattered light.

**Parameters:**

- **`u`** (*float*) — cosine of scattering angle  
- **`n_max`** (*int*) — maximum multipole order

**Returns:**  
A 2×N array: `[π_n(u), τ_n(u)]`.

**Internal Notes:**

- Initializes dipole (n=1) and quadrupole (n=2) terms.
- Higher-order terms computed via recurrence:
  \[
  \pi_n = \frac{(2n-1)u\,\pi_{n-1} - n\,\pi_{n-2}}{n-1}
  \]
  \[
  \tau_n = n u \pi_n - (n+1)\pi_{n-1}
  \]

---

### `mie_s12(m: complex, x: float, u: float) -> np.ndarray`

Compute the Mie scattering amplitude functions \(S_1\) and \(S_2\), which fully characterize scattering for perpendicular and parallel polarization states.

**Parameters:**

- **`m`** (*complex*) — refractive index
- **`x`** (*float*) — size parameter
- **`u`** (*float*) — \(\cos(\theta)\)

**Returns:**  
A 2-element array: `[S1, S2]`.

**Computation Outline:**

1. Determine `n_max` using the standard approximation.
2. Compute Mie coefficients `a_n`, `b_n`.
3. Compute angular functions `π_n(u)`, `τ_n(u)`.
4. Apply the weighting:
   \[
   \frac{2n + 1}{n(n+1)}
   \]
5. Form the scattering amplitudes:
   \[
   S_1 = \sum_n \left(a_n \pi_n + b_n \tau_n\right)
   \]
   \[
   S_2 = \sum_n \left(a_n \tau_n + b_n \pi_n\right)
   \]

These can be further used to compute intensity patterns:

\[
I(\theta) \propto |S_1|^2 \quad \text{or} \quad |S_2|^2
\]

---

## References

- **Bohren, C. F. & Huffman, D. R. (1983)**  
  *Absorption and Scattering of Light by Small Particles.*

- **C. Mätzler (2002)**  
  MATLAB codes for Mie scattering, IAP Research Report.

