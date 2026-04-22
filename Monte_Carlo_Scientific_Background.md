# Monte Carlo Proton Transport: Scientific Background

## Natural Sciences Computing Club, UNC Chapel Hill

This document provides the scientific background for implementing Monte Carlo proton transport in the Bragg Peak simulation, replacing the simplified CSDA (Continuous Slowing Down Approximation) approach.

---

## Table of Contents

1. [Why Monte Carlo?](#1-why-monte-carlo)
2. [Multiple Coulomb Scattering](#2-multiple-coulomb-scattering)
3. [Energy Straggling](#3-energy-straggling)
4. [Implementation: Condensed History Method](#4-implementation-condensed-history-method)
5. [Validation Against Reference Codes](#5-validation-against-reference-codes)
6. [Mathematical Appendix](#6-mathematical-appendix)

---

## 1. Why Monte Carlo?

### Limitations of CSDA

The original simulation uses the **Continuous Slowing Down Approximation (CSDA)**, which assumes:

| Assumption | Reality |
|------------|---------|
| Protons travel in straight lines | Protons scatter due to Coulomb interactions |
| Energy loss is continuous and deterministic | Energy loss is discrete and stochastic |
| All protons stop at the same depth | Range varies statistically (range straggling) |

### What Monte Carlo Provides

Monte Carlo simulation models the **stochastic nature** of particle transport by:

1. **Sampling random interactions** from probability distributions
2. **Tracking individual particle histories**
3. **Building up dose distributions** from many histories

This produces physically realistic features:
- **Lateral beam spreading** (penumbra)
- **Range straggling** (distal dose falloff)
- **Dose fluctuations** (statistical noise)

---

## 2. Multiple Coulomb Scattering

### Physical Mechanism

As protons traverse matter, they undergo many small-angle deflections from **elastic Coulomb scattering** with atomic nuclei. A 150 MeV proton experiences ~10⁶ scattering events in water.

### Molière Theory

The angular distribution of scattered particles was derived by Molière (1947-1948). The theory predicts:

- A **Gaussian core** containing ~98% of particles (small angles)
- **Non-Gaussian tails** from single large-angle scatters (rare events)

The full Molière distribution is:

$$f(\theta) = \frac{1}{2\pi\theta_M^2} \left[ f^{(0)}(\theta') + \frac{f^{(1)}(\theta')}{B} + \frac{f^{(2)}(\theta')}{B^2} + \cdots \right]$$

where $\theta' = \theta/(\theta_M\sqrt{B})$ and $B$ is a screening parameter.

### Highland Formula (Practical Approximation)

For Monte Carlo, we use the **Highland formula**, which approximates the RMS scattering angle:

$$\theta_{\text{rms}} = \frac{14.1 \text{ MeV}}{p \cdot \beta \cdot c} \sqrt{\frac{x}{X_0}} \left[ 1 + \frac{1}{9}\log_{10}\left(\frac{x}{X_0}\right) \right]$$

**Where:**
- $p$ = proton momentum [MeV/c]
- $\beta$ = velocity / speed of light
- $x$ = path length [cm]
- $X_0$ = radiation length (36.08 cm for water)

### Implementation Notes

In our code (`Braggs_Peak_MC.py`), we implement this in `highland_theta_rms()`:

```python
def highland_theta_rms(E_MeV, dz_cm, X0_cm=36.08):
    beta, gamma, p_MeV = get_relativistic_params(E_MeV)
    x_over_X0 = dz_cm / X0_cm
    log_term = 1.0 + np.log10(x_over_X0) / 9.0
    theta_rms = (14.1 / (p_MeV * beta)) * np.sqrt(x_over_X0) * log_term
    return theta_rms
```

The scattering angle is then sampled from a 2D Gaussian:
- $\theta_x \sim N(0, \theta_{\text{rms}})$
- $\theta_y \sim N(0, \theta_{\text{rms}})$
- $\theta = \sqrt{\theta_x^2 + \theta_y^2}$

---

## 3. Energy Straggling

### Physical Mechanism

Energy loss occurs through discrete ionization events. While the **average** energy loss is given by the Bethe-Bloch formula (stopping power), individual protons lose slightly different amounts of energy due to statistical fluctuations.

This leads to **range straggling**: protons with identical initial energies stop at different depths.

### Three Classical Theories

| Theory | Year | Regime | Distribution Shape |
|--------|------|--------|-------------------|
| **Landau** | 1944 | Thin absorbers | Asymmetric, long tail |
| **Vavilov** | 1957 | Intermediate | Interpolates Landau↔Bohr |
| **Bohr** | 1915 | Thick absorbers | Gaussian (CLT applies) |

### Bohr's Theory (Our Implementation)

For **thick absorbers** where many collisions occur, the Central Limit Theorem applies and the energy loss distribution becomes Gaussian.

**Bohr's variance formula:**

$$\sigma_E^2 = 4\pi N_A r_e^2 m_e c^2 \cdot Z \cdot z^2 \cdot \rho \cdot \Delta x \cdot \frac{Z_{\text{target}}}{A_{\text{target}}}$$

**Simplified for water:**

$$\sigma_E^2 \approx 0.157 \cdot \frac{Z}{A} \cdot \rho \cdot \Delta x \quad [\text{MeV}^2]$$

where $Z/A \approx 10/18$ for water (H₂O).

### Implementation

In `bohr_straggling_sigma()`:

```python
def bohr_straggling_sigma(dz_cm, rho_g_cm3=1.0):
    Z_over_A = 10.0 / 18.0  # Water
    K_straggle = 0.157      # MeV²·cm²/g
    sigma_squared = K_straggle * Z_over_A * rho_g_cm3 * dz_cm
    return np.sqrt(sigma_squared)
```

The actual energy loss is then sampled:
$$\Delta E = \Delta E_{\text{mean}} + N(0, \sigma_E)$$

### Range Straggling

The cumulative effect of energy straggling produces a distribution of stopping depths. The range straggling (σ_R) is approximately:

$$\sigma_R \approx 0.012 \cdot R_0$$

where $R_0$ is the CSDA range. For 150 MeV protons in water ($R_0 \approx 15.8$ cm), this gives $\sigma_R \approx 0.2$ cm.

---

## 4. Implementation: Condensed History Method

### The Challenge

Simulating every individual Coulomb scattering event (~10⁶ per proton) would be computationally prohibitive.

### Solution: Condensed History

The **Condensed History** or **Condensed Random Walk** method (Berger, 1963) groups many small interactions into single steps:

1. **Divide the path** into discrete steps (Δz ~ 0.01 cm)
2. **At each step:**
   - Calculate mean energy loss from stopping power
   - Sample energy fluctuation from straggling distribution
   - Sample angular deflection from MCS distribution
   - Update position and direction
3. **Repeat** until energy falls below cutoff

### Algorithm Pseudocode

```
For each proton history:
    Initialize: position = (0,0,0), direction = (0,0,1), E = E₀

    While E > E_cutoff:
        1. Get stopping power S(E) from PSTAR data
        2. Calculate mean energy loss: ΔE_mean = S × Δz
        3. Sample straggling: ΔE = ΔE_mean + N(0, σ_E)
        4. Sample scattering angle: θ from Highland formula
        5. Rotate direction vector by (θ, φ)
        6. Step forward: position += direction × Δz
        7. Update energy: E -= ΔE
        8. Score dose at position
```

### Direction Vector Rotation

When a particle scatters by angle θ with azimuthal angle φ, we must rotate the direction vector. This requires transforming to a local coordinate system where the current direction is the z-axis.

The rotation is performed using the **rotation matrix** approach implemented in `rotate_direction()`.

---

## 5. Validation Against Reference Codes

### Reference Monte Carlo Codes

| Code | Organization | Strengths |
|------|--------------|-----------|
| **Geant4** | CERN | Gold standard, comprehensive physics |
| **TOPAS** | MGH/SLAC | User-friendly Geant4 wrapper |
| **FLUKA** | CERN/INFN | Excellent hadron physics |
| **MCNP** | LANL | Neutron/photon, some proton |

### Validation Metrics

1. **Depth-dose curve shape**
   - Compare Bragg peak position (should match within 1-2 mm)
   - Compare peak-to-entrance ratio
   - Compare distal falloff width

2. **Range accuracy**
   - Mean range should match CSDA range closely
   - Range straggling σ should be ~1.2% of range

3. **Lateral profiles**
   - Beam width vs. depth
   - Penumbra at various depths

4. **Angular distributions**
   - Exit angle distribution
   - Comparison to Highland prediction

### Expected Results

For 150 MeV protons in water:

| Parameter | Expected Value |
|-----------|---------------|
| CSDA Range | ~15.8 cm |
| Mean MC Range | ~15.7-15.9 cm |
| Range straggling (σ) | ~0.15-0.20 cm |
| Peak depth | ~15.5 cm |
| Lateral spread at peak | ~0.3-0.5 cm (σ) |

### Getting TOPAS for Validation

TOPAS is free for research:
1. Register at https://www.topasmc.org/
2. Download and install
3. Create simple water phantom geometry
4. Run proton beam simulation
5. Export depth-dose curve for comparison

---

## 6. Mathematical Appendix

### A. Relativistic Kinematics

For a proton with kinetic energy $E$:

**Lorentz factor:**
$$\gamma = 1 + \frac{E}{m_p c^2}$$

**Velocity (as fraction of c):**
$$\beta = \sqrt{1 - \frac{1}{\gamma^2}}$$

**Momentum:**
$$p = m_p c \cdot \gamma \cdot \beta$$

where $m_p c^2 = 938.272$ MeV.

### B. Stopping Power

The Bethe-Bloch formula gives the mean energy loss per unit path length:

$$-\frac{dE}{dx} = K \cdot \frac{Z}{A} \cdot \frac{z^2}{\beta^2} \left[ \ln\frac{2m_e c^2 \beta^2 \gamma^2 T_{\max}}{I^2} - 2\beta^2 - \delta - 2\frac{C}{Z} \right]$$

In practice, we use tabulated PSTAR data from NIST.

### C. Radiation Length

The radiation length $X_0$ characterizes electromagnetic interactions:

$$\frac{1}{X_0} = 4\alpha r_e^2 \frac{N_A}{A} \left[ Z^2 (L_{rad} - f(Z)) + Z \cdot L'_{rad} \right]$$

For water: $X_0 = 36.08$ cm

### D. Gaussian Sampling

To sample from a Gaussian distribution $N(\mu, \sigma)$:

Using Box-Muller transform:
$$X = \mu + \sigma \sqrt{-2\ln(U_1)} \cos(2\pi U_2)$$

where $U_1, U_2$ are uniform random numbers on (0,1).

In NumPy: `np.random.normal(mu, sigma)`

### E. 3D Direction Rotation

Given current direction $\vec{d} = (d_x, d_y, d_z)$ and scattering angles $(\theta, \phi)$:

For $|d_z| < 0.99999$:
$$d'_x = d_x \cos\theta + \frac{(d_x d_z \cos\phi - d_y \sin\phi) \sin\theta}{\sqrt{1-d_z^2}}$$

$$d'_y = d_y \cos\theta + \frac{(d_y d_z \cos\phi + d_x \sin\phi) \sin\theta}{\sqrt{1-d_z^2}}$$

$$d'_z = d_z \cos\theta - \sqrt{1-d_z^2} \sin\theta \cos\phi$$

---

## References

1. Molière, G. (1947). "Theorie der Streuung schneller geladener Teilchen I." *Z. Naturforsch.* 2a, 133.

2. Highland, V. L. (1975). "Some practical remarks on multiple scattering." *Nucl. Instrum. Meth.* 129, 497.

3. Bohr, N. (1915). "On the decrease of velocity of swiftly moving electrified particles." *Phil. Mag.* 30, 581.

4. Vavilov, P. V. (1957). "Ionization losses of high-energy heavy particles." *JETP* 5, 749.

5. Berger, M. J. (1963). "Monte Carlo calculation of the penetration and diffusion of fast charged particles." *Methods Comput. Phys.* 1, 135.

6. ICRU Report 49 (1993). "Stopping Powers and Ranges for Protons and Alpha Particles."

7. Gottschalk, B. (2010). "On the scattering power of radiotherapy protons." *Med. Phys.* 37, 352.

8. Perl, J. et al. (2012). "TOPAS: An innovative proton Monte Carlo platform." *Med. Phys.* 39, 6818.

---

## Code Files

| File | Description |
|------|-------------|
| `Braggs Peak.py` | Original CSDA simulation |
| `Braggs_Peak_MC.py` | Monte Carlo implementation |
| `Monte_Carlo_Scientific_Background.md` | This document |

---

*Document version: 1.0*
*Last updated: 2026*
*Natural Sciences Computing Club, UNC Chapel Hill*
