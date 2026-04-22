# -*- coding: utf-8 -*-
"""
Monte Carlo Bragg Peak Simulation
Based on original CSDA implementation by reichen schaller

This version implements:
- 3D proton tracking (position + direction)
- Multiple Coulomb Scattering (Highland formula)
- Energy straggling (Bohr model)
- Multi-history Monte Carlo sampling

@author: Natural Sciences Computing Club, UNC Chapel Hill
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Part 1: Configuration Controls
# =============================================================================
CSV_FILE = "C:/Users/reich/Downloads/Water PSTAR Data.csv"
E0_MEV = 150.0                  # Initial proton energy [MeV]
DZ_CM = 0.01                    # Step size [cm] (larger for MC efficiency)
ZMAX_CM = None                  # Maximum depth (auto-calculated if None)
E_CUT_MEV = 1.0e-3              # Energy cutoff [MeV]
RHO_WATER_G_CM3 = 1.0           # Water density [g/cm³]

# Monte Carlo parameters
N_HISTORIES = 1000              # Number of proton histories to simulate
USE_SCATTERING = True           # Enable multiple Coulomb scattering
USE_STRAGGLING = True           # Enable energy straggling
RANDOM_SEED = 42                # For reproducibility (set None for random)

# Physics constants
X0_WATER_CM = 36.08             # Radiation length of water [cm]
M_PROTON_MEV = 938.272          # Proton rest mass [MeV/c²]

# Dose scoring grid
NX_BINS = 50                    # Number of bins in x
NY_BINS = 50                    # Number of bins in y
NZ_BINS = 200                   # Number of bins in z (depth)
XY_HALF_WIDTH_CM = 2.0          # Half-width of scoring region in x,y [cm]


# =============================================================================
# Part 2: Data Loading (unchanged from original)
# =============================================================================
def load_pstar_csv(path):
    """
    Load NIST PSTAR stopping power data from CSV.

    Returns:
        E_MeV: Energy array [MeV]
        S_mass: Mass stopping power [MeV·cm²/g]
        R_csda_mass: CSDA range [g/cm²]
    """
    raw = pd.read_csv(path, header=None, dtype=str)
    raw = raw.apply(lambda col: col.str.strip() if col.dtype == object else col)

    energy = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    stop = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    csda = pd.to_numeric(raw.iloc[:, 2], errors="coerce")

    mask = energy.notna() & stop.notna() & csda.notna()

    E_MeV = energy[mask].to_numpy(dtype=float)
    S_mass = stop[mask].to_numpy(dtype=float)
    R_csda_mass = csda[mask].to_numpy(dtype=float)

    order = np.argsort(E_MeV)
    E_MeV = E_MeV[order]
    S_mass = S_mass[order]
    R_csda_mass = R_csda_mass[order]

    keep = np.concatenate(([True], np.diff(E_MeV) > 0))
    E_MeV = E_MeV[keep]
    S_mass = S_mass[keep]
    R_csda_mass = R_csda_mass[keep]

    if len(E_MeV) < 5:
        raise ValueError("CSV file does not contain enough numeric PSTAR rows.")

    return E_MeV, S_mass, R_csda_mass


def make_log_interp(x, y):
    """
    Create log-log interpolation function for smooth physics data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("Interpolation data must be strictly positive.")

    lx = np.log(x)
    ly = np.log(y)

    def interp(x_new):
        x_new = np.asarray(x_new, dtype=float)
        if np.any(x_new <= 0.0):
            raise ValueError("Interpolation input must be strictly positive.")
        return np.exp(np.interp(np.log(x_new), lx, ly))

    return interp


# =============================================================================
# Part 3: Monte Carlo Physics Functions
# =============================================================================

def get_relativistic_params(E_MeV):
    """
    Calculate relativistic parameters for a proton.

    Args:
        E_MeV: Kinetic energy [MeV]

    Returns:
        beta: v/c (velocity as fraction of speed of light)
        gamma: Lorentz factor
        p_MeV: Momentum [MeV/c]
    """
    gamma = 1.0 + E_MeV / M_PROTON_MEV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_MeV = M_PROTON_MEV * gamma * beta
    return beta, gamma, p_MeV


def highland_theta_rms(E_MeV, dz_cm, X0_cm=X0_WATER_CM):
    """
    Calculate RMS scattering angle using Highland formula.

    The Highland formula approximates Moliere's multiple scattering theory
    for the central Gaussian part of the angular distribution.

    θ_rms = (14.1 MeV / pv) × √(x/X₀) × [1 + (1/9)·log₁₀(x/X₀)]

    Args:
        E_MeV: Proton kinetic energy [MeV]
        dz_cm: Step length [cm]
        X0_cm: Radiation length [cm]

    Returns:
        theta_rms: RMS scattering angle [radians]
    """
    if E_MeV <= 0 or dz_cm <= 0:
        return 0.0

    beta, gamma, p_MeV = get_relativistic_params(E_MeV)

    x_over_X0 = dz_cm / X0_cm

    # Highland formula
    # Note: log₁₀ term can be negative for small steps, but formula still valid
    log_term = 1.0 + np.log10(x_over_X0) / 9.0
    theta_rms = (14.1 / (p_MeV * beta)) * np.sqrt(x_over_X0) * log_term

    return max(theta_rms, 0.0)


def sample_scattering_angle(E_MeV, dz_cm, X0_cm=X0_WATER_CM):
    """
    Sample a scattering angle from the multiple scattering distribution.

    Uses Gaussian approximation to Moliere theory (valid for ~98% of particles).

    Args:
        E_MeV: Proton kinetic energy [MeV]
        dz_cm: Step length [cm]
        X0_cm: Radiation length [cm]

    Returns:
        theta: Polar scattering angle [radians]
        phi: Azimuthal angle [radians]
    """
    theta_rms = highland_theta_rms(E_MeV, dz_cm, X0_cm)

    # Sample from 2D Gaussian (projected angles)
    # theta_x and theta_y are independent Gaussian with same sigma
    theta_x = np.random.normal(0, theta_rms)
    theta_y = np.random.normal(0, theta_rms)

    # Convert to polar coordinates
    theta = np.sqrt(theta_x**2 + theta_y**2)
    phi = np.arctan2(theta_y, theta_x)

    return theta, phi


def bohr_straggling_sigma(dz_cm, rho_g_cm3=RHO_WATER_G_CM3):
    """
    Calculate energy straggling sigma using Bohr's theory.

    Bohr's theory gives the variance of energy loss for thick absorbers
    where the central limit theorem applies (many collisions).

    σ²_Bohr = 4π × N_A × Z² × z² × e⁴ × ρ × Δx / A

    For water (Z_eff ≈ 7.42, A_eff ≈ 14.3):
    σ² ≈ 0.157 × (Z/A) × ρ × Δx  [MeV²]

    Args:
        dz_cm: Step length [cm]
        rho_g_cm3: Material density [g/cm³]

    Returns:
        sigma_E: Standard deviation of energy loss [MeV]
    """
    # Effective Z/A for water (H₂O)
    # Z_eff = (2×1 + 8) / 3 ≈ 3.33 for electrons
    # But for straggling we use: Z/A ≈ 10/18 = 0.556
    Z_over_A = 10.0 / 18.0  # Water: Z=10 (2H + O), A=18

    # Bohr straggling constant (derived from fundamental constants)
    # K = 4π × N_A × r_e² × m_e × c² = 0.1535 MeV·cm²/g
    K_straggle = 0.157  # MeV²·cm²/g (empirical constant for Bohr)

    sigma_squared = K_straggle * Z_over_A * rho_g_cm3 * dz_cm

    return np.sqrt(sigma_squared)


def sample_energy_loss(E_MeV, S_lin_MeV_cm, dz_cm, use_straggling=True):
    """
    Sample energy loss with optional Bohr straggling.

    Args:
        E_MeV: Current proton energy [MeV]
        S_lin_MeV_cm: Linear stopping power [MeV/cm]
        dz_cm: Step length [cm]
        use_straggling: Whether to include stochastic fluctuations

    Returns:
        dE: Energy loss in this step [MeV]
    """
    # Mean energy loss (CSDA)
    dE_mean = S_lin_MeV_cm * dz_cm

    if not use_straggling:
        return min(dE_mean, E_MeV)

    # Add Bohr straggling
    sigma_E = bohr_straggling_sigma(dz_cm)
    dE = np.random.normal(dE_mean, sigma_E)

    # Physical constraints: can't lose negative energy or more than we have
    dE = max(0.0, min(dE, E_MeV))

    return dE


def rotate_direction(direction, theta, phi):
    """
    Rotate a direction vector by polar angle theta and azimuthal angle phi.

    Uses rotation matrix approach to deflect the particle direction.
    The rotation is performed in a local coordinate system where the
    current direction is the z-axis.

    Args:
        direction: Current direction unit vector [dx, dy, dz]
        theta: Polar deflection angle [radians]
        phi: Azimuthal angle [radians]

    Returns:
        new_direction: Rotated unit vector
    """
    dx, dy, dz = direction

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Handle case when direction is nearly along z-axis
    if abs(dz) > 0.99999:
        # Special case: direction is nearly vertical
        sign_z = np.sign(dz) if dz != 0 else 1.0
        new_dx = sin_theta * cos_phi
        new_dy = sin_theta * sin_phi * sign_z
        new_dz = cos_theta * sign_z
    else:
        # General case: construct local coordinate system
        dz_perp = np.sqrt(1.0 - dz**2)

        # New direction components
        new_dx = (dx * cos_theta +
                  (dx * dz * cos_phi - dy * sin_phi) * sin_theta / dz_perp)
        new_dy = (dy * cos_theta +
                  (dy * dz * cos_phi + dx * sin_phi) * sin_theta / dz_perp)
        new_dz = dz * cos_theta - dz_perp * sin_theta * cos_phi

    # Normalize to ensure unit vector
    norm = np.sqrt(new_dx**2 + new_dy**2 + new_dz**2)

    return np.array([new_dx / norm, new_dy / norm, new_dz / norm])


# =============================================================================
# Part 4: Monte Carlo Simulation Engine
# =============================================================================

def simulate_single_proton(E0_MeV, S_mass_of_E, dz_cm, E_cut_MeV, rho,
                           use_scattering=True, use_straggling=True):
    """
    Simulate a single proton history through water.

    Args:
        E0_MeV: Initial kinetic energy [MeV]
        S_mass_of_E: Interpolation function for mass stopping power
        dz_cm: Step size [cm]
        E_cut_MeV: Energy cutoff [MeV]
        rho: Material density [g/cm³]
        use_scattering: Enable MCS
        use_straggling: Enable energy straggling

    Returns:
        positions: Array of (x, y, z) positions [cm]
        energies: Array of energies at each position [MeV]
        dose_deposits: Array of energy deposited at each step [MeV]
    """
    # Initialize proton state
    position = np.array([0.0, 0.0, 0.0])  # Start at origin
    direction = np.array([0.0, 0.0, 1.0])  # Initially traveling along +z
    energy = E0_MeV

    # Storage for history
    positions = [position.copy()]
    energies = [energy]
    dose_deposits = [0.0]  # No dose at entry point

    # Transport loop
    while energy > E_cut_MeV:
        # Get stopping power at current energy
        S_lin = rho * float(S_mass_of_E(energy))  # MeV/cm

        # Sample energy loss (with optional straggling)
        dE = sample_energy_loss(energy, S_lin, dz_cm, use_straggling)

        # Sample scattering angle (with optional MCS)
        if use_scattering:
            theta, phi = sample_scattering_angle(energy, dz_cm)
            direction = rotate_direction(direction, theta, phi)

        # Move proton
        position = position + direction * dz_cm
        energy = energy - dE

        # Record state
        positions.append(position.copy())
        energies.append(max(energy, 0.0))
        dose_deposits.append(dE)

        # Safety check for runaway
        if position[2] > 50.0 or position[2] < -1.0:
            break

    return np.array(positions), np.array(energies), np.array(dose_deposits)


def simulate_mc_bragg_peak(
    csv_file,
    E0_MeV,
    dz_cm,
    zmax_cm,
    E_cut_MeV,
    rho_water_g_cm3,
    n_histories=N_HISTORIES,
    use_scattering=USE_SCATTERING,
    use_straggling=USE_STRAGGLING,
    random_seed=RANDOM_SEED,
    xy_half_width=XY_HALF_WIDTH_CM,
    nx_bins=NX_BINS,
    ny_bins=NY_BINS,
    nz_bins=NZ_BINS,
):
    """
    Run full Monte Carlo Bragg peak simulation.

    Args:
        csv_file: Path to PSTAR data
        E0_MeV: Initial energy [MeV]
        dz_cm: Step size [cm]
        zmax_cm: Maximum depth [cm]
        E_cut_MeV: Energy cutoff [MeV]
        rho_water_g_cm3: Water density [g/cm³]
        n_histories: Number of proton histories
        use_scattering: Enable MCS
        use_straggling: Enable energy straggling
        random_seed: RNG seed for reproducibility
        xy_half_width: Half-width of scoring region [cm]
        nx_bins, ny_bins, nz_bins: Number of dose grid bins

    Returns:
        Dictionary with simulation results
    """
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load PSTAR data
    E_tab, S_mass_tab, R_csda_mass_tab = load_pstar_csv(csv_file)

    if not (E_tab[0] <= E0_MeV <= E_tab[-1]):
        raise ValueError(f"E0_MeV must lie between {E_tab[0]} and {E_tab[-1]} MeV.")

    # Create interpolation functions
    S_mass_of_E = make_log_interp(E_tab, S_mass_tab)
    R_cm_of_E = make_log_interp(E_tab, R_csda_mass_tab / rho_water_g_cm3)

    # CSDA range for reference
    R0_csda_cm = float(R_cm_of_E(E0_MeV))

    if zmax_cm is None:
        zmax_cm = 1.2 * R0_csda_cm

    # Initialize dose scoring grid
    x_edges = np.linspace(-xy_half_width, xy_half_width, nx_bins + 1)
    y_edges = np.linspace(-xy_half_width, xy_half_width, ny_bins + 1)
    z_edges = np.linspace(0, zmax_cm, nz_bins + 1)

    dose_3d = np.zeros((nx_bins, ny_bins, nz_bins))

    # Storage for sample trajectories
    sample_trajectories = []

    # Run Monte Carlo histories
    print(f"Running {n_histories} proton histories...")
    print(f"  Initial energy: {E0_MeV} MeV")
    print(f"  CSDA range: {R0_csda_cm:.3f} cm")
    print(f"  Scattering: {'ON' if use_scattering else 'OFF'}")
    print(f"  Straggling: {'ON' if use_straggling else 'OFF'}")

    all_stop_depths = []

    for i in range(n_histories):
        # Progress indicator
        if (i + 1) % (n_histories // 10) == 0:
            print(f"  Completed {i + 1}/{n_histories} histories...")

        # Simulate single proton
        positions, energies, dose_deps = simulate_single_proton(
            E0_MeV, S_mass_of_E, dz_cm, E_cut_MeV, rho_water_g_cm3,
            use_scattering, use_straggling
        )

        # Record stopping depth
        all_stop_depths.append(positions[-1, 2])

        # Save first few trajectories for visualization
        if i < 20:
            sample_trajectories.append(positions)

        # Score dose to grid
        for j in range(1, len(positions)):
            x, y, z = positions[j]
            dE = dose_deps[j]

            # Find bin indices
            ix = np.searchsorted(x_edges, x) - 1
            iy = np.searchsorted(y_edges, y) - 1
            iz = np.searchsorted(z_edges, z) - 1

            # Check bounds
            if 0 <= ix < nx_bins and 0 <= iy < ny_bins and 0 <= iz < nz_bins:
                dose_3d[ix, iy, iz] += dE

    print("  Done!")

    # Calculate depth-dose curve (central axis)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # Sum dose in central region
    central_ix = nx_bins // 2
    central_iy = ny_bins // 2
    half_width_bins = 3  # Average over central 6x6 bins

    dose_central = np.zeros(nz_bins)
    for ix in range(central_ix - half_width_bins, central_ix + half_width_bins):
        for iy in range(central_iy - half_width_bins, central_iy + half_width_bins):
            if 0 <= ix < nx_bins and 0 <= iy < ny_bins:
                dose_central += dose_3d[ix, iy, :]

    # Normalize
    dose_norm = dose_central / dose_central.max() if dose_central.max() > 0 else dose_central

    # Find peak and statistics
    z_peak_cm = z_centers[np.argmax(dose_central)] if dose_central.max() > 0 else np.nan
    z_stop_mean = np.mean(all_stop_depths)
    z_stop_std = np.std(all_stop_depths)

    return {
        "z_cm": z_centers,
        "dose_3d": dose_3d,
        "dose_central": dose_central,
        "dose_norm": dose_norm,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "R0_csda_cm": R0_csda_cm,
        "z_peak_cm": z_peak_cm,
        "z_stop_mean_cm": z_stop_mean,
        "z_stop_std_cm": z_stop_std,
        "sample_trajectories": sample_trajectories,
        "n_histories": n_histories,
        "use_scattering": use_scattering,
        "use_straggling": use_straggling,
    }


# =============================================================================
# Part 5: Visualization
# =============================================================================

def plot_mc_results(result):
    """
    Plot Monte Carlo simulation results.
    """
    z = result["z_cm"]
    D = result["dose_norm"]
    trajectories = result["sample_trajectories"]

    # Figure 1: Depth-dose curve
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(z, D, 'b-', linewidth=2, label="MC Depth-dose")
    ax1.axvline(result["R0_csda_cm"], color='g', linestyle='--',
                label=f"CSDA range ({result['R0_csda_cm']:.2f} cm)")
    ax1.axvline(result["z_peak_cm"], color='r', linestyle=':',
                label=f"Peak depth ({result['z_peak_cm']:.2f} cm)")
    ax1.axvline(result["z_stop_mean_cm"], color='orange', linestyle='-.',
                label=f"Mean stop ({result['z_stop_mean_cm']:.2f} ± {result['z_stop_std_cm']:.2f} cm)")

    ax1.set_xlabel("Depth in water [cm]", fontsize=12)
    ax1.set_ylabel("Relative dose [a.u.]", fontsize=12)
    ax1.set_title(f"Monte Carlo Bragg Peak ({result['n_histories']} histories)\n"
                  f"Scattering: {'ON' if result['use_scattering'] else 'OFF'}, "
                  f"Straggling: {'ON' if result['use_straggling'] else 'OFF'}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 2: Sample trajectories (3D)
    if len(trajectories) > 0:
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')

        for traj in trajectories[:10]:  # Plot first 10
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.7, linewidth=0.8)

        ax2.set_xlabel("X [cm]")
        ax2.set_ylabel("Y [cm]")
        ax2.set_zlabel("Depth Z [cm]")
        ax2.set_title("Sample Proton Trajectories (first 10)")
        plt.tight_layout()
        plt.show()

    # Figure 3: Trajectories projected onto XZ plane
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for traj in trajectories:
        ax3.plot(traj[:, 2], traj[:, 0], alpha=0.5, linewidth=0.5)

    ax3.axvline(result["R0_csda_cm"], color='g', linestyle='--',
                label="CSDA range", linewidth=2)
    ax3.set_xlabel("Depth Z [cm]", fontsize=12)
    ax3.set_ylabel("Lateral position X [cm]", fontsize=12)
    ax3.set_title("Proton Trajectories (XZ projection)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 4: Lateral dose profile at peak
    dose_3d = result["dose_3d"]
    x_edges = result["x_edges"]
    z_peak_idx = np.argmax(result["dose_central"])

    lateral_profile = dose_3d[:, :, z_peak_idx].sum(axis=1)  # Sum over y
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(x_centers, lateral_profile / lateral_profile.max(), 'b-', linewidth=2)
    ax4.set_xlabel("Lateral position X [cm]", fontsize=12)
    ax4.set_ylabel("Relative dose [a.u.]", fontsize=12)
    ax4.set_title(f"Lateral Dose Profile at Peak (z = {result['z_peak_cm']:.2f} cm)")
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_range_histogram(result):
    """
    Plot histogram of proton stopping depths (range straggling).
    """
    # Re-extract stopping depths from trajectories
    stop_depths = [traj[-1, 2] for traj in result["sample_trajectories"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stop_depths, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(result["R0_csda_cm"], color='g', linestyle='--',
               label=f"CSDA range", linewidth=2)
    ax.axvline(result["z_stop_mean_cm"], color='r', linestyle='-',
               label=f"Mean: {result['z_stop_mean_cm']:.3f} cm", linewidth=2)

    ax.set_xlabel("Stopping depth [cm]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Range Straggling Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Part 6: Main Execution
# =============================================================================

if __name__ == "__main__":
    # Run Monte Carlo simulation
    result = simulate_mc_bragg_peak(
        csv_file=CSV_FILE,
        E0_MeV=E0_MEV,
        dz_cm=DZ_CM,
        zmax_cm=ZMAX_CM,
        E_cut_MeV=E_CUT_MEV,
        rho_water_g_cm3=RHO_WATER_G_CM3,
        n_histories=N_HISTORIES,
        use_scattering=USE_SCATTERING,
        use_straggling=USE_STRAGGLING,
        random_seed=RANDOM_SEED,
    )

    # Print summary
    print("\n" + "="*50)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*50)
    print(f"Number of histories: {result['n_histories']}")
    print(f"CSDA range (reference): {result['R0_csda_cm']:.4f} cm")
    print(f"Mean stopping depth: {result['z_stop_mean_cm']:.4f} cm")
    print(f"Range straggling (σ): {result['z_stop_std_cm']:.4f} cm")
    print(f"Peak depth: {result['z_peak_cm']:.4f} cm")
    print("="*50)

    # Plot results
    plot_mc_results(result)
    plot_range_histogram(result)
