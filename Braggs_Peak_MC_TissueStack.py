# -*- coding: utf-8 -*-
"""
Monte Carlo Bragg Peak Simulation Through Tissue Stack
Merged implementation combining:
- Monte Carlo proton transport (Osman Taka)
- Multi-layer tissue stack modeling (Reichen Schaller)

Features:
- 3D proton tracking (position + direction) through heterogeneous tissue
- Multiple Coulomb Scattering (Highland formula) with material-specific radiation lengths
- Energy straggling (Bohr model) with material-specific parameters
- Multi-history Monte Carlo sampling
- 3D visualization with tissue layer boundaries

@author: Natural Sciences Computing Club, UNC Chapel Hill
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================================================================
# Part 1: Configuration Controls
# =============================================================================
# File paths - relative to script location
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(_SCRIPT_DIR, "Flesh PSTAR Data.csv")    # Multi-material PSTAR data
STACK_FILE = os.path.join(_SCRIPT_DIR, "tissue stack.csv")       # Tissue layer definition

# Simulation parameters
E0_MEV = 100.0                     # Initial proton energy [MeV]
DZ_CM = 0.005                      # Step size [cm] (smaller for tissue accuracy)
ZMAX_CM = None                     # Maximum depth (auto-calculated if None)
E_CUT_MEV = 1.0e-3                 # Energy cutoff [MeV]

# Monte Carlo parameters
N_HISTORIES = 500                  # Number of proton histories to simulate
USE_SCATTERING = True              # Enable multiple Coulomb scattering
USE_STRAGGLING = True              # Enable energy straggling
RANDOM_SEED = 42                   # For reproducibility (set None for random)

# Physics constants
M_PROTON_MEV = 938.272             # Proton rest mass [MeV/c²]

# Dose scoring grid
NX_BINS = 50                       # Number of bins in x
NY_BINS = 50                       # Number of bins in y
NZ_BINS = 200                      # Number of bins in z (depth)
XY_HALF_WIDTH_CM = 1.5             # Half-width of scoring region in x,y [cm]

# =============================================================================
# Part 2: Material Database with Physical Properties
# =============================================================================
# Radiation lengths from NIST/PDG data
# Z/A ratios for Bohr straggling

MATERIALS = {
    "air_dry": {
        "pstar_name": "Air, Dry",
        "density_g_cm3": 0.00120479,
        "X0_cm": 30420.0,           # Radiation length [cm]
        "Z_over_A": 0.499,          # Z/A ratio for straggling
    },
    "water_liquid": {
        "pstar_name": "Water, Liquid",
        "density_g_cm3": 1.00,
        "X0_cm": 36.08,
        "Z_over_A": 0.556,
    },
    "adipose_tissue": {
        "pstar_name": "Adipose Tissue",
        "density_g_cm3": 0.92,
        "X0_cm": 42.2,              # Similar to soft tissue
        "Z_over_A": 0.558,
    },
    "muscle_skeletal": {
        "pstar_name": "Muscle, Skeletal",
        "density_g_cm3": 1.04,
        "X0_cm": 36.5,              # Close to water
        "Z_over_A": 0.550,
    },
    "muscle_striated": {
        "pstar_name": "Muscle, Striated",
        "density_g_cm3": 1.04,
        "X0_cm": 36.5,
        "Z_over_A": 0.550,
    },
    "bone_compact": {
        "pstar_name": "Bone, Compact",
        "density_g_cm3": 1.85,
        "X0_cm": 16.1,              # Denser, shorter radiation length
        "Z_over_A": 0.530,
    },
    "bone_cortical": {
        "pstar_name": "Bone, Cortical",
        "density_g_cm3": 1.85,
        "X0_cm": 16.1,
        "Z_over_A": 0.530,
    },
}

# Colors for tissue visualization
MATERIAL_COLORS = {
    "air_dry": "#E0FFFF",           # Light cyan
    "water_liquid": "#ADD8E6",       # Light blue
    "adipose_tissue": "#FFE4B5",     # Moccasin (fat)
    "muscle_skeletal": "#FFA07A",    # Light salmon
    "muscle_striated": "#FA8072",    # Salmon
    "bone_compact": "#A9A9A9",       # Dark gray
    "bone_cortical": "#808080",      # Gray
}

MATERIAL_COLORS_3D = {
    "air_dry": (0.878, 1.0, 1.0, 0.2),
    "water_liquid": (0.678, 0.847, 0.902, 0.2),
    "adipose_tissue": (1.0, 0.894, 0.710, 0.3),
    "muscle_skeletal": (1.0, 0.627, 0.478, 0.3),
    "muscle_striated": (0.980, 0.502, 0.447, 0.3),
    "bone_compact": (0.663, 0.663, 0.663, 0.4),
    "bone_cortical": (0.502, 0.502, 0.502, 0.4),
}

# =============================================================================
# Part 3: Data Loading Functions
# =============================================================================

def _normalize_material_name(name):
    """Normalize material name for matching."""
    name = "" if pd.isna(name) else str(name)
    name = name.strip().strip('"').lower().replace(",", " ")
    name = " ".join(name.split())
    return name


def _get_material_blocks(raw, block_width=4):
    """
    Find each 4-column material block in the wide PSTAR CSV.
    Each block: Energy | Total Stp. Pow. | CSDA Range | Projected Range
    """
    blocks = {}
    for c0 in range(0, raw.shape[1], block_width):
        pretty_name = "" if pd.isna(raw.iat[0, c0]) else str(raw.iat[0, c0]).strip().strip('"')
        key = _normalize_material_name(pretty_name)
        if key:
            blocks[key] = (pretty_name, c0)
    return blocks


def load_pstar_csv(path, material="Water, Liquid"):
    """
    Load one material from the wide multi-material PSTAR CSV.

    Returns:
        E_MeV: Energy array [MeV]
        S_mass: Mass stopping power [MeV·cm²/g]
        R_csda_mass: CSDA range [g/cm²]
    """
    raw = pd.read_csv(path, header=None, dtype=str)
    raw = raw.apply(lambda col: col.str.strip() if col.dtype == object else col)

    blocks = _get_material_blocks(raw)
    key = _normalize_material_name(material)

    if key not in blocks:
        available = [pretty for pretty, _ in blocks.values()]
        raise ValueError(
            f"Material '{material}' not found in CSV.\n"
            f"Available materials: {available}"
        )

    pretty_name, c0 = blocks[key]
    block = raw.iloc[:, c0:c0 + 4]

    energy = pd.to_numeric(block.iloc[:, 0], errors="coerce")
    stop = pd.to_numeric(block.iloc[:, 1], errors="coerce")
    csda = pd.to_numeric(block.iloc[:, 2], errors="coerce")

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
        raise ValueError(f"Material '{pretty_name}' does not contain enough numeric PSTAR rows.")

    return E_MeV, S_mass, R_csda_mass


def make_log_interp(x, y):
    """Create log-log interpolation function for smooth physics data."""
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
# Part 4: Tissue Stack Loading
# =============================================================================

def load_stack_csv(path, materials_db=MATERIALS):
    """
    Read tissue stack CSV and create ordered layer list.

    Expected CSV columns:
        label, material_key, thickness_cm, density_g_cm3 (optional)
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"material_key", "thickness_cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stack CSV is missing required columns: {sorted(missing)}")

    if "label" not in df.columns:
        df["label"] = [f"layer_{i+1}" for i in range(len(df))]

    if "density_g_cm3" not in df.columns:
        df["density_g_cm3"] = np.nan

    layers = []
    z_cursor = 0.0

    for i, row in df.iterrows():
        label = str(row["label"]).strip()
        material_key = str(row["material_key"]).strip().lower()
        thickness_cm = float(row["thickness_cm"])

        if material_key not in materials_db:
            raise ValueError(
                f"Unknown material_key '{material_key}' on row {i+2}. "
                f"Allowed keys: {sorted(materials_db.keys())}"
            )

        if thickness_cm <= 0.0:
            raise ValueError(f"thickness_cm must be > 0 on row {i+2}")

        if pd.notna(row["density_g_cm3"]) and row["density_g_cm3"] != "":
            density_g_cm3 = float(row["density_g_cm3"])
        else:
            density_g_cm3 = float(materials_db[material_key]["density_g_cm3"])

        layers.append({
            "layer_index": i,
            "label": label,
            "material_key": material_key,
            "pstar_name": materials_db[material_key]["pstar_name"],
            "thickness_cm": thickness_cm,
            "density_g_cm3": density_g_cm3,
            "X0_cm": materials_db[material_key]["X0_cm"],
            "Z_over_A": materials_db[material_key]["Z_over_A"],
            "z_start_cm": z_cursor,
            "z_end_cm": z_cursor + thickness_cm,
        })

        z_cursor += thickness_cm

    return layers


def build_tissue_stack(csv_file, layers, dz_cm, zmax_cm=None):
    """
    Build the tissue stack arrays for Monte Carlo simulation.

    Returns dict with depth-indexed arrays and interpolation functions.
    """
    if dz_cm <= 0.0:
        raise ValueError("dz_cm must be > 0.")

    total_thickness_cm = layers[-1]["z_end_cm"]

    if zmax_cm is None:
        zmax_cm = total_thickness_cm

    zmax_cm = min(zmax_cm, total_thickness_cm)

    # Get unique materials and load PSTAR data
    unique_material_keys = []
    for layer in layers:
        key = layer["material_key"]
        if key not in unique_material_keys:
            unique_material_keys.append(key)

    material_index_map = {key: i for i, key in enumerate(unique_material_keys)}

    # Load stopping power data for each material
    material_data = {}
    for key in unique_material_keys:
        first_layer = next(layer for layer in layers if layer["material_key"] == key)
        pstar_name = first_layer["pstar_name"]

        E_tab, S_mass_tab, R_csda_mass_tab = load_pstar_csv(csv_file, material=pstar_name)

        material_data[key] = {
            "S_mass_interp": make_log_interp(E_tab, S_mass_tab),
            "R_csda_mass_interp": make_log_interp(E_tab, R_csda_mass_tab),
            "E_min": E_tab[0],
            "E_max": E_tab[-1],
        }

    # Create depth arrays
    z_cm = np.arange(0.0, zmax_cm + dz_cm, dz_cm)
    n_points = len(z_cm)

    # Arrays to hold material properties at each depth
    rho_g_cm3 = np.zeros(n_points)
    X0_cm = np.zeros(n_points)
    Z_over_A = np.zeros(n_points)
    material_index = np.full(n_points, -1, dtype=int)
    layer_index = np.full(n_points, -1, dtype=int)
    material_key_arr = np.empty(n_points, dtype=object)
    label_arr = np.empty(n_points, dtype=object)

    # Fill arrays based on layer boundaries
    for j, layer in enumerate(layers):
        z0 = layer["z_start_cm"]
        z1 = min(layer["z_end_cm"], zmax_cm)

        if z1 < z0:
            continue

        is_last = (j == len(layers) - 1) or (layer["z_end_cm"] >= zmax_cm)

        if is_last:
            mask = (z_cm >= z0) & (z_cm <= z1)
        else:
            mask = (z_cm >= z0) & (z_cm < z1)

        key = layer["material_key"]
        m = material_index_map[key]

        rho_g_cm3[mask] = layer["density_g_cm3"]
        X0_cm[mask] = layer["X0_cm"]
        Z_over_A[mask] = layer["Z_over_A"]
        material_index[mask] = m
        layer_index[mask] = j
        material_key_arr[mask] = key
        label_arr[mask] = layer["label"]

    return {
        "z_cm": z_cm,
        "rho_g_cm3": rho_g_cm3,
        "X0_cm": X0_cm,
        "Z_over_A": Z_over_A,
        "material_index": material_index,
        "layer_index": layer_index,
        "material_key": material_key_arr,
        "label": label_arr,
        "material_data": material_data,
        "unique_material_keys": unique_material_keys,
        "total_thickness_cm": total_thickness_cm,
        "layers": layers,
    }


# =============================================================================
# Part 5: Monte Carlo Physics Functions
# =============================================================================

def get_relativistic_params(E_MeV):
    """
    Calculate relativistic parameters for a proton.

    Returns:
        beta: v/c (velocity as fraction of speed of light)
        gamma: Lorentz factor
        p_MeV: Momentum [MeV/c]
    """
    gamma = 1.0 + E_MeV / M_PROTON_MEV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_MeV = M_PROTON_MEV * gamma * beta
    return beta, gamma, p_MeV


def highland_theta_rms(E_MeV, dz_cm, X0_cm):
    """
    Calculate RMS scattering angle using Highland formula.
    Material-specific radiation length X0_cm.

    θ_rms = (14.1 MeV / pv) × sqrt(x/X₀) × [1 + (1/9)·log₁₀(x/X₀)]
    """
    if E_MeV <= 0 or dz_cm <= 0:
        return 0.0

    beta, gamma, p_MeV = get_relativistic_params(E_MeV)

    x_over_X0 = dz_cm / X0_cm

    log_term = 1.0 + np.log10(x_over_X0) / 9.0
    theta_rms = (14.1 / (p_MeV * beta)) * np.sqrt(x_over_X0) * log_term

    return max(theta_rms, 0.0)


def sample_scattering_angle(E_MeV, dz_cm, X0_cm):
    """
    Sample scattering angle from multiple scattering distribution.
    Uses Gaussian approximation to Moliere theory.
    """
    theta_rms = highland_theta_rms(E_MeV, dz_cm, X0_cm)

    theta_x = np.random.normal(0, theta_rms)
    theta_y = np.random.normal(0, theta_rms)

    theta = np.sqrt(theta_x**2 + theta_y**2)
    phi = np.arctan2(theta_y, theta_x)

    return theta, phi


def bohr_straggling_sigma(dz_cm, rho_g_cm3, Z_over_A):
    """
    Calculate energy straggling sigma using Bohr's theory.
    Material-specific Z/A ratio.

    σ² ≈ 0.157 × (Z/A) × ρ × Δx  [MeV²]
    """
    K_straggle = 0.157  # MeV²·cm²/g
    sigma_squared = K_straggle * Z_over_A * rho_g_cm3 * dz_cm
    return np.sqrt(sigma_squared)


def sample_energy_loss(E_MeV, S_lin_MeV_cm, dz_cm, rho_g_cm3, Z_over_A, use_straggling=True):
    """
    Sample energy loss with optional Bohr straggling.
    """
    dE_mean = S_lin_MeV_cm * dz_cm

    if not use_straggling:
        return min(dE_mean, E_MeV)

    sigma_E = bohr_straggling_sigma(dz_cm, rho_g_cm3, Z_over_A)
    dE = np.random.normal(dE_mean, sigma_E)

    dE = max(0.0, min(dE, E_MeV))
    return dE


def rotate_direction(direction, theta, phi):
    """
    Rotate direction vector by polar angle theta and azimuthal angle phi.
    """
    dx, dy, dz = direction

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    if abs(dz) > 0.99999:
        sign_z = np.sign(dz) if dz != 0 else 1.0
        new_dx = sin_theta * cos_phi
        new_dy = sin_theta * sin_phi * sign_z
        new_dz = cos_theta * sign_z
    else:
        dz_perp = np.sqrt(1.0 - dz**2)
        new_dx = (dx * cos_theta +
                  (dx * dz * cos_phi - dy * sin_phi) * sin_theta / dz_perp)
        new_dy = (dy * cos_theta +
                  (dy * dz * cos_phi + dx * sin_phi) * sin_theta / dz_perp)
        new_dz = dz * cos_theta - dz_perp * sin_theta * cos_phi

    norm = np.sqrt(new_dx**2 + new_dy**2 + new_dz**2)
    return np.array([new_dx / norm, new_dy / norm, new_dz / norm])


# =============================================================================
# Part 6: Monte Carlo Simulation Engine
# =============================================================================

def get_material_at_depth(z, tissue_stack):
    """
    Get material properties at a given depth z.
    Returns tuple: (rho, X0, Z_over_A, material_key, S_mass_interp)
    """
    z_arr = tissue_stack["z_cm"]

    if z < 0:
        z = 0
    if z > z_arr[-1]:
        z = z_arr[-1]

    idx = np.searchsorted(z_arr, z)
    if idx >= len(z_arr):
        idx = len(z_arr) - 1

    rho = tissue_stack["rho_g_cm3"][idx]
    X0 = tissue_stack["X0_cm"][idx]
    Z_A = tissue_stack["Z_over_A"][idx]
    mat_key = tissue_stack["material_key"][idx]
    S_mass_interp = tissue_stack["material_data"][mat_key]["S_mass_interp"]

    return rho, X0, Z_A, mat_key, S_mass_interp


def simulate_single_proton_tissue(E0_MeV, tissue_stack, dz_cm, E_cut_MeV,
                                   use_scattering=True, use_straggling=True):
    """
    Simulate a single proton history through the tissue stack.

    Returns:
        positions: Array of (x, y, z) positions [cm]
        energies: Array of energies at each position [MeV]
        dose_deposits: Array of energy deposited at each step [MeV]
        materials_visited: List of material keys at each step
    """
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    energy = E0_MeV

    positions = [position.copy()]
    energies = [energy]
    dose_deposits = [0.0]
    materials_visited = ["entry"]

    total_thickness = tissue_stack["total_thickness_cm"]

    while energy > E_cut_MeV:
        z = position[2]

        # Stop if we've exited the tissue stack
        if z < 0 or z > total_thickness:
            break

        # Get material properties at current depth
        rho, X0, Z_A, mat_key, S_mass_interp = get_material_at_depth(z, tissue_stack)

        # Get stopping power at current energy
        S_lin = rho * float(S_mass_interp(energy))

        # Sample energy loss
        dE = sample_energy_loss(energy, S_lin, dz_cm, rho, Z_A, use_straggling)

        # Sample scattering angle
        if use_scattering:
            theta, phi = sample_scattering_angle(energy, dz_cm, X0)
            direction = rotate_direction(direction, theta, phi)

        # Move proton
        position = position + direction * dz_cm
        energy = energy - dE

        # Record state
        positions.append(position.copy())
        energies.append(max(energy, 0.0))
        dose_deposits.append(dE)
        materials_visited.append(mat_key)

        # Safety check
        if len(positions) > 100000:
            break

    return (np.array(positions), np.array(energies),
            np.array(dose_deposits), materials_visited)


def simulate_mc_tissue_stack(
    csv_file,
    stack_file,
    E0_MeV,
    dz_cm,
    zmax_cm,
    E_cut_MeV,
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
    Run full Monte Carlo Bragg peak simulation through tissue stack.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load tissue stack
    print("Loading tissue stack...")
    layers = load_stack_csv(stack_file)
    tissue_stack = build_tissue_stack(csv_file, layers, dz_cm, zmax_cm)

    total_thickness = tissue_stack["total_thickness_cm"]
    if zmax_cm is None:
        zmax_cm = total_thickness

    # Print tissue stack info
    print("\nTissue Stack Configuration:")
    print("-" * 60)
    for layer in layers:
        print(f"  {layer['label']:15s} | {layer['pstar_name']:20s} | "
              f"{layer['thickness_cm']:.2f} cm | rho={layer['density_g_cm3']:.2f} g/cm³")
    print(f"  {'Total':15s} | {total_thickness:.2f} cm")
    print("-" * 60)

    # Initialize dose scoring grid
    x_edges = np.linspace(-xy_half_width, xy_half_width, nx_bins + 1)
    y_edges = np.linspace(-xy_half_width, xy_half_width, ny_bins + 1)
    z_edges = np.linspace(0, zmax_cm, nz_bins + 1)

    dose_3d = np.zeros((nx_bins, ny_bins, nz_bins))

    sample_trajectories = []
    all_stop_depths = []

    print(f"\nRunning {n_histories} proton histories...")
    print(f"  Initial energy: {E0_MeV} MeV")
    print(f"  Scattering: {'ON' if use_scattering else 'OFF'}")
    print(f"  Straggling: {'ON' if use_straggling else 'OFF'}")

    for i in range(n_histories):
        if n_histories >= 10 and (i + 1) % (n_histories // 10) == 0:
            print(f"  Completed {i + 1}/{n_histories} histories...")

        positions, energies, dose_deps, mats = simulate_single_proton_tissue(
            E0_MeV, tissue_stack, dz_cm, E_cut_MeV,
            use_scattering, use_straggling
        )

        all_stop_depths.append(positions[-1, 2])

        if i < 30:
            sample_trajectories.append({
                "positions": positions,
                "energies": energies,
                "materials": mats,
            })

        # Score dose to grid
        for j in range(1, len(positions)):
            x, y, z = positions[j]
            dE = dose_deps[j]

            ix = np.searchsorted(x_edges, x) - 1
            iy = np.searchsorted(y_edges, y) - 1
            iz = np.searchsorted(z_edges, z) - 1

            if 0 <= ix < nx_bins and 0 <= iy < ny_bins and 0 <= iz < nz_bins:
                dose_3d[ix, iy, iz] += dE

    print("  Done!")

    # Calculate depth-dose curve
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    central_ix = nx_bins // 2
    central_iy = ny_bins // 2
    half_width_bins = 3

    dose_central = np.zeros(nz_bins)
    for ix in range(central_ix - half_width_bins, central_ix + half_width_bins):
        for iy in range(central_iy - half_width_bins, central_iy + half_width_bins):
            if 0 <= ix < nx_bins and 0 <= iy < ny_bins:
                dose_central += dose_3d[ix, iy, :]

    dose_norm = dose_central / dose_central.max() if dose_central.max() > 0 else dose_central

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
        "z_peak_cm": z_peak_cm,
        "z_stop_mean_cm": z_stop_mean,
        "z_stop_std_cm": z_stop_std,
        "sample_trajectories": sample_trajectories,
        "n_histories": n_histories,
        "use_scattering": use_scattering,
        "use_straggling": use_straggling,
        "layers": layers,
        "tissue_stack": tissue_stack,
        "all_stop_depths": all_stop_depths,
    }


# =============================================================================
# Part 7: Visualization Functions
# =============================================================================

def plot_depth_dose_with_layers(result):
    """Plot depth-dose curve with tissue layer backgrounds."""
    z = result["z_cm"]
    D = result["dose_norm"]
    layers = result["layers"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw layer backgrounds
    material_patches = []
    seen_materials = set()

    for layer in layers:
        x0 = layer["z_start_cm"]
        x1 = layer["z_end_cm"]
        key = layer["material_key"]
        color = MATERIAL_COLORS.get(key, "whitesmoke")

        ax.axvspan(x0, x1, facecolor=color, alpha=0.3, zorder=0)

        if key not in seen_materials:
            material_patches.append(Patch(facecolor=color, alpha=0.3,
                                          label=layer["pstar_name"]))
            seen_materials.add(key)

    # Plot depth-dose curve
    ax.plot(z, D, 'b-', linewidth=2, label="MC Depth-dose", zorder=3)
    ax.axvline(result["z_peak_cm"], color='r', linestyle=':', linewidth=2,
               label=f"Peak ({result['z_peak_cm']:.2f} cm)", zorder=4)
    ax.axvline(result["z_stop_mean_cm"], color='orange', linestyle='--', linewidth=2,
               label=f"Mean stop ({result['z_stop_mean_cm']:.2f} cm)", zorder=4)

    ax.set_xlabel("Depth [cm]", fontsize=12)
    ax.set_ylabel("Relative Dose [a.u.]", fontsize=12)
    ax.set_title(f"Monte Carlo Bragg Peak Through Tissue Stack\n"
                 f"({result['n_histories']} histories, "
                 f"Scattering: {'ON' if result['use_scattering'] else 'OFF'}, "
                 f"Straggling: {'ON' if result['use_straggling'] else 'OFF'})")

    # Legends
    legend1 = ax.legend(loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=material_patches, title="Tissues", loc="lower right")

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, layers[-1]["z_end_cm"])
    plt.tight_layout()
    plt.show()


def plot_3d_trajectories_with_layers(result, n_trajectories=15):
    """
    Plot 3D proton trajectories with tissue layer visualization.
    """
    trajectories = result["sample_trajectories"][:n_trajectories]
    layers = result["layers"]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get bounds
    xy_half = result["x_edges"][-1]
    z_max = layers[-1]["z_end_cm"]

    # Draw tissue layers as semi-transparent boxes
    for layer in layers:
        z0 = layer["z_start_cm"]
        z1 = layer["z_end_cm"]
        key = layer["material_key"]
        color = MATERIAL_COLORS_3D.get(key, (0.9, 0.9, 0.9, 0.2))

        # Create vertices for box
        vertices = [
            [[-xy_half, -xy_half, z0], [xy_half, -xy_half, z0],
             [xy_half, xy_half, z0], [-xy_half, xy_half, z0]],  # bottom
            [[-xy_half, -xy_half, z1], [xy_half, -xy_half, z1],
             [xy_half, xy_half, z1], [-xy_half, xy_half, z1]],  # top
        ]

        # Draw only top and bottom faces (less cluttered)
        for face in vertices:
            poly = Poly3DCollection([face], alpha=color[3],
                                     facecolor=color[:3], edgecolor='gray', linewidth=0.5)
            ax.add_collection3d(poly)

    # Plot trajectories with color gradient based on energy
    for traj_data in trajectories:
        positions = traj_data["positions"]
        energies = traj_data["energies"]

        # Normalize energies for color mapping
        e_norm = energies / energies[0] if energies[0] > 0 else energies

        # Plot line segments with energy-based colors
        for i in range(len(positions) - 1):
            color_val = plt.cm.plasma(e_norm[i])
            ax.plot3D(
                [positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]],
                [positions[i, 2], positions[i+1, 2]],
                color=color_val, linewidth=0.8, alpha=0.7
            )

    # Mark stopping points
    for traj_data in trajectories:
        pos = traj_data["positions"][-1]
        ax.scatter(pos[0], pos[1], pos[2], c='red', s=20, marker='x', alpha=0.8)

    ax.set_xlabel("X [cm]", fontsize=10)
    ax.set_ylabel("Y [cm]", fontsize=10)
    ax.set_zlabel("Depth Z [cm]", fontsize=10)
    ax.set_title(f"Proton Trajectories Through Tissue Stack\n"
                 f"(Color: energy, red X: stopping point)")

    ax.set_xlim(-xy_half, xy_half)
    ax.set_ylim(-xy_half, xy_half)
    ax.set_zlim(0, z_max)

    # Add layer labels
    for layer in layers:
        z_mid = (layer["z_start_cm"] + layer["z_end_cm"]) / 2
        ax.text(xy_half * 1.1, 0, z_mid, layer["label"], fontsize=8, ha='left')

    plt.tight_layout()
    plt.show()


def plot_xz_projection_with_layers(result):
    """Plot XZ projection of trajectories with tissue layers."""
    trajectories = result["sample_trajectories"]
    layers = result["layers"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw layer backgrounds
    y_min, y_max = -2, 2  # Lateral extent for display

    for layer in layers:
        z0 = layer["z_start_cm"]
        z1 = layer["z_end_cm"]
        key = layer["material_key"]
        color = MATERIAL_COLORS.get(key, "whitesmoke")
        ax.axvspan(z0, z1, facecolor=color, alpha=0.25, zorder=0)

    # Plot trajectories
    for traj_data in trajectories:
        positions = traj_data["positions"]
        ax.plot(positions[:, 2], positions[:, 0], alpha=0.5, linewidth=0.6, zorder=1)

    # Mark layer boundaries
    for layer in layers:
        ax.axvline(layer["z_end_cm"], color='gray', linestyle=':', alpha=0.5, linewidth=0.5)

    ax.axvline(result["z_stop_mean_cm"], color='orange', linestyle='--', linewidth=2,
               label=f"Mean stop: {result['z_stop_mean_cm']:.2f} cm")

    ax.set_xlabel("Depth Z [cm]", fontsize=12)
    ax.set_ylabel("Lateral position X [cm]", fontsize=12)
    ax.set_title("Proton Trajectories (XZ Projection) Through Tissue Stack")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, layers[-1]["z_end_cm"])

    plt.tight_layout()
    plt.show()


def plot_lateral_profile_at_depths(result):
    """Plot lateral dose profiles at different depths."""
    dose_3d = result["dose_3d"]
    x_edges = result["x_edges"]
    z_cm = result["z_cm"]
    layers = result["layers"]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Select depths at middle of each layer
    fig, ax = plt.subplots(figsize=(10, 6))

    depths_to_plot = []
    for layer in layers:
        z_mid = (layer["z_start_cm"] + layer["z_end_cm"]) / 2
        iz = np.argmin(np.abs(z_cm - z_mid))
        depths_to_plot.append((layer["label"], iz, z_mid))

    for label, iz, z_mid in depths_to_plot:
        if iz < dose_3d.shape[2]:
            lateral = dose_3d[:, :, iz].sum(axis=1)
            if lateral.max() > 0:
                lateral_norm = lateral / lateral.max()
                ax.plot(x_centers, lateral_norm, label=f"{label} (z={z_mid:.1f} cm)")

    ax.set_xlabel("Lateral Position X [cm]", fontsize=12)
    ax.set_ylabel("Relative Dose [a.u.]", fontsize=12)
    ax.set_title("Lateral Dose Profiles in Each Tissue Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_range_straggling(result):
    """Plot histogram of stopping depths (range straggling)."""
    stop_depths = result["all_stop_depths"]
    layers = result["layers"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw layer backgrounds
    for layer in layers:
        z0 = layer["z_start_cm"]
        z1 = layer["z_end_cm"]
        key = layer["material_key"]
        color = MATERIAL_COLORS.get(key, "whitesmoke")
        ax.axvspan(z0, z1, facecolor=color, alpha=0.25, zorder=0)

    # Histogram
    ax.hist(stop_depths, bins=30, edgecolor='black', alpha=0.7, zorder=2)
    ax.axvline(result["z_stop_mean_cm"], color='r', linestyle='-', linewidth=2,
               label=f"Mean: {result['z_stop_mean_cm']:.3f} cm")
    ax.axvline(result["z_stop_mean_cm"] - result["z_stop_std_cm"],
               color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(result["z_stop_mean_cm"] + result["z_stop_std_cm"],
               color='r', linestyle='--', linewidth=1, alpha=0.7,
               label=f"σ: {result['z_stop_std_cm']:.3f} cm")

    ax.set_xlabel("Stopping Depth [cm]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Range Straggling Distribution Through Tissue Stack")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_dose_heatmap(result):
    """Plot 2D dose heatmap (XZ plane through center)."""
    dose_3d = result["dose_3d"]
    x_edges = result["x_edges"]
    z_edges = result["z_edges"]
    layers = result["layers"]

    # Sum over Y to get XZ projection
    dose_xz = dose_3d.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot dose heatmap
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    im = ax.pcolormesh(z_centers, x_centers, dose_xz, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax, label='Dose [a.u.]')

    # Add layer boundaries
    for layer in layers:
        ax.axvline(layer["z_end_cm"], color='white', linestyle='--', alpha=0.5, linewidth=1)
        z_mid = (layer["z_start_cm"] + layer["z_end_cm"]) / 2
        ax.text(z_mid, x_centers[-1] * 0.9, layer["label"],
                color='white', fontsize=8, ha='center', va='top')

    ax.set_xlabel("Depth Z [cm]", fontsize=12)
    ax.set_ylabel("Lateral X [cm]", fontsize=12)
    ax.set_title("Dose Distribution (XZ Plane)")
    ax.set_xlim(0, layers[-1]["z_end_cm"])

    plt.tight_layout()
    plt.show()


# =============================================================================
# Part 8: Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MONTE CARLO PROTON SCATTERING THROUGH TISSUE STACK")
    print("Natural Sciences Computing Club, UNC Chapel Hill")
    print("=" * 70)

    # Run simulation
    result = simulate_mc_tissue_stack(
        csv_file=CSV_FILE,
        stack_file=STACK_FILE,
        E0_MeV=E0_MEV,
        dz_cm=DZ_CM,
        zmax_cm=ZMAX_CM,
        E_cut_MeV=E_CUT_MEV,
        n_histories=N_HISTORIES,
        use_scattering=USE_SCATTERING,
        use_straggling=USE_STRAGGLING,
        random_seed=RANDOM_SEED,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"Number of histories:    {result['n_histories']}")
    print(f"Mean stopping depth:    {result['z_stop_mean_cm']:.4f} cm")
    print(f"Range straggling (σ):   {result['z_stop_std_cm']:.4f} cm")
    print(f"Peak dose depth:        {result['z_peak_cm']:.4f} cm")
    print("=" * 70)

    # Generate all visualizations
    print("\nGenerating visualizations...")

    plot_depth_dose_with_layers(result)
    plot_3d_trajectories_with_layers(result)
    plot_xz_projection_with_layers(result)
    plot_lateral_profile_at_depths(result)
    plot_range_straggling(result)
    plot_dose_heatmap(result)

    print("\nSimulation complete!")
