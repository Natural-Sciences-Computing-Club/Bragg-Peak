# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:08:33 2026

@author: reichen schaller
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Part 1: controls
CSV_FILE = "C:/Users/reich/Downloads/Flesh PSTAR Data.csv"
STACK = "C:/Users/reich/Downloads/tissue stack.csv"
E0_MEV = 100.0 #initial energy
DZ_CM = 0.001 #step size
ZMAX_CM = None
PHI0 = 1.0 #more attenutation stuff idk we kinda assuming stuff here
LAMBDA_PER_CM = 0.0 #attenuation constant
SIGMA_CM = 0.03 #smoothing factor
E_CUT_MEV = 1.0e-3 #cutoff for when the energy is too low
PLOT_IDEAL = True #where to show an idea curve or not

MATERIALS = {
    "air_dry": {
        "pstar_name": "Air, Dry",
        "density_g_cm3": 0.00120479,
    },
    "water_liquid": {
        "pstar_name": "Water, Liquid",
        "density_g_cm3": 1.00,
    },
    "adipose_tissue": {
        "pstar_name": "Adipose Tissue",
        "density_g_cm3": 0.92,
    },
    "muscle_skeletal": {
        "pstar_name": "Muscle, Skeletal",
        "density_g_cm3": 1.04,
    },
    "muscle_striated": {
        "pstar_name": "Muscle, Striated",
        "density_g_cm3": 1.04,
    },
    "bone_compact": {
        "pstar_name": "Bone, Compact",
        "density_g_cm3": 1.85,
    },
    "bone_cortical": {
        "pstar_name": "Bone, Cortical",
        "density_g_cm3": 1.85,
    },
}

MATERIAL_COLORS = {
    "air_dry": "lightcyan",
    "water_liquid": "lightblue",
    "adipose_tissue": "plum",
    "muscle_skeletal": "bisque",
    "muscle_striated": "salmon",
    "bone_compact": "lightgray",
    "bone_cortical": "silver",
}
####################################################################################################
# Part 2: data / thing to read the csv file

def _normalize_material_name(name):
    name = "" if pd.isna(name) else str(name)
    name = name.strip().strip('"').lower().replace(",", " ")
    name = " ".join(name.split())
    return name


def _get_material_blocks(raw, block_width=4):
    """
    Finds each 4-column material block in the wide PSTAR CSV.
    Each block is:
        Energy | Total Stp. Pow. | CSDA Range | Projected Range
    """
    blocks = {}

    for c0 in range(0, raw.shape[1], block_width):
        pretty_name = "" if pd.isna(raw.iat[0, c0]) else str(raw.iat[0, c0]).strip().strip('"')
        key = _normalize_material_name(pretty_name)

        if key:
            blocks[key] = (pretty_name, c0)

    return blocks


def load_pstar_csv(path, material="Water, Liquid", return_projected=False):
    """
    Load one material from the wide multi-material PSTAR CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    material : str, optional
        Material name to load. Examples:
        "Water, Liquid"
        "Adipose Tissue"
        "Muscle, Skeletal"
        "Muscle, Striated"
        "Bone, Cortical"
        "Bone, Compact"
        "Air, Dry"
    return_projected : bool, optional
        If True, also return projected range.

    Returns
    -------
    If return_projected is False:
        E_MeV, S_mass, R_csda_mass

    If return_projected is True:
        E_MeV, S_mass, R_csda_mass, R_proj_mass
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
    proj = pd.to_numeric(block.iloc[:, 3], errors="coerce")

    mask = energy.notna() & stop.notna() & csda.notna()
    if return_projected:
        mask = mask & proj.notna()

    E_MeV = energy[mask].to_numpy(dtype=float)
    S_mass = stop[mask].to_numpy(dtype=float)
    R_csda_mass = csda[mask].to_numpy(dtype=float)

    order = np.argsort(E_MeV)
    E_MeV = E_MeV[order]
    S_mass = S_mass[order]
    R_csda_mass = R_csda_mass[order]

    if return_projected:
        R_proj_mass = proj[mask].to_numpy(dtype=float)
        R_proj_mass = R_proj_mass[order]

    keep = np.concatenate(([True], np.diff(E_MeV) > 0))
    E_MeV = E_MeV[keep]
    S_mass = S_mass[keep]
    R_csda_mass = R_csda_mass[keep]

    if return_projected:
        R_proj_mass = R_proj_mass[keep]

    if len(E_MeV) < 5:
        raise ValueError(f"Material '{pretty_name}' does not contain enough numeric PSTAR rows.")

    if return_projected:
        return E_MeV, S_mass, R_csda_mass, R_proj_mass

    return E_MeV, S_mass, R_csda_mass


def make_log_interp(x, y):  # do a log log interpolation since data range is big and curves a lot
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
####################################################################################################
# Part 2.5: read the tissue stack csv


def load_stack_csv(path, materials_db=MATERIALS):
    """
    Reads the stack CSV and turns it into a clean ordered layer list.

    Expected CSV columns:
        label, material_key, thickness_cm, density_g_cm3

    density_g_cm3 can be left blank and it will use the default value
    from MATERIALS.
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

        if pd.notna(row["density_g_cm3"]):
            density_g_cm3 = float(row["density_g_cm3"])
        else:
            density_g_cm3 = float(materials_db[material_key]["density_g_cm3"])

        if density_g_cm3 <= 0.0:
            raise ValueError(f"density_g_cm3 must be > 0 on row {i+2}")

        z_start_cm = z_cursor
        z_end_cm = z_cursor + thickness_cm

        layers.append({
            "layer_index": i,
            "label": label,
            "material_key": material_key,
            "pstar_name": materials_db[material_key]["pstar_name"],
            "thickness_cm": thickness_cm,
            "density_g_cm3": density_g_cm3,
            "z_start_cm": z_start_cm,
            "z_end_cm": z_end_cm,
        })

        z_cursor = z_end_cm

    if len(layers) == 0:
        raise ValueError("Stack CSV has no layers.")

    return layers

####################################################################################################
# Part 2.8: turn the stack into arrays that Part 3 can use

def build_stack_array(csv_file, layers, dz_cm, zmax_cm=None):
    """
    Preloads the PSTAR data for each material in the stack and builds
    depth-indexed arrays for the simulation loop.

    Returns a dict with:
        z_cm
        rho_g_cm3
        material_index
        layer_index
        label
        material_key
        pstar_name

        material_key_by_index
        pstar_name_by_index
        S_mass_interp_by_index
        R_csda_mass_interp_by_index
    """
    if dz_cm <= 0.0:
        raise ValueError("dz_cm must be > 0.")

    total_thickness_cm = layers[-1]["z_end_cm"]

    if zmax_cm is None:
        zmax_cm = total_thickness_cm

    if zmax_cm <= 0.0:
        raise ValueError("zmax_cm must be > 0.")

    # keep only the part of the stack actually inside the requested depth
    zmax_cm = min(zmax_cm, total_thickness_cm)

    # unique materials in the order they first appear
    unique_material_keys = []
    for layer in layers:
        key = layer["material_key"]
        if key not in unique_material_keys:
            unique_material_keys.append(key)

    material_index_map = {key: i for i, key in enumerate(unique_material_keys)}

    material_key_by_index = []
    pstar_name_by_index = []
    S_mass_interp_by_index = []
    R_csda_mass_interp_by_index = []

    for key in unique_material_keys:
        # use the first layer with this key to get its PSTAR name
        first_layer = next(layer for layer in layers if layer["material_key"] == key)
        pstar_name = first_layer["pstar_name"]

        E_tab, S_mass_tab, R_csda_mass_tab = load_pstar_csv(
            csv_file,
            material=pstar_name,
            return_projected=False,
        )

        material_key_by_index.append(key)
        pstar_name_by_index.append(pstar_name)
        S_mass_interp_by_index.append(make_log_interp(E_tab, S_mass_tab))
        R_csda_mass_interp_by_index.append(make_log_interp(E_tab, R_csda_mass_tab))

    z_cm = np.arange(0.0, zmax_cm + dz_cm, dz_cm)

    rho_g_cm3 = np.full(z_cm.shape, np.nan, dtype=float)
    material_index = np.full(z_cm.shape, -1, dtype=int)
    layer_index = np.full(z_cm.shape, -1, dtype=int)

    label = np.empty(z_cm.shape, dtype=object)
    material_key = np.empty(z_cm.shape, dtype=object)
    pstar_name = np.empty(z_cm.shape, dtype=object)

    label[:] = None
    material_key[:] = None
    pstar_name[:] = None

    n_layers = len(layers)

    for j, layer in enumerate(layers):
        z0 = layer["z_start_cm"]
        z1 = min(layer["z_end_cm"], zmax_cm)

        if z1 < z0:
            continue

        # include the right edge only for the last active layer
        is_last_active_layer = (j == n_layers - 1) or (layer["z_end_cm"] >= zmax_cm)

        if is_last_active_layer:
            mask = (z_cm >= z0) & (z_cm <= z1)
        else:
            mask = (z_cm >= z0) & (z_cm < z1)

        m = material_index_map[layer["material_key"]]

        rho_g_cm3[mask] = layer["density_g_cm3"]
        material_index[mask] = m
        layer_index[mask] = layer["layer_index"]
        label[mask] = layer["label"]
        material_key[mask] = layer["material_key"]
        pstar_name[mask] = layer["pstar_name"]

    if np.any(material_index < 0):
        bad_depths = z_cm[material_index < 0]
        raise ValueError(
            "Some depth points were not assigned a material. "
            f"First bad depth = {bad_depths[0]:.6f} cm"
        )

    return {
        "z_cm": z_cm,
        "rho_g_cm3": rho_g_cm3,
        "material_index": material_index,
        "layer_index": layer_index,
        "label": label,
        "material_key": material_key,
        "pstar_name": pstar_name,
        "material_key_by_index": material_key_by_index,
        "pstar_name_by_index": pstar_name_by_index,
        "S_mass_interp_by_index": S_mass_interp_by_index,
        "R_csda_mass_interp_by_index": R_csda_mass_interp_by_index,
        "total_thickness_cm": total_thickness_cm,
    }

####################################################################################################
# Part 3: physics
#blur the peak with a guassian to not make it too idealized
#this is one of the bigger assumptions since our whole system lacks any noise
def gaussian_kernel(dz_cm, sigma_cm, n_sigma=6): #6 is the guassian deviation cutoff 
    if sigma_cm <= 0.0: #ie if no blur do nothing
        return np.array([1.0])

    half_width = int(np.ceil(n_sigma * sigma_cm / dz_cm))
    offsets = np.arange(-half_width, half_width + 1) * dz_cm

    kernel = np.exp(-0.5 * (offsets / sigma_cm) ** 2)
    kernel /= kernel.sum() * dz_cm

    return kernel

# the real meat
def simulate_bragg_peak(
    stack_array,
    E0_MeV,
    dz_cm,
    phi0,
    lambda_per_cm,
    sigma_cm,
    E_cut_MeV,
):
    z_cm = stack_array["z_cm"]

    E_MeV = np.zeros_like(z_cm)
    S_lin_MeV_per_cm = np.zeros_like(z_cm)
    fluence = phi0 * np.exp(-lambda_per_cm * z_cm)
    D_ideal = np.zeros_like(z_cm)

    E_MeV[0] = E0_MeV

    for i in range(len(z_cm) - 1): #iterate through depth points
        Ei = E_MeV[i]

        if Ei <= E_cut_MeV:
            break

        m = stack_array["material_index"][i]
        rho_i = stack_array["rho_g_cm3"][i]
        S_mass_of_E = stack_array["S_mass_interp_by_index"][m]
        S_lin_i = rho_i * float(S_mass_of_E(Ei)) #finds stopping power at i depth
        S_lin_MeV_per_cm[i] = S_lin_i
        D_ideal[i] = fluence[i] * S_lin_i

        E_next = Ei - S_lin_i * dz_cm # finds new energy after stopping power decreases it
        E_MeV[i + 1] = max(E_next, 0.0)

    live = np.where(E_MeV > E_cut_MeV)[0] #where we are above the cutoff
    if len(live) > 0:
        j = live[-1]
        m = stack_array["material_index"][j]
        rho_j = stack_array["rho_g_cm3"][j]
        S_mass_of_E = stack_array["S_mass_interp_by_index"][m]

        S_lin_MeV_per_cm[j] = rho_j * float(S_mass_of_E(E_MeV[j]))
        D_ideal[j] = fluence[j] * S_lin_MeV_per_cm[j]

    kernel = gaussian_kernel(dz_cm, sigma_cm)
    D_broad = np.convolve(D_ideal, kernel, mode="same") * dz_cm
    D_norm = D_broad / D_broad.max() if D_broad.max() > 0 else D_broad.copy()
#smears and normalizes the curve
    stopped = np.where(E_MeV <= E_cut_MeV)[0] #finds where the energy hits cutoff
    z_stop_cm = z_cm[stopped[0]] if len(stopped) > 0 else np.nan
    z_peak_cm = z_cm[np.argmax(D_broad)] if D_broad.max() > 0 else np.nan
 #finds  value for stop and peak
    return {
        "z_cm": z_cm,
        "E_MeV": E_MeV,
        "S_lin_MeV_per_cm": S_lin_MeV_per_cm,
        "fluence": fluence,
        "D_ideal": D_ideal,
        "D_broad": D_broad,
        "D_norm": D_norm,
        "z_stop_cm": z_stop_cm,
        "z_peak_cm": z_peak_cm,
    }


# Part 4: plots and run
def plot_result(result, layers, plot_ideal=True):
    z = result["z_cm"]
    E = result["E_MeV"]
    D0 = result["D_ideal"]
    D = result["D_norm"]

    # dose plot
    fig, ax = plt.subplots(figsize=(9, 5))

    material_patches = []
    seen_materials = set()

    for layer in layers:
        x0 = layer["z_start_cm"]
        x1 = layer["z_end_cm"]
        key = layer["material_key"]
        color = MATERIAL_COLORS.get(key, "whitesmoke")

        ax.axvspan(x0, x1, facecolor=color, alpha=0.25, zorder=0)

        if key not in seen_materials:
            pretty_name = layer["pstar_name"]
            material_patches.append(Patch(facecolor=color, edgecolor="none", alpha=0.25, label=pretty_name))
            seen_materials.add(key)

    line_handles = []
    line_labels = []

    h1, = ax.plot(z, D, label="Broadened depth-dose", zorder=3)
    line_handles.append(h1)
    line_labels.append("Broadened depth-dose")

    if plot_ideal and np.max(D0) > 0:
        h2, = ax.plot(z, D0 / np.max(D0), alpha=0.7, label="Ideal depth-dose", zorder=3)
        line_handles.append(h2)
        line_labels.append("Ideal depth-dose")

    h3 = ax.axvline(result["z_peak_cm"], linestyle=":", label="Peak depth", zorder=4)
    line_handles.append(h3)
    line_labels.append("Peak depth")

    ax.set_xlabel("Depth in stack [cm]")
    ax.set_ylabel("Relative dose [a.u.]")
    ax.set_title("1D Bragg Peak Through Tissue Stack")

    # first legend = lines
    legend1 = ax.legend(line_handles, line_labels, loc="lower left")
    ax.add_artist(legend1)

    # second legend = materials
    ax.legend(handles=material_patches, title="Materials", loc="lower right")

    plt.show()

    # energy plot
    fig, ax = plt.subplots(figsize=(9, 5))

    material_patches = []
    seen_materials = set()

    for layer in layers:
        x0 = layer["z_start_cm"]
        x1 = layer["z_end_cm"]
        key = layer["material_key"]
        color = MATERIAL_COLORS.get(key, "whitesmoke")

        ax.axvspan(x0, x1, facecolor=color, alpha=0.25, zorder=0)

        if key not in seen_materials:
            pretty_name = layer["pstar_name"]
            material_patches.append(Patch(facecolor=color, edgecolor="none", alpha=0.25, label=pretty_name))
            seen_materials.add(key)

    ax.plot(z, E, zorder=3)
    ax.set_xlabel("Depth in stack [cm]")
    ax.set_ylabel("Proton energy [MeV]")
    ax.set_title("Proton Energy vs Depth")
    ax.legend(handles=material_patches, title="Materials", loc="lower right")

    plt.show()


if __name__ == "__main__":
    layers = load_stack_csv(STACK)
    stack_array = build_stack_array(CSV_FILE, layers, DZ_CM, ZMAX_CM)

    result = simulate_bragg_peak(
        stack_array=stack_array,
        E0_MeV=E0_MEV,
        dz_cm=DZ_CM,
        phi0=PHI0,
        lambda_per_cm=LAMBDA_PER_CM,
        sigma_cm=SIGMA_CM,
        E_cut_MeV=E_CUT_MEV,
    )

    print(f"Stopping depth from stepping: {result['z_stop_cm']:.4f} cm")
    print(f"Peak depth of broadened curve: {result['z_peak_cm']:.4f} cm")

    plot_result(result,layers, plot_ideal=PLOT_IDEAL)