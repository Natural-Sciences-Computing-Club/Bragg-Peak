# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:08:33 2026

@author: reichen schaller
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1: controls
CSV_FILE = "C:/Users/reich/Downloads/Water PSTAR Data.csv"
E0_MEV = 150.0 #initial energy
DZ_CM = 0.001 #step size
ZMAX_CM = None
PHI0 = 1.0 #more attenutation stuff idk we kinda assuming stuff here
LAMBDA_PER_CM = 0.0 #attenuation constant
SIGMA_CM = 0.03 #smoothing factor
E_CUT_MEV = 1.0e-3 #cutoff for when the energy is too low
RHO_WATER_G_CM3 = 1.0 #water density
PLOT_IDEAL = True #where to show an idea curve or not


# Part 2: data / thing to read the csv file 
def load_pstar_csv(path):
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


def make_log_interp(x, y): # do a log log interpretation since data range hella big and curve a lot, logs make them smoother
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
    csv_file,
    E0_MeV,
    dz_cm,
    zmax_cm,
    phi0,
    lambda_per_cm,
    sigma_cm,
    E_cut_MeV,
    rho_water_g_cm3,
):
    E_tab, S_mass_tab, R_csda_mass_tab = load_pstar_csv(csv_file)
 #load stuff from file
    if not (E_tab[0] <= E0_MeV <= E_tab[-1]): #checking if energy is actually in csv
        raise ValueError(f"E0_MeV must lie between {E_tab[0]} and {E_tab[-1]} MeV.")

    S_mass_of_E = make_log_interp(E_tab, S_mass_tab) #making our logs of the data insto a nice smooth distribution of values
    R_cm_of_E = make_log_interp(E_tab, R_csda_mass_tab / rho_water_g_cm3)

    R0_cm = float(R_cm_of_E(E0_MeV)) #evaluates the CSDA range at given E0

    if zmax_cm is None: #making a max depth if not given one
        zmax_cm = 1.2 * R0_cm

    z_cm = np.arange(0.0, zmax_cm + dz_cm, dz_cm)

    E_MeV = np.zeros_like(z_cm)
    S_lin_MeV_per_cm = np.zeros_like(z_cm) #initializing arrays
    fluence = phi0 * np.exp(-lambda_per_cm * z_cm)
    D_ideal = np.zeros_like(z_cm) #idea curve

    E_MeV[0] = E0_MeV

    for i in range(len(z_cm) - 1): #iterate through depth points
        Ei = E_MeV[i]

        if Ei <= E_cut_MeV:
            break

        S_lin_i = rho_water_g_cm3 * float(S_mass_of_E(Ei)) #finds stopping power at i depth
        S_lin_MeV_per_cm[i] = S_lin_i
        D_ideal[i] = fluence[i] * S_lin_i

        E_next = Ei - S_lin_i * dz_cm # finds new energy after stopping power decreases it
        E_MeV[i + 1] = max(E_next, 0.0)

    live = np.where(E_MeV > E_cut_MeV)[0] #where we are above the cutoff
    if len(live) > 0:
        j = live[-1]
        S_lin_MeV_per_cm[j] = rho_water_g_cm3 * float(S_mass_of_E(E_MeV[j]))
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
        "R0_csda_cm": R0_cm,
        "z_stop_cm": z_stop_cm,
        "z_peak_cm": z_peak_cm,
    }


# Part 4: plots and run
def plot_result(result, plot_ideal=True):
    z = result["z_cm"]
    E = result["E_MeV"]
    D0 = result["D_ideal"]
    D = result["D_norm"]

    plt.figure(figsize=(8, 5))
    plt.plot(z, D, label="Broadened depth-dose")
    if plot_ideal and np.max(D0) > 0:
        plt.plot(z, D0 / np.max(D0), label="Ideal depth-dose", alpha=0.7)
    plt.axvline(result["R0_csda_cm"], linestyle="--", label="PSTAR CSDA range")
    plt.axvline(result["z_peak_cm"], linestyle=":", label="Peak depth")
    plt.xlabel("Depth in water [cm]")
    plt.ylabel("Relative dose [a.u.]")
    plt.title("1D Bragg Peak in Liquid Water")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(z, E)
    plt.xlabel("Depth in water [cm]")
    plt.ylabel("Proton energy [MeV]")
    plt.title("Proton Energy vs Depth")
    plt.show()


if __name__ == "__main__": #this just runs the file, main part is because chat says you do this if you are sharing stuff idk
    result = simulate_bragg_peak(
        csv_file=CSV_FILE,
        E0_MeV=E0_MEV,
        dz_cm=DZ_CM,
        zmax_cm=ZMAX_CM,
        phi0=PHI0,
        lambda_per_cm=LAMBDA_PER_CM,
        sigma_cm=SIGMA_CM,
        E_cut_MeV=E_CUT_MEV,
        rho_water_g_cm3=RHO_WATER_G_CM3,
    )

    print(f"PSTAR CSDA range at E0: {result['R0_csda_cm']:.4f} cm")
    print(f"Stopping depth from stepping: {result['z_stop_cm']:.4f} cm")
    print(f"Peak depth of broadened curve: {result['z_peak_cm']:.4f} cm")

    plot_result(result, plot_ideal=PLOT_IDEAL)
