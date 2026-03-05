# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import numpy as np
from scipy.optimize import curve_fit
from .engine import BeadType, MultiBeadMeasurement
import yaml
import pickle as pkl
import pandas as pd


def prepare_multibeadmeasurement(state_manager):

    measurements = state_manager.get_state("measurements")
    if measurements is not None:
        outdated = state_manager.get_state("measurements_outdated", True)
        if not outdated:
            return measurements

    plateaus = state_manager.get_state("plateaus")
    zmag_pos = state_manager.get_state("mag_pos")
    bead_radius = state_manager.get_state("bead_radius")
    axis = state_manager.get_state("axis")
    fsample = state_manager.get_state("fsample")
    include = state_manager.get_state("inclusion")
    bead_spec = state_manager.get_state("bead_specs")
    include_col = (
        bead_spec["Include"].to_numpy().repeat(len(plateaus)).reshape(-1, len(plateaus))
    )
    include = include & include_col

    bead_pos = state_manager.get_state("bead_pos").copy()
    if state_manager.get_state("median_filter"):
        bead_pos[:, :, 2] = state_manager.get_state("bead_filtered_z_pos")
    offsets = state_manager.get_state("bead_specs")["Offset"].to_numpy()
    bead_pos[:, :, 2] = bead_pos[:, :, 2] - offsets
    allBeads = np.arange(bead_pos.shape[1])
    bead_pos = np.einsum("ijk->jki", bead_pos)
    refbeadmask = state_manager.get_state("bead_specs")["Type"] == BeadType.REFERENCE
    assert refbeadmask.any(), "No reference beads found in bead_specs."

    refbeads = allBeads[refbeadmask]
    bead_pos = bead_pos[allBeads, :, :]
    measurements = []
    for pid, pl in enumerate(plateaus):
        magz = zmag_pos[pl]
        magz = magz.mean()
        m = MultiBeadMeasurement(
            fsample,  # sampling frequency
            magz,  # magnet position
            0.0,  # magnet rotation
            axis,  # measurement axis
            bead_pos[:, :, pl],  # traces in the plateau
            allBeads,  # all beads (references are included)
            refbeads,  # reference bead ids
            bead_radius=bead_radius,
        )
        for bid in allBeads:
            m[bid].good = include[bid, pid]

        m.subtract_reference()  # subtraces mean reference if there is more than one reference
        measurements.append(m)

    state_manager.set_state("measurements", measurements)
    state_manager.set_state("measurements_outdated", False)
    return measurements


def two_term_exp(x, fmax, l1, l2, c):
    return fmax * (c * np.exp(-x / l1) + (1 - c) * np.exp(-x / l2))


def single_exp(x, fmax, l1):
    return fmax * np.exp(-x / l1)


def wlcfunc(ext, Lp, Lc, T):
    """
    % Given a vector of extensions and the parameter
    % Lp = persistence length  (in nano-m)
    % Lc = contour length      (in nano-m)
    % T  = absolute temperature (in Kelvin)
    % This function returns the forces computed from a
    % 7 parameter model of the WLC,
    % using the model by Bouchiat, et al. Biophys J 76:409 (1999)
    """
    kT = 1.3806503e-2 * T  # k_B T in units pN nano-m
    z_scaled = ext / Lc
    coef = np.array(
        [-0.25, 1, -0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718]
    )
    Fwlc = 1.0 / (4.0 * (1.0 - z_scaled) ** 2)
    for p, a in enumerate(coef):
        Fwlc += a * z_scaled**p
    return Fwlc * kT / Lp


def fit_wlc(x, y, temperatue, p0=None, wlcfit_cutoff_force=10, n_trial=100):
    wlcfit = lambda x, Lp, Lc: wlcfunc(x, Lp, Lc, temperatue)
    mask = (y > 0) & (y < wlcfit_cutoff_force) & (x > 0)

    maxext = np.max(x[mask])
    bounds = ([1e-8, maxext + 1e-8], [1e3, 100 * maxext])

    if p0 is not None:
        popt, pcov = curve_fit(wlcfit, x[mask], y[mask], p0=p0, bounds=bounds)
        return popt

    # seed rng for reproducibility
    rng = np.random.default_rng(seed=0x250796)
    try:
        p0 = [45, maxext * 1.2]
        bestpopt, _ = curve_fit(wlcfit, x[mask], y[mask], p0=p0, bounds=bounds)
        residuals = y[mask] - wlcfit(x[mask], *bestpopt)
        bestresidual = np.sum(residuals**2)
    except:
        bestresidual = np.inf
        bestpopt = None
    for _ in range(n_trial):
        # randomly sample p0 from bounds
        Lp0 = rng.uniform(bounds[0][0], bounds[1][0])
        Lc0 = rng.uniform(bounds[0][1], bounds[1][1])
        p0 = [Lp0, Lc0]
        try:
            popt, pcov = curve_fit(wlcfit, x[mask], y[mask], p0=p0, bounds=bounds)
            residuals = y[mask] - wlcfit(x[mask], *popt)
            ss_res = np.sum(residuals**2)
            if ss_res < bestresidual:
                bestresidual = ss_res
                bestpopt = popt
        except RuntimeError:
            continue

    return bestpopt


def get_all_forces_v_magpos(state_manager):
    measurements = prepare_multibeadmeasurement(state_manager)

    fullmagpos = np.array([])
    fullforces = {"PSD": [], "AV": [], "HV": []}
    for m in measurements:
        for method in ["PSD", "AV", "HV"]:
            force = m.get_forces(method)
            if force.size == 0:
                continue
            fullforces[method].append(force)
        magpos = m.mag_pos * np.ones(force.shape[0])
        fullmagpos = np.hstack((fullmagpos, magpos))

    extmagpos = state_manager.get_state("ext_mag_pos")
    if extmagpos is not None:
        fullmagpos = np.hstack((extmagpos, fullmagpos))

    for method in ["PSD", "AV", "HV"]:
        fullforces[method] = np.vstack(fullforces[method])

    extforces = state_manager.get_state("ext_forces")
    if extforces is not None:
        for method in ["PSD", "AV", "HV"]:
            fullforces[method] = np.vstack((extforces[method], fullforces[method]))

    return fullmagpos, fullforces


def export_calibration(method, path, state_manager):
    fitparams = state_manager.get_state("master_curve_params", None)
    if fitparams is None:
        raise ValueError("No master curve fit available.")
    fitparams = fitparams[method]
    fitmodel = state_manager.get_state("master_curve_model", None)

    fullmagpos, fullforces = get_all_forces_v_magpos(state_manager)
    data = {
        "mag_pos": fullmagpos,
        "forces": fullforces,
    }
    picklepath = path.with_suffix(".pkl")
    with open(picklepath, "wb") as f:
        pkl.dump(data, f)

    magposrange = np.linspace(0, 15, 100)
    if fitmodel == "Double Exponential":
        modeltext = r"$ F(z) = F_{max} (c e^{-z/a} + (1-c) e^{-z/b}) $"
        yamlout = {
            "Model": modeltext,
            "F_{max}": float(fitparams[0]),
            "a": float(fitparams[1]),
            "b": float(fitparams[2]),
            "c": float(fitparams[3]),
            "Calibration_method": method,
            "Force_unit": "pN",
            "Length_unit": "mm",
        }
        forces = two_term_exp(magposrange, *(fitparams[:-1]))

    elif fitmodel == "Single Exponential":
        modeltext = r"$ F(z) = F_{max} e^{-z/l} $"
        yamlout = {
            "Model": modeltext,
            "F_{max}": float(fitparams[0]),
            "l": float(fitparams[1]),
            "Calibration_method": method,
            "Force_unit": "pN",
            "Length_unit": "mm",
        }
        forces = single_exp(magposrange, *(fitparams[:-1]))

    else:
        raise ValueError(f"Unknown fit model: {fitmodel}")

    with open(path, "w") as f:
        yaml.dump(yamlout, f, sort_keys=False)

    forcetable = np.zeros((len(magposrange), 2))
    forcetable[:, 0] = magposrange
    forcetable[:, 1] = forces
    forcetablepath = path.parent / (path.stem + "_forcetable.csv")
    np.savetxt(
        forcetablepath,
        forcetable,
        delimiter=",",
        header="Magnet_Position[mm],Force[pN]",
        comments="",
    )

    export_all_bead_data(path, state_manager)  # Export bead data to CSV
    beadspecs = state_manager.get_state("bead_specs")
    beadtable_path = path.parent / (path.stem + "_bead_property_table.csv")
    beadspecs.to_csv(beadtable_path, index=False)

    return


def export_all_bead_data(path, state_manager):
    """
    Export all bead data to pandas DataFrame and save it to a CSV file
    :param path: Path to save the CSV file
    :param state_manager: StateManager instance containing the bead data
    :return: None
    """
    measurements = prepare_multibeadmeasurement(state_manager)
    include = state_manager.get_state("inclusion")
    nbeads = state_manager.get_state("#beads")
    type = state_manager.get_state("bead_specs")["Type"].to_numpy()
    all_bead_data = []

    for pid, m in enumerate(measurements):
        for bead_id in range(nbeads):
            if not include[bead_id, pid]:
                continue
            if not m[bead_id].good:
                continue
            if type[bead_id] == BeadType.REFERENCE:
                continue
            forceav = m[bead_id].get_force("AV")
            forcepsd = m[bead_id].get_force("PSD")
            forcehv = m[bead_id].get_force("HV")

            extension = m[bead_id].get_extension()

            bead_data = {
                "#Bead_ID": bead_id,
                "Magnet_Position": m.mag_pos,
                "Extension": extension["mean"],
                "Extension_Error": extension["stderr"],
                "Force_PSD": forcepsd[0],
                "Force_Error_PSD": forcepsd[1],
                "Force_AV": forceav[0],
                "Force_Error_AV": forceav[1],
                "Force_HV": forcehv[0],
                "Force_Error_HV": forcehv[1],
            }

            all_bead_data.append(bead_data)
    csvpath = path.with_suffix(".csv")
    df = pd.DataFrame(all_bead_data)
    df.to_csv(csvpath, index=False, header=True)


def load_force_calibration_data(path, state_manager):
    """
    Load force calibration data from a pickle
    :param path: Path to the pickle file
    :return: DataFrame containing the force calibration data
    """
    with open(path, "rb") as f:
        data = pkl.load(f)
    mag_pos = data["mag_pos"]
    forces = data["forces"]

    ext_magpos = state_manager.get_state("ext_mag_pos")
    if ext_magpos is None:
        ext_magpos = mag_pos
    else:
        ext_magpos = np.hstack([ext_magpos, mag_pos])

    state_manager.set_state("ext_mag_pos", ext_magpos)

    ext_forces = state_manager.get_state("ext_forces")
    if ext_forces is None:
        ext_forces = forces
    else:
        for key in forces.keys():
            ext_forces[key] = np.vstack([ext_forces[key], forces[key]])
    state_manager.set_state("ext_forces", ext_forces)


def clear_external_force_calibration_data(state_manager):
    """
    Clear external force calibration data from the state manager
    :param state_manager: StateManager instance
    :return: None
    """
    state_manager.delete_state("ext_mag_pos")
    state_manager.delete_state("ext_forces")
