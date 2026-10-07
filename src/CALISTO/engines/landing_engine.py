# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import numpy as np
from scipy.interpolate import interp1d
import h5py
import pandas as pd

from .engine import (
    AttachmentType,
    BeadType,
    TetherType,
    TestResult,
    process_bead_data_time_units,
    process_bead_data_position_units,
    process_motor_data_time_units,
    process_motor_data_position_units,
)


def load_bead_datafile(path, state_manager):
    rawdata = pd.read_csv(
        path,
        sep=r"\s+",  # space / multiple spaces / tabs
        engine="python",
        header=None,
        na_values=["-1.#IND000"],  # treat this token as NaN
    )

    # Interpolate only numeric columns, within each column (linear)
    num_cols = rawdata.select_dtypes(include=[np.number]).columns
    rawdata[num_cols] = rawdata[num_cols].interpolate(
        method="linear", limit_direction="both"
    )

    # Optional: if any NaNs remain (e.g. entire column missing), decide what to do:
    rawdata[num_cols] = rawdata[num_cols].fillna(0)  # or leave as NaN

    rawdata = rawdata.to_numpy()

    time = rawdata[:, 1]
    time = process_bead_data_time_units(time, state_manager)

    data = rawdata[:, 2:]
    data = data.reshape([data.shape[0], -1, 3])
    data = process_bead_data_position_units(data, state_manager)

    state_manager.set_state("time", time)
    state_manager.set_state("bead_pos", data)
    state_manager.set_state("#beads", data.shape[1])

    fsample = (len(time) - 1) / (time[-1] - time[0])
    state_manager.set_state("computed_fsample", fsample)
    state_manager.set_state("bead_specs_outdated", True)
    state_manager.set_state("measurements_outdated", True)


def load_motor_datafile(path, state_manager):
    rawdata = pd.read_csv(
        path,
        sep=r"\s+",  # space / multiple spaces / tabs
        engine="python",
        header=None,
        na_values=["-1.#IND000"],  # treat this token as NaN
    )

    # Interpolate only numeric columns, within each column (linear)
    num_cols = rawdata.select_dtypes(include=[np.number]).columns
    rawdata[num_cols] = rawdata[num_cols].interpolate(
        method="linear", limit_direction="both"
    )

    rawdata[num_cols] = rawdata[num_cols].fillna(0)

    rawdata = rawdata.to_numpy()

    mag_time = rawdata[:, 1]
    mag_time = process_motor_data_time_units(mag_time, state_manager)

    mag_pos = rawdata[:, 2]
    mag_pos = process_motor_data_position_units(mag_pos, state_manager)

    state_manager.set_state("mag_pos", mag_pos)
    state_manager.set_state("mag_time", mag_time)
    state_manager.set_state("bead_specs_outdated", True)
    state_manager.set_state("measurements_outdated", True)


def verify_data_consistency(state_manager):
    time = state_manager.get_state("time")
    bead_pos = state_manager.get_state("bead_pos")

    mag_time = state_manager.get_state("mag_time")
    mag_pos = state_manager.get_state("mag_pos")

    if bead_pos.shape[0] != len(time):
        raise ValueError("Time and data are not consistent.")

    magposinterp = interp1d(mag_time, mag_pos, kind="nearest")
    mag_pos = magposinterp(time)
    state_manager.set_state("mag_pos", mag_pos)
    state_manager.set_state("mag_time", time)


def prepare_dataframe(state_manager):

    outdated = state_manager.get_state("bead_specs_outdated")
    bead_specs = state_manager.get_state("bead_specs")
    if not outdated and bead_specs is not None:
        return

    nbeads = state_manager.get_state("#beads")
    bead_specs = pd.DataFrame(
        columns=[
            "Type",
            "Tether Type",
            "Offset",
            "Test Result",
            "Include",
            # "Attachment",
        ]
    )

    bead_type = state_manager.get_state("bead_type")
    if bead_type is None:
        bead_type = np.repeat(BeadType.MAGNETIC, nbeads)

    bead_specs["Type"] = bead_type
    bead_specs["Offset"] = np.zeros(nbeads)

    bead_specs["Tether Type"] = np.repeat(TetherType.NICKED, nbeads)
    bead_specs["Test Result"] = np.repeat(TestResult.NOT_TESTED, nbeads)
    bead_specs["Include"] = np.repeat(True, nbeads)
    state_manager.delete_state("bead_type")
    state_manager.set_state("bead_specs", bead_specs)
    state_manager.set_state("bead_specs_outdated", False)
    state_manager.set_state("measurements_outdated", True)

    keys_not_to_delete = [
        "time",
        "bead_pos",
        "#beads",
        "computed_fsample",
        "bead_specs_outdated",
        "measurements_outdated",
        "mag_pos",
        "mag_time",
        "bead_specs",
        "bead_radius",
        "temperature",
        "fsample",
        "axis",
        "config_setup",
        "config",
    ]
    for key in list(state_manager.keys()):
        if key not in keys_not_to_delete:
            state_manager.delete_state(key)

def isbeadkey(k):
    start = 0
    if k[start] == "M" or k[start] == "R":
        start = 1

    try:
        int(k[start:])
        return True
    except ValueError:
        return False

def read_hdf_beads(f):
    """Read bead arrays indexed by unique, contiguous zero-based bead IDs."""
    keys = [key for key in f.keys() if isbeadkey(key)]
    ids = [int(key[1:] if key[0] in "MR" else key) for key in keys]
    if not keys or sorted(ids) != list(range(len(keys))):
        raise ValueError("HDF5 bead IDs must be unique and contiguous starting at 0.")

    time = np.asarray(f["timestamp"][...], dtype=np.float64)
    bead_type = np.empty(len(keys), dtype=object)
    # Each dataset is one time series; keep its destination contiguous.
    bead_data = np.empty((len(time), len(keys), 3), dtype=np.float64, order="F")
    for key, beadid in zip(keys, ids):
        bead_type[beadid] = BeadType.REFERENCE if key.startswith("R") else BeadType.MAGNETIC
        bead = f[key]
        if isinstance(bead, h5py.Dataset):
            if bead.shape != time.shape:
                raise ValueError(f"HDF5 bead {key} must match timestamp shape.")
            # Read compound coordinates together: decompress each chunk only once.
            values = bead.fields(("x_nm", "y_nm", "z_nm"))[...]
            for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
                bead_data[:, beadid, axis] = values[name]
            continue
        for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
            dataset = bead[name]
            if dataset.shape != time.shape:
                raise ValueError(f"HDF5 bead {key}/{name} must match timestamp shape.")
            dataset.read_direct(bead_data[:, beadid, axis])
    return time, bead_data, bead_type


def load_hdf_datafile(path, state_manager):
    with h5py.File(path, "r") as f:
        time, bead_data, bead_type = read_hdf_beads(f)
        mag_time = np.asarray(f["stage"]["t_s"][...])
        mag_pos = np.asarray(f["stage"]["mag_pos_mm"][...])

    time = process_bead_data_time_units(time, state_manager)
    mag_time = process_motor_data_time_units(mag_time, state_manager)
    mag_pos = process_motor_data_position_units(mag_pos, state_manager)
    nbeads = bead_data.shape[1]

    bead_data = process_bead_data_position_units(bead_data, state_manager, copy=False)

    state_manager.set_state("time", time)
    state_manager.set_state("bead_pos", bead_data)
    state_manager.set_state("#beads", nbeads)
    state_manager.set_state("mag_time", mag_time)
    state_manager.set_state("mag_pos", mag_pos)
    state_manager.set_state("bead_type", bead_type)

    fsample = (len(time) - 1) / (time[-1] - time[0])
    state_manager.set_state("computed_fsample", fsample)
    state_manager.set_state("bead_specs_outdated", True)
    state_manager.set_state("measurements_outdated", True)
