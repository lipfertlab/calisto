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


def load_hdf_datafile(path, state_manager):
    f = h5py.File(path, "r")

    time = np.array(f["timestamp"]).astype(np.float64)
    time = process_bead_data_time_units(time, state_manager)

    mag_time = f["stage"]["t_s"]
    mag_time = process_motor_data_time_units(mag_time, state_manager)

    mag_pos = f["stage"]["mag_pos_mm"]
    mag_pos = process_motor_data_position_units(mag_pos, state_manager)

    nbeads = len(f.keys()) - 3
    bead_type = np.empty(nbeads, dtype=BeadType)

    bead_data = np.empty((time.shape[0], nbeads, 3), dtype=np.float64)

    for k in f.keys():
        print(k[0])
        if k[0] == "M" or k[0] == "R":
            type = BeadType.MAGNETIC if k[0] == "M" else BeadType.REFERENCE
            beadid = int(k[1:])
        else:
            type = BeadType.MAGNETIC
            beadid = int(k)
        bead_type[beadid] = type
        bead = f[k]
        bead_data[:, beadid, 0] = bead["x_nm"]
        bead_data[:, beadid, 1] = bead["y_nm"]
        bead_data[:, beadid, 2] = bead["z_nm"]
    f.close()

    bead_data = process_bead_data_position_units(bead_data, state_manager)

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
