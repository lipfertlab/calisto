# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from enum import Enum
from turtle import pd
import numpy as np
import h5py
import PySide6.QtWidgets as QtWidgets
from scipy.interpolate import interp1d
from .engine import (
    BeadType,
    TetherType,
    FileType,
    TestResult,
    process_bead_data_position_units,
    process_bead_data_time_units,
    process_motor_data_time_units,
    process_motor_data_position_units,
)
import pandas as pd


class OffsetType(Enum):
    OFFSET_TABLE = 1
    OFFSET_DATA = 2
    OFFSET_CONSTANT = 3


def prepare_table(table, status_manager):

    bead_specs = status_manager.get_state("bead_specs")
    columns = bead_specs.columns
    table.setColumnCount(len(columns))
    table.setRowCount(len(bead_specs))
    table.setHorizontalHeaderLabels(columns)
    table.setVerticalHeaderLabels([str(i) for i in range(len(bead_specs))])

    for i in range(len(bead_specs)):
        item = QtWidgets.QComboBox()
        item.addItems(["Magnetic", "Reference"])
        item.setCurrentIndex(bead_specs.iloc[i, 0].value)
        item.currentIndexChanged.connect(
            lambda index, row=i, col=0: table_cell_changed(
                row, col, status_manager, table
            )
        )
        table.setCellWidget(i, 0, item)

    for i in range(len(bead_specs)):
        item = QtWidgets.QComboBox()
        item.addItems(
            [
                "Nicked",
                "Supercoilable",
                "Multiple tethered",
                "No Tether",
            ]
        )
        item.setCurrentIndex(bead_specs.iloc[i, 1].value)
        item.currentIndexChanged.connect(
            lambda index, row=i, col=1: table_cell_changed(
                row, col, status_manager, table
            )
        )
        table.setCellWidget(i, 1, item)

    for i in range(len(bead_specs)):
        item = QtWidgets.QLineEdit(str(round(bead_specs.iloc[i, 2], 6)))
        item.textChanged.connect(
            lambda text, row=i, col=2: table_cell_changed(
                row, col, status_manager, table
            )
        )
        table.setCellWidget(i, 2, item)

    for i in range(len(bead_specs)):
        item = QtWidgets.QLabel(str(bead_specs.iloc[i, 3]))
        table.setCellWidget(i, 3, item)

    for i in range(len(bead_specs)):
        item = QtWidgets.QCheckBox()
        item.setChecked(bead_specs.iloc[i, 4])
        item.stateChanged.connect(
            lambda state, row=i, col=4: table_cell_changed(
                row, col, status_manager, table
            )
        )
        table.setCellWidget(i, 4, item)


def table_changed(table, status_manager):
    # the table has been changed, update the bead_specs
    bead_specs = status_manager.get_state("bead_specs")

    for i in range(len(bead_specs)):
        bead_specs.iloc[i, 0] = BeadType(table.cellWidget(i, 0).currentIndex())
        bead_specs.iloc[i, 1] = TetherType(table.cellWidget(i, 1).currentIndex())
        bead_specs.iloc[i, 2] = float(table.cellWidget(i, 2).text())
        bead_specs.iloc[i, 4] = table.cellWidget(i, 4).isChecked()

    status_manager.set_state("bead_specs", bead_specs)


def table_cell_changed(row, column, status_manager, table):
    # the cell in the table has been changed, update the bead_specs
    bead_specs = status_manager.get_state("bead_specs")
    if column == 0:
        beadtype = BeadType(table.cellWidget(row, 0).currentIndex())
        bead_specs.iloc[row, 0] = beadtype
        if beadtype == BeadType.REFERENCE:
            bead_specs.iloc[row, 1] = TetherType.NO_TETHER
    elif column == 1:
        bead_specs.iloc[row, 1] = TetherType(table.cellWidget(row, 1).currentIndex())
    elif column == 2:
        bead_specs.iloc[row, 2] = float(table.cellWidget(row, 2).text())
    elif column == 4:
        checked = table.cellWidget(row, 4).isChecked()
        bead_specs.iloc[row, 4] = checked
        if checked:
            bead_specs.iloc[row, 3] = TestResult.INCLUDED_BY_USER
        else:
            bead_specs.iloc[row, 3] = TestResult.EXCLUDED_BY_USER

    status_manager.set_state("bead_specs", bead_specs)


def offset_type_set(offset_type, status_manager):
    status_manager.set_state("offset_type", offset_type)


def offset_table_load(offset_table_path, status_manager):

    rawdata = pd.read_csv(
        offset_table_path,
        sep=r"\s+",  # space / multiple spaces / tabs
        engine="python",
        header=None,
        na_values=["-1.#IND000"],  # treat this token as NaN
        comment="#",
    )

    # Interpolate only numeric columns, within each column (linear)
    num_cols = rawdata.select_dtypes(include=[np.number]).columns
    rawdata[num_cols] = rawdata[num_cols].interpolate(
        method="linear", limit_direction="both"
    )

    rawdata[num_cols] = rawdata[num_cols].fillna(0)

    rawdata = rawdata.to_numpy()

    table = rawdata[:, 1]

    bead_specs = status_manager.get_state("bead_specs")

    if len(table) != bead_specs.shape[0]:
        raise ValueError(
            "Number of bead in the offset table and the data set do not match."
        )

    bead_specs["Offset"] = table
    status_manager.set_state("bead_specs", bead_specs)
    status_manager.set_state("measurements_outdated", True)

    offset_type_set(OffsetType.OFFSET_TABLE, status_manager)


def offset_data_load(offset_data_path, file_type, status_manager):
    offset_type_set(OffsetType.OFFSET_DATA, status_manager)
    if file_type == FileType.PLAINTEXT:
        rawdata = pd.read_csv(
            offset_data_path,
            sep=r"\s+",  # space / multiple spaces / tabs
            engine="python",
            header=None,
            na_values=["-1.#IND000"],  # treat this token as NaN
            comment="#",
        )

        # Interpolate only numeric columns, within each column (linear)
        num_cols = rawdata.select_dtypes(include=[np.number]).columns
        rawdata[num_cols] = rawdata[num_cols].interpolate(
            method="linear", limit_direction="both"
        )

        rawdata[num_cols] = rawdata[num_cols].fillna(0)

        data = rawdata.to_numpy()

        time = data[:, 1]
        data = data[:, 2:]
        data = data.reshape([data.shape[0], -1, 3])
    elif file_type == FileType.CSV:
        raise NotImplementedError("CSV file type not implemented yet.")
    elif file_type == FileType.HDF5:
        time, data = offset_data_hdf_load(offset_data_path)
    else:
        raise ValueError("File type not recognized.")

    nbeads = status_manager.get_state("#beads")
    if data.shape[1] != nbeads:
        raise ValueError(
            "Number of bead in the offset data and the data set do not match."
        )

    data = process_bead_data_position_units(data, status_manager)
    time = process_bead_data_time_units(time, status_manager)

    status_manager.set_state("offset_rawdata", data)
    status_manager.set_state("offset_time", time)


def offset_data_hdf_load(path):
    f = h5py.File(path, "r")

    time = np.array(f["timestamp"]).astype(np.float64)

    nbeads = len(f.keys()) - 3
    bead_type = np.empty(nbeads, dtype=BeadType)
    bead_data = np.empty((time.shape[0], nbeads, 3), dtype=np.float64)

    for k in f.keys():
        print(k[0])
        if not (k[0] == "M" or k[0] == "R"):
            continue
        type = BeadType.MAGNETIC if k[0] == "M" else BeadType.REFERENCE
        beadid = int(k[1:])
        bead_type[beadid] = type
        bead = f[k]
        bead_data[:, beadid, 0] = bead["x_nm"]
        bead_data[:, beadid, 1] = bead["y_nm"]
        bead_data[:, beadid, 2] = bead["z_nm"]
    f.close()

    return time, bead_data


def offset_constant_set(offset_constant, status_manager):
    offset_type_set(OffsetType.OFFSET_CONSTANT, status_manager)
    status_manager.set_state("offset_constant", offset_constant)
    bead_specs = status_manager.get_state("bead_specs")
    # set magnetic beads to the offset constant
    bead_specs.loc[bead_specs["Type"] == BeadType.MAGNETIC, "Offset"] = offset_constant

    status_manager.set_state("bead_specs", bead_specs)
    status_manager.set_state("measurements_outdated", True)


def prepare_mean_reference(status_manager):
    bead_specs = status_manager.get_state("bead_specs")
    ref_beads_mask = (bead_specs["Type"] == BeadType.REFERENCE).to_numpy()
    if not ref_beads_mask.any():
        mean_ref = np.zeros((status_manager.get_state("time").shape[0], 3))
    else:
        mean_ref = status_manager.get_state("data")[:, ref_beads_mask, :].mean(axis=1)
    status_manager.set_state("ref_trace", mean_ref)


def load_rot_hdfdatafile(path, state_manager):
    f = h5py.File(path, "r")

    time = np.array(f["timestamp"]).astype(np.float64)
    time = process_bead_data_time_units(time, state_manager)

    mag_time = f["stage"]["t_s"]
    mag_time = process_motor_data_time_units(mag_time, state_manager)

    mag_pos = f["stage"]["mag_pos_mm"]
    mag_pos = process_motor_data_position_units(mag_pos, state_manager)

    mag_rot = f["stage"]["mag_rot_turn"]

    nbeads = len(f.keys()) - 3
    bead_data = np.empty((time.shape[0], nbeads, 3), dtype=np.float64)

    for k in f.keys():
        if not (k[0] == "M" or k[0] == "R"):
            continue

        beadid = int(k[1:])

        bead = f[k]
        bead_data[:, beadid, 0] = bead["x_nm"]
        bead_data[:, beadid, 1] = bead["y_nm"]
        bead_data[:, beadid, 2] = bead["z_nm"]
    f.close()

    bead_data = process_bead_data_position_units(bead_data, state_manager)

    state_manager.set_state("rot_time", time)
    state_manager.set_state("rot_bead_pos", bead_data)
    state_manager.set_state("rot_#beads", nbeads)
    state_manager.set_state("rot_mag_time", mag_time)
    state_manager.set_state("rot_mag_pos", mag_pos)
    state_manager.set_state("rot_mag_rot", mag_rot)


def load_rot_datafile(path, state_manager):
    rawdata = pd.read_csv(
        path,
        sep=r"\s+",  # space / multiple spaces / tabs
        engine="python",
        header=None,
        na_values=["-1.#IND000"],  # treat this token as NaN
        comment="#",
    )

    # Interpolate only numeric columns, within each column (linear)
    num_cols = rawdata.select_dtypes(include=[np.number]).columns
    rawdata[num_cols] = rawdata[num_cols].interpolate(
        method="linear", limit_direction="both"
    )

    rawdata[num_cols] = rawdata[num_cols].fillna(0)

    rawdata = rawdata.to_numpy()

    time = rawdata[:, 1]
    time = process_bead_data_time_units(time, state_manager)

    data = rawdata[:, 2:]
    data = data.reshape([data.shape[0], -1, 3])
    data = process_bead_data_position_units(data, state_manager)

    state_manager.set_state("rot_time", time)
    state_manager.set_state("rot_bead_pos", data)
    state_manager.set_state("rot_#beads", data.shape[1])


def load_rot_motor_datafile(path, state_manager):
    rawdata = pd.read_csv(
        path,
        sep=r"\s+",  # space / multiple spaces / tabs
        engine="python",
        header=None,
        na_values=["-1.#IND000"],  # treat this token as NaN
        comment="#",
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

    mag_rot = rawdata[:, 3]

    state_manager.set_state("rot_mag_pos", mag_pos)
    state_manager.set_state("rot_mag_time", mag_time)
    state_manager.set_state("rot_mag_rot", mag_rot)


def verify_rot_data_consistency(state_manager):
    time = state_manager.get_state("rot_time")
    bead_pos = state_manager.get_state("rot_bead_pos")

    mag_time = state_manager.get_state("rot_mag_time")
    mag_pos = state_manager.get_state("rot_mag_pos")
    mag_rot = state_manager.get_state("rot_mag_rot")

    rotnbeads = state_manager.get_state("rot_#beads")
    nbeads = state_manager.get_state("#beads")

    if nbeads != rotnbeads:
        bead_pos = bead_pos[:, :nbeads, :]
        state_manager.set_state("rot_bead_pos", bead_pos)
        state_manager.set_state("rot_#beads", nbeads)
        raise ValueError(
            f"Number of beads in the rotation data set ({rotnbeads}) does not match with the force calibration data set ({nbeads})."
        )

    if bead_pos.shape[0] != len(time):
        raise ValueError("Time and data are not consistent.")

    magposinterp = interp1d(mag_time, mag_pos, kind="linear", fill_value="extrapolate")
    mag_pos = magposinterp(time)

    magrotinterp = interp1d(mag_time, mag_rot, kind="linear", fill_value="extrapolate")
    mag_rot = magrotinterp(time)

    state_manager.set_state("rot_mag_pos", mag_pos)
    state_manager.set_state("rot_mag_time", time)
    state_manager.set_state("rot_mag_rot", mag_rot)


def process_offset_filter(state_manager, min_offset, max_offset):
    bead_specs = state_manager.get_state("bead_specs")
    offsets = bead_specs["Offset"].to_numpy()
    inclusion = state_manager.get_state("backup_include")
    reasons = state_manager.get_state("backup_reason")
    mag_beads_mask = (bead_specs["Type"] == BeadType.MAGNETIC).to_numpy()
    # mask everything, filter works on magnetic beads
    offsets = offsets[mag_beads_mask]
    inclusion = inclusion[mag_beads_mask]
    reasons = reasons[mag_beads_mask]

    offset_criterion = (offsets >= min_offset) & (offsets <= max_offset)
    currently_excluded_by_filter = reasons == TestResult.EXCLUDED_BY_OFFSET_FILTER
    new_inclusion = (inclusion | currently_excluded_by_filter) & offset_criterion

    excluded_by_filter = ~new_inclusion & inclusion

    reasons[excluded_by_filter] = TestResult.EXCLUDED_BY_OFFSET_FILTER

    bead_specs.loc[mag_beads_mask, "Test Result"] = reasons
    bead_specs.loc[mag_beads_mask, "Include"] = new_inclusion
    state_manager.set_state("bead_specs", bead_specs)
    state_manager.set_state("measurements_outdated", True)


def backup_inclusion_and_reason(state_manager):
    bead_specs = state_manager.get_state("bead_specs")
    inclusion = bead_specs["Include"].to_numpy()
    reasons = bead_specs["Test Result"].to_numpy()

    state_manager.set_state("backup_include", inclusion)
    state_manager.set_state("backup_reason", reasons)


def restore_inclusion_and_reason(state_manager):
    backup_include = state_manager.get_state("backup_include")
    backup_reason = state_manager.get_state("backup_reason")

    bead_specs = state_manager.get_state("bead_specs")
    bead_specs["Include"] = backup_include
    bead_specs["Test Result"] = backup_reason
    state_manager.set_state("bead_specs", bead_specs)
