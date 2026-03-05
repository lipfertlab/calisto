# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import pyqtgraph as pyg
from .engine import BeadType, TestResult
from bottleneck import move_median

cmap64 = [
    (0, 0, 0),
    (1, 0, 103),
    (213, 255, 0),
    (255, 0, 86),
    (158, 0, 142),
    (14, 76, 161),
    (255, 229, 2),
    (0, 95, 57),
    (0, 255, 0),
    (149, 0, 58),
    (255, 147, 126),
    (164, 36, 0),
    (0, 21, 68),
    (145, 208, 203),
    (98, 14, 0),
    (107, 104, 130),
    (0, 0, 255),
    (0, 125, 181),
    (106, 130, 108),
    (0, 174, 126),
    (194, 140, 159),
    (190, 153, 112),
    (0, 143, 156),
    (95, 173, 78),
    (255, 0, 0),
    (255, 0, 246),
    (255, 2, 157),
    (104, 61, 59),
    (255, 116, 163),
    (150, 138, 232),
    (152, 255, 82),
    (167, 87, 64),
    (1, 255, 254),
    (255, 238, 232),
    (254, 137, 0),
    (189, 198, 255),
    (1, 208, 255),
    (187, 136, 0),
    (117, 68, 177),
    (165, 255, 210),
    (255, 166, 254),
    (119, 77, 0),
    (122, 71, 130),
    (38, 52, 0),
    (0, 71, 84),
    (67, 0, 44),
    (181, 0, 255),
    (255, 177, 103),
    (255, 219, 102),
    (144, 251, 146),
    (126, 45, 210),
    (189, 211, 147),
    (229, 111, 254),
    (222, 255, 116),
    (0, 255, 120),
    (0, 155, 255),
    (0, 100, 1),
    (0, 118, 255),
    (133, 169, 0),
    (0, 185, 23),
    (120, 130, 49),
    (0, 255, 198),
    (255, 110, 65),
    (232, 94, 190),
]

pens = [pyg.mkPen(color=c, width=1) for c in cmap64]


def table_cell_changed(row, column, status_manager, table):
    # the cell in the table has been changed, update the bead_specs
    bead_specs = status_manager.get_state("bead_specs")
    if column == 0:  # Plot
        pass  # plot
    elif column == 1:  # bead type
        bead_specs.iloc[row, 0] = BeadType(table.cellWidget(row, 1).currentIndex())
    elif column == 2:  # include
        checked = table.cellWidget(row, 2).isChecked()
        # update the bead_specs DataFrame with the checked value
        bead_specs.iloc[row, 4] = checked
        if checked:
            bead_specs.loc[row, "Test Result"] = (
                TestResult.INCLUDED_BY_USER_TRACEPLOTTER
            )
        else:
            bead_specs.loc[row, "Test Result"] = (
                TestResult.EXCLUDED_BY_USER_TRACEPLOTTER
            )

    else:
        raise ValueError("Invalid column index")

    status_manager.set_state("bead_specs", bead_specs)


def filter_z_positions(value, state_manager):
    """
    Filter the z positions based on the value and state manager.
    """
    if value % 2 == 0:
        # even
        value += 1

    beadzpos = state_manager.get_state("bead_pos")[:, :, 2].copy()
    beadzpos = move_median(beadzpos, window=value, axis=0, min_count=1)
    # Update the state manager with the filtered z positions
    state_manager.set_state("bead_filtered_z_pos", beadzpos)
