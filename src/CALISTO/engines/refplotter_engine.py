# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import pyqtgraph as pyg
from .engine import BeadType, TestResult

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


def table_cell_changed(row, column, status_manager, table, beadids, agent="user"):
    # the cell in the table has been changed, update the bead_specs
    bead_specs = status_manager.get_state("bead_specs")
    beadid = beadids[row]
    if column == 0:  # Plot
        pass  # plot
    elif column == 1:  # include
        checked = table.cellWidget(row, 1).isChecked()
        bead_specs.iloc[beadid, 4] = checked
        if checked:
            bead_specs.loc[beadid, "Test Result"] = (
                TestResult.INCLUDED_BY_USER_REFPLOTTER
                if agent == "user"
                else TestResult.INCLUDED_BY_ALGO_REFPLOTTER
            )
        else:
            bead_specs.loc[beadid, "Test Result"] = (
                TestResult.EXCLUDED_BY_USER_REFPLOTTER
                if agent == "user"
                else TestResult.EXCLUDED_BY_ALGO_REFPLOTTER
            )

    else:
        raise ValueError("Invalid column index")

    status_manager.set_state("bead_specs", bead_specs)
