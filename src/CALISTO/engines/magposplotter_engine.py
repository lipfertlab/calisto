# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import numpy as np
import pyqtgraph as pyg
from .engine import BeadType

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

pens = [pyg.mkPen(color=c, width=5) for c in cmap64]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def identify_plateaus(zmag_pos, tolerance, min_plateau_length):
    indices = np.arange(zmag_pos.shape[0])
    zmag_const = np.abs(zmag_pos[:-1] - zmag_pos[1:]) < tolerance
    constindices = (indices[:-1])[zmag_const]

    plateaus = np.array(consecutive(constindices), dtype=object)
    plateau_sizes = np.array([gr.size for gr in plateaus])
    plateaus = plateaus[plateau_sizes > min_plateau_length]

    return plateaus


def get_bead_inclusion_array(state_manager, plateaus):
    nbeads = state_manager.get_state("#beads")
    nplt = len(plateaus)

    inclusion_array = np.ones((nbeads, nplt), dtype=bool)
    return inclusion_array
