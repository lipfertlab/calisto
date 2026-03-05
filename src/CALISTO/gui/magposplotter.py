# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtWidgets import QLabel, QLineEdit, QGridLayout, QGroupBox, QWidget
import pyqtgraph as pyg

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
from itertools import cycle
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType

import engines.magposplotter_engine as engine

checkmark = "\u2713"
crossmark = "\u2717"


class MagPosPlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        self.state_manager.stateChanged.connect(parent.on_state_changed)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.plotter = self.create_plotter()
        self.plot_magpos()
        self.layout.addWidget(self.plotter, 2, 0, 21, 21)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group, 0, 0, 2, 21)

        # Synchronize the zoom and pan of the plot widgets
        # self.xplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        # self.yplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        # self.zplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)

    def create_plotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Time (s)")
        plotter.setLabel("left", "Magnet Position (mm)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        return plotter

    def create_controls(self):
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        tolerance_label = QLabel("Tolerance (mm)")
        # give additional info about the tolerance value
        tolerance_label.setToolTip(
            """Tolerance for plateau detection. Smaller values yield more plateaus.
            Tolance is the maximum deviation of consequtive magnet position values to be considered part of the same plateau."""
        )
        control_layout.addWidget(tolerance_label, 0, 0)

        self.tolerance_input = QLineEdit()
        tolerance = self.state_manager.get_state("tolerance")
        self.tolerance_input.setText(str(tolerance))
        control_layout.addWidget(self.tolerance_input, 0, 1)
        self.tolerance_input.textChanged.connect(self.tolerance_changed)

        plateou_label = QLabel("Minimum Plateau Length (s) ")
        # give additional info about the minimum plateau length
        plateou_label.setToolTip(
            """Minimum length of a plateau in seconds. Shorter plateaus are ignored."""
        )
        control_layout.addWidget(plateou_label, 0, 2)

        self.plateau_input = QLineEdit()
        minplateau = self.state_manager.get_state("minplateau")
        fsample = self.state_manager.get_state("fsample")
        self.plateau_input.setText(str(round(minplateau / fsample, 4)))
        control_layout.addWidget(self.plateau_input, 0, 3)
        self.plateau_input.textChanged.connect(self.minplateau_changed)

        return control_group

    def tolerance_changed(self):
        try:
            tolerance = float(self.tolerance_input.text())
            self.state_manager.set_state("tolerance", tolerance)

            minplateau = self.state_manager.get_state("minplateau")
            magpos = self.state_manager.get_state("mag_pos")
            plateaus = engine.identify_plateaus(magpos, tolerance, minplateau)
            self.state_manager.set_state("plateaus", plateaus)
            inclusion = engine.get_bead_inclusion_array(self.state_manager, plateaus)
            self.state_manager.set_state("inclusion", inclusion)

            self.plot_magpos()
        except ValueError:
            print("Invalid value for tolerance")

    def minplateau_changed(self):
        try:
            minplateau = float(self.plateau_input.text())
            fsample = self.state_manager.get_state("fsample")
            self.state_manager.set_state("minplateau", int(minplateau * fsample))
            tolerance = self.state_manager.get_state("tolerance")
            magpos = self.state_manager.get_state("mag_pos")
            plateaus = engine.identify_plateaus(
                magpos, tolerance, int(minplateau * fsample)
            )
            self.state_manager.set_state("plateaus", plateaus)
            inclusion = engine.get_bead_inclusion_array(self.state_manager, plateaus)
            self.state_manager.set_state("inclusion", inclusion)

            self.plot_magpos()
        except ValueError:
            print("Invalid value for minimum plateau length")

    def plot_magpos(self):
        time = self.state_manager.get_state("time")
        magpos = self.state_manager.get_state("mag_pos")
        self.plotter.clear()
        full = self.plotter.plot(
            time,
            magpos,
            pen=pyg.mkPen((0, 0, 0), width=1),
            autoDownsample=True,
            downsample=10,
        )
        full.setDownsampling(auto=True, ds=True)
        plateaus = self.state_manager.get_state("plateaus")
        if plateaus is None:
            tolerance = self.state_manager.get_state("tolerance")
            minplateau = self.state_manager.get_state("minplateau")

            if tolerance is None:
                tolerance = 2.5e-4
                self.state_manager.set_state("tolerance", tolerance)
            if minplateau is None:
                minplateau = 1000
                self.state_manager.set_state("minplateau", minplateau)

            plateaus = engine.identify_plateaus(magpos, tolerance, minplateau)
            self.state_manager.set_state("plateaus", plateaus)
            inclusion = engine.get_bead_inclusion_array(self.state_manager, plateaus)
            self.state_manager.set_state("inclusion", inclusion)

        for pl, pen in zip(plateaus, cycle(engine.pens[1:])):
            partial = self.plotter.plot(
                time[pl], magpos[pl], pen=pen, autoDownsample=True, downsample=10
            )
            partial.setDownsampling(auto=True, ds=True)
