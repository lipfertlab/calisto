# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtWidgets import (
    QLabel,
    QComboBox,
    QPushButton,
    QWidget,
    QGridLayout,
    QGroupBox,
    QRadioButton,
)
from PySide6.QtCore import Qt, Signal, QObject
import pyqtgraph as pyg
import numpy as np
import matplotlib.cm as cm

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
from itertools import cycle
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType, TestResult, TetherType  # AttachmentType

import engines.rotation_engine as engine
from PySide6.QtWidgets import QSizePolicy

checkmark = "\u2713"
crossmark = "\u2717"


class RotationPlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        if parent is not None:
            self.state_manager.stateChanged.connect(parent.on_state_changed)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Create plot widgets with appropriate size policies
        self.magrotplotter = self.create_mag_rot_plotter()
        self.magrotplotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.magposplotter = self.create_mag_pos_plotter()
        self.magposplotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.beadzplotter = self.create_bead_z_plotter()
        self.beadzplotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.beadxyplotter = self.create_bead_xy_plotter()
        self.beadxyplotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.control_group = self.create_controls()
        self.control_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Add widgets to layout
        self.layout.addWidget(self.magrotplotter, 0, 0)
        self.layout.addWidget(self.magposplotter, 0, 1)
        self.layout.addWidget(self.beadzplotter, 1, 0)
        self.layout.addWidget(self.beadxyplotter, 1, 1)
        self.layout.addWidget(self.control_group, 2, 0, 1, 2)

        # Set row and column stretches to maintain proportions
        # Rows: top:middle:bottom = 8:13:2
        self.layout.setRowStretch(0, 8)
        self.layout.setRowStretch(1, 13)
        self.layout.setRowStretch(2, 2)

        # Columns: left:right = 21:13
        self.layout.setColumnStretch(0, 21)
        self.layout.setColumnStretch(1, 13)

        # Synchronize the zoom and pan of the plot widgets
        # self.xplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        # self.yplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        # self.zplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        self.magposplotter.setXLink(self.beadzplotter)
        self.magrotplotter.setXLink(self.beadzplotter)
        self.beadzplotter.getViewBox().sigRangeChanged.connect(
            self.update_xyplot_time_range
        )

    def create_mag_pos_plotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("left", "Magnet Position (mm)")
        plotter.setLabel("bottom", "Time (s)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        mag_time = self.state_manager.get_state("rot_mag_time")
        mag_pos = self.state_manager.get_state("rot_mag_pos")
        plotter.plot(mag_time, mag_pos, pen=pyg.mkPen(color="b", width=3))

        return plotter

    def create_mag_rot_plotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Time (s)")
        plotter.setLabel("left", "Rotation (turns)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        mag_rot = self.state_manager.get_state("rot_mag_rot")
        mag_time = self.state_manager.get_state("rot_mag_time")
        plotter.plot(mag_time, mag_rot, pen=pyg.mkPen(color="b", width=3))

        return plotter

    def create_bead_z_plotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Time (s)")
        plotter.setLabel("left", "Z Position (nm)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        return plotter

    def create_bead_xy_plotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "X Position (nm)")
        plotter.setLabel("left", "Y Position (nm)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        return plotter

    def create_controls(self):
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        bead_label = QLabel("Viewing Magnetic Bead ")
        control_layout.addWidget(bead_label, 0, 0)

        self.bead_combo = QComboBox()
        magbeads = self.get_magnetic_beads()
        self.bead_combo.addItems(magbeads)
        control_layout.addWidget(self.bead_combo, 0, 1)
        self.bead_combo.currentIndexChanged.connect(self.bead_id_changed)

        tether_type_label = QLabel("Tether Type: ")
        control_layout.addWidget(tether_type_label, 0, 2)

        self.tether_type_combo = QComboBox()
        self.tether_type_combo.addItems(["Nicked", "Supercoilable", "Double tethered"])
        control_layout.addWidget(self.tether_type_combo, 0, 3)
        self.tether_type_combo.currentIndexChanged.connect(self.bead_type_changed)

        self.include_radiobutton = QRadioButton("Include in the calibration")
        self.include_radiobutton.clicked.connect(self.include_bead)
        control_layout.addWidget(self.include_radiobutton, 1, 0)

        self.exclude_radiobutton = QRadioButton("Exclude from the calibration")
        self.exclude_radiobutton.clicked.connect(self.exclude_bead)
        control_layout.addWidget(self.exclude_radiobutton, 1, 1)

        self.previous_button = QPushButton("Previous")
        control_layout.addWidget(self.previous_button, 1, 2)
        self.previous_button.clicked.connect(self.previous_bead)
        self.previous_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        control_layout.addWidget(self.next_button, 1, 3)
        self.next_button.clicked.connect(self.next_bead)

        self.bead_id_changed()

        return control_group

    def include_bead(self):
        bead_id = int(self.bead_combo.currentText())
        bead_specs = self.state_manager.get_state("bead_specs")
        bead_specs.loc[bead_id, "Include"] = True
        bead_specs.loc[bead_id, "Test Result"] = TestResult.INCLUDED_BY_USER_ROTATION
        self.state_manager.set_state("bead_specs", bead_specs)

    def exclude_bead(self):
        bead_id = int(self.bead_combo.currentText())
        bead_specs = self.state_manager.get_state("bead_specs")
        bead_specs.loc[bead_id, "Include"] = False
        bead_specs.loc[bead_id, "Test Result"] = TestResult.EXCLUDED_BY_USER_ROTATION
        self.state_manager.set_state("bead_specs", bead_specs)

    def bead_type_changed(self):
        bead_id = int(self.bead_combo.currentText())
        bead_specs = self.state_manager.get_state("bead_specs")
        bead_specs.iloc[bead_id, 1] = TetherType(self.tether_type_combo.currentIndex())
        self.state_manager.set_state("bead_specs", bead_specs)

    def get_magnetic_beads(self):
        bead_specs = self.state_manager.get_state("bead_specs")
        magbeads = []
        for i in range(len(bead_specs)):
            if bead_specs.iloc[i, 0] == BeadType.MAGNETIC:
                magbeads.append(str(i))
        return np.array(magbeads)

    def bead_id_changed(self):
        bead_id = int(self.bead_combo.currentText())
        bead_specs = self.state_manager.get_state("bead_specs")
        include = bead_specs.loc[bead_id, "Include"]
        if include:
            self.include_radiobutton.setChecked(True)
            self.exclude_radiobutton.setChecked(False)
        else:
            self.include_radiobutton.setChecked(False)
            self.exclude_radiobutton.setChecked(True)

        if self.bead_combo.currentIndex() == 0:
            self.previous_button.setEnabled(False)
        else:
            self.previous_button.setEnabled(True)
        if self.bead_combo.currentIndex() == self.bead_combo.count() - 1:
            self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)

        # update the tether type combo box
        tether_type = bead_specs.loc[bead_id, "Tether Type"]
        # first disconnect the signal to avoid triggering bead_type_changed when setting the index
        self.tether_type_combo.currentIndexChanged.disconnect(self.bead_type_changed)
        self.tether_type_combo.setCurrentIndex(tether_type.value)
        self.tether_type_combo.currentIndexChanged.connect(self.bead_type_changed)

        self.plot_bead(bead_id)

    def plot_bead(self, id):
        bead_data = self.state_manager.get_state("rot_bead_pos")
        bead_time = self.state_manager.get_state("rot_time")
        bead = bead_data[:, id, :]
        self.beadzplotter.clear()
        self.beadzplotter.plot(bead_time, bead[:, 2], pen=pyg.mkPen(color="b", width=3))
        self.beadxyplotter.clear()

        self.update_xyplot_time_range()

    def previous_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() - 1)

    def next_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() + 1)

    def update_xyplot_time_range(self):
        # get time range from zplotter
        t_range = self.beadzplotter.getViewBox().viewRange()[0]
        # get the bead id
        bead_id = int(self.bead_combo.currentText())
        # get the bead data
        bead_data = self.state_manager.get_state("rot_bead_pos")
        bead_time = self.state_manager.get_state("rot_time")
        bead = bead_data[:, bead_id, :]
        # get the indices of the time range
        indices = np.where(
            np.logical_and(bead_time >= t_range[0], bead_time <= t_range[1])
        )[0]
        # plot the data
        # Extract the visible data points and their times
        x_data = bead[indices, 0]
        y_data = bead[indices, 1]
        times = bead_time[indices]

        # Clear previous plot
        self.beadxyplotter.clear()

        # Create a single ScatterPlotItem
        if len(times) > 0:
            # Normalize time values to [0, 1] for color mapping
            norm_times = (
                (times - times.min()) / (times.max() - times.min())
                if times.max() > times.min()
                else np.zeros_like(times)
            )

            # Create an array of brushes using viridis colormap
            colors = cm.viridis(norm_times)  # This returns RGBA values between 0-1
            brushes = [
                pyg.mkBrush(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
                for r, g, b, a in colors
            ]
            # Create spots data structure (much more efficient)
            spots = [
                {"pos": (x, y), "brush": brush, "size": 5}
                for x, y, brush in zip(x_data, y_data, brushes)
            ]

            # Create and add scatter plot with all points at once
            scatter = pyg.ScatterPlotItem(spots=spots, pxMode=True)
            self.beadxyplotter.addItem(scatter)

            # set the range of the xy plotter
            xmid = (bead[indices, 0].min() + bead[indices, 0].max()) / 2
            ymid = (bead[indices, 1].min() + bead[indices, 1].max()) / 2
            xwidth = bead[indices, 0].max() - bead[indices, 0].min()
            ywidth = bead[indices, 1].max() - bead[indices, 1].min()
            width = max(xwidth, ywidth)
            self.beadxyplotter.setXRange(xmid - width / 2, xmid + width / 2)
            self.beadxyplotter.setYRange(ymid - width / 2, ymid + width / 2)

            # Add a simple text legend to show color mapping
            legend_text = pyg.TextItem(
                text="Purple → Yellow = Early → Late time (viridis)", color=(0, 0, 0)
            )
            legend_text.setPos(xmid - width / 2, ymid - width / 2)
            self.beadxyplotter.addItem(legend_text)

            # set the axis scaling to be equal
            self.beadxyplotter.getPlotItem().setAspectLocked(True)
