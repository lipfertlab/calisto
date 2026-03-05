# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtWidgets import (
    QLabel,
    QComboBox,
    QLineEdit,
    QSlider,
    QPushButton,
    QWidget,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QObject
import pyqtgraph as pyg
import numpy as np

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
from itertools import cycle
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType

import engines.offset_engine as engine

checkmark = "\u2713"
crossmark = "\u2717"


class OffsetPlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        if parent is not None:
            self.state_manager.stateChanged.connect(parent.on_state_changed)

        engine.process_raw_data(self.state_manager)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.plotter = self.create_plotter()

        self.layout.addWidget(self.plotter)
        self.zrange = self.plotter.getAxis("left").range

        self.offset_slider = QSlider(Qt.Vertical)
        self.offset_slider_step_count = 1000
        self.offset_slider.setRange(0, self.offset_slider_step_count)

        self.slider_interaction = False
        self.layout.addWidget(self.offset_slider)
        self.offset_slider.sliderPressed.connect(
            lambda: setattr(self, "slider_interaction", True)
        )
        self.offset_slider.sliderReleased.connect(self.slider_release)
        self.offset_slider.valueChanged.connect(self.offset_changed)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group)

    def closeEvent(self, event):
        # pop up a message box to ask if the user wants to save the current offsets
        # if yes, make user select a file to save the offsets
        # if no, just close the window
        reply = QMessageBox.question(
            self,
            "Close Offset Plotter",
            "Do you want to save the current offsets?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            # ask user where to save the offsets
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Offsets",
                "",
                "Text Files (*.txt);;All Files (*)",
                options=options,
            )
            if filename:
                bead_specs = self.state_manager.get_state("bead_specs")
                offsets = bead_specs.iloc[:, 2].to_numpy()
                offsets = offsets.reshape(-1, 1)
                ids = np.arange(len(offsets)).reshape(-1, 1)
                offsets = np.hstack((ids, offsets))
                # save the offsets to a text file
                # the first column is the bead id, the second column is the offset
                np.savetxt(
                    filename, offsets, header="Bead ID, Offset (nm)", fmt="%d %.6f"
                )

        event.accept()

    def create_plotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Time (s)")
        plotter.setLabel("left", "Z Position (nm)")
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

        offset_method_label = QLabel("Offset Identification Method")
        control_layout.addWidget(offset_method_label, 1, 0)

        self.offset_method_combo = QComboBox()
        self.offset_method_combo.addItems(["Minimum", "GMM", "Manual"])
        self.offset_method_combo.setCurrentIndex(1)
        control_layout.addWidget(self.offset_method_combo, 1, 1)
        self.offset_method_combo.currentIndexChanged.connect(self.method_changed)

        offset_label = QLabel("Offset (nm)")
        control_layout.addWidget(offset_label, 2, 0)

        self.offset_input = QLineEdit()
        control_layout.addWidget(self.offset_input, 2, 1)
        self.offset_input.textChanged.connect(self.offset_changed)

        self.previous_button = QPushButton("Previous")
        control_layout.addWidget(self.previous_button, 3, 0)
        self.previous_button.clicked.connect(self.previous_bead)
        self.previous_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        control_layout.addWidget(self.next_button, 3, 1)
        self.next_button.clicked.connect(self.next_bead)

        self.bead_id_changed()

        return control_group

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
        offset = bead_specs.iloc[bead_id, 2]
        # check if offset is a number
        if not np.isnan(offset):
            self.offset_input.setText(str(round(offset, 6)))
            self.offset_method_combo.setCurrentIndex(2)

        if self.bead_combo.currentIndex() == 0:
            self.previous_button.setEnabled(False)
        else:
            self.previous_button.setEnabled(True)
        if self.bead_combo.currentIndex() == self.bead_combo.count() - 1:
            self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)

        self.plot_bead(bead_id)
        self.method_changed()  # update offset

    def plot_bead(self, id):
        time = self.state_manager.get_state("offset_time")
        traces = self.state_manager.get_state("offset_traces")
        self.plotter.clear()
        plt = self.plotter.plot(
            time, traces[:, id], pen=pyg.mkPen((0, 0, 0), width=3), autoDownsample=True
        )
        plt.setDownsampling(auto=True, ds=True)
        self.zrange = self.plotter.getAxis("left").range
        self.offset_slider.setValue(0)

    def get_slider_value(self):
        offset_int = self.offset_slider.value()

        zmin = self.zrange[0]
        zmax = self.zrange[1]
        dz = (zmax - zmin) / self.offset_slider_step_count
        offset = zmin + offset_int * dz

        return offset

    def set_slider_value(self, offset):
        zmin = self.zrange[0]
        zmax = self.zrange[1]
        dz = (zmax - zmin) / self.offset_slider_step_count
        offset_int = int((offset - zmin) / dz)
        self.offset_slider.setValue(offset_int)

    def offset_changed(self):
        if self.slider_interaction:
            offset = self.get_slider_value()

            self.offset_input.setText(str(round(offset, 6)))
            self.offset_method_combo.setCurrentIndex(2)

            # plot a horizontal infinite line at the offset
            if hasattr(self, "manualline"):
                self.plotter.removeItem(self.manualline)
            self.manualline = self.plotter.addLine(
                y=offset, pen=pyg.mkPen("r", width=5), name="offset"
            )
            self.manualline.setZValue(10)
        else:
            offset = float(self.offset_input.text())
            beadid = int(self.bead_combo.currentText())

            bead_specs = self.state_manager.get_state("bead_specs")
            bead_specs.iloc[beadid, 2] = offset
            self.state_manager.set_state("bead_specs", bead_specs)

        pass

    def previous_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() - 1)

    def next_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() + 1)

    def slider_release(self):
        self.slider_interaction = False

        offset = float(self.offset_input.text())
        beadid = int(self.bead_combo.currentText())

        bead_specs = self.state_manager.get_state("bead_specs")
        bead_specs.iloc[beadid, 2] = offset
        self.state_manager.set_state("bead_specs", bead_specs)

    def method_changed(self):
        method = self.offset_method_combo.currentText()

        if method == "Manual":
            offset = float(self.offset_input.text())
            self.set_slider_value(offset)
            self.offset_changed()
            if hasattr(self, "manualline"):
                self.plotter.removeItem(self.manualline)
            self.manualline = self.plotter.addLine(
                y=offset, pen=pyg.mkPen("r", width=5), name="offset"
            )
            return

        offset = None

        id = int(self.bead_combo.currentText())
        traces = self.state_manager.get_state("offset_traces")
        trace = traces[:, id]

        if method == "Minimum":
            offset = engine.get_minimum_offset(trace)
        elif method == "GMM":
            offset = engine.get_gmm_offset(trace)
        else:
            raise ValueError("Invalid offset method")

        self.offset_input.setText(str(round(offset, 6)))
        self.offset_changed()

        if hasattr(self, "manualline"):
            self.plotter.removeItem(self.manualline)
        self.manualline = self.plotter.addLine(
            y=offset, pen=pyg.mkPen("r", width=5), name="offset"
        )
        self.manualline.setZValue(10)
        self.set_slider_value(offset)
