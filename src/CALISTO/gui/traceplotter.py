# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtWidgets import (
    QLabel,
    QComboBox,
    QPushButton,
    QWidget,
    QGridLayout,
    QGroupBox,
    QTableWidget,
    QCheckBox,
    QSpinBox,
)
from PySide6.QtCore import Qt, Signal, QObject
import pyqtgraph as pyg

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType

import engines.traceplotter_engine as engine

checkmark = "\u2713"
crossmark = "\u2717"


class TracePlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.first = True
        self.parent = parent
        self.state_manager = state_manager
        self.state_manager.stateChanged.connect(parent.on_state_changed)
        self.plot_items = {"x": {}, "y": {}, "z": {}}

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.xplotter = self.create_xplotter()
        self.layout.addWidget(self.xplotter, 0, 0, 2, 8)

        self.yplotter = self.create_yplotter()
        self.layout.addWidget(self.yplotter, 2, 0, 2, 8)

        self.zplotter = self.create_zplotter()
        self.layout.addWidget(self.zplotter, 4, 0, 2, 8)

        self.bead_table = self.create_plotter_bead_table()
        self.bead_table.resizeColumnsToContents()
        self.layout.addWidget(self.bead_table, 0, 8, 5, 4)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group, 5, 8, 1, 4)

        # Synchronize the zoom and pan of the plot widgets
        # self.xplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        # self.yplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)
        # self.zplotter.getViewBox().sigRangeChanged.connect(self.update_plot_ranges)

    def create_xplotter(self):
        xplotter = pyg.PlotWidget()
        xplotter.setLabel("bottom", "Time (s)")
        xplotter.setLabel("left", "X Position (nm)")
        xplotter.showGrid(x=True, y=True)
        xplotter.setMouseEnabled(x=True, y=True)
        return xplotter

    def create_yplotter(self):
        yplotter = pyg.PlotWidget()
        yplotter.setLabel("bottom", "Time (s)")
        yplotter.setLabel("left", "Y Position (nm)")
        yplotter.showGrid(x=True, y=True)
        yplotter.setMouseEnabled(x=True, y=True)
        return yplotter

    def create_zplotter(self):
        zplotter = pyg.PlotWidget()
        zplotter.setLabel("bottom", "Time (s)")
        zplotter.setLabel("left", "Z Position (nm)")
        zplotter.showGrid(x=True, y=True)
        zplotter.setMouseEnabled(x=True, y=True)
        return zplotter

    def create_plotter_bead_table(self):
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Plot", "Type", "Include", "Color"])

        table.setRowCount(self.state_manager.get_state("#beads"))
        bead_specs = self.state_manager.get_state("bead_specs")

        table.setVerticalHeaderLabels([str(i) for i in range(len(bead_specs))])

        # everything in the plot column is a checkbox
        for i in range(table.rowCount()):
            item = QCheckBox()
            item.stateChanged.connect(
                lambda state, id=i: self.plot_state_changed(id, state)
            )
            table.setCellWidget(i, 0, item)

        # everything in the type column is a combobox
        for i in range(len(bead_specs)):
            item = QComboBox()
            item.addItems(["Magnetic", "Reference"])
            item.setCurrentIndex(bead_specs.iloc[i, 0].value)
            item.currentIndexChanged.connect(
                lambda index, row=i, col=1: engine.table_cell_changed(
                    row, col, self.state_manager, table
                )
            )
            table.setCellWidget(i, 1, item)

        for i in range(len(bead_specs)):
            item = QCheckBox()
            item.setChecked(bead_specs.iloc[i, 4])
            item.stateChanged.connect(
                lambda state, row=i, col=2: engine.table_cell_changed(
                    row, col, self.state_manager, table
                )
            )

            table.setCellWidget(i, 2, item)

        # this column is for the color of the bead in the plot, it is not editable, just solid color
        for i in range(len(bead_specs)):
            item = QLabel()
            item.setStyleSheet(
                f"background-color: {engine.pens[i % len(engine.pens)].color().name()}"
            )
            table.setCellWidget(i, 3, item)

        return table

    def create_controls(self):
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all)
        control_layout.addWidget(self.select_all_button, 0, 0, 1, 2)

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.clear_all)
        control_layout.addWidget(self.clear_all_button, 0, 2, 1, 2)

        self.select_magnetic_button = QPushButton("Select Magnetic")
        self.select_magnetic_button.clicked.connect(self.select_magnetic)
        control_layout.addWidget(self.select_magnetic_button, 1, 0, 1, 2)

        self.select_reference_button = QPushButton("Select Reference")
        self.select_reference_button.clicked.connect(self.select_reference)
        control_layout.addWidget(self.select_reference_button, 1, 2, 1, 2)

        return control_group

    def select_all(self):
        for i in range(self.bead_table.rowCount()):
            self.bead_table.cellWidget(i, 0).setChecked(True)

    def clear_all(self):
        for i in range(self.bead_table.rowCount()):
            self.bead_table.cellWidget(i, 0).setChecked(False)

    def select_magnetic(self):
        self.clear_all()
        for i in range(self.bead_table.rowCount()):
            if (
                self.bead_table.cellWidget(i, 1).currentIndex()
                == BeadType.MAGNETIC.value
            ):
                self.bead_table.cellWidget(i, 0).setChecked(True)

    def select_reference(self):
        self.clear_all()
        for i in range(self.bead_table.rowCount()):
            if (
                self.bead_table.cellWidget(i, 1).currentIndex()
                == BeadType.REFERENCE.value
            ):
                self.bead_table.cellWidget(i, 0).setChecked(True)

    def plot_state_changed(self, beadid, state):
        if Qt.CheckState(state) == Qt.Checked:

            for i, (ax, pwidget) in enumerate(
                zip(self.plot_items, [self.xplotter, self.yplotter, self.zplotter])
            ):
                if beadid in self.plot_items[ax]:
                    self.plot_items[ax][beadid].show()
                else:
                    pen = engine.pens[beadid % len(engine.pens)]
                    beadpos = self.state_manager.get_state("bead_pos")
                    time = self.state_manager.get_state("time")
                    r = beadpos[:, beadid, i]

                    self.plot_items[ax][beadid] = pwidget.plot(
                        time, r, pen=pen, autoDownsample=True
                    )
                    self.plot_items[ax][beadid].setDownsampling(ds=True, auto=True)
        else:

            for ax in self.plot_items:
                self.plot_items[ax][beadid].hide()

    def refresh_z_plotter(self):
        self.zplotter.clear()
        beadzpos = self.state_manager.get_state("bead_pos")[:, :, 2]

        time = self.state_manager.get_state("time")
        bead_specs = self.state_manager.get_state("bead_specs")
        for beadid in range(len(bead_specs)):
            if beadid in self.plot_items["z"]:
                visible = self.plot_items["z"][beadid].isVisible()
                self.plot_items["z"][beadid].clear()

                pen = engine.pens[beadid % len(engine.pens)]
                time = self.state_manager.get_state("time")
                r = beadzpos[:, beadid]

                self.plot_items["z"][beadid] = self.zplotter.plot(
                    time, r, pen=pen, autoDownsample=True
                )
                self.plot_items["z"][beadid].setDownsampling(ds=True, auto=True)

                if not visible:
                    self.plot_items["z"][beadid].hide()
