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
)
from PySide6.QtCore import Qt, Signal, QObject
import pyqtgraph as pyg

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType

import engines.refplotter_engine as engine
from engines.refbead_processor import choose_beads

checkmark = "\u2713"
crossmark = "\u2717"


class RefPlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        self.state_manager.stateChanged.connect(parent.on_state_changed)
        self.plot_items = {
            int(True): {"x": {}, "y": {}, "z": {}},
            int(False): {"x": {}, "y": {}, "z": {}},
        }

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        included_label = QLabel("Included Reference Beads")
        self.layout.addWidget(included_label, 0, 0, 1, 8)

        excluded_label = QLabel("Excluded Reference Beads")
        self.layout.addWidget(excluded_label, 0, 8, 1, 8)

        self.xplotter_in = self.create_xplotter()
        self.layout.addWidget(self.xplotter_in, 1, 0, 2, 8)

        self.yplotter_in = self.create_yplotter()
        self.layout.addWidget(self.yplotter_in, 3, 0, 2, 8)

        self.zplotter_in = self.create_zplotter()
        self.layout.addWidget(self.zplotter_in, 5, 0, 2, 8)

        self.xplotter_out = self.create_xplotter()
        self.layout.addWidget(self.xplotter_out, 1, 8, 2, 8)

        self.yplotter_out = self.create_yplotter()
        self.layout.addWidget(self.yplotter_out, 3, 8, 2, 8)

        self.zplotter_out = self.create_zplotter()
        self.layout.addWidget(self.zplotter_out, 5, 8, 2, 8)

        self.bead_table = self.create_plotter_bead_table()
        self.bead_table.resizeColumnsToContents()
        self.layout.addWidget(self.bead_table, 0, 16, 6, 4)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group, 6, 16, 1, 4)

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
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Plot", "Include", "Color"])

        bead_specs = self.state_manager.get_state("bead_specs")
        refmask = bead_specs["Type"] == BeadType.REFERENCE
        self.refids = bead_specs[refmask].index
        table.setRowCount(len(self.refids))
        table.setVerticalHeaderLabels([str(i) for i in self.refids])

        # everything in the plot column is a checkbox
        for i, id in enumerate(self.refids):
            item = QCheckBox()
            item.stateChanged.connect(
                lambda state, id=id: self.plot_state_changed(id, state)
            )
            table.setCellWidget(i, 0, item)

        for i, id in enumerate(self.refids):
            item = QCheckBox()
            item.setChecked(bead_specs.iloc[id, 4])
            item.stateChanged.connect(
                lambda state, row=i: self.include_state_changed(row, state)
            )
            table.setCellWidget(i, 1, item)

        # this column is for the color of the bead in the plot, it is not editable, just solid color
        for i, id in enumerate(self.refids):
            item = QLabel()
            item.setStyleSheet(
                f"background-color: {engine.pens[id % len(engine.pens)].color().name()}"
            )
            table.setCellWidget(i, 2, item)

        return table

    def include_state_changed(self, row, state):
        beadid = self.refids[row]
        plotstate = self.bead_table.cellWidget(row, 0).isChecked()
        if plotstate:
            self.bead_table.cellWidget(row, 0).setChecked(False)

        engine.table_cell_changed(
            row, 1, self.state_manager, self.bead_table, self.refids
        )

        if plotstate:
            self.bead_table.cellWidget(row, 0).setChecked(True)

    def create_controls(self):
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        self.select_all_button = QPushButton("Plot All")
        self.select_all_button.clicked.connect(self.select_all)
        control_layout.addWidget(self.select_all_button, 0, 0, 1, 2)

        self.clear_all_button = QPushButton("Plot None")
        self.clear_all_button.clicked.connect(self.clear_all)
        control_layout.addWidget(self.clear_all_button, 0, 2, 1, 2)

        self.include_all_button = QPushButton("Include All")
        self.include_all_button.clicked.connect(self.include_all)
        control_layout.addWidget(self.include_all_button, 1, 0, 1, 2)

        self.exclude_all_button = QPushButton("Exclude All")
        self.exclude_all_button.clicked.connect(self.exclude_all)
        control_layout.addWidget(self.exclude_all_button, 1, 2, 1, 2)

        self.auto_identify_button = QPushButton("Identify Suitable Reference Beads")
        self.auto_identify_button.clicked.connect(self.identify_reference_beads)
        control_layout.addWidget(self.auto_identify_button, 2, 0, 1, 4)

        return control_group

    def identify_reference_beads(self):
        refids = self.refids
        beadpos = self.state_manager.get_state("bead_pos")
        beadpos = beadpos[:, refids, :]
        # reshape to [bead, time, axis]
        beadpos = beadpos.transpose([1, 0, 2])
        axis = self.state_manager.get_state("axis")
        chosen_mask = choose_beads(beadpos, axis)

        # block signals to avoid triggering the plot_state_changed function

        for i in range(self.bead_table.rowCount()):
            checkbox = self.bead_table.cellWidget(i, 1)
            checkbox.setChecked(chosen_mask[i])
            engine.table_cell_changed(
                i, 1, self.state_manager, self.bead_table, self.refids, agent="algo"
            )
        self.select_all()

    def include_all(self):
        for i in range(self.bead_table.rowCount()):
            checkbox = self.bead_table.cellWidget(i, 1)
            checkbox.setChecked(True)

    def exclude_all(self):
        for i in range(self.bead_table.rowCount()):
            checkbox = self.bead_table.cellWidget(i, 1)
            checkbox.setChecked(False)

    def select_all(self):
        for i in range(self.bead_table.rowCount()):
            self.bead_table.cellWidget(i, 0).setChecked(True)

    def clear_all(self):
        for i in range(self.bead_table.rowCount()):
            self.bead_table.cellWidget(i, 0).setChecked(False)

    def plot_state_changed(self, beadid, state):
        bead_specs = self.state_manager.get_state("bead_specs")
        include = bead_specs["Include"].to_numpy()
        included = include[beadid]
        if included:
            plotters = [self.xplotter_in, self.yplotter_in, self.zplotter_in]
        else:
            plotters = [self.xplotter_out, self.yplotter_out, self.zplotter_out]

        plot_items = self.plot_items[int(included)]

        if Qt.CheckState(state) == Qt.Checked:

            for i, (ax, pwidget) in enumerate(
                zip(
                    plot_items,
                    plotters,
                )
            ):
                if beadid in plot_items[ax]:
                    plot_items[ax][beadid].show()
                else:
                    pen = engine.pens[beadid % len(engine.pens)]
                    beadpos = self.state_manager.get_state("bead_pos")
                    time = self.state_manager.get_state("time")
                    r = beadpos[:, beadid, i]

                    plot_items[ax][beadid] = pwidget.plot(
                        time, r, pen=pen, autoDownsample=True
                    )
                    plot_items[ax][beadid].setDownsampling(ds=True, auto=True)
        else:

            for ax in plot_items:
                plot_items[ax][beadid].hide()
