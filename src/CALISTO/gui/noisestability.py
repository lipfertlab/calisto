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
    QApplication,
)
from PySide6.QtCore import QItemSelectionModel, Qt
import pyqtgraph as pyg
import numpy as np

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType

import engines.noisestability_engine as engine
from gui.worker import WorkerManager

checkmark = "\u2713"
crossmark = "\u2717"


class NoiseStabilityWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        if parent is not None:
            self.state_manager.stateChanged.connect(parent.on_state_changed)
        self.plot_items = {"x": {}, "y": {}, "z": {}}
        axis = self.state_manager.get_state("axis")
        
        # Initialize worker manager
        self.worker_manager = WorkerManager(self)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.bead_table = self.create_plateau_table()
        self.bead_table.resizeColumnsToContents()
        self.layout.addWidget(self.bead_table, 0, 0, 21, 21)

        self.plotter = self.create_plotter(axis)
        self.layout.addWidget(self.plotter, 0, 21, 13, 13)

        self.plot_control_group = self.create_plot_controls()
        self.layout.addWidget(self.plot_control_group, 13, 21, 3, 13)

        self.identify_control_group = self.create_identify_controls()
        self.layout.addWidget(self.identify_control_group, 16, 21, 5, 13)

    def create_plotter(self, axis):
        axlabel = ["X", "Y", "Z"]

        xplotter = pyg.PlotWidget()
        xplotter.setLabel("bottom", "Time (s)")

        xplotter.setLabel("left", axlabel[axis] + " Position (nm)")
        xplotter.showGrid(x=True, y=True)
        xplotter.setMouseEnabled(x=True, y=True)
        return xplotter

    def create_plateau_table(self):

        include = self.state_manager.get_state("inclusion")
        table = QTableWidget()
        table.setColumnCount(include.shape[1])

        bead_specs = self.state_manager.get_state("bead_specs")
        bead_type = bead_specs["Type"]
        magnetic = bead_type == BeadType.MAGNETIC
        self.magnetic_idx = np.arange(len(bead_type))[magnetic]
        table.setRowCount(self.magnetic_idx.shape[0])

        table.setVerticalHeaderLabels(
            [str(self.magnetic_idx[i]) for i in range(len(self.magnetic_idx))]
        )

        for i in range(table.rowCount()):
            bead_id = self.magnetic_idx[i]
            for j in range(table.columnCount()):
                item = QCheckBox()
                item.setChecked(include[i, j])
                item.stateChanged.connect(
                    lambda state, bid=bead_id, pid=j: engine.table_cell_changed(
                        bid, pid, state, self.state_manager
                    )
                )
                table.setCellWidget(i, j, item)
        table.cellClicked.connect(self.cellClicked)
        return table

    def cellClicked(self, row, column):
        pid = column
        bead_id = self.magnetic_idx[row]
        self.bead_combo.setCurrentIndex(row)
        self.plateau_combo.setCurrentIndex(column)

    def create_plot_controls(self):
        control_group = QGroupBox("Plot")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        beadlabel = QLabel("Bead:")
        control_layout.addWidget(beadlabel, 0, 0)

        self.bead_combo = QComboBox()
        self.bead_combo.addItems([str(i) for i in self.magnetic_idx])
        control_layout.addWidget(self.bead_combo, 0, 1)
        self.bead_combo.setCurrentIndex(-1)
        self.bead_combo.currentIndexChanged.connect(self.refresh_plot)

        self.plateaulabel = QLabel("Magnet step:")
        control_layout.addWidget(self.plateaulabel, 0, 2)

        self.plateau_combo = QComboBox()
        self.plateau_combo.addItems(
            [str(i) for i in range(self.bead_table.columnCount())]
        )
        control_layout.addWidget(self.plateau_combo, 0, 3)
        self.plateau_combo.setCurrentIndex(-1)
        self.plateau_combo.currentIndexChanged.connect(self.refresh_plot)

        return control_group

    def create_identify_controls(self):
        control_group = QGroupBox("Identify Stable Regions")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        methodlabel = QLabel("Method:")
        control_layout.addWidget(methodlabel, 0, 0)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Hadamard Variance (HV)", "Allan Variance (AV)"])
        control_layout.addWidget(self.method_combo, 0, 1, 1, 2)
        self.method_combo.setCurrentIndex(-1)

        self.identify_button = QPushButton("Identify")
        control_layout.addWidget(self.identify_button, 1, 0, 1, 2)
        self.identify_button.clicked.connect(self.identify_clicked)

        self.select_all_button = QPushButton("Select All")
        control_layout.addWidget(self.select_all_button, 1, 2, 1, 1)
        self.select_all_button.clicked.connect(self.select_all_clicked)

        self.infolabel = QLabel("Number of Stable Regions:")
        control_layout.addWidget(self.infolabel, 2, 0, 1, 2)

        self.stable_regions = QLabel("N/A")
        control_layout.addWidget(self.stable_regions, 2, 1, 1, 1)

        return control_group

    def refresh_plot(self):
        try:
            bid = int(self.bead_combo.currentText())
            pid = int(self.plateau_combo.currentText())
            alpha = self.state_manager.get_state("alphas")[bid, pid]

            # highlight the selected cell
            self.bead_table.blockSignals(True)
            row = int(self.bead_combo.currentText())
            column = pid

            index = self.bead_table.model().index(row, column)
            self.bead_table.selectionModel().select(
                index, QItemSelectionModel.Select | QItemSelectionModel.Current
            )
            self.bead_table.blockSignals(False)
        except:
            return

        bead_pos = self.state_manager.get_state("bead_pos")
        axis = self.state_manager.get_state("axis")
        time = self.state_manager.get_state("time")
        plateaus = self.state_manager.get_state("plateaus")

        bead_pos = bead_pos[plateaus[pid], :, axis]
        time = time[plateaus[pid]]

        bead_specs = self.state_manager.get_state("bead_specs")
        refbead_mask = (bead_specs["Type"] == BeadType.REFERENCE).to_numpy()
        include_mask = bead_specs["Include"].to_numpy()
        refbead_mask = refbead_mask & include_mask
        reftrace = bead_pos[:, refbead_mask].mean(axis=1)

        trace = bead_pos[:, bid] - reftrace

        self.plotter.clear()
        self.plotter.plot(time, trace, pen=pyg.mkPen("k"), name=f"α: {alpha}")
        self.plotter.setRange(
            xRange=[time[0], time[-1]], yRange=[trace.min(), trace.max()]
        )
        # add a legend
        self.plotter.addLegend(labelTextSize="16pt")

    def identify_clicked(self):
        """Start stability identification in background thread."""
        method = self.method_combo.currentText()
        if method == "Hadamard Variance (HV)":
            method = "HV"
        elif method == "Allan Variance (AV)":
            method = "AV"
        else:
            return
        
        # Disable button during computation
        self.identify_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Run computation in background
        self.worker_manager.run_async(
            self._compute_stability,
            method,
            on_result=self._update_stability_results,
            on_error=self._handle_stability_error,
            on_finished=lambda: self.identify_button.setEnabled(True)
        )
    
    def _compute_stability(self, method):
        """Compute trace stability (runs in worker thread)."""
        bead_pos = self.state_manager.get_state("bead_pos")
        axis = self.state_manager.get_state("axis")
        bead_pos = bead_pos[:, :, axis]
        plateaus = self.state_manager.get_state("plateaus")
        include = self.state_manager.get_state("inclusion").copy()
        
        bead_specs = self.state_manager.get_state("bead_specs")
        refbead_mask = (bead_specs["Type"] == BeadType.REFERENCE).to_numpy()
        include_mask = bead_specs["Include"].to_numpy()
        refmask = refbead_mask & include_mask
        alphas = np.ones((len(self.magnetic_idx), len(plateaus))) * np.nan

        for pid, pl in enumerate(plateaus):
            trace_g = bead_pos[pl]
            reftrace = trace_g[:, refmask].mean(axis=1)

            for rid, bid in enumerate(self.magnetic_idx):
                trace = trace_g[:, bid] - reftrace
                stable, alphaint = engine.is_trace_stable(trace, method)
                include[bid, pid] = stable
                alphas[rid, pid] = alphaint
        
        return {'include': include, 'alphas': alphas}
    
    def _update_stability_results(self, result):
        """Update UI with stability results (runs in GUI thread)."""
        include = result['include']
        alphas = result['alphas']
        
        # Update checkboxes
        for pid in range(include.shape[1]):
            for rid, bid in enumerate(self.magnetic_idx):
                checkbox = self.bead_table.cellWidget(rid, pid)
                checkbox.blockSignals(True)
                checkbox.setChecked(include[bid, pid])
                checkbox.blockSignals(False)
        
        self.state_manager.set_state("inclusion", include)
        self.state_manager.set_state("alphas", alphas)
        self.stable_regions.setText(str(include.sum()))
        QApplication.restoreOverrideCursor()
    
    def _handle_stability_error(self, error_msg, traceback):
        """Handle errors during stability computation."""
        print(f"Error computing stability: {error_msg}")
        print(traceback)
        QApplication.restoreOverrideCursor()

    def select_all_clicked(self):
        """
        Reselect all beads in the bead table.
        """
        bead_pos = self.state_manager.get_state("bead_pos")
        axis = self.state_manager.get_state("axis")
        bead_pos = bead_pos[:, :, axis]
        plateaus = self.state_manager.get_state("plateaus")
        include = self.state_manager.get_state("inclusion")

        for pid, pl in enumerate(plateaus):

            for rid, bid in enumerate(self.magnetic_idx):
                include[bid, pid] = True
                checkbox = self.bead_table.cellWidget(rid, pid)
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)

        self.state_manager.set_state("inclusion", include)
        self.stable_regions.setText("N/A")
    
    def closeEvent(self, event):
        """Clean up threads when window is closed."""
        self.worker_manager.cleanup()
        event.accept()
