# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtWidgets import (
    QLabel,
    QComboBox,
    QPushButton,
    QWidget,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QApplication,
)
from PySide6.QtCore import Qt

import pyqtgraph as pyg
import numpy as np


pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


from pathlib import Path
from itertools import cycle
from scipy.stats import kstest
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType

import engines.calibration_engine as engine
import engines.fit_engine as fit
from gui.worker import WorkerManager

checkmark = "\u2713"
crossmark = "\u2717"


class MasterCurvePlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        if parent is not None:
            self.state_manager.stateChanged.connect(parent.on_state_changed)

        # Initialize worker manager
        self.worker_manager = WorkerManager(self)
        self.measurements = None

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.fcplotter = self.create_fcplotter()
        self.layout.addWidget(self.fcplotter)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group)
        # self.zrange = self.plotter.getAxis("left").range
        self.force_ub = None
        self.fitparemeters = {}
        
        # Prepare measurements asynchronously
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.worker_manager.run_async(
            engine.prepare_multibeadmeasurement,
            state_manager,
            on_result=self._measurements_ready,
            on_error=self._handle_measurement_error
        )
    
    def _measurements_ready(self, measurements):
        """Called when measurements are ready (runs in GUI thread)."""
        self.measurements = measurements
        self.plot_curves()
        QApplication.restoreOverrideCursor()
    
    def _handle_measurement_error(self, error_msg, traceback):
        """Handle errors during measurement preparation."""
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "Error", f"Error preparing measurements: {error_msg}")
        print(traceback)

    def create_fcplotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Magnet Position (mm)")
        plotter.setLabel("left", "Force (pN)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        return plotter

    def create_controls(self):
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        self.fituntillabel = QLabel("Fit until (pN)")
        control_layout.addWidget(self.fituntillabel, 0, 0, 1, 1)
        self.fituntillineedit = QLineEdit()
        control_layout.addWidget(self.fituntillineedit, 0, 1, 1, 2)
        self.fituntilbutton = QPushButton("Apply")
        self.fituntilbutton.clicked.connect(self.set_fit_upperbound)
        control_layout.addWidget(self.fituntilbutton, 0, 3, 1, 1)

        self.modelcombo = QComboBox()
        self.modelcombo.addItems(["Single Exponential", "Double Exponential"])
        # set index to 1
        self.modelcombo.setCurrentIndex(1)
        self.state_manager.set_state("master_curve_model", "Double Exponential")
        control_layout.addWidget(self.modelcombo, 1, 0, 1, 4)
        # connect to onchange
        self.modelcombo.currentIndexChanged.connect(self.on_model_changed)
        self.double_exponential_label_text = (
            "F(z) = F<sub>max</sub> (c e<sup>-z/Λ</sup> + (1-c) e<sup>-z/λ</sup>)"
        )
        self.single_exponential_label_text = "F(z) = F<sub>max</sub> e<sup>-z/Λ</sup>"
        self.modellabel = QLabel(self.double_exponential_label_text)
        control_layout.addWidget(self.modellabel, 2, 0, 1, 5)

        hvlabel = QLabel("HV")
        control_layout.addWidget(hvlabel, 3, 1)

        avlabel = QLabel("AV")
        control_layout.addWidget(avlabel, 3, 2)

        psdlabel = QLabel("PSD")
        control_layout.addWidget(psdlabel, 3, 3)

        fmaxlabel = QLabel("F<sub>max</sub> (pN)")
        control_layout.addWidget(fmaxlabel, 4, 0)

        l1label = QLabel("Λ (mm)")
        control_layout.addWidget(l1label, 5, 0)

        l2label = QLabel("λ (mm)")
        control_layout.addWidget(l2label, 6, 0)

        clabel = QLabel("c")
        control_layout.addWidget(clabel, 7, 0)

        goodensslabel = QLabel("AICc")
        control_layout.addWidget(goodensslabel, 8, 0)

        importbutton = QPushButton("Import Force Calibration Data")
        control_layout.addWidget(importbutton, 9, 0, 1, 5)
        importbutton.clicked.connect(self.import_calibration)

        methodlabel = QLabel("Export Method")
        control_layout.addWidget(methodlabel, 11, 0)

        self.methodcombo = QComboBox()
        self.methodcombo.addItems(["HV", "AV", "PSD"])
        control_layout.addWidget(self.methodcombo, 11, 1, 1, 4)

        exportbutton = QPushButton("Export Force Calibration")
        control_layout.addWidget(exportbutton, 12, 0, 1, 5)
        exportbutton.clicked.connect(self.export_calibration)

        self.hvfmax = QLabel()
        control_layout.addWidget(self.hvfmax, 4, 1)

        self.hvl1 = QLabel()
        control_layout.addWidget(self.hvl1, 5, 1)

        self.hvl2 = QLabel()
        control_layout.addWidget(self.hvl2, 6, 1)

        self.hvc = QLabel()
        control_layout.addWidget(self.hvc, 7, 1)

        self.hvgoodness = QLabel()
        control_layout.addWidget(self.hvgoodness, 8, 1)

        self.avfmax = QLabel()
        control_layout.addWidget(self.avfmax, 4, 2)

        self.avl1 = QLabel()
        control_layout.addWidget(self.avl1, 5, 2)

        self.avl2 = QLabel()
        control_layout.addWidget(self.avl2, 6, 2)

        self.avc = QLabel()
        control_layout.addWidget(self.avc, 7, 2)

        self.avgoodness = QLabel()
        control_layout.addWidget(self.avgoodness, 8, 2)

        self.psdfmax = QLabel()
        control_layout.addWidget(self.psdfmax, 4, 3)

        self.psdl1 = QLabel()
        control_layout.addWidget(self.psdl1, 5, 3)

        self.psdl2 = QLabel()
        control_layout.addWidget(self.psdl2, 6, 3)

        self.psdc = QLabel()
        control_layout.addWidget(self.psdc, 7, 3)

        self.psdgoodness = QLabel()
        control_layout.addWidget(self.psdgoodness, 8, 3)

        self.fitparemeterslabel = {  # method: [fmax, l1, l2, c, goodness]
            "HV": [self.hvfmax, self.hvl1, self.hvl2, self.hvc, self.hvgoodness],
            "AV": [self.avfmax, self.avl1, self.avl2, self.avc, self.avgoodness],
            "PSD": [self.psdfmax, self.psdl1, self.psdl2, self.psdc, self.psdgoodness],
        }

        return control_group

    def get_magnetic_beads(self):
        bead_specs = self.state_manager.get_state("bead_specs")
        magbeads = []
        for i in range(len(bead_specs)):
            if bead_specs.iloc[i, 0] == BeadType.MAGNETIC:
                magbeads.append(str(i + 1))
        return np.array(magbeads)

    def plot_curves(self):
        # Guard: Don't plot if measurements aren't ready yet
        if self.measurements is None:
            return
        
        colors = {"PSD": "g", "AV": "r", "HV": "b"}
        fullmagpos, fullforces = engine.get_all_forces_v_magpos(self.state_manager)

        self.fcplotter.clear()
        for idx, method in enumerate(["PSD", "AV", "HV"]):
            errorbar = pyg.ErrorBarItem(
                x=fullmagpos,
                y=fullforces[method][:, 0],
                top=fullforces[method][:, 1],
                bottom=fullforces[method][:, 1],
                beam=0.1,
                pen=pyg.mkPen(color=colors[method], width=5),
            )
            self.fcplotter.plot(
                fullmagpos,
                fullforces[method][:, 0],
                symbol="o",
                pen=None,
                symbolPen=pyg.mkPen(color=colors[method], width=1),
                symbolBrush=pyg.mkBrush(color=colors[method]),
                name=method,
            )
            self.fcplotter.addItem(errorbar)
            fitmagpos = np.linspace(fullmagpos.min(), fullmagpos.max(), 300)
            try:
                if self.force_ub is None:
                    mask = np.ones(fullforces[method].shape[0], dtype=bool)
                else:
                    mask = fullforces[method][:, 0] < self.force_ub

                if (
                    self.state_manager.get_state("master_curve_model")
                    == "Double Exponential"
                ):
                    dexpfit = fit.fit_double_exp_multiplicative(
                        fullmagpos[mask], fullforces[method][mask, 0]
                    )
                    fmax = dexpfit["Fmax"]
                    l1 = dexpfit["tau_fast"]
                    l2 = dexpfit["tau_slow"]
                    c = dexpfit["c"]

                    fitforces = engine.two_term_exp(fitmagpos, fmax, l1, l2, c)
                    expforces = engine.two_term_exp(fullmagpos[mask], fmax, l1, l2, c)
                    dof = 4

                    info = fit.info_criteria(
                        fullforces[method][mask, 0], expforces, dof
                    )
                    aicc = info["AICc"]

                    self.fitparemeters[method] = [
                        fmax,
                        l1,
                        l2,
                        c,
                        aicc,
                    ]

                    self.fitparemeterslabel[method][0].setText(str(round(fmax, 3)))
                    self.fitparemeterslabel[method][1].setText(str(round(l1, 3)))
                    self.fitparemeterslabel[method][2].setText(str(round(l2, 3)))
                    self.fitparemeterslabel[method][3].setText(str(round(c, 3)))
                    self.fitparemeterslabel[method][4].setText(f"{aicc:.2E}")

                if (
                    self.state_manager.get_state("master_curve_model")
                    == "Single Exponential"
                ):
                    sexpfit = fit.fit_single_exp_multiplicative(
                        fullmagpos[mask], fullforces[method][mask, 0]
                    )
                    fmax = sexpfit["A"]
                    l1 = sexpfit["tau"]

                    dof = 2
                    fitforces = engine.single_exp(fitmagpos, fmax, l1)
                    expforces = engine.single_exp(fullmagpos[mask], fmax, l1)

                    info = fit.info_criteria(
                        fullforces[method][mask, 0], expforces, dof
                    )
                    aicc = info["AICc"]

                    self.fitparemeters[method] = [
                        fmax,
                        l1,
                        aicc,
                    ]

                    self.fitparemeterslabel[method][0].setText(str(round(fmax, 3)))
                    self.fitparemeterslabel[method][1].setText(str(round(l1, 3)))
                    self.fitparemeterslabel[method][2].setText("N/A")
                    self.fitparemeterslabel[method][3].setText("N/A")
                    self.fitparemeterslabel[method][4].setText(str(f"{aicc:.2E}"))

                self.fcplotter.plot(
                    fitmagpos,
                    fitforces,
                    pen=pyg.mkPen(color=colors[method], width=5),
                    name=f"{method} fit",
                )

            except:
                self.fitparemeterslabel[method][0].setText("N/A")
                self.fitparemeterslabel[method][2].setText("N/A")
                self.fitparemeterslabel[method][1].setText("N/A")
                self.fitparemeterslabel[method][3].setText("N/A")
                self.fitparemeterslabel[method][4].setText("N/A")

            # plot legend
            self.state_manager.set_state("master_curve_params", self.fitparemeters)
            self.fcplotter.addLegend()

    def export_calibration(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Force Calibration",
            "",
            "YAML Files (*.yaml);;All Files (*)",
            options=QFileDialog.Options(),
        )
        method = self.methodcombo.currentText()
        if path:
            path = Path(path)
            try:
                engine.export_calibration(method, path, self.state_manager)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def import_calibration(self):
        """Import force calibration data from a pickle file(s)."""

        # the user can chose multiple pickle files
        path, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Force Calibration Data",
            "",
            "Pickle Files (*.pkl)",
            options=QFileDialog.Options(),
        )
        if not path:
            return
        path = [Path(p) for p in path]
        try:
            for p in path:
                engine.load_force_calibration_data(p, self.state_manager)
            self.plot_curves()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_model_changed(self, index):
        model = self.modelcombo.itemText(index)
        if index == 0:
            model = "Single Exponential"
            self.modellabel.setText(self.single_exponential_label_text)
        elif index == 1:
            model = "Double Exponential"
            self.modellabel.setText(self.double_exponential_label_text)

        self.state_manager.set_state("master_curve_model", model)
        self.plot_curves()

    def closeEvent(self, event):
        """
        Override the closeEvent to handle window closing.
        """
        self.worker_manager.cleanup()
        engine.clear_external_force_calibration_data(self.state_manager)
        event.accept()

    def set_fit_upperbound(self):
        try:
            ub = float(self.fituntillineedit.text())
            if ub < 0:
                raise ValueError
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Value has to be a positive real number"
            )
            return
        self.force_ub = ub
        self.plot_curves()
