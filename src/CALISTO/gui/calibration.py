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

import pyqtgraph as pyg
import numpy as np

from pathlib import Path
from itertools import cycle
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType, TestResult

import engines.calibration_engine as engine
from engines.fit_engine import fit_wlc_multiplicative, fit_double_exp_multiplicative

checkmark = "\u2713"
crossmark = "\u2717"

pyg.setConfigOption("background", "w")
pyg.setConfigOption("foreground", "k")


class CalibrationPlotterWindow(QWidget):
    def __init__(self, parent, state_manager):
        super().__init__()
        self.parent = parent
        self.state_manager = state_manager
        if parent is not None:
            self.state_manager.stateChanged.connect(parent.on_state_changed)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.fcplotter = self.create_fcplotter()
        self.layout.addWidget(self.fcplotter, 0, 0, 21, 21)

        self.wlcplotter = self.create_wlcplotter()
        self.layout.addWidget(self.wlcplotter, 0, 21, 13, 13)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group, 13, 21, 8, 13)

    def create_fcplotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Magnet Position (nm)")
        plotter.setLabel("left", "Force (pN)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        return plotter

    def create_wlcplotter(self):
        plotter = pyg.PlotWidget()
        plotter.setLabel("bottom", "Extension (nm)")
        plotter.setLabel("left", "Force (pN)")
        plotter.showGrid(x=True, y=True)
        plotter.setMouseEnabled(x=True, y=True)

        return plotter

    def create_controls(self):
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        bead_label = QLabel("Viewing Magnetic Bead ")
        control_layout.addWidget(bead_label, 0, 0, 1, 2)

        self.bead_combo = QComboBox()
        magbeads = self.get_magnetic_beads()
        self.bead_combo.addItems(magbeads)
        control_layout.addWidget(self.bead_combo, 0, 1)
        self.bead_combo.currentIndexChanged.connect(self.bead_id_changed)

        self.include_radiobutton = QRadioButton("Include in the calibration")
        self.include_radiobutton.clicked.connect(self.include_bead)
        control_layout.addWidget(self.include_radiobutton, 1, 0)

        self.exclude_radiobutton = QRadioButton("Exclude from the calibration")
        self.exclude_radiobutton.clicked.connect(self.exclude_bead)
        control_layout.addWidget(self.exclude_radiobutton, 1, 1)

        self.previous_button = QPushButton("Previous")
        control_layout.addWidget(self.previous_button, 2, 0)
        self.previous_button.clicked.connect(self.previous_bead)
        self.previous_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        control_layout.addWidget(self.next_button, 2, 1)
        self.next_button.clicked.connect(self.next_bead)

        self.bead_id_changed()

        return control_group

    def include_bead(self):
        bead_id = int(self.bead_combo.currentText())
        bead_specs = self.state_manager.get_state("bead_specs")
        bead_specs.loc[bead_id, "Include"] = True
        bead_specs.loc[bead_id, "Test Result"] = TestResult.INCLUDED_BY_USER_CALIBRATION
        self.state_manager.set_state("bead_specs", bead_specs)

    def exclude_bead(self):
        bead_id = int(self.bead_combo.currentText())
        bead_specs = self.state_manager.get_state("bead_specs")
        bead_specs.loc[bead_id, "Include"] = False
        bead_specs.loc[bead_id, "Test Result"] = TestResult.EXCLUDED_BY_USER_CALIBRATION
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

        self.plot_bead(bead_id)

    def plot_bead(self, id):
        measurements = engine.prepare_multibeadmeasurement(self.state_manager)
        include = self.state_manager.get_state("inclusion")
        colors = {"PSD": "b", "AV": "g", "HV": "r"}
        fullmagpos = []
        extensions = []
        fullforces = {"PSD": [], "AV": [], "HV": []}
        for pid, m in enumerate(measurements):
            if not include[id, pid]:
                continue

            fullmagpos.append(m.mag_pos)
            m = m[id]
            extensions.append(m.get_extension()["mean"])
            for method in ["PSD", "AV", "HV"]:
                force = m.get_force(method)
                fullforces[method].append(force)

        fullmagpos = np.array(fullmagpos).flatten()
        extensions = np.array(extensions).flatten()
        self.fcplotter.clear()
        self.wlcplotter.clear()
        for idx, method in enumerate(["PSD", "AV", "HV"]):
            fullforces[method] = np.vstack(fullforces[method])
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

            dexpfit = fit_double_exp_multiplicative(
                fullmagpos, fullforces[method][:, 0]
            )
            fmax = dexpfit["Fmax"]
            l1 = dexpfit["tau_fast"]
            l2 = dexpfit["tau_slow"]
            c = dexpfit["c"]
            fitmagpos = np.linspace(fullmagpos.min(), fullmagpos.max(), 300)
            fitforces = engine.two_term_exp(fitmagpos, fmax, l1, l2, c)
            fmax = round(fmax, 3)
            l1 = round(l1, 3)
            l2 = round(l2, 3)

            self.fcplotter.plot(
                fitmagpos,
                fitforces,
                pen=pyg.mkPen(color=colors[method], width=2),
                name=f"{method} fit: F<sub>max</sub>={fmax} pN,\n     l<sub>1</sub>={l1} nm\n     l<sub>2</sub>={l2} nm",
            )

            self.wlcplotter.plot(
                extensions,
                fullforces[method][:, 0],
                symbol="o",
                pen=None,
                symbolPen=pyg.mkPen(color=colors[method], width=1),
                symbolBrush=pyg.mkBrush(color=colors[method]),
                name=method,
            )
            errorbar2 = pyg.ErrorBarItem(
                x=extensions,
                y=fullforces[method][:, 0],
                top=fullforces[method][:, 1],
                bottom=fullforces[method][:, 1],
                beam=0.1,
                pen=pyg.mkPen(color=colors[method], width=5),
            )
            self.wlcplotter.addItem(errorbar2)
            temp = self.state_manager.get_state("temperature")
            try:
                wlcfit = fit_wlc_multiplicative(
                    extensions, fullforces[method][:, 0], temp
                )
                fitext = np.linspace(extensions.min(), extensions.max(), 300)
                Lp, Lc = wlcfit
                fitforces = engine.wlcfunc(fitext, Lp, Lc, temp)
                mask = (fitforces < fullforces[method][:, 0].max()) & (fitext < Lc)
                fitforces = fitforces[mask]
                fitext = fitext[mask]

                Lp = round(Lp, 2)
                Lc = round(Lc, 2)

                self.wlcplotter.plot(
                    fitext,
                    fitforces,
                    pen=pyg.mkPen(color=colors[method], width=2),
                    name=f"{method} fit: L<sub>p</sub>={Lp} nm, L<sub>c</sub>={Lc} nm",
                )
            except Exception as e:
                print(f"Error fitting WLC: {e}")

            self.fcplotter.addLegend()
            self.wlcplotter.addLegend()

    def previous_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() - 1)

    def next_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() + 1)
