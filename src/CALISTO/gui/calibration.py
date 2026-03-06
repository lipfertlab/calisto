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
    QApplication,
)
from PySide6.QtCore import Qt

import pyqtgraph as pyg
import numpy as np

from pathlib import Path
from itertools import cycle
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.engine import BeadType, TestResult

import engines.calibration_engine as engine
from engines.fit_engine import fit_wlc_multiplicative, fit_double_exp_multiplicative
from gui.worker import WorkerManager

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

        # Initialize worker manager for async operations
        self.worker_manager = WorkerManager(self)

        # Cache for measurements (prepared once, used for all beads)
        self.measurements = None
        self.measurements_ready = False

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.fcplotter = self.create_fcplotter()
        self.layout.addWidget(self.fcplotter, 0, 0, 21, 21)

        self.wlcplotter = self.create_wlcplotter()
        self.layout.addWidget(self.wlcplotter, 0, 21, 13, 13)

        self.control_group = self.create_controls()
        self.layout.addWidget(self.control_group, 13, 21, 8, 13)

        # Prepare measurements asynchronously when window is created
        self._prepare_measurements()

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

        # Don't plot immediately - wait for measurements to be ready

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
        # Guard: Check if there are any beads to plot
        if self.bead_combo.count() == 0:
            return

        # Guard: Check if measurements are ready
        if not self.measurements_ready:
            return

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

        # Plot the current bead
        self.plot_bead(bead_id)

    def _prepare_measurements(self):
        """Prepare measurements when window is created.

        Note: This is done synchronously in the main thread because
        prepare_multibeadmeasurement needs access to state_manager (a Qt object).
        This only happens once at window creation, so the brief delay is acceptable.
        The per-bead plotting operations run asynchronously.
        """
        try:
            self.measurements = engine.prepare_multibeadmeasurement(self.state_manager)
            self.measurements_ready = True

            # Now plot the first bead if we have any
            if self.bead_combo.count() > 0:
                self.bead_id_changed()
        except Exception as e:
            print(f"Error preparing measurements: {e}")
            import traceback

            traceback.print_exc()

    def plot_bead(self, bead_id):
        """Plot bead data asynchronously to avoid blocking the GUI."""
        # Guard: Check if measurements are ready
        if not self.measurements_ready or self.measurements is None:
            return

        # Extract ALL data from state_manager and measurements in main thread
        # (Qt objects and complex objects are not thread-safe!)
        try:
            include = self.state_manager.get_state("inclusion").copy()
            temp = self.state_manager.get_state("temperature")

            # Extract all needed data from measurements in main thread
            measurement_data = []
            for pid, m in enumerate(self.measurements):
                if not include[bead_id, pid]:
                    continue

                # Extract all data we need from this measurement
                mag_pos = m.mag_pos
                m_bead = m[bead_id]
                extension_mean = m_bead.get_extension()["mean"]

                forces = {}
                for method in ["PSD", "AV", "HV"]:
                    forces[method] = m_bead.get_force(method)

                measurement_data.append(
                    {"mag_pos": mag_pos, "extension": extension_mean, "forces": forces}
                )
        except Exception as e:
            print(f"Error preparing data: {e}")
            import traceback

            traceback.print_exc()
            return

        # Disable controls while computing
        self.set_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def cleanup():
            QApplication.restoreOverrideCursor()
            self.set_controls_enabled(True)

        # Run computation in background thread with pure Python/NumPy data only
        self.worker_manager.run_async(
            self._compute_plot_data,
            measurement_data,
            temp,
            on_result=self._update_plots,
            on_error=self._handle_plot_error,
            on_finished=cleanup,
        )

    def _compute_plot_data(self, measurement_data, temp):
        """Compute all data needed for plotting (runs in worker thread)."""
        # All parameters are now pure Python/NumPy objects, safe to use in worker thread
        colors = {"PSD": "b", "AV": "g", "HV": "r"}

        # Check if we have any valid measurements
        if len(measurement_data) == 0:
            # Return empty data structure if no valid measurements
            return {
                "fullmagpos": np.array([]),
                "extensions": np.array([]),
                "plot_data": {
                    method: {"forces": np.array([]), "dexp_fit": None, "wlc_fit": None}
                    for method in ["PSD", "AV", "HV"]
                },
                "colors": colors,
                "has_data": False,
            }

        # Extract arrays from pre-processed data
        fullmagpos = []
        extensions = []
        fullforces = {"PSD": [], "AV": [], "HV": []}

        for data in measurement_data:
            fullmagpos.append(data["mag_pos"])
            extensions.append(data["extension"])
            for method in ["PSD", "AV", "HV"]:
                fullforces[method].append(data["forces"][method])

        fullmagpos = np.array(fullmagpos).flatten()
        extensions = np.array(extensions).flatten()

        # Prepare plot data for each method
        plot_data = {}
        for method in ["PSD", "AV", "HV"]:
            fullforces[method] = np.vstack(fullforces[method])

            # Fit double exponential
            dexpfit = fit_double_exp_multiplicative(
                fullmagpos, fullforces[method][:, 0]
            )
            fmax = dexpfit["Fmax"]
            l1 = dexpfit["tau_fast"]
            l2 = dexpfit["tau_slow"]
            c = dexpfit["c"]
            fitmagpos = np.linspace(fullmagpos.min(), fullmagpos.max(), 300)
            fitforces = engine.two_term_exp(fitmagpos, fmax, l1, l2, c)

            # Try to fit WLC
            wlc_success = False
            wlc_fitext = None
            wlc_fitforces = None
            Lp = Lc = None
            try:
                wlcfit = fit_wlc_multiplicative(
                    extensions, fullforces[method][:, 0], temp
                )
                fitext = np.linspace(extensions.min(), extensions.max(), 300)
                Lp, Lc = wlcfit
                wlc_fitforces = engine.wlcfunc(fitext, Lp, Lc, temp)
                mask = (wlc_fitforces < fullforces[method][:, 0].max()) & (fitext < Lc)
                wlc_fitforces = wlc_fitforces[mask]
                wlc_fitext = fitext[mask]
                wlc_success = True
            except Exception as e:
                print(f"Error fitting WLC for {method}: {e}")

            plot_data[method] = {
                "forces": fullforces[method],
                "dexp_fit": {
                    "x": fitmagpos,
                    "y": fitforces,
                    "fmax": round(fmax, 3),
                    "l1": round(l1, 3),
                    "l2": round(l2, 3),
                    "c": c,
                },
                "wlc_fit": {
                    "success": wlc_success,
                    "x": wlc_fitext,
                    "y": wlc_fitforces,
                    "Lp": round(Lp, 2) if Lp else None,
                    "Lc": round(Lc, 2) if Lc else None,
                },
            }

        return {
            "fullmagpos": fullmagpos,
            "extensions": extensions,
            "plot_data": plot_data,
            "colors": colors,
            "has_data": True,
        }

    def _update_plots(self, result):
        """Update plots with computed data (runs in GUI thread)."""
        # Clear plots first
        self.fcplotter.clear()
        self.wlcplotter.clear()

        # Check if we have data to plot
        if not result.get("has_data", True):
            return

        fullmagpos = result["fullmagpos"]
        extensions = result["extensions"]
        plot_data = result["plot_data"]
        colors = result["colors"]

        for method in ["PSD", "AV", "HV"]:
            data = plot_data[method]
            forces = data["forces"]

            # Plot force-magpos
            errorbar = pyg.ErrorBarItem(
                x=fullmagpos,
                y=forces[:, 0],
                top=forces[:, 1],
                bottom=forces[:, 1],
                beam=0.1,
                pen=pyg.mkPen(color=colors[method], width=5),
            )
            self.fcplotter.plot(
                fullmagpos,
                forces[:, 0],
                symbol="o",
                pen=None,
                symbolPen=pyg.mkPen(color=colors[method], width=1),
                symbolBrush=pyg.mkBrush(color=colors[method]),
                name=method,
            )
            self.fcplotter.addItem(errorbar)

            # Plot double exp fit
            dexp = data["dexp_fit"]
            self.fcplotter.plot(
                dexp["x"],
                dexp["y"],
                pen=pyg.mkPen(color=colors[method], width=2),
                name=f"{method} fit: F<sub>max</sub>={dexp['fmax']} pN,\n     l<sub>1</sub>={dexp['l1']} nm\n     l<sub>2</sub>={dexp['l2']} nm",
            )

            # Plot force-extension
            self.wlcplotter.plot(
                extensions,
                forces[:, 0],
                symbol="o",
                pen=None,
                symbolPen=pyg.mkPen(color=colors[method], width=1),
                symbolBrush=pyg.mkBrush(color=colors[method]),
                name=method,
            )
            errorbar2 = pyg.ErrorBarItem(
                x=extensions,
                y=forces[:, 0],
                top=forces[:, 1],
                bottom=forces[:, 1],
                beam=0.1,
                pen=pyg.mkPen(color=colors[method], width=5),
            )
            self.wlcplotter.addItem(errorbar2)

            # Plot WLC fit if available
            wlc = data["wlc_fit"]
            if wlc["success"]:
                self.wlcplotter.plot(
                    wlc["x"],
                    wlc["y"],
                    pen=pyg.mkPen(color=colors[method], width=2),
                    name=f"{method} fit: L<sub>p</sub>={wlc['Lp']} nm, L<sub>c</sub>={wlc['Lc']} nm",
                )

        self.fcplotter.addLegend()
        self.wlcplotter.addLegend()

    def _handle_plot_error(self, error_msg, traceback):
        """Handle errors from plotting computation."""
        print(f"Error computing plot data: {error_msg}")
        print(traceback)

    def set_controls_enabled(self, enabled):
        """Enable or disable controls during computation."""
        self.bead_combo.setEnabled(enabled)
        self.include_radiobutton.setEnabled(enabled)
        self.exclude_radiobutton.setEnabled(enabled)
        self.previous_button.setEnabled(enabled and self.bead_combo.currentIndex() > 0)
        self.next_button.setEnabled(
            enabled and self.bead_combo.currentIndex() < self.bead_combo.count() - 1
        )

    def previous_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() - 1)

    def next_bead(self):
        self.bead_combo.setCurrentIndex(self.bead_combo.currentIndex() + 1)
