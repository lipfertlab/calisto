#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtGui import QGuiApplication, QCursor
from PySide6.QtWidgets import (
    QMainWindow,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QWidget,
    QGridLayout,
    QGroupBox,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import engines.landing_engine as engine
from .fcalibration import FCWindow


checkmark = "\u2713"
crossmark = "\u2717"


class MainWindow(QMainWindow):

    def __init__(self, state_manager, app_name="CALISTO"):
        super().__init__()
        self.state_manager = state_manager
        self.state_manager.stateChanged.connect(self.on_state_changed)
        self.path_maxlength = 27

        self.setWindowTitle(app_name)

        self.layout = QGridLayout()

        self.configdependentwidgets = []

        # CHOOSE SETUP
        self.setup_label = QLabel("Choose MT Setup:")
        self.layout.addWidget(self.setup_label, 0, 0)

        self.setup_combobox = QComboBox()
        setups = state_manager.get_state("config").keys()
        self.setup_combobox.addItems(setups)
        self.setup_combobox.setCurrentIndex(-1)
        self.layout.addWidget(self.setup_combobox, 0, 1)
        self.setup_combobox.currentIndexChanged.connect(self.setSetup)

        # CHOOSE FORMAT
        self.datatype_label = QLabel("Choose File Format (*.txt or *.h5):")
        self.layout.addWidget(self.datatype_label, 1, 0)

        self.datatype_combobox = QComboBox()
        self.datatype_combobox.addItem("Plain Text (.txt)")
        self.datatype_combobox.addItem("HDF5 (.h5)")

        self.layout.addWidget(self.datatype_combobox, 1, 1)

        ##################################################################

        # CHOOSE TXT FILES
        self.txtbox = QGroupBox()
        txtbox_layout = QGridLayout()
        self.txtbox.setLayout(txtbox_layout)

        self.txt_bead_button = QPushButton("Trace Data")
        self.txt_bead_button.clicked.connect(self.getBeadDatafile)
        txtbox_layout.addWidget(self.txt_bead_button, 0, 0)
        self.configdependentwidgets.append(self.txt_bead_button)

        self.txt_bead_tracepath_label = QLabel("not chosen")
        txtbox_layout.addWidget(self.txt_bead_tracepath_label, 0, 1, 1, 2)

        self.txt_motor_button = QPushButton("Motor Data")
        self.txt_motor_button.clicked.connect(self.getMotorDatafile)
        txtbox_layout.addWidget(self.txt_motor_button, 1, 0)
        self.configdependentwidgets.append(self.txt_motor_button)

        self.txt_bead_motorpath_label = QLabel("not chosen")
        txtbox_layout.addWidget(self.txt_bead_motorpath_label, 1, 1, 1, 2)

        self.layout.addWidget(self.txtbox, 2, 0, 2, 2)
        ##################################################################

        # CHOOSE HDF5 FILE
        self.hdfbox = QGroupBox()
        hdfbox_layout = QGridLayout()
        self.hdfbox.setLayout(hdfbox_layout)

        self.hdf_data_button = QPushButton("Data")
        self.hdf_data_button.clicked.connect(self.gethdfDatafile)
        hdfbox_layout.addWidget(self.hdf_data_button, 0, 0)
        self.configdependentwidgets.append(self.hdf_data_button)

        self.hdf_datapath_label = QLabel("not chosen")
        hdfbox_layout.addWidget(self.hdf_datapath_label, 0, 1, 1, 2)

        self.layout.addWidget(self.hdfbox, 2, 0, 1, 2)
        ##################################################################

        self.parameterbox = QGroupBox()
        parameterbox_layout = QGridLayout()
        self.parameterbox.setLayout(parameterbox_layout)

        ### sampling frequency:
        self.samplingfreq_label = QLabel("Acq. frequency:")
        parameterbox_layout.addWidget(self.samplingfreq_label, 0, 0)

        self.samplingfreq_lineedit = QLineEdit()
        self.samplingfreq_lineedit.setPlaceholderText("unit: Hz & use decimal dot (.)")
        parameterbox_layout.addWidget(self.samplingfreq_lineedit, 0, 1)
        self.configdependentwidgets.append(self.samplingfreq_lineedit)

        self.samplingfreq_postlabel = QLabel("  ")
        parameterbox_layout.addWidget(self.samplingfreq_postlabel, 0, 2)

        self.samplingfreq_lineedit.textChanged.connect(self.getSamplingFreq)

        ### bead radius:
        self.beadradius_label = QLabel("Bead radius:")
        parameterbox_layout.addWidget(self.beadradius_label, 1, 0)

        self.beadradius_lineedit = QLineEdit()
        self.beadradius_lineedit.setPlaceholderText("unit: nm & use decimal dot (.)")
        parameterbox_layout.addWidget(self.beadradius_lineedit, 1, 1)
        self.configdependentwidgets.append(self.beadradius_lineedit)

        self.beadradius_postlabel = QLabel("  ")
        parameterbox_layout.addWidget(self.beadradius_postlabel, 1, 2)

        self.beadradius_lineedit.textChanged.connect(self.getBeadRadius)
        ###

        ### temperature:
        self.temperature_label = QLabel("Temperature:")
        parameterbox_layout.addWidget(self.temperature_label, 2, 0)

        self.temperature_lineedit = QLineEdit()
        self.temperature_lineedit.setPlaceholderText("unit: K & use decimal dot (.)")
        parameterbox_layout.addWidget(self.temperature_lineedit, 2, 1)
        self.configdependentwidgets.append(self.temperature_lineedit)

        self.temperature_postlabel = QLabel("  ")
        parameterbox_layout.addWidget(self.temperature_postlabel, 2, 2)

        self.temperature_lineedit.textChanged.connect(self.getTemperature)
        ###

        ### axis
        self.axisinputlabel = QLabel("Variance axis:")
        parameterbox_layout.addWidget(self.axisinputlabel, 3, 0)

        self.axis_combobox = QComboBox()
        self.axis_combobox.addItems(["x", "y"])
        self.axis_combobox.setCurrentIndex(-1)
        parameterbox_layout.addWidget(self.axis_combobox, 3, 1)
        self.configdependentwidgets.append(self.axis_combobox)

        self.axis_postlabel = QLabel("  ")
        parameterbox_layout.addWidget(self.axis_postlabel, 3, 2)

        self.axis_combobox.currentIndexChanged.connect(self.getaxis)

        self.layout.addWidget(self.parameterbox, 4, 0, 3, 2)

        ##################################################################

        # START FORCE CALIBRATION
        self.FCbtn = QPushButton("Start Force Calibration Session")
        # disable button until files & parameters are chosen
        self.FCbtn.setEnabled(False)
        self.layout.addWidget(self.FCbtn, 7, 0, 1, 2)
        ##################################################################

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        # self.layout.addWidget(self.DirBox, 1,0,3,2)

        self.datatype_combobox.currentIndexChanged.connect(self.datatype_changed)
        self.datatype_changed(0)

        self.setCentralWidget(self.widget)

        self.FCbtn.clicked.connect(self.activate_FC)
        for widget in self.configdependentwidgets:
            widget.setEnabled(False)

    def datatype_changed(self, index):
        self.state_manager.delete_state("bead_data")
        self.state_manager.delete_state("motor_data")
        self.state_manager.set_state("measurements_outdated", True)
        self.state_manager.set_state("bead_specs_outdated", True)
        if index == 0:
            self.hdfbox.hide()
            self.txtbox.show()
        else:
            self.txtbox.hide()
            self.hdfbox.show()

    def getBeadDatafile(self):
        file_filter = "Text File (*.txt)"
        self.txt_bead_tracepath, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open file",
            filter=file_filter,
        )
        pathtext = self.txt_bead_tracepath
        if pathtext == "":
            return

        if len(pathtext) > self.path_maxlength:
            pathtext = "..." + pathtext[-self.path_maxlength :]
        self.txt_bead_tracepath_label.setText(pathtext)

        self.txt_bead_tracepath = Path(self.txt_bead_tracepath)
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            engine.load_bead_datafile(self.txt_bead_tracepath, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")

        return

    def getMotorDatafile(self):
        file_filter = "Text File (*.txt)"
        self.txt_bead_motorpath, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open file",
            filter=file_filter,
        )
        pathtext = self.txt_bead_motorpath
        if pathtext == "":
            return
        if len(pathtext) > self.path_maxlength:
            pathtext = "..." + pathtext[-self.path_maxlength :]
        self.txt_bead_motorpath_label.setText(pathtext)

        self.txt_bead_motorpath = Path(self.txt_bead_motorpath)
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            engine.load_motor_datafile(self.txt_bead_motorpath, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")

        return

    def gethdfDatafile(self):
        file_filter = "HDF5 (*.h5)"
        self.hdf_datapath, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open file",
            filter=file_filter,
        )
        pathtext = self.hdf_datapath
        if pathtext == "":
            return

        if len(pathtext) > self.path_maxlength:
            pathtext = "..." + pathtext[-self.path_maxlength :]
        self.hdf_datapath_label.setText(pathtext)

        self.hdf_datapath = Path(self.hdf_datapath)

        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            engine.load_hdf_datafile(self.hdf_datapath, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")
        return

    def getBeadRadius(self):
        try:
            bead_radius = float(self.beadradius_lineedit.text())
            if bead_radius <= 0:
                raise ValueError
            self.state_manager.set_state("bead_radius", bead_radius)
            self.state_manager.set_state("measurements_outdated", True)
            self.beadradius_postlabel.setText(checkmark)
        except ValueError:
            self.beadradius_postlabel.setText(crossmark)
            # if the key exists, delete it
            self.state_manager.delete_state("bead_radius")

        return

    def getTemperature(self):
        try:
            temperature = float(self.temperature_lineedit.text())
            if temperature <= 0:
                raise ValueError
            self.state_manager.set_state("temperature", temperature)
            self.state_manager.set_state("measurements_outdated", True)
            self.temperature_postlabel.setText(checkmark)
        except ValueError:
            self.temperature_postlabel.setText(crossmark)
            # if the key exists, delete it
            self.state_manager.delete_state("temperature")

        return

    def getSamplingFreq(self):
        try:
            sampling_freq = float(self.samplingfreq_lineedit.text())
            if sampling_freq <= 0:
                raise ValueError
            self.state_manager.set_state("fsample", sampling_freq)
            self.state_manager.set_state("measurements_outdated", True)
            self.samplingfreq_postlabel.setText(checkmark)
        except ValueError:
            self.samplingfreq_postlabel.setText(crossmark)
            # if the key exists, delete it
            self.state_manager.delete_state("fsample")

        return

    def getaxis(self):
        axis = self.axis_combobox.currentIndex()
        self.state_manager.set_state("axis", axis)
        self.state_manager.set_state("measurements_outdated", True)
        self.axis_postlabel.setText(checkmark)
        return

    def on_state_changed(self, state):
        required_keys = [
            "bead_radius",
            "fsample",
            "axis",
            "bead_pos",
            "mag_pos",
            "time",
            "temperature",
        ]
        if all(key in state.keys() for key in required_keys):
            self.FCbtn.setEnabled(True)
        else:
            self.FCbtn.setEnabled(False)

        if "computed_fsample" in state.keys():
            self.samplingfreq_lineedit.setText(str(round(state["computed_fsample"], 2)))
            self.state_manager.delete_state("computed_fsample")

    def activate_FC(self):
        try:
            engine.verify_data_consistency(self.state_manager)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Error verifying data consistency: {e}"
            )
            return
        engine.prepare_dataframe(self.state_manager)
        self.FCWindow = FCWindow(self.state_manager, self)
        self.FCWindow.show()
        self.hide()

    def setSetup(self, index):
        if index == -1:
            return
        setup = self.setup_combobox.currentText()
        self.state_manager.set_state("config_setup", setup)
        setups = self.state_manager.get_state("config")
        try:
            axist = setups[setup]["variance_axis"]
        except KeyError:
            axist = ""
        if axist in ["x", "y"]:
            axisint = 0 if axist == "x" else 1
            self.axis_combobox.setCurrentIndex(axisint)
        else:
            axisint = -1
            self.axis_combobox.setCurrentIndex(axisint)
            self.axis_postlabel.setText("")

        for widget in self.configdependentwidgets:
            widget.setEnabled(True)
