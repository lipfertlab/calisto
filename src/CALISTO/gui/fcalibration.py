# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from PySide6.QtGui import QGuiApplication, QCursor
from PySide6.QtWidgets import (
    QFrame,
    QSizePolicy,
    QSpacerItem,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QWidget,
    QGridLayout,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QTableWidget,
)
from PySide6.QtCore import Qt
import pickle
from .rotation import RotationPlotterWindow
from .traceplotter import TracePlotterWindow
from .offset import OffsetPlotterWindow
from .refplotter import RefPlotterWindow
from .mastercurve import MasterCurvePlotterWindow
from .noisestability import NoiseStabilityWindow
from .calibration import CalibrationPlotterWindow
from .magposplotter import MagPosPlotterWindow

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from engines.engine import FileType, BeadType
import engines.fcalibration_engine as engine
from engines.calibration_engine import prepare_multibeadmeasurement
import numpy as np

# import pickle

checkmark = "\u2713"
crossmark = "\u2717"


class FCWindow(QWidget):
    def __init__(self, state_manager, parent=None):
        super().__init__()

        self.path_maxlength = 27
        self.state_manager = state_manager
        self.state_manager.stateChanged.connect(self.on_state_changed)

        self.parent = parent

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.leftcol = self.create_leftcol()
        self.leftcol.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addWidget(self.leftcol, 0, 0)

        self.tablecol = self.create_tablecol()
        self.tablecol.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.tablecol, 0, 1)

        self.layout.setColumnStretch(0, 0)  # Left column does not stretch
        self.layout.setColumnStretch(1, 1)  # Right column stretches

        self.traceplotter = None
        self.magposplotter = None
        self.offsetplotter = None
        self.forces_per_bead_plotter = None
        self.master_curve_plotter = None
        self.noise_stability_plotter = None
        self.refplotter = None

    def create_leftcol(self):
        col = QGroupBox()
        layout = QGridLayout()
        col.setLayout(layout)

        self.manual_check_box = self.create_manual_check_box()
        layout.addWidget(self.manual_check_box, 0, 0)
        layout.addItem(
            QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding), 1, 0
        )

        self.offset_box = self.create_offset_box()
        layout.addWidget(self.offset_box, 2, 0)
        layout.addItem(
            QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding), 3, 0
        )

        self.rotation_curve_box = self.create_rotation_curve_box()
        layout.addWidget(self.rotation_curve_box, 4, 0)
        layout.addItem(
            QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding), 5, 0
        )

        self.force_calibration_box = self.create_force_calibration_box()
        layout.addWidget(self.force_calibration_box, 6, 0)

        return col

    def closeEvent(self, event):
        self.parent.show()
        event.accept()

    def create_manual_check_box(self):
        layout = QGridLayout()
        self.manual_check_box = QGroupBox("Check Data:")
        self.manual_check_box.setStyleSheet(
            "QGroupBox {padding-top: 20 px; margin-top: 20 px; font: bold 14px Arial }"
        )
        self.manual_check_box.setLayout(layout)

        self.plot_traces_button = QPushButton("Choose && Plot Traces")
        self.plot_traces_button.setToolTip(
            "Plot bead traces, mark them as tethered/reference, or kick them out"
        )
        self.plot_traces_button.clicked.connect(self.plot_traces_button_clicked)
        layout.addWidget(self.plot_traces_button, 0, 0)

        self.plot_magnet_button = QPushButton("Plot && Partition Magnet Positions")
        self.plot_magnet_button.setToolTip(
            "Plot magnet positions and partition them into constant force plateaus"
        )
        self.plot_magnet_button.clicked.connect(self.plot_magnet_button_clicked)
        layout.addWidget(self.plot_magnet_button, 1, 0)

        self.identify_suitable_reference_beads_button = QPushButton(
            "Identify Suitable Reference Beads"
        )
        self.identify_suitable_reference_beads_button.setToolTip(
            "Identify Suitable Reference Beads"  # TODO: add more info
        )
        self.identify_suitable_reference_beads_button.clicked.connect(
            self.identify_suitable_reference_beads_button_clicked
        )
        layout.addWidget(self.identify_suitable_reference_beads_button, 2, 0)

        self.identify_stable_regions_button = QPushButton("Identify Stable Regions")
        self.identify_stable_regions_button.setToolTip("Identify stable regions")
        self.identify_stable_regions_button.clicked.connect(
            self.identify_stable_regions_button_clicked
        )
        layout.addWidget(self.identify_stable_regions_button, 3, 0)

        return self.manual_check_box

    def create_offset_box(self):
        layout = QGridLayout()
        self.OffsetBox = QGroupBox("Offsets:")
        self.OffsetBox.setStyleSheet(
            "QGroupBox {padding-top: 20 px; margin-top: 20 px; font: bold 14px Arial }"
        )
        self.OffsetBox.setLayout(layout)

        self.define_offset_label = QLabel("Define offsets:")
        self.define_offset_label.setToolTip(
            """Define offsets from:
        - Data analysis: Load traces for offset measurement, analyze and define offsets
        - Existing table: Load an existing offset table
        - Constant value: Set a constant offset value for all beads"""
        )
        layout.addWidget(self.define_offset_label, 0, 0)

        self.offset_method_combobox = QComboBox()
        self.offset_method_combobox.addItems(
            ["from data analysis", "from existing table", "set constant value"]
        )
        self.offset_method_combobox.setCurrentIndex(-1)
        self.offset_method_combobox.setToolTip(
            """Define offsets from:
        - Data analysis: Load traces for offset measurement, analyze and define offsets
        - Existing table: Load an existing offset table
        - Constant value: Set a constant offset value for all beads"""
        )
        self.offset_method_combobox.currentIndexChanged.connect(
            self.offset_method_changed
        )
        layout.addWidget(self.offset_method_combobox, 0, 1, 1, 2)

        # Global offset section
        self.offset_global_label = QLabel("Offset value (nm):")
        self.offset_global_label.setToolTip(
            "Define a constant offset value for all beads"
        )
        layout.addWidget(self.offset_global_label, 1, 0)

        self.offset_global_value = QLineEdit()
        self.offset_global_value.setToolTip(
            "Define a constant offset value for all beads"
        )
        layout.addWidget(self.offset_global_value, 1, 1)

        self.offset_apply_button = QPushButton("Apply")
        self.offset_apply_button.setToolTip(
            "Apply the defined offset value to all beads"
        )
        self.offset_apply_button.clicked.connect(self.offset_apply_button_clicked)
        layout.addWidget(self.offset_apply_button, 1, 2)

        self.offset_global_widgets = [
            self.offset_global_label,
            self.offset_global_value,
            self.offset_apply_button,
        ]
        for widget in self.offset_global_widgets:
            widget.hide()

        ###

        # Use existing offset table section
        self.offset_table_choose_button = QPushButton("Choose file")
        self.offset_table_choose_button.setToolTip("Load an existing offset table")
        self.offset_table_choose_button.clicked.connect(
            self.offset_table_choose_button_clicked
        )
        layout.addWidget(self.offset_table_choose_button, 1, 0)

        self.offset_table_path_label = QLabel("no file chosen")
        layout.addWidget(self.offset_table_path_label, 1, 1, 1, 2)

        self.offset_table_widgets = [
            self.offset_table_choose_button,
            self.offset_table_path_label,
        ]
        for widget in self.offset_table_widgets:
            widget.hide()

        ###

        # Data analysis section
        self.offset_data_choose_button = QPushButton("Choose file")
        self.offset_data_choose_button.clicked.connect(
            self.offset_data_choose_button_clicked
        )
        layout.addWidget(self.offset_data_choose_button, 1, 0)

        self.offset_data_path_label = QLabel("no file chosen")
        layout.addWidget(self.offset_data_path_label, 1, 1, 1, 2)

        self.offset_data_analyze_button = QPushButton("Analyze")
        self.offset_data_analyze_button.setEnabled(False)
        self.offset_data_analyze_button.clicked.connect(
            self.offset_data_analyze_button_clicked
        )
        layout.addWidget(self.offset_data_analyze_button, 2, 0, 1, 3)

        self.offset_data_widgets = [
            self.offset_data_choose_button,
            self.offset_data_path_label,
            self.offset_data_analyze_button,
        ]
        for widget in self.offset_data_widgets:
            widget.hide()
        ###

        # put a horizonotal line between the define offset section and the filter section
        self.line = QLabel()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self.line, 3, 0, 1, 3)

        # filter section

        self.offset_filter_label = QLabel("Filter Beads:")
        self.offset_filter_label.setToolTip("Filter beads based on offset interval")
        layout.addWidget(self.offset_filter_label, 4, 0)

        self.offset_filter_min_label = QLabel("min (nm):")
        self.offset_filter_min_label.setToolTip("Minimum offset value")
        layout.addWidget(self.offset_filter_min_label, 4, 1)

        self.offset_filter_min = QLineEdit()
        self.offset_filter_min.setToolTip("Minimum offset value")
        self.offset_filter_min.textChanged.connect(self.offset_filter_changed)
        layout.addWidget(self.offset_filter_min, 4, 2)

        self.offset_filter_min_verified_label = QLabel("")
        layout.addWidget(self.offset_filter_min_verified_label, 4, 3)

        self.offset_filter_max_label = QLabel("max (nm):")
        self.offset_filter_max_label.setToolTip("Maximum offset value")
        layout.addWidget(self.offset_filter_max_label, 5, 1)

        self.offset_filter_max = QLineEdit()
        self.offset_filter_max.setToolTip("Maximum offset value")
        self.offset_filter_max.textChanged.connect(self.offset_filter_changed)
        layout.addWidget(self.offset_filter_max, 5, 2)

        self.offset_filter_max_verified_label = QLabel("")
        layout.addWidget(self.offset_filter_max_verified_label, 5, 3)
        ###
        return self.OffsetBox

    def create_rotation_curve_box(self):
        layout = QGridLayout()
        self.rotation_curve_box = QGroupBox("Rotation Test:")
        self.rotation_curve_box.setStyleSheet(
            "QGroupBox {padding-top: 20 px; margin-top: 20 px; font: bold 14px Arial }"
        )
        self.rotation_curve_box.setLayout(layout)

        # CHOOSE FORMAT
        self.datatype_label = QLabel("Choose File Format:")
        layout.addWidget(self.datatype_label, 0, 0)

        self.rotdatatype_combobox = QComboBox()
        self.rotdatatype_combobox.addItem("Plain Text (.txt)")
        self.rotdatatype_combobox.addItem("HDF5 (.h5)")

        layout.addWidget(self.rotdatatype_combobox, 0, 1)

        ##################################################################

        # CHOOSE TXT FILES
        self.rottxtbox = QGroupBox()
        txtbox_layout = QGridLayout()
        self.rottxtbox.setLayout(txtbox_layout)

        self.txt_bead_button = QPushButton("Trace Data")
        self.txt_bead_button.clicked.connect(self.getRotDatafile)
        txtbox_layout.addWidget(self.txt_bead_button, 0, 0)

        self.txt_bead_tracepath_label = QLabel("not chosen")
        txtbox_layout.addWidget(self.txt_bead_tracepath_label, 0, 1, 1, 2)

        self.txt_motor_button = QPushButton("Motor Data")
        self.txt_motor_button.clicked.connect(self.getRotMotorDatafile)
        txtbox_layout.addWidget(self.txt_motor_button, 1, 0)

        self.txt_bead_motorpath_label = QLabel("not chosen")
        txtbox_layout.addWidget(self.txt_bead_motorpath_label, 1, 1, 1, 2)

        layout.addWidget(self.rottxtbox, 1, 0, 2, 2)
        ##################################################################

        # CHOOSE HDF5 FILE
        self.rothdfbox = QGroupBox()
        hdfbox_layout = QGridLayout()
        self.rothdfbox.setLayout(hdfbox_layout)

        self.hdf_data_button = QPushButton("Data")
        self.hdf_data_button.clicked.connect(self.gethdfRotDatafile)
        hdfbox_layout.addWidget(self.hdf_data_button, 0, 0)

        self.hdf_datapath_label = QLabel("not chosen")
        hdfbox_layout.addWidget(self.hdf_datapath_label, 0, 1, 1, 2)

        layout.addWidget(self.rothdfbox, 1, 0, 1, 2)
        ##################################################################

        self.rotation_test_button = QPushButton("Choose && Plot")
        self.rotation_test_button.clicked.connect(self.rotation_test_button_clicked)
        layout.addWidget(self.rotation_test_button, 3, 0)

        self.rotation_test_button.setEnabled(False)  # TODO: not implemented yet

        self.rotdatatype_combobox.currentIndexChanged.connect(self.rotdatatype_changed)
        self.rotdatatype_changed(0)

        return self.rotation_curve_box

    def rotdatatype_changed(self, index):
        self.state_manager.delete_state("bead_data")
        self.state_manager.delete_state("motor_data")
        if index == 0:
            self.rothdfbox.hide()
            self.rottxtbox.show()
        else:
            self.rottxtbox.hide()
            self.rothdfbox.show()

    def create_force_calibration_box(self):
        form_layout = QGridLayout()

        self.force_calibration_box = QGroupBox("Force Calibration:")
        self.force_calibration_box.setStyleSheet(
            "QGroupBox {padding-top: 20 px; margin-top: 20 px; font: bold 14px Arial }"
        )
        self.force_calibration_box.setLayout(form_layout)

        self.force_calibration_status_label = QLabel()

        form_layout.addWidget(self.force_calibration_status_label, 0, 0)

        self.forces_per_bead_button = QPushButton("Get Forces per Bead")
        self.forces_per_bead_button.clicked.connect(self.forces_per_bead_button_clicked)
        form_layout.addWidget(self.forces_per_bead_button, 1, 0)

        self.force_calibration_curve_button = QPushButton(
            "Get Master Force Calibration Curve"
        )
        self.force_calibration_curve_button.clicked.connect(
            self.force_calibration_curve_button_clicked
        )
        form_layout.addWidget(self.force_calibration_curve_button, 2, 0)
        self.get_force_calibration_status()
        return self.force_calibration_box

    def create_tablecol(self):
        col = QGroupBox()
        layout = QGridLayout()
        col.setLayout(layout)
        self.table = QTableWidget()
        engine.prepare_table(self.table, self.state_manager)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.resizeColumnsToContents()

        self.table.itemChanged.connect(
            lambda row, col: engine.table_cell_changed(
                row, col, self.state_manager, self.table
            )
        )

        layout.addWidget(self.table, 0, 0)

        total_width = sum(
            self.table.columnWidth(i) for i in range(self.table.columnCount())
        )
        total_width += self.table.verticalHeader().width()
        total_width *= 1.15

        screen_width = QGuiApplication.primaryScreen().geometry().width()

        current_width = self.frameGeometry().width()

        # get width of the qgroupbox
        col_width = col.frameGeometry().width()
        rest_width = current_width - col_width

        if total_width > screen_width * 0.5:
            col.setMinimumWidth(screen_width * 0.5)
        else:
            col.setMinimumWidth(total_width)

        return col

    def plot_traces_button_clicked(self):

        self.traceplotter = TracePlotterWindow(self, self.state_manager)
        self.traceplotter.show()

    def plot_magnet_button_clicked(self):
        self.magposplotter = MagPosPlotterWindow(self, self.state_manager)
        self.magposplotter.show()

    def offset_method_changed(self, index):
        match index:
            case 0:
                for widget in self.offset_global_widgets:
                    widget.hide()
                for widget in self.offset_table_widgets:
                    widget.hide()

                for widget in self.offset_data_widgets:
                    widget.show()
            case 1:
                for widget in self.offset_global_widgets:
                    widget.hide()
                for widget in self.offset_data_widgets:
                    widget.hide()

                for widget in self.offset_table_widgets:
                    widget.show()
            case 2:
                for widget in self.offset_table_widgets:
                    widget.hide()
                for widget in self.offset_data_widgets:
                    widget.hide()

                for widget in self.offset_global_widgets:
                    widget.show()
            case _:
                raise ValueError("Invalid offset method index")

    def offset_apply_button_clicked(self):
        try:
            offset_const = float(self.offset_global_value.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid offset value")
            return
        self.state_manager.set_state("measurements_outdated", True)
        engine.offset_constant_set(offset_const, self.state_manager)

    def offset_table_choose_button_clicked(self):
        file_filter = "Text File (*.txt)"
        self.offset_table_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open file",
            filter=file_filter,
        )
        pathtext = self.offset_table_path
        if pathtext == "":
            return

        if len(pathtext) > self.path_maxlength:
            pathtext = "..." + pathtext[-self.path_maxlength :]
        self.offset_table_path_label.setText(pathtext)

        self.offset_table_path = Path(self.offset_table_path)
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            engine.offset_table_load(self.offset_table_path, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")
            return
        self.state_manager.set_state("measurements_outdated", True)
        return

    def offset_data_choose_button_clicked(self):
        file_filter = "Text File (*.txt);;HDF5 File (*.h5)"
        self.offset_data_path, file_type = QFileDialog.getOpenFileName(
            parent=self, caption="Open file", filter=file_filter
        )
        file_info = None
        if file_type == "Text File (*.txt)":
            file_type = FileType.PLAINTEXT
        elif file_type == "HDF5 File (*.h5)":
            file_type = FileType.HDF5
        else:
            QMessageBox.critical(
                self, "Error", f"File type not recognized: {file_type}"
            )
            return

        pathtext = self.offset_data_path
        if pathtext == "":
            return

        if len(pathtext) > self.path_maxlength:
            pathtext = "..." + pathtext[-self.path_maxlength :]
        self.offset_data_path_label.setText(pathtext)

        self.offset_data_path = Path(self.offset_data_path)
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            engine.offset_data_load(
                self.offset_data_path, file_type, self.state_manager
            )
            QGuiApplication.restoreOverrideCursor()
            self.offset_data_analyze_button.setEnabled(True)
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")
            return

        if file_type == FileType.HDF5:
            self.offset_data_hdf5_info(self.state_manager)

        return

    def offset_data_hdf5_info(self):
        pass

    def offset_data_analyze_button_clicked(self):
        self.state_manager.set_state("measurements_outdated", True)
        self.offsetplotter = OffsetPlotterWindow(self, self.state_manager)
        self.offsetplotter.show()

    def offset_filter_changed(self):
        try:
            mintext = self.offset_filter_min.text()
            min_offset = -np.inf if mintext == "" else float(mintext)
            maxtext = self.offset_filter_max.text()
            max_offset = np.inf if maxtext == "" else float(maxtext)
        except:
            return

        last_mod = self.state_manager.get_state("last_modified")
        print(last_mod)
        if last_mod != "offset_filter":
            engine.backup_inclusion_and_reason(self.state_manager)

        engine.process_offset_filter(self.state_manager, min_offset, max_offset)
        self.state_manager.set_state("offset_filter", (min_offset, max_offset))

    def rotation_test_button_clicked(self):
        try:
            engine.verify_rot_data_consistency(self.state_manager)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Error verifying data consistency: {e}"
            )
            return
        self.rotation_plotter = RotationPlotterWindow(self, self.state_manager)
        self.rotation_plotter.show()

    def forces_per_bead_button_clicked(self):
        _ = prepare_multibeadmeasurement(self.state_manager)

        self.forces_per_bead_plotter = CalibrationPlotterWindow(
            self, self.state_manager
        )
        measurements_outdated = self.state_manager.get_state("measurements_outdated")
        if measurements_outdated:
            self.forces_per_bead_plotter.refresh_measurements(self.state_manager)
        self.forces_per_bead_plotter.show()

    def force_calibration_curve_button_clicked(self):
        _ = prepare_multibeadmeasurement(self.state_manager)

        self.master_curve_plotter = MasterCurvePlotterWindow(self, self.state_manager)

        self.master_curve_plotter.show()

    def identify_stable_regions_button_clicked(self):

        self.noise_stability_plotter = NoiseStabilityWindow(self, self.state_manager)
        self.noise_stability_plotter.show()

    def identify_suitable_reference_beads_button_clicked(self):
        bead_specs = self.state_manager.get_state("bead_specs")
        bead_types = bead_specs["Type"]
        nref = (bead_types == BeadType.REFERENCE).sum()
        if nref == 0:
            QMessageBox.critical(
                self,
                "Error",
                "No reference beads selected. Select some reference beads",
            )
            return

        self.refplotter = RefPlotterWindow(self, self.state_manager)
        self.refplotter.show()

    def get_force_calibration_status(self):
        waiting = []
        if self.state_manager.get_state("plateaus") is None:
            waiting.append("force plateaus")
        if self.state_manager.get_state("mag_pos") is None:
            waiting.append("magnet positions")

        bead_spec = self.state_manager.get_state("bead_specs")
        include = bead_spec["Include"]
        # is there at least one bead that is included?:
        if not include.any():
            waiting.append("bead inclusion")

        offsets = bead_spec["Offset"]
        if offsets.isnull().values.any():
            waiting.append("offsets")

        refbeadmask = (bead_spec["Type"] == BeadType.REFERENCE) & include
        if refbeadmask.sum() == 0:
            waiting.append("reference beads")

        magbeadmask = (bead_spec["Type"] == BeadType.MAGNETIC) & include
        if magbeadmask.sum() == 0:
            waiting.append("magnetic beads")

        if len(waiting) == 0:
            text = "Ready"
            self.forces_per_bead_button.setEnabled(True)
            self.force_calibration_curve_button.setEnabled(True)
        else:
            text = "Waiting for: " + ", ".join(waiting)
            self.forces_per_bead_button.setEnabled(False)
            self.force_calibration_curve_button.setEnabled(False)

        self.force_calibration_status_label.setText(text)

    def gethdfRotDatafile(self):
        file_filter = "HDF5 File (*.h5)"
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
            engine.load_rot_hdfdatafile(self.hdf_datapath, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")

        return

    def getRotDatafile(self):
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
            engine.load_rot_datafile(self.txt_bead_tracepath, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")

        return

    def getRotMotorDatafile(self):
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
            engine.load_rot_motor_datafile(self.txt_bead_motorpath, self.state_manager)
            QGuiApplication.restoreOverrideCursor()
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Error loading data file: {e}")

        return

    def on_state_changed(self, state):
        self.get_force_calibration_status()

        if state["last_modified"] == "bead_specs":
            engine.prepare_table(self.table, self.state_manager)
            self.state_manager.set_state("measurements_outdated", True)
        if state["last_modified"] == "inclusion":
            self.state_manager.set_state("measurements_outdated", True)

        if state["last_modified"] == "offsets":
            self.state_manager.set_state("measurements_outdated", True)
        if state["last_modified"] == "median_filter":
            self.state_manager.set_state("measurements_outdated", True)
        if state["last_modified"] == "median_filter_size":
            self.state_manager.set_state("measurements_outdated", True)

        rotation_dependent_states = [
            "rot_bead_pos",
            "rot_mag_pos",
            "rot_mag_rot",
            "rot_mag_time",
            "rot_time",
        ]
        if state["last_modified"] in rotation_dependent_states:
            rot_enabled = all(
                [
                    self.state_manager.get_state(state) is not None
                    for state in rotation_dependent_states
                ]
            )
            self.rotation_test_button.setEnabled(rot_enabled)
