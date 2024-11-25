import os
import json
import sys
import cv2
import pickle
import numpy as np

from PyQt5.QtGui import QColor, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QHeaderView, QPushButton, QToolButton

from configs.load_config import CONFIG
from interface.functions.gui_mixins import MessageBoxMixin
from interface.functions.trajectory_thread import TrajectoryThread


class TrajecotryGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(TrajecotryGUI, self).__init__()

        # previous step result name
        self.centerline_result_name = "centerline_data.pkl"

        # set saving subfolder
        self.trajectory_saving_subfolder = "trajectory_planning"

        # previous step variables
        self.mask_action_binaries = None
        self.line_dict = None
        self.reverse_mask_dict = None

        # output trajecotry holders
        # store the trajectory of line cutting and coarse bulk cutting
        self.coarse_trajectory_holders = []
        # store the trajectory of fine bulk cutting
        self.fine_trajectory_holders = []
        # store the trajectory of ultra fine bulk cutting
        self.ultra_fine_trajectory_holders = []
        # [ ] store the depth map of all cutting (current just for bulk cutting)
        self.depth_map_holders = []

        # setup layout
        page_layout = self.create_layout()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(page_layout)

        if message_box is not None:
            self.message_box = message_box
        else:
            # define left message box
            self.message_box = QtWidgets.QTextEdit()
            self.message_box.setReadOnly(True)
            self.message_box.setFixedHeight(100)
            main_layout.addWidget(self.message_box)

        self.setLayout(main_layout)

        # setup thread
        self.traj_thread = TrajectoryThread()

    def create_layout(self):
        # image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(1280, 800)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.image_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: lightgray;
        """
        )

        # right side control panel

        font = QtGui.QFont()
        font.setBold(True)

        # case select part
        self.case_select_label = QtWidgets.QLabel("Select Case")
        self.case_select_label.setFont(font)
        self.init_case_select_widgets()

        # cutting define part
        self.cutting_setting_label = QtWidgets.QLabel("Trajectory Settings")
        self.cutting_setting_label.setFont(font)

        line_cutting_depth_Hlayout = QtWidgets.QHBoxLayout()
        self.line_cutting_depth_label = QtWidgets.QLabel("Line Cutting Depth (mm): ")
        self.line_cutting_depth_spin = QtWidgets.QDoubleSpinBox()
        self.line_cutting_depth_spin.setRange(0.1, 100)
        self.line_cutting_depth_spin.setDecimals(1)
        self.line_cutting_depth_spin.setSingleStep(0.1)
        self.line_cutting_depth_spin.setValue(CONFIG["line_cutting_depth"])
        line_cutting_depth_Hlayout.addWidget(self.line_cutting_depth_label)
        line_cutting_depth_Hlayout.addWidget(self.line_cutting_depth_spin)

        self.start_trajectory_button = QtWidgets.QPushButton(
            "Start Trajectory Planning"
        )
        self.start_trajectory_button.clicked.connect(self.start_trajectory_planning)

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.case_select_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_select_combo)
        case_path_layout.addWidget(self.case_refresh_button)
        controls_layout.addLayout(case_path_layout)
        controls_layout.addLayout(line_cutting_depth_Hlayout)

        controls_layout.addWidget(self.start_trajectory_button)

        controls_layout.addStretch()

        # horizontal layout for all
        all_layout = QtWidgets.QHBoxLayout()
        all_layout.addWidget(self.image_label)
        all_layout.addLayout(controls_layout)

        return all_layout

    def start_trajectory_planning(self):
        if hasattr(self, "traj_thread") and self.traj_thread.isRunning():
            self.traj_thread.terminate()
            self.traj_thread.wait()

        # get case name and data path
        self.get_case_info()

        # try to load centerline results and assign to variable
        try:
            with open(
                os.path.join(self.temp_file_path, self.centerline_result_name), "rb"
            ) as f:
                loaded_centerline_data = pickle.load(f)
            self.mask_action_binaries = loaded_centerline_data["mask_action_binaries"]
            self.line_dict = loaded_centerline_data["line_dict"]
            self.reverse_mask_dict = loaded_centerline_data["reverse_mask_dict"]
            self.append_message("Centerline results loaded successfully", "info")
        except Exception as e:
            self.append_message(f"Failed to load centerline results: {e}", "error")
            return

        self.traj_thread = TrajectoryThread()
        self.traj_thread.line_dict = self.line_dict
        self.traj_thread.reverse_mask_dict = self.reverse_mask_dict
        self.traj_thread.mask_action_binaries = self.mask_action_binaries
        self.traj_thread.temp_file_path = os.path.join(
            self.temp_file_path, self.trajectory_saving_subfolder
        )
        self.traj_thread.line_cutting_depth = self.line_cutting_depth_spin.value()
        self.traj_thread.depth_forward_steps = CONFIG["depth_forward_steps"]

        # connect signals
        self.traj_thread.coarse_trajectory_signal.connect(self.update_coarse_trajectory)
        self.traj_thread.fine_trajectory_signal.connect(self.update_fine_trajectory)
        self.traj_thread.ultra_fine_trajectory_signal.connect(
            self.update_ultra_fine_trajectory
        )
        self.traj_thread.depth_map_signal.connect(self.update_depth_map)
        self.traj_thread.coarse_trajectory_drawing_signal.connect(
            self.update_coarse_traj_image
        )
        self.traj_thread.message_signal.connect(self.append_message)

        # start trajectory planning thread
        self.traj_thread.start()

    @QtCore.pyqtSlot(list)
    def update_coarse_trajectory(self, coarse_trajectory):
        del self.coarse_trajectory_holders
        self.coarse_trajectory_holders = coarse_trajectory

    @QtCore.pyqtSlot(list)
    def update_fine_trajectory(self, fine_trajectory):
        del self.fine_trajectory_holders
        self.fine_trajectory_holders = fine_trajectory

    @QtCore.pyqtSlot(list)
    def update_ultra_fine_trajectory(self, ultra_fine_trajectory):
        del self.ultra_fine_trajectory_holders
        self.ultra_fine_trajectory_holders = ultra_fine_trajectory

    @QtCore.pyqtSlot(list)
    def update_depth_map(self, depth_map):
        del self.depth_map_holders
        self.depth_map_holders = depth_map

    @QtCore.pyqtSlot(np.ndarray)
    def update_coarse_traj_image(self, coarse_traj_image):
        qt_pixmap = self.convert_cv_qt(coarse_traj_image)
        self.image_label.setPixmap(qt_pixmap)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.append_message("Coarse trajectory image visualized", "info")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrajecotryGUI()
    window.show()
    sys.exit(app.exec_())
