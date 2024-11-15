import os
import sys
import cv2
import time
import shutil
import datetime
import threading
import numpy as np
import open3d as o3d
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from separate_wood_surface import seperate_wood_surface

from PyQt5 import QtCore, QtGui, QtWidgets

from interface.functions.gui_mixins import MessageBoxMixin

from configs.load_config import CONFIG

# - Vis Panel: show separated surface
# - Control: 
#   - Select case
#   - origin_label = CONFIG["origin_label"]
#   - x_axis_label = CONFIG["x_axis_label"]
#   - y_axis_label = CONFIG["y_axis_label"]

class SeperateGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self,message_box: QtWidgets.QTextEdit = None):
        super(SeperateGUI, self).__init__()

        # set close event
        self.stop_event = threading.Event()
        self.thread = None

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

    def create_layout(self):
        # create layout
        layout = QtWidgets.QHBoxLayout()

        # image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: lightgray;
        """
        )

        self.case_label = QtWidgets.QLabel("Select Case:")
        self.case_path = ""
        self.case_path_eidt = QtWidgets.QLineEdit(self.case_path)
        self.case_choose_button = QtWidgets.QPushButton()
        folder_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon)
        self.case_choose_button.setIcon(folder_icon)
        self.case_choose_button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #A0A0A0;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #A0A0A0;
            }
        """)
        self.case_choose_button.clicked.connect(self.select_folder)

        self.origin_label = QtWidgets.QLabel("origin_label:")
        self.origin_label_spin = QtWidgets.QSpinBox()
        self.origin_label_spin.setRange(0, 10)
        self.origin = int(CONFIG["origin_label"])
        self.origin_label_spin.setValue(self.origin)

        self.x_axis_label = QtWidgets.QLabel("x_axis_label:")
        self.x_axis = ",".join(CONFIG["x_axis_label"])
        self.x_axis_label_input = QtWidgets.QLineEdit(self.x_axis)

        self.y_axis_label = QtWidgets.QLabel("y_axis_label:")
        self.y_axis = ",".join(CONFIG["y_axis_label"])
        self.y_axis_label_input = QtWidgets.QLineEdit(self.y_axis)

        self.start_button = QtWidgets.QPushButton("Start Separation")
        self.start_button.clicked.connect(self.start_separation)


        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.case_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_path_eidt)
        case_path_layout.addWidget(self.case_choose_button)
        control_layout.addLayout(case_path_layout)
        control_layout.addWidget(self.origin_label)
        control_layout.addWidget(self.origin_label_spin)
        control_layout.addWidget(self.x_axis_label)
        control_layout.addWidget(self.x_axis_label_input)
        control_layout.addWidget(self.y_axis_label)
        control_layout.addWidget(self.y_axis_label_input)
        control_layout.addStretch()
        control_layout.addWidget(self.start_button)

        # horizontal layout for seperate
        seperate_layout = QtWidgets.QHBoxLayout()
        seperate_layout.addWidget(self.image_label)
        seperate_layout.addLayout(control_layout)
        return seperate_layout

    def select_folder(self):
        self.case_path = QtWidgets.QFileDialog.getExistingDirectory(self, "select case", options=QtWidgets.QFileDialog.ShowDirsOnly)
        self.case_path_eidt.setText(self.case_path)

    def start_separation(self):
        self.origin = str(self.origin_label_spin.value())
        self.x_axis = self.x_axis_label_input.text()
        self.y_axis = self.y_axis_label_input.text()
        self.case_path = self.case_path_eidt.text()
        if self.case_path != "":
            case_name = os.path.basename(self.case_path.rstrip("/"))
            CONFIG["temp_file_path"] = CONFIG["temp_file_path_template"].format(
            case_name=case_name
            )
        self.message_box.append("Start Separation")
        self.message_box.append(f"Origin Label: {self.origin}")
        self.message_box.append(f"X Axis Label: {self.x_axis}")
        self.message_box.append(f"Y Axis Label: {self.y_axis}")
        self.message_box.append(f"Case Path: {self.case_path}")
        if self.case_path != "":
            data_path = os.path.join(self.case_path, "point_cloud.npz")
        else:
            data_path = CONFIG["data_path"]
        sys.stdout = self
        seperate_wood_surface(data_path,self.origin,self.x_axis,self.y_axis)

        temp_output_path = CONFIG["temp_file_path"]
        seperate_image_path = os.path.join(temp_output_path, "wrapped_image_zoom.png")
        if os.path.exists(seperate_image_path):
            self.show_image(seperate_image_path)

    def show_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SeperateGUI()
    window.show()
    sys.exit(app.exec_())



