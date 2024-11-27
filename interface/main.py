import os
import sys
import cv2
import shutil
import datetime
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from interface.pages.capture_gui import CaptureGUI
from interface.pages.separate_gui import SeperateGUI
from interface.pages.mask_extract_gui import MaskExtractGUI
from interface.pages.trajectory_gui import TrajectoryGUI


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Draw2Cut")
        self.setFixedSize(1600, 1000)

        # define message box
        self.message_box = QtWidgets.QTextEdit()
        self.message_box.setReadOnly(True)
        self.message_box.setFixedSize(1500, 100)

        # create pages
        self.pages = QtWidgets.QTabWidget()
        self.pages.setStyleSheet(
            """
            QTabWidget::pane {
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """
        )

        # Page1: capture
        self.capture_layout = CaptureGUI(message_box=self.message_box)
        self.pages.addTab(self.capture_layout, "Step1: Capture Pointcloud")

        # Page2: separate surface
        self.separate_layout = SeperateGUI(message_box=self.message_box)
        self.pages.addTab(self.separate_layout, "Step2: Separate Cutting Surface")

        # Page3: color mask extraction & Centerline extraction
        self.mask_extract_layout = MaskExtractGUI(message_box=self.message_box)
        self.pages.addTab(
            self.mask_extract_layout, "Step3: Mask && Centerline Extraction"
        )

        # Page4: cutting visualization & replay + gcode generation + save
        self.trajectory_layout = TrajectoryGUI(message_box=self.message_box)
        self.pages.addTab(
            self.trajectory_layout,
            "Step 4: Trajectory && Visualization && Gcode Generation",
        )

        # main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.pages, alignment=QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.message_box, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
