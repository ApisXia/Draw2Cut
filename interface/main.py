import os
import sys
import cv2
import shutil
import datetime
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from interface.pages.capture_gui import CaptureGUI


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Draw2Cut Interface")

        # define message box
        self.message_box = QtWidgets.QTextEdit()
        self.message_box.setReadOnly(True)
        self.message_box.setFixedHeight(100)

        # create pages
        self.pages = QtWidgets.QTabWidget()

        # page1: capture layout
        self.capture_layout = CaptureGUI(message_box=self.message_box)
        self.pages.addTab(self.capture_layout, "Step1: Capture")

        # page2: generate trajectory
        self.trajectory_layout = CaptureGUI(message_box=self.message_box)
        self.pages.addTab(self.trajectory_layout, "Step2: Planning")

        # main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.pages)
        main_layout.addWidget(self.message_box)

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
