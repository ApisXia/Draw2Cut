import os
import sys
import cv2
import shutil
import datetime
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from configs.load_config import CONFIG
from interface.functions.capture_thread import CaptureThread
from interface.functions.gui_mixins import MessageBoxMixin


class CaptureGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(CaptureGUI, self).__init__()

        # debugging setting
        self.use_ordinary_camera = False

        # setup layout
        page_layout = self.create_layout()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(page_layout, stretch=7)

        if message_box is not None:
            self.message_box = message_box
        else:
            # define left message box
            self.message_box = QtWidgets.QTextEdit()
            self.message_box.setReadOnly(True)
            self.message_box.setFixedHeight(100)
            main_layout.addWidget(self.message_box, stretch=1)

        self.setLayout(main_layout)

        self.capture_thread = CaptureThread()

    def create_layout(self):
        # image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(1080, 800)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.image_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: lightgray;
        """
        )
        # depth display
        self.depth_label = QtWidgets.QLabel()
        self.depth_label.setMinimumSize(1080, 800)
        self.depth_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.depth_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: darkgray;
        """
        )

        # use stacked layout to switch between image and depth
        self.stacked_layout = QtWidgets.QStackedLayout()
        self.stacked_layout.addWidget(self.image_label)
        self.stacked_layout.addWidget(self.depth_label)

        # add switch button
        self.switch_button = QtWidgets.QPushButton("Show Depth")
        self.switch_button.clicked.connect(self.switch_display)

        # add overlay description for each display
        self.display_widget = QtWidgets.QWidget()
        self.display_layout = QtWidgets.QStackedLayout()
        self.display_widget.setLayout(self.display_layout)

        # right side control panel
        font = QtGui.QFont()
        font.setBold(True)

        self.capture_label = QtWidgets.QLabel("Capture Settings")
        self.capture_label.setFont(font)

        if self.use_ordinary_camera:
            self.camera_label = QtWidgets.QLabel("Select Camera:")
            self.camera_combo = QtWidgets.QComboBox()
            self.populate_cameras()

        case_name_Hlayout = QtWidgets.QHBoxLayout()
        self.case_name_label = QtWidgets.QLabel("Case Name:")
        self.case_name_edit = QtWidgets.QLineEdit(CONFIG["case_name"])
        case_name_Hlayout.addWidget(self.case_name_label, 1)
        case_name_Hlayout.addWidget(self.case_name_edit, 1)

        exposure_Hlayout = QtWidgets.QHBoxLayout()
        self.exposure_label = QtWidgets.QLabel("Exposure Level:")
        self.exposure_spin = QtWidgets.QSpinBox()
        self.exposure_spin.setRange(1, 500)
        self.exposure_spin.setValue(CONFIG["exposure_level"])
        exposure_Hlayout.addWidget(self.exposure_label, 1)
        exposure_Hlayout.addWidget(self.exposure_spin, 1)

        samping_number_Hlayout = QtWidgets.QHBoxLayout()
        self.sampling_number_label = QtWidgets.QLabel("Image Sampling Number:")
        self.sampling_number_spin = QtWidgets.QSpinBox()
        self.sampling_number_spin.setRange(1, 50)
        self.sampling_number_spin.setValue(CONFIG["image_sampling_size"])
        samping_number_Hlayout.addWidget(self.sampling_number_label, 1)
        samping_number_Hlayout.addWidget(self.sampling_number_spin, 1)

        depth_queue_Hlayout = QtWidgets.QHBoxLayout()
        self.depth_queue_label = QtWidgets.QLabel("Depth Queue Size:")
        self.depth_queue_spin = QtWidgets.QSpinBox()
        self.depth_queue_spin.setRange(1, 100)
        self.depth_queue_spin.setValue(CONFIG["depth_queue_size"])
        depth_queue_Hlayout.addWidget(self.depth_queue_label, 1)
        depth_queue_Hlayout.addWidget(self.depth_queue_spin, 1)

        self.save_checkbox = QtWidgets.QCheckBox("Save Data")
        self.save_checkbox.setChecked(True)

        self.start_button = QtWidgets.QPushButton("Start Capture")
        self.start_button.clicked.connect(self.start_capture)

        self.stop_button = QtWidgets.QPushButton("Stop Capture")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)

        # vertical layout for controls
        controls_widget = QtWidgets.QWidget()
        controls_widget.setFixedWidth(400)
        controls_widget.setFixedHeight(800)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)

        controls_layout.addWidget(self.capture_label)
        if self.use_ordinary_camera:
            controls_layout.addWidget(self.camera_label)
            controls_layout.addWidget(self.camera_combo)

        controls_layout.addLayout(case_name_Hlayout)
        controls_layout.addLayout(exposure_Hlayout)
        controls_layout.addLayout(samping_number_Hlayout)
        controls_layout.addLayout(depth_queue_Hlayout)

        controls_layout.addWidget(self.switch_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.save_checkbox)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        # horizontal layout for capture
        capture_layout = QtWidgets.QHBoxLayout()
        capture_layout.addLayout(self.stacked_layout)
        capture_layout.addWidget(controls_widget)

        return capture_layout

    def start_capture(self):
        # if has running thread, stop it first, then start a new one
        if hasattr(self, "capture_thread") and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread.wait()
            del self.capture_thread

        self.capture_thread = CaptureThread()
        if self.use_ordinary_camera:
            selected_camera = self.camera_combo.currentData()
            self.capture_thread.camera_index = selected_camera
        self.capture_thread.case_name = self.case_name_edit.text()
        self.capture_thread.exposure_level = self.exposure_spin.value()
        self.capture_thread.sampling_number = self.sampling_number_spin.value()
        self.capture_thread.depth_queue_size = self.depth_queue_spin.value()
        self.capture_thread.saving_opt = self.save_checkbox.isChecked()
        self.capture_thread.use_ordinary_camera = self.use_ordinary_camera

        # connect the signal to slot
        self.capture_thread.image_updated.connect(self.update_image)
        self.capture_thread.depth_updated.connect(self.update_depth)
        self.capture_thread.message_signal.connect(self.append_message)

        # Pop up windows to ask
        if self.save_checkbox.isChecked():
            saving_path = os.path.join("data", self.capture_thread.case_name)
            if os.path.exists(saving_path):
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Folder Exists",
                    "The folder already exists. Do you want to overwrite?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.No:
                    self.append_message("Capture aborted.", "warning")
                    return
                else:
                    self.append_message("Overwriting the existing folder...", "warning")
                    # delete and recreate folders
                    shutil.rmtree(saving_path)
                    os.makedirs(saving_path)
                    self.append_message("Folder overwritten successfully.", "info")
            else:
                os.makedirs(saving_path)

        self.capture_thread.start()
        if self.use_ordinary_camera:
            self.camera_combo.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_checkbox.setEnabled(False)

    def stop_capture(self):
        if hasattr(self, "capture_thread") and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread.wait()
            # del the thread
            del self.capture_thread

        if self.use_ordinary_camera:
            self.camera_combo.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_checkbox.setEnabled(True)
        self.append_message("Capture stopped.", "info")

    # receive the images from the capture thread
    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """update the image in the label"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setPixmap(
            qt_img.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    @QtCore.pyqtSlot(np.ndarray)
    def update_depth(self, cv_img):
        """update the depth in the label"""
        qt_img = self.convert_cv_qt(cv_img)
        self.depth_label.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_label.setPixmap(
            qt_img.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def switch_display(self):
        """switch between image and depth display"""
        current_index = self.stacked_layout.currentIndex()
        next_index = (current_index + 1) % self.stacked_layout.count()
        self.stacked_layout.setCurrentIndex(next_index)
        # set button text
        if next_index == 0:
            self.switch_button.setText("Show Depth")
        else:
            self.switch_button.setText("Show Color Image")

    def populate_cameras(self):
        """list all available cameras"""
        self.camera_combo.clear()
        available_cameras = self.get_available_cameras()
        for index in available_cameras:
            self.camera_combo.addItem(f"Cam {index}", index)

    def get_available_cameras(self, max_cameras=10):
        """detect all available cameras"""
        available = []
        for index in range(max_cameras):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available.append(index)
                cap.release()
        return available


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CaptureGUI()
    window.show()
    sys.exit(app.exec_())
