import os
import sys
import cv2
import shutil
import datetime
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from interface.capture import CaptureThread


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Draw2Cut Interface")

        # ? set up for page 1

        # set tab pages
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.East)
        self.image_tab = QtWidgets.QWidget()
        self.depth_tab = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QVBoxLayout()
        self.depth_layout = QtWidgets.QVBoxLayout()

        # left side image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: lightgray;
        """
        )

        # left side depth display
        self.depth_label = QtWidgets.QLabel()
        self.depth_label.setFixedSize(1280, 720)
        self.depth_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: lightgray;
        """
        )

        # add all to tabs
        self.image_layout.addWidget(self.image_label)
        self.depth_layout.addWidget(self.depth_label)

        # 设置 image_layout 的边距和间距
        self.image_layout.setContentsMargins(0, 0, 0, 0)  # 左、上、右、下边距
        self.image_layout.setSpacing(0)  # 控件之间的间距

        # 设置 depth_layout 的边距和间距
        self.depth_layout.setContentsMargins(0, 0, 0, 0)
        self.depth_layout.setSpacing(0)

        self.image_tab.setLayout(self.image_layout)
        self.depth_tab.setLayout(self.depth_layout)
        self.tab_widget.addTab(self.image_tab, "Color Image")
        self.tab_widget.addTab(self.depth_tab, "Depth Image")

        # define left message box
        self.message_box = QtWidgets.QTextEdit()
        self.message_box.setReadOnly(True)
        self.message_box.setFixedHeight(100)

        # right side control panel
        self.case_name_label = QtWidgets.QLabel("Case Name:")
        self.case_name_edit = QtWidgets.QLineEdit("temp_case")

        self.exposure_label = QtWidgets.QLabel("Exposure Level:")
        self.exposure_spin = QtWidgets.QSpinBox()
        self.exposure_spin.setRange(1, 500)
        self.exposure_spin.setValue(140)

        self.sampling_number_label = QtWidgets.QLabel("Image Sampling Number:")
        self.sampling_number_spin = QtWidgets.QSpinBox()
        self.sampling_number_spin.setRange(1, 50)
        self.sampling_number_spin.setValue(20)

        self.depth_queue_label = QtWidgets.QLabel("Depth Queue Size:")
        self.depth_queue_spin = QtWidgets.QSpinBox()
        self.depth_queue_spin.setRange(1, 100)
        self.depth_queue_spin.setValue(50)

        self.save_checkbox = QtWidgets.QCheckBox("Save Data")
        self.save_checkbox.setChecked(True)

        self.start_button = QtWidgets.QPushButton("Start Capture")
        self.start_button.clicked.connect(self.start_capture)

        self.stop_button = QtWidgets.QPushButton("Stop Capture")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)

        # vertical layout for image display + message box
        image_layout = QtWidgets.QVBoxLayout()
        image_layout.addWidget(self.tab_widget)
        image_layout.addWidget(self.message_box)

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.case_name_label)
        controls_layout.addWidget(self.case_name_edit)
        controls_layout.addWidget(self.exposure_label)
        controls_layout.addWidget(self.exposure_spin)
        controls_layout.addWidget(self.sampling_number_label)
        controls_layout.addWidget(self.sampling_number_spin)
        controls_layout.addWidget(self.depth_queue_label)
        controls_layout.addWidget(self.depth_queue_spin)
        controls_layout.addStretch()
        controls_layout.addWidget(self.save_checkbox)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        # horizontal layout for main window
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

        self.capture_thread = CaptureThread()

    def start_capture(self):
        # if has running thread, stop it first, then start a new one
        if hasattr(self, "capture_thread") and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread.wait()
            del self.capture_thread

        self.capture_thread = CaptureThread()
        self.capture_thread.case_name = self.case_name_edit.text()
        self.capture_thread.exposure_level = self.exposure_spin.value()
        self.capture_thread.sampling_number = self.sampling_number_spin.value()
        self.capture_thread.depth_queue_size = self.depth_queue_spin.value()
        self.capture_thread.saving_opt = self.save_checkbox.isChecked()

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
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_checkbox.setEnabled(False)

    def stop_capture(self):
        if hasattr(self, "capture_thread") and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread.wait()
            # 清理线程实例
            del self.capture_thread
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_checkbox.setEnabled(True)

    # receive the images from the capture thread
    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """update the image in the label"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @QtCore.pyqtSlot(np.ndarray)
    def update_depth(self, cv_img):
        """update the depth in the label"""
        qt_img = self.convert_cv_qt(cv_img)
        self.depth_label.setPixmap(qt_img)

    @QtCore.pyqtSlot(str, str)
    def append_message(self, message, msg_type="info"):
        """在消息框中追加带格式的消息"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        # 根据消息类型设置颜色
        color = "black"
        if msg_type == "info":
            color = "blue"
        elif msg_type == "warning":
            color = "orange"
        elif msg_type == "error":
            color = "red"
        # 使用 HTML 格式化消息
        formatted_message = (
            f'<span style="color:{color};">[{current_time}] {message}</span>'
        )
        self.message_box.append(formatted_message)

        # 限制消息框中的最大消息数量
        max_blocks = 100  # 设置最大消息条数
        if self.message_box.document().blockCount() > max_blocks:
            cursor = self.message_box.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def convert_cv_qt(self, cv_img):
        """convert cv image to qt image"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
        qt_pixmap = qt_pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            QtCore.Qt.KeepAspectRatio,
        )
        return qt_pixmap


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
