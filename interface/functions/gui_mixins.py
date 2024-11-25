import os
import cv2
import datetime

from glob import glob
from PyQt5 import QtCore, QtGui, QtWidgets

from configs.load_config import CONFIG


class MessageBoxMixin:
    @QtCore.pyqtSlot(str, str)
    def append_message(self, message, msg_type="info"):
        """add formatted message to the message box"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        # based on message type, set color
        color = "black"

        if msg_type == "step":
            color = "lime"  # more visible green
        elif msg_type == "info":
            color = "green"
        elif msg_type == "warning":
            color = "orange"
        elif msg_type == "error":
            color = "red"
        # use HTML to format message
        formatted_message = (
            f'<span style="color:{color};">[{current_time}] {message}</span>'
        )
        self.message_box.append(formatted_message)

        # limit the number of messages
        max_blocks = 100  # maximum number of blocks
        if self.message_box.document().blockCount() > max_blocks:
            cursor = self.message_box.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def write(self, message):
        message = message.rstrip()
        if message:
            self.append_message(message)

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

    def define_separator(self):
        """define the separator"""
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        return separator

    """ Universal case selection function """

    def init_case_select_widgets(self):
        self.case_select_combo = QtWidgets.QComboBox()
        self.refresh_folder_list()
        self.case_select_combo.currentIndexChanged.connect(
            self.clean_result_variables
        )  # need to define for each class

        self.case_refresh_button = QtWidgets.QPushButton()
        self.case_refresh_button.setText("Refresh")
        self.case_refresh_button.clicked.connect(self.refresh_folder_list)

    def clean_result_variables(self):
        self.append_message("Cleaning result variables function not defined", "warning")
        pass

    def get_case_info(self):
        # get case name and data path
        self.case_name = self.case_select_combo.currentText()
        if not self.case_name:
            self.message_box.append("Can not get viable case name", "error")
            return

        self.data_path = CONFIG["data_path_template"].format(case_name=self.case_name)
        self.temp_file_path = CONFIG["temp_file_path_template"].format(
            case_name=self.case_name
        )
        if not os.path.exists(self.temp_file_path):
            os.makedirs(self.temp_file_path)

    # add function to refresh folder list
    def refresh_folder_list(self):
        case_list = [
            os.path.basename(os.path.normpath(path))
            for path in sorted(
                glob(CONFIG["case_folder_template"].format(case_name="*")),
                key=os.path.getmtime,
                reverse=True,
            )
        ]

        if len(case_list) > 0:
            self.case_select_combo.clear()
            self.case_select_combo.addItems(case_list)
        else:
            self.append_message("No case folder found", "error")
