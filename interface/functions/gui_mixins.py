import cv2
import datetime

from PyQt5 import QtCore, QtGui, QtWidgets


class MessageBoxMixin:
    @QtCore.pyqtSlot(str, str)
    def append_message(self, message, msg_type="info"):
        """add formatted message to the message box"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        # based on message type, set color
        color = "black"
        if msg_type == "info":
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
