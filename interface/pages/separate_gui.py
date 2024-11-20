import os
import sys
import threading

from glob import glob
from PyQt5 import QtGui, QtWidgets

from configs.load_config import CONFIG
from separate_wood_surface import seperate_wood_surface
from interface.functions.gui_mixins import MessageBoxMixin


class SeperateGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
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
        self.image_label.setFixedSize(1280, 800)
        self.image_label.setStyleSheet(
            """
            border: 1px solid black;
            background-color: lightgray;
        """
        )

        # control layout
        self.case_select_label = QtWidgets.QLabel("Select Case:")
        self.case_select_combo = QtWidgets.QComboBox()
        self.refresh_folder_list()

        self.case_refresh_button = QtWidgets.QPushButton()
        self.case_refresh_button.setText("Refresh")
        self.case_refresh_button.clicked.connect(self.refresh_folder_list)

        self.origin_label = QtWidgets.QLabel("origin_label:")
        self.origin_label_input = QtWidgets.QLineEdit(CONFIG["origin_label"])

        self.x_axis_label = QtWidgets.QLabel("x_axis_label:")
        self.x_axis_label_input = QtWidgets.QLineEdit(",".join(CONFIG["x_axis_label"]))

        self.y_axis_label = QtWidgets.QLabel("y_axis_label:")
        self.y_axis_label_input = QtWidgets.QLineEdit(",".join(CONFIG["y_axis_label"]))

        self.start_button = QtWidgets.QPushButton("Start Separation")
        self.start_button.clicked.connect(self.start_separation)

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.case_select_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_select_combo)
        case_path_layout.addWidget(self.case_refresh_button)
        control_layout.addLayout(case_path_layout)
        control_layout.addWidget(self.origin_label)
        control_layout.addWidget(self.origin_label_input)
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

    # check the x_axis and y_axis label input format, at least 1
    def check_x_axis_label(self):
        x_axis_label = self.x_axis_label_input.text().split(",")
        if len(x_axis_label) <= 1 and x_axis_label[0] == "":
            self.append_message("Please input correct x axis labels", "error")
            self.x_axis_label_input.setFocus()
            return False
        return True

    def check_y_axis_label(self):
        y_axis_label = self.y_axis_label_input.text().split(",")
        if len(y_axis_label) <= 1 and y_axis_label[0] == "":
            self.append_message("Please input correct y axis labels", "error")
            self.y_axis_label_input.setFocus()
            return False
        return True

    def start_separation(self):
        # check x_axis and y_axis label
        if not self.check_x_axis_label() or not self.check_y_axis_label():
            return

        # get case name and data path
        case_name = self.case_select_combo.currentText()
        if not case_name:
            self.message_box.append("Can not get viable case name", "error")
            return

        data_path = CONFIG["data_path_template"].format(case_name=case_name)
        temp_file_path = CONFIG["temp_file_path_template"].format(case_name=case_name)
        if not os.path.exists(temp_file_path):
            os.makedirs(temp_file_path)

        # [ ]: need to be modified
        # self.origin = str(self.origin_label_spin.value())
        # self.x_axis = self.x_axis_label_input.text()
        # self.y_axis = self.y_axis_label_input.text()
        # self.message_box.append("Start Separation")
        # self.message_box.append(f"Origin Label: {self.origin}")
        # self.message_box.append(f"X Axis Label: {self.x_axis}")
        # self.message_box.append(f"Y Axis Label: {self.y_axis}")

        seperate_wood_surface(
            data_path,
            temp_file_path,
            self.origin_label_input.text(),
            self.x_axis_label_input.text().split(","),
            self.y_axis_label_input.text().split(","),
        )

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
