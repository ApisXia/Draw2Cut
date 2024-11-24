import os
import json
import sys
import cv2
import numpy as np

from glob import glob
from functools import partial
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QHeaderView, QPushButton, QToolButton

from configs.load_config import CONFIG
from interface.functions.gui_mixins import MessageBoxMixin
from interface.functions.mask_extract_thread import MaskExtractThread


"""_summary_
    Two functions:
    1. Extract color mask and visualize, also visualize different semantic masks by clicking list.
    2. Can check the centerline of each type.
"""


class MaskExtractGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(MaskExtractGUI, self).__init__()

        # color type values setting
        with open(CONFIG["color_value_setting_file"], "r") as f:
            self.color_type_values = json.load(f)
        self.color_type_saving_path = "saving_test.json"

        # set saving subfolder
        self.mask_saving_subfolder = "mask_extract"

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

        # result variables
        self.colored_mask = None
        self.semantic_mask_dict = None
        self.separate_image_name = "wrapped_image_zoom.png"

        self.mask_extract_thread = MaskExtractThread()

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

        # ? case select part
        self.case_select_label = QtWidgets.QLabel("Select Case:")
        self.case_select_combo = QtWidgets.QComboBox()
        self.refresh_folder_list()

        self.case_refresh_button = QtWidgets.QPushButton()
        self.case_refresh_button.setText("Refresh")
        self.case_refresh_button.clicked.connect(self.refresh_folder_list)

        # ? color value control part
        self.color_value_table = QTableWidget()
        self.color_value_table.setColumnCount(4)
        self.color_value_table.setFixedWidth(320)
        self.color_value_table.setFixedHeight(180)
        self.color_value_table.setHorizontalHeaderLabels(
            ["Type", "HSV Value", "Action", ""]
        )

        self.color_value_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.color_value_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.color_value_table.horizontalHeader().setStretchLastSection(True)
        self.color_value_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.color_value_table.resizeRowsToContents()
        self.color_value_table.setWordWrap(False)

        # Set column widths
        self.color_value_table.setColumnWidth(0, 50)  # Type
        self.color_value_table.setColumnWidth(1, 100)  # HSV Value
        self.color_value_table.setColumnWidth(2, 80)  # Action
        self.color_value_table.setColumnWidth(3, 20)  # Remove

        self.color_value_table.cellChanged.connect(self.handle_cell_changed)

        self.update_color_table()

        # add new color button
        self.add_color_button = QPushButton("+")
        self.add_color_button.setFixedHeight(180)
        self.add_color_button.setFixedWidth(40)
        self.add_color_button.clicked.connect(self.add_color)

        # add buttom to the left of the table
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.setSpacing(5)
        color_layout.addWidget(self.color_value_table)
        color_layout.addWidget(self.add_color_button)

        # add saving color type values button
        self.save_color_button = QPushButton("Save color type values")
        self.save_color_button.clicked.connect(self.save_color_type_values)

        # start color mask extraction
        self.mask_extract_button = QPushButton("Start Mask Extraction")
        self.mask_extract_button.clicked.connect(self.start_extract_mask)

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.case_select_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_select_combo)
        case_path_layout.addWidget(self.case_refresh_button)
        controls_layout.addLayout(case_path_layout)
        controls_layout.addLayout(color_layout)
        controls_layout.addWidget(self.save_color_button)
        controls_layout.addWidget(self.mask_extract_button)
        controls_layout.addStretch()
        # controls_layout.addWidget(self.case_name_edit)
        # controls_layout.addWidget(self.exposure_label)
        # controls_layout.addWidget(self.exposure_spin)
        # controls_layout.addWidget(self.sampling_number_label)
        # controls_layout.addWidget(self.sampling_number_spin)
        # controls_layout.addWidget(self.depth_queue_label)
        # controls_layout.addWidget(self.depth_queue_spin)
        # controls_layout.addWidget(self.switch_button)
        # controls_layout.addStretch()
        # controls_layout.addWidget(self.save_checkbox)
        # controls_layout.addWidget(self.start_button)
        # controls_layout.addWidget(self.stop_button)

        # horizontal layout for capture
        capture_layout = QtWidgets.QHBoxLayout()
        capture_layout.addWidget(self.image_label)
        capture_layout.addLayout(controls_layout)

        return capture_layout

    def start_extract_mask(self):
        if (
            hasattr(self, "mask_extract_thread")
            and self.mask_extract_thread.isRunning()
        ):
            self.mask_extract_thread.terminate()
            self.mask_extract_thread.wait()

        # get case name and data path
        self.get_case_info()

        self.mask_extract_thread = MaskExtractThread()
        self.mask_extract_thread.color_type_values = self.color_type_values
        self.mask_extract_thread.temp_file_path = os.path.join(
            self.temp_file_path, self.mask_saving_subfolder
        )

        self.mask_extract_thread.separated_image = cv2.imread(
            os.path.join(self.temp_file_path, self.separate_image_name)
        )

        # connect signals
        self.mask_extract_thread.colored_mask_signal.connect(self.update_colored_mask)
        self.mask_extract_thread.semantic_mask_dict_signal.connect(
            self.update_semantic_mask_dict
        )

        # start the thread
        self.mask_extract_thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def update_colored_mask(self, colored_mask):
        del self.colored_mask
        self.colored_mask = colored_mask
        # also update the image label
        qt_img = self.convert_cv_qt(colored_mask)
        self.image_label.setPixmap(qt_img)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

    @QtCore.pyqtSlot(dict)
    def update_semantic_mask_dict(self, semantic_mask_dict):
        del self.semantic_mask_dict
        self.semantic_mask_dict = semantic_mask_dict

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

    def update_color_table(self):
        # avoid signal emitting
        self.color_value_table.blockSignals(True)

        # set the color value table length
        self.color_value_table.setRowCount(len(self.color_type_values))

        for i, color_info in enumerate(self.color_type_values):
            self.color_value_table.setItem(
                i, 0, QtWidgets.QTableWidgetItem(color_info["type"])
            )
            self.color_value_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(str(color_info["HSV_value"]))
            )
            self.color_value_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(color_info["action"])
            )

            # Convert HSV to RGB
            hsv = color_info["HSV_value"]
            hsv_np = np.uint8([[hsv]])
            rgb_np = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2RGB)
            rgb = rgb_np[0][0]
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))

            # Apply background color to the row
            for col in range(self.color_value_table.columnCount()):
                item = self.color_value_table.item(i, col)
                if item:
                    item.setBackground(color)

            # Add action button
            action_button = QToolButton()
            action_button.setText("Del")
            action_button.clicked.connect(partial(self.delete_color, i))
            self.color_value_table.setCellWidget(i, 3, action_button)

        # re-enable signal emitting
        self.color_value_table.blockSignals(False)

    def delete_color(self, row):
        # remove the color from the list
        if 0 <= row < len(self.color_type_values):
            del self.color_type_values[row]
            # update the table
            self.update_color_table()

    def add_color(self):
        # add a new color to the list
        self.color_type_values.insert(
            0,
            {
                "type": "New",
                "HSV_value": [0, 0, 255],
                "action": "not defined",
            },
        )
        # update the table
        self.update_color_table()

    def handle_cell_changed(self, row, column):
        if row < 0 or row >= len(self.color_type_values):
            return

        item = self.color_value_table.item(row, column)
        if not item:
            return

        self.append_message(f"Cell {row}, {column} changed to {item.text()}", "info")

        text = item.text()
        if column == 0:
            self.color_type_values[row]["type"] = text
            self.append_message("Type updated.", "info")
        elif column == 1:
            try:
                hsv = list(map(int, text.strip("[]").split(",")))
                if len(hsv) == 3:
                    self.color_type_values[row]["HSV_value"] = hsv
                    self.append_message("HSV value updated.", "info")
            except:
                self.append_message("HSV value format error.", "error")
                pass
        elif column == 2:
            self.color_type_values[row]["action"] = text
            self.append_message("Action updated.", "info")

        self.update_color_table()

    def save_color_type_values(self):
        with open(self.color_type_saving_path, "w") as file:
            json.dump(self.color_type_values, file, indent=4)
        self.append_message("Color type values saved.", "info")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MaskExtractGUI()
    window.show()
    sys.exit(app.exec_())
