import os
import json
import sys
import cv2
import numpy as np

from glob import glob
from functools import partial
from PyQt5.QtGui import QColor, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QHeaderView, QPushButton, QToolButton

from configs.load_config import CONFIG
from interface.functions.gui_mixins import MessageBoxMixin
from interface.functions.mask_extract_thread import MaskExtractThread
from interface.functions.centerline_thread import CenterlineThread


"""_summary_
    Two functions:
    1. (Done) Extract color mask and visualize, also visualize different semantic masks by clicking list.
    2. (Done) Can check the centerline of each type.
"""


class MaskExtractGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(MaskExtractGUI, self).__init__()

        # color type values setting
        with open(CONFIG["color_value_setting_file"], "r") as f:
            self.color_type_values = json.load(f)
        self.color_type_saving_path = "saving_test.json"

        self.all_type_lines = CONFIG["contour_mark"] + CONFIG["behavior_mark"]

        # predefined separated image name
        self.separate_image_name = "wrapped_image_zoom.png"

        # set saving subfolder
        self.mask_saving_subfolder = "mask_extract"
        self.centerline_saving_subfolder = "centerline_extract"

        # result variables of mask extraction
        self.colored_mask = None
        self.semantic_mask_dict = None
        self.action_mapping_dict = None

        # result variables of centerline extraction
        self.mask_action_binaries = None
        self.line_dict = None
        self.centerline_images_dict = None
        self.reverse_mask_dict = None

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

        # define threads
        self.mask_extract_thread = MaskExtractThread()
        self.centerline_thread = CenterlineThread()

    def create_layout(self):
        # image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(1080, 800)
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

        # ? case select part
        self.case_select_label = QtWidgets.QLabel("Select Case")
        self.case_select_label.setFont(font)

        self.case_select_combo = QtWidgets.QComboBox()
        self.refresh_folder_list()
        self.case_select_combo.currentIndexChanged.connect(self.clean_result_variables)

        self.case_refresh_button = QtWidgets.QPushButton()
        self.case_refresh_button.setText("Refresh")
        self.case_refresh_button.clicked.connect(self.refresh_folder_list)

        # ? color value control part
        self.color_label = QtWidgets.QLabel("Color Mask Extraction")
        self.color_label.setFont(font)

        self.color_value_table = QTableWidget()
        self.color_value_table.setColumnCount(5)
        self.color_value_table.setFixedWidth(350)
        self.color_value_table.setFixedHeight(180)
        self.color_value_table.setHorizontalHeaderLabels(
            ["Type", "HSV Value", "Action", "", ""]
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
        self.color_value_table.setColumnWidth(4, 20)  # Remove

        self.color_value_table.cellChanged.connect(self.handle_cell_changed)

        self.update_color_table()

        # add new color button
        self.add_color_button = QPushButton("Add New Color")
        self.add_color_button.clicked.connect(self.add_color)

        # add saving color type values button
        self.save_color_button = QPushButton("Save Current")
        self.save_color_button.clicked.connect(self.save_color_type_values)

        # botton layout
        color_botton_layout = QtWidgets.QHBoxLayout()
        color_botton_layout.setSpacing(5)
        color_botton_layout.addWidget(self.add_color_button)
        color_botton_layout.addWidget(self.save_color_button)

        # start color mask extraction
        self.mask_extract_button = QPushButton("Start Mask Extraction")
        self.mask_extract_button.clicked.connect(self.start_extract_mask)

        # centerline extraction
        self.centerline_section_label = QtWidgets.QLabel("Centerline Extraction")
        self.centerline_section_label.setFont(font)

        self.centerline_smooth_label = QtWidgets.QLabel("Smooth Size:")
        self.centerline_smooth_spin = QtWidgets.QSpinBox()
        self.centerline_smooth_spin.setRange(0, 20)
        self.centerline_smooth_spin.setSingleStep(1)
        self.centerline_smooth_spin.setValue(CONFIG["smooth_size"])
        centerline_smooth_Hlayout = QtWidgets.QHBoxLayout()
        centerline_smooth_Hlayout.addWidget(self.centerline_smooth_label)
        centerline_smooth_Hlayout.addWidget(self.centerline_smooth_spin)

        self.centerline_extract_button = QPushButton("Start\nCenterline\nExtraction")
        self.centerline_extract_button.setFixedHeight(95)
        self.centerline_extract_button.setFixedWidth(100)
        self.centerline_extract_button.clicked.connect(self.start_extract_centerline)

        self.centerline_vis_table = QTableWidget()
        self.centerline_vis_table.setColumnCount(2)
        self.centerline_vis_table.setFixedWidth(250)
        self.centerline_vis_table.setFixedHeight(90)
        self.centerline_vis_table.setHorizontalHeaderLabels(["Action", "Visualize"])

        self.centerline_vis_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.centerline_vis_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        self.centerline_vis_table.horizontalHeader().setStretchLastSection(True)
        self.centerline_vis_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.centerline_vis_table.resizeRowsToContents()
        self.centerline_vis_table.setWordWrap(False)

        self.centerline_vis_table.setColumnWidth(0, 100)  # Type
        self.centerline_vis_table.setColumnWidth(1, 50)  # Visualize

        self.update_centerline_table()

        centerline_Hlayout = QtWidgets.QHBoxLayout()
        centerline_Hlayout.addWidget(self.centerline_extract_button)
        centerline_Hlayout.setSpacing(7)
        centerline_Hlayout.addWidget(self.centerline_vis_table)

        # saving button
        self.save_section_label = QtWidgets.QLabel("Save Results")
        self.save_section_label.setFont(font)
        self.save_button = QPushButton("Save to File")
        self.save_button.clicked.connect(self.save_centerline_results)

        # create horizontal separator
        separator1 = self.define_separator()
        separator2 = self.define_separator()
        separator3 = self.define_separator()

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.case_select_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_select_combo)
        case_path_layout.addWidget(self.case_refresh_button)
        controls_layout.addLayout(case_path_layout)

        controls_layout.addWidget(separator1)

        controls_layout.addWidget(self.color_label)
        controls_layout.addWidget(self.color_value_table)
        controls_layout.addSpacing(-10)
        controls_layout.addLayout(color_botton_layout)
        controls_layout.addWidget(self.mask_extract_button)

        controls_layout.addWidget(separator2)

        controls_layout.addWidget(self.centerline_section_label)
        controls_layout.addLayout(centerline_smooth_Hlayout)
        controls_layout.addLayout(centerline_Hlayout)

        controls_layout.addWidget(separator3)

        controls_layout.addWidget(self.save_section_label)
        controls_layout.addWidget(self.save_button)

        controls_layout.addStretch()

        # horizontal layout for capture
        capture_layout = QtWidgets.QHBoxLayout()
        capture_layout.addWidget(self.image_label)
        capture_layout.addLayout(controls_layout)

        return capture_layout

    """ Extract Mask Part """

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
        if self.mask_extract_thread.separated_image is None:
            self.append_message("Can not read separated image", "error")
            return

        # connect signals
        self.mask_extract_thread.colored_mask_signal.connect(self.update_colored_mask)
        self.mask_extract_thread.semantic_mask_dict_signal.connect(
            self.update_semantic_mask_dict
        )

        # start the thread
        self.mask_extract_thread.finished.connect(self.update_color_table)
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

    """ Centerline Part """

    def start_extract_centerline(self):
        if hasattr(self, "centerline_thread") and self.centerline_thread.isRunning():
            self.centerline_thread.terminate()
            self.centerline_thread.wait()

        # get case name and data path
        self.get_case_info()

        if self.semantic_mask_dict is None:
            self.append_message("No semantic mask found", "error")
            return

        self.centerline_thread = CenterlineThread()
        self.centerline_thread.semantic_mask_dict = self.semantic_mask_dict
        self.centerline_thread.temp_file_path = os.path.join(
            self.temp_file_path, self.centerline_saving_subfolder
        )
        self.centerline_thread.action_mapping_dict = self.action_mapping_dict
        self.centerline_thread.smooth_size = self.centerline_smooth_spin.value()

        # connect signals
        self.centerline_thread.mask_action_binaries_signal.connect(
            self.update_mask_action_binaries
        )
        self.centerline_thread.line_dict_signal.connect(self.update_line_dict)
        self.centerline_thread.centerline_images_dict_signal.connect(
            self.update_centerline_images_dict
        )
        self.centerline_thread.reverse_mask_dict_signal.connect(
            self.update_reverse_mask_dict
        )
        self.centerline_thread.message_signal.connect(self.append_message)

        # start the thread
        self.centerline_thread.finished.connect(self.update_centerline_table)
        self.centerline_thread.start()

    @QtCore.pyqtSlot(dict)
    def update_mask_action_binaries(self, mask_action_binaries):
        del self.mask_action_binaries
        self.mask_action_binaries = mask_action_binaries

    @QtCore.pyqtSlot(dict)
    def update_line_dict(self, line_dict):
        del self.line_dict
        self.line_dict = line_dict

    @QtCore.pyqtSlot(dict)
    def update_centerline_images_dict(self, centerline_images_dict):
        del self.centerline_images_dict
        self.centerline_images_dict = centerline_images_dict

    @QtCore.pyqtSlot(dict)
    def update_reverse_mask_dict(self, reverse_mask_dict):
        del self.reverse_mask_dict
        self.reverse_mask_dict = reverse_mask_dict

    """ Common Functions """

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

    def update_action_mapping_dict(self):
        action_mapping_dict = {}
        for item in self.color_type_values:
            if (
                item["action"] in CONFIG["contour_mark"] + CONFIG["behavior_mark"]
            ):  # currently only support these two functions
                action_mapping_dict[item["type"]] = item["action"]
        self.action_mapping_dict = action_mapping_dict

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

            # Add visualize binary mask button
            font = QFont()
            font.setPointSize(7)
            visualize_button = QToolButton()
            if (
                self.semantic_mask_dict is not None
                and color_info["type"] in self.semantic_mask_dict.keys()
            ):
                visualize_button.setText("💡")
                visualize_button.setFont(font)
                visualize_button.clicked.connect(
                    partial(self.visualize_each_mask, color_info["type"])
                )
            else:
                visualize_button.setText("❌")
                visualize_button.setFont(font)
                visualize_button.setDisabled(True)
            self.color_value_table.setCellWidget(i, 3, visualize_button)

            # Add action button
            action_button = QToolButton()
            action_button.setText("Del")
            action_button.clicked.connect(partial(self.delete_color, i))
            self.color_value_table.setCellWidget(i, 4, action_button)

        # re-enable signal emitting
        self.color_value_table.blockSignals(False)

        # also update the action mapping dict
        self.update_action_mapping_dict()

    def update_centerline_table(self):
        # avoid signal emitting
        self.centerline_vis_table.blockSignals(True)

        # set the color value table length
        self.centerline_vis_table.setRowCount(len(self.all_type_lines))

        for i, mark_type_name in enumerate(self.all_type_lines):
            self.centerline_vis_table.setItem(
                i, 0, QtWidgets.QTableWidgetItem(mark_type_name)
            )

            # Add visualize binary mask button
            font = QFont()
            font.setPointSize(7)
            visualize_button = QToolButton()
            if self.centerline_images_dict is not None:
                if mark_type_name in self.centerline_images_dict.keys():
                    visualize_button.setText("💡")
                    visualize_button.setFont(font)
                    visualize_button.clicked.connect(
                        partial(self.visualize_each_centerline, mark_type_name)
                    )
                else:
                    visualize_button.setText("❌")
                    visualize_button.setFont(font)
                    visualize_button.setDisabled(True)
            else:
                visualize_button.setText("❌")
                visualize_button.setFont(font)
                visualize_button.setDisabled(True)

            self.centerline_vis_table.setCellWidget(i, 1, visualize_button)

        # re-enable signal emitting
        self.centerline_vis_table.blockSignals(False)

    def visualize_each_mask(self, color_type):
        if self.semantic_mask_dict is None:
            self.append_message("No semantic mask found", "error")
            return

        if color_type not in self.semantic_mask_dict.keys():
            self.append_message(f"No {color_type} mask found", "error")
            return

        mask = self.semantic_mask_dict[color_type]
        qt_img = self.convert_cv_qt(mask)
        self.image_label.setPixmap(qt_img)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

    def visualize_each_centerline(self, mark_type_name):
        if self.centerline_images_dict is None:
            self.append_message("No centerline images found", "error")
            return

        if mark_type_name not in self.centerline_images_dict.keys():
            self.append_message(f"No {mark_type_name} centerline found", "error")
            return

        mask = self.centerline_images_dict[mark_type_name]
        qt_img = self.convert_cv_qt(mask)
        self.image_label.setPixmap(qt_img)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

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

    def clean_result_variables(self):
        del self.colored_mask
        del self.semantic_mask_dict
        self.colored_mask = None
        self.semantic_mask_dict = None

        self.update_color_table()

    def save_centerline_results(self):
        if (
            self.mask_action_binaries is None
            or self.line_dict is None
            or self.reverse_mask_dict is None
        ):
            self.append_message("No centerline results found", "error")
            return

        # used to save all variables of centerline extraction
        self.append_message("Start saving centerline results", "info")

        np.savez(
            os.path.join(self.temp_file_path, "centerline_results.npz"),
            mask_action_binaries=self.mask_action_binaries,
            line_dict=self.line_dict,
            reverse_mask_dict=self.reverse_mask_dict,
        )

        self.append_message("Centerline results saved", "info")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MaskExtractGUI()
    window.show()
    sys.exit(app.exec_())
