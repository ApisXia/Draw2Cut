import os
import sys
import cv2
import threading
import numpy as np
import pyqtgraph.opengl as gl

from glob import glob
from copy import deepcopy
from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtWidgets, QtCore
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, disk

from configs.load_config import CONFIG
from src.space_finding.plane import calculate_points_plane
from interface.functions.gui_mixins import MessageBoxMixin


class SeperateGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(SeperateGUI, self).__init__()

        # intermediate variables
        self.pointcloud_data = None
        self.x_direction = None
        self.y_direction = None
        self.z_direction = None
        self.x_length = None
        self.y_length = None
        self.origin_point = None

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

        # create GLViewWidget
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setFixedSize(1080, 800)
        self.gl_view.opts["distance"] = 320  # set camera distance

        # add grid
        self.glo = gl.GLGridItem()
        self.glo.scale(2, 2, 1)
        self.glo.setDepthValue(10)  # set grid depth
        self.gl_view.addItem(self.glo)

        # use stacked layout to switch between image and depth
        self.stacked_layout = QtWidgets.QStackedLayout()
        self.stacked_layout.setAlignment(Qt.AlignCenter)
        self.stacked_layout.addWidget(self.image_label)
        self.stacked_layout.addWidget(self.gl_view)

        font = QtGui.QFont()
        font.setBold(True)

        # control layout
        self.case_select_label = QtWidgets.QLabel("Select Case")
        self.case_select_label.setFont(font)

        self.init_case_select_widgets()

        # visualize point cloud
        vis_point_cloud_label = QtWidgets.QLabel("Visualization")
        vis_point_cloud_label.setFont(font)

        self.vis_point_cloud_button = QtWidgets.QPushButton("Show Collected Pointcloud")
        self.vis_point_cloud_button.clicked.connect(self.show_collected_pointcloud)

        axis_label = QtWidgets.QLabel("Axis Setting && Separation Parameters")
        axis_label.setFont(font)
        origin_HLayout = QtWidgets.QHBoxLayout()
        self.origin_label = QtWidgets.QLabel("origin_label:")
        self.origin_label_input = QtWidgets.QLineEdit(CONFIG["origin_label"])
        origin_HLayout.addWidget(self.origin_label, 1)
        origin_HLayout.addWidget(self.origin_label_input, 1)

        x_axis_HLayout = QtWidgets.QHBoxLayout()
        self.x_axis_label = QtWidgets.QLabel("x_axis_label:")
        self.x_axis_label_input = QtWidgets.QLineEdit(",".join(CONFIG["x_axis_label"]))
        x_axis_HLayout.addWidget(self.x_axis_label, 1)
        x_axis_HLayout.addWidget(self.x_axis_label_input, 1)

        y_axis_HLayout = QtWidgets.QHBoxLayout()
        self.y_axis_label = QtWidgets.QLabel("y_axis_label:")
        self.y_axis_label_input = QtWidgets.QLineEdit(",".join(CONFIG["y_axis_label"]))
        y_axis_HLayout.addWidget(self.y_axis_label, 1)
        y_axis_HLayout.addWidget(self.y_axis_label_input, 1)

        z_max_HLayout = QtWidgets.QHBoxLayout()
        self.z_max_label = QtWidgets.QLabel("z_max_level (Upper Range):")
        self.z_max_label_input = QtWidgets.QDoubleSpinBox()
        self.z_max_label_input.setRange(10, 200)
        self.z_max_label_input.setDecimals(1)
        self.z_max_label_input.setSingleStep(0.5)
        self.z_max_label_input.setValue(CONFIG["z_max_level"])
        z_max_HLayout.addWidget(self.z_max_label, 1)
        z_max_HLayout.addWidget(self.z_max_label_input, 1)

        z_min_HLayout = QtWidgets.QHBoxLayout()
        self.z_min_label = QtWidgets.QLabel("z_min_level (Lower Range):")
        self.z_min_label_input = QtWidgets.QDoubleSpinBox()
        self.z_min_label_input.setRange(0, 100)
        self.z_min_label_input.setDecimals(1)
        self.z_min_label_input.setSingleStep(0.5)
        self.z_min_label_input.setValue(CONFIG["z_min_level"])
        z_min_HLayout.addWidget(self.z_min_label, 1)
        z_min_HLayout.addWidget(self.z_min_label_input, 1)

        self.dilation_label = QtWidgets.QLabel("Dilation Option:")
        self.dilation_checkbox = QtWidgets.QCheckBox()
        self.dilation_checkbox.setChecked(False)

        self.dilation_value_label = QtWidgets.QLabel("Dilation Size:")
        self.dilation_spin = QtWidgets.QSpinBox()
        self.dilation_spin.setRange(1, 10)
        self.dilation_spin.setValue(CONFIG["dilation_size"])

        dilation_sub_HLayout = QtWidgets.QHBoxLayout()
        dilation_sub_HLayout.setSpacing(0)
        dilation_sub_HLayout.setContentsMargins(0, 0, 0, 0)
        dilation_sub_HLayout.addWidget(self.dilation_checkbox, 1)
        dilation_sub_HLayout.addWidget(self.dilation_spin, 7)

        dilation_HLayout = QtWidgets.QHBoxLayout()
        dilation_HLayout.addWidget(self.dilation_label, 1)
        dilation_HLayout.addLayout(dilation_sub_HLayout, 1)

        self.start_separate_button = QtWidgets.QPushButton("Start Separation")
        self.start_separate_button.clicked.connect(self.start_separation)

        # Define separator
        seperator1 = self.define_separator()
        seperator2 = self.define_separator()

        # vertical layout for controls
        controls_widget = QtWidgets.QWidget()
        controls_widget.setFixedWidth(400)
        controls_widget.setFixedHeight(800)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)

        controls_layout.addWidget(self.case_select_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_select_combo)
        case_path_layout.addWidget(self.case_refresh_button)
        controls_layout.addLayout(case_path_layout)

        controls_layout.addWidget(seperator1)

        controls_layout.addWidget(vis_point_cloud_label)
        controls_layout.addWidget(self.vis_point_cloud_button)

        controls_layout.addWidget(seperator2)

        controls_layout.addWidget(axis_label)
        controls_layout.addLayout(origin_HLayout)
        controls_layout.addLayout(x_axis_HLayout)
        controls_layout.addLayout(y_axis_HLayout)
        controls_layout.addLayout(z_max_HLayout)
        controls_layout.addLayout(z_min_HLayout)
        controls_layout.addLayout(dilation_HLayout)
        controls_layout.addWidget(self.start_separate_button)

        controls_layout.addStretch(1)

        # horizontal layout for seperate
        seperate_layout = QtWidgets.QHBoxLayout()
        seperate_layout.addLayout(self.stacked_layout)
        seperate_layout.addWidget(controls_widget)
        return seperate_layout

    """ Show collected pointcloud """

    def show_collected_pointcloud(self):
        self.get_case_info()
        self.switch_display(1)

        # load data
        self.load_pointcloud_data(check_exist=True)

        # create point cloud object
        points = self.pointcloud_data["points_pos"]
        colors = self.pointcloud_data["transformed_color"].reshape((-1, 3))
        colors = colors / 255.0
        depth = self.pointcloud_data["depth"].reshape((-1))

        # keep depth larger than 0
        mask = depth > 0
        points = points[mask, :]
        colors = colors[mask, :]
        depth = depth[mask]

        # remove points with depth larger than threshold
        threshold = 1000
        mask = depth < threshold
        points = points[mask, :]
        colors = colors[mask, :]
        # point center
        center = np.mean(points, axis=0)
        points -= center

        scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=2, pxMode=True)

        self.gl_view.clear()
        self.gl_view.addItem(scatter)

    """ Seperate wood surface """

    def start_separation(self):
        # check x_axis and y_axis label
        if not self.check_x_axis_label() or not self.check_y_axis_label():
            return

        # get case name and data path
        self.get_case_info()
        if not os.path.exists(self.temp_file_path):
            os.makedirs(self.temp_file_path)

        self.append_message("Start separation", "step")

        # load data
        self.load_pointcloud_data(check_exist=True)

        # get values from input
        z_max = self.z_max_label_input.value()
        z_min = self.z_min_label_input.value()
        origin_label = self.origin_label_input.text()
        x_axis_label = self.x_axis_label_input.text().split(",")
        y_axis_label = self.y_axis_label_input.text().split(",")
        dilation_option = self.dilation_checkbox.isChecked()
        dilation_size = self.dilation_spin.value()

        # calculate the coordinates
        self.calculate_coordinates(origin_label, x_axis_label, y_axis_label)

        # seperate wood surface
        self.separate_wood_surface(
            z_max, z_min, dilate_option=dilation_option, dilation_size=dilation_size
        )

        # show the wrapped image
        self.show_wrapped_image()

        self.append_message("Separation finished", "step")

    def calculate_coordinates(self, origin_label, x_axis_label, y_axis_label):
        # calculate the plane of the points
        points_plane, _ = calculate_points_plane(
            origin_label, self.pointcloud_data, resize_factor=2
        )
        self.append_message(f"Find points in the plane: {points_plane.keys()}", "info")
        if origin_label not in points_plane:
            self.append_message(
                f"Can not find the origin label: {origin_label}", "error"
            )
            return

        # assert at least one label in x_axis_label can be found in points_plane
        if not any([label in points_plane.keys() for label in x_axis_label]):
            self.append_message(
                f"Can not find any x axis label: {x_axis_label}", "error"
            )
            return
        # assert at least one label in y_axis_label can be found in points_plane
        if not any([label in points_plane.keys() for label in y_axis_label]):
            self.append_message(
                f"Can not find any y axis label: {y_axis_label}", "error"
            )
            return

        # calculate the x, y, z direction and length
        x_direction = 0
        for label in x_axis_label:
            if label in points_plane:
                direction = points_plane[label] - points_plane[origin_label]
                direction = direction / np.linalg.norm(direction)
                x_direction += direction
        x_direction = x_direction / np.linalg.norm(x_direction)

        if x_axis_label[-1] not in points_plane:
            x_length = CONFIG["default_x_length"]
        else:
            x_length = np.linalg.norm(
                points_plane[x_axis_label[-1]] - points_plane[origin_label]
            )

        y_direction = 0
        for label in y_axis_label:
            if label in points_plane:
                direction = points_plane[label] - points_plane[origin_label]
                direction = direction / np.linalg.norm(direction)
                y_direction += direction
        y_direction = y_direction / np.linalg.norm(y_direction)

        if y_axis_label[-1] not in points_plane:
            y_length = CONFIG["default_y_length"]
        else:
            y_length = np.linalg.norm(
                points_plane[y_axis_label[-1]] - points_plane[origin_label]
            )

        z_direction = np.cross(y_direction, x_direction)

        # using z and y axis to calculate orthogonal x axis again
        x_direction = np.cross(z_direction, y_direction)

        origin_point = points_plane[origin_label]

        self.append_message("Calculate the following parameters:", "step")
        self.append_message(f"x_direction: {x_direction}", "info")
        self.append_message(f"y_direction: {y_direction}", "info")
        self.append_message(f"z_direction: {z_direction}", "info")
        self.append_message(f"x_length: {x_length}", "info")
        self.append_message(f"y_length: {y_length}", "info")
        self.append_message(f"origin_point: {origin_point}", "info")

        # save the calculated parameters
        self.x_direction = x_direction
        self.y_direction = y_direction
        self.z_direction = z_direction
        self.x_length = x_length
        self.y_length = y_length
        self.origin_point = origin_point

    def separate_wood_surface(self, z_max, z_min, dilate_option=False, dilation_size=7):
        if z_max <= z_min:
            self.append_message("z_max should be larger than z_min", "error")
            return

        # load data
        points = self.pointcloud_data["points_pos"]
        colors = self.pointcloud_data["transformed_color"].reshape((-1, 3))

        # transform the points to the standard plane
        points_transformed = points - self.origin_point
        points_transformed = np.dot(
            points_transformed,
            np.array([self.x_direction, self.y_direction, self.z_direction]).T,
        )

        # get the mask of points_transformed within the standard box
        mask_box = (
            (points_transformed[:, 0] > 0)
            & (points_transformed[:, 0] < self.x_length)
            & (points_transformed[:, 1] > 0)
            & (points_transformed[:, 1] < self.y_length)
            & (points_transformed[:, 2] < z_max)
            & (points_transformed[:, 2] > z_min)
        )

        # get z surface level and mask plane
        mask_plane = np.zeros_like(mask_box)
        z = z_max
        z_tolerance_value = 5
        z_tolerance = z_tolerance_value
        z_step = 0.5
        while z > z_min:
            mask_plane = (points_transformed[:, 2] > z) & mask_box
            if np.sum(mask_plane) > 0.5 * np.sum(mask_box):
                z_tolerance -= z_step
            if z_tolerance <= 0:
                break
            z -= z_step
        z_surface = z
        z_surface += z_tolerance_value + z_step

        self.append_message(f"Find the surface level: {z_surface} mm", "step")

        # get the plane range on x and y axis
        x_plane_min = np.min(points_transformed[mask_plane, 0])
        x_plane_max = np.max(points_transformed[mask_plane, 0])
        y_plane_min = np.min(points_transformed[mask_plane, 1])
        y_plane_max = np.max(points_transformed[mask_plane, 1])

        # within the plane range, get the 2D mask for wrapping
        inrange_2D_mask = (
            (points_transformed[:, 0] > x_plane_min)
            & (points_transformed[:, 0] < x_plane_max)
            & (points_transformed[:, 1] > y_plane_min)
            & (points_transformed[:, 1] < y_plane_max)
            & mask_plane
        )
        inrange_2D_mask = inrange_2D_mask & mask_plane

        # ? fix holes for the 2D mask
        inrange_2D_mask = inrange_2D_mask.reshape(
            CONFIG["depth_map_height"], CONFIG["depth_map_width"]
        )

        # remove small objects and holes
        inrange_2D_mask = remove_small_objects(inrange_2D_mask, min_size=50)
        inrange_2D_mask = remove_small_holes(inrange_2D_mask, area_threshold=50)

        if dilate_option:
            inrange_2D_mask = dilation(inrange_2D_mask, disk(dilation_size))

        inrange_2D_mask = inrange_2D_mask.reshape(-1)
        # inrange_2D_mask = inrange_2D_mask & mask

        # update x_min, x_max, y_min, y_max
        x_plane_min = np.min(points_transformed[inrange_2D_mask, 0])
        x_plane_max = np.max(points_transformed[inrange_2D_mask, 0])
        y_plane_min = np.min(points_transformed[inrange_2D_mask, 1])
        y_plane_max = np.max(points_transformed[inrange_2D_mask, 1])

        # ? get the corresponding color for these points, and wrap them into a 2d image
        extract_color = colors[inrange_2D_mask, :]
        extract_points = points_transformed[inrange_2D_mask, :]

        x_size = int((x_plane_max - x_plane_min) / CONFIG["device_precision"])
        y_size = int((y_plane_max - y_plane_min) / CONFIG["device_precision"])
        wrapped_image = np.zeros((x_size + 1, y_size + 1, 3), dtype=np.uint8)
        for point, color in zip(extract_points, extract_color):
            x = int((point[0] - x_plane_min) / CONFIG["device_precision"])
            y = int((point[1] - y_plane_min) / CONFIG["device_precision"])
            wrapped_image[x_size - x - 1, y, :] = color.astype(np.uint8)

        # ? fill the holes in the wrapped_image with the average of the nearest colors
        zero_pixel_mask = np.all(wrapped_image == 0, axis=2).astype(np.uint8)
        # use the nearest 3x3 pixels to fill the holes
        wrapped_image = cv2.inpaint(wrapped_image, zero_pixel_mask, 3, cv2.INPAINT_NS)
        wrapped_image = cv2.cvtColor(wrapped_image, cv2.COLOR_BGR2RGB)

        # ? increase the size by 10 and save the wrapped image
        wrapped_image = cv2.resize(wrapped_image, (0, 0), fx=1, fy=1)
        cv2.imwrite(
            os.path.join(self.temp_file_path, "wrapped_image.png"), wrapped_image
        )
        wrapped_image = cv2.resize(
            wrapped_image,
            (0, 0),
            fx=CONFIG["surface_upscale"],
            fy=CONFIG["surface_upscale"],
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(
            os.path.join(self.temp_file_path, "wrapped_image_zoom.png"), wrapped_image
        )

        self.warpped_image = wrapped_image

        # ? save left bottom point position, including x, y, z and x_length, y_length
        left_bottom_point = np.array([x_plane_min, y_plane_min, z_surface])
        self.append_message(f"Left bottom point position: {left_bottom_point}", "info")
        left_bottom_saving_path = os.path.join(
            self.temp_file_path, "left_bottom_point.npz"
        )
        np.savez(
            left_bottom_saving_path,
            left_bottom_point=left_bottom_point,
            x_length=self.x_length,
            y_length=self.y_length,
        )
        self.append_message(
            f"Save left bottom point position to {left_bottom_saving_path}", "info"
        )

        # ? save pointclouds for later visualization
        colors = colors.astype(np.float32)
        colors /= 255.0

        # smoothed surface
        points_smoothed = deepcopy(points_transformed)
        points_smoothed[inrange_2D_mask, 2] = z_surface

        mask_smoothed = (
            (points_smoothed[:, 0] > 0)
            & (points_smoothed[:, 0] < self.x_length)
            & (points_smoothed[:, 1] > 0)
            & (points_smoothed[:, 1] < self.y_length)
            & (points_smoothed[:, 2] < z_max + 5)
            & (points_smoothed[:, 2] > -5)
        )

        points_smoothed = points_smoothed[mask_smoothed, :]
        colors = colors[mask_smoothed, :]

        # save points_transformed
        point_transformed_saving_path = os.path.join(
            self.temp_file_path, "points_transformed.npz"
        )
        np.savez(
            point_transformed_saving_path,
            points_smoothed=points_smoothed,
            colors=colors,
            left_bottom_point=left_bottom_point,
            x_length=self.x_length,
            y_length=self.y_length,
        )
        self.append_message(
            f"Save transformed points to {point_transformed_saving_path}", "info"
        )

    """ Common functions """

    def switch_display(self, display_index: int):
        self.stacked_layout.setCurrentIndex(display_index)

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

    def load_pointcloud_data(self, check_exist=True):
        if self.pointcloud_data is not None and check_exist:
            self.append_message("Pointcloud data already loaded", "warning")
            return
        self.pointcloud_data = np.load(self.data_path)
        self.append_message("Pointcloud data loaded", "step")

    def show_wrapped_image(self):
        self.switch_display(0)
        qt_pixmap = self.convert_cv_qt(self.warpped_image)
        self.image_label.setPixmap(qt_pixmap)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.append_message("Wrapped image displayed", "info")

    def clean_result_variables(self):
        self.pointcloud_data = None
        self.x_direction = None
        self.y_direction = None
        self.z_direction = None
        self.x_length = None
        self.y_length = None
        self.origin_point = None
        self.warpped_image = None


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SeperateGUI()
    window.show()
    sys.exit(app.exec_())
