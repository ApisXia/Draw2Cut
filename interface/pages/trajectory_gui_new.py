import os
import json
import sys
import cv2
import time
import pickle
import numpy as np
import open3d as o3d
import pyqtgraph.opengl as gl

from copy import deepcopy
from scipy.spatial import KDTree
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QHeaderView, QPushButton, QToolButton

from configs.load_config import CONFIG
from utils.trajectory_transform import down_scaling_to_real, vis_points_transformation
from interface.functions.gui_mixins import MessageBoxMixin
from interface.functions.trajectory_thread import TrajectoryThread


class TrajecotryGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(TrajecotryGUI, self).__init__()

        # previous step result name
        self.centerline_result_name = "centerline_data.pkl"
        self.original_points_name = "points_transformed.npz"

        # set saving subfolder
        self.trajectory_saving_subfolder = "trajectory_planning"

        # previous step variables
        self.mask_action_binaries = None
        self.line_dict = None
        self.reverse_mask_dict = None

        # step1: trajecotry holders
        self.spindle_radius = CONFIG["spindle_radius"]
        # store the trajectory of line cutting and coarse bulk cutting
        self.coarse_trajectory_holders = []
        # store the trajectory of fine bulk cutting
        self.fine_trajectory_holders = []
        # store the trajectory of ultra fine bulk cutting
        self.ultra_fine_trajectory_holders = []
        # [ ] store the depth map of all cutting (current just for bulk cutting)
        self.depth_map_holders = []

        # step2: mesh holders
        self.vertices_offset = np.asarray([[150, 260, 0]])
        self.left_bottom_point = None
        self.original_mesh_vertices = None
        self.original_mesh_triangles = None
        self.original_mesh_colors = None

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

        # setup thread
        self.traj_thread = TrajectoryThread()

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

        # create GLViewWidget
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setFixedSize(1280, 720)
        self.gl_view.opts["distance"] = 320  # set camera distance

        # add grid
        self.glo = gl.GLGridItem()
        self.glo.scale(2, 2, 1)
        self.glo.setDepthValue(10)  # set grid depth
        self.gl_view.addItem(self.glo)

        # use stacked layout to switch between image and depth
        self.stacked_layout = QtWidgets.QStackedLayout()
        self.stacked_layout.addWidget(self.image_label)
        self.stacked_layout.addWidget(self.gl_view)

        # right side control panel

        font = QtGui.QFont()
        font.setBold(True)

        # case select part
        self.case_select_label = QtWidgets.QLabel("Select Case")
        self.case_select_label.setFont(font)
        self.init_case_select_widgets()

        # cutting define part
        self.cutting_setting_label = QtWidgets.QLabel("Trajectory Settings")
        self.cutting_setting_label.setFont(font)

        line_cutting_depth_Hlayout = QtWidgets.QHBoxLayout()
        self.line_cutting_depth_label = QtWidgets.QLabel("Line Cutting Depth (mm): ")
        self.line_cutting_depth_spin = QtWidgets.QDoubleSpinBox()
        self.line_cutting_depth_spin.setRange(0.1, 100)
        self.line_cutting_depth_spin.setDecimals(1)
        self.line_cutting_depth_spin.setSingleStep(0.1)
        self.line_cutting_depth_spin.setValue(CONFIG["line_cutting_depth"])
        line_cutting_depth_Hlayout.addWidget(self.line_cutting_depth_label)
        line_cutting_depth_Hlayout.addWidget(self.line_cutting_depth_spin)

        self.start_trajectory_button = QtWidgets.QPushButton(
            "Start Trajectory Planning"
        )
        self.start_trajectory_button.clicked.connect(self.start_trajectory_planning)

        # visualization part
        self.visualization_label = QtWidgets.QLabel("Visualization")
        self.visualization_label.setFont(font)

        self.vis_original_button = QtWidgets.QPushButton("Original Pointcloud")
        self.vis_original_button.clicked.connect(self.visualize_original_mesh)

        self.visualize_animation_button = QtWidgets.QPushButton("Animated Trajectory")
        self.visualize_animation_button.clicked.connect(
            self.visualize_animated_trajectory
        )

        # separator
        separator1 = self.define_separator()

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.case_select_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_select_combo)
        case_path_layout.addWidget(self.case_refresh_button)
        controls_layout.addLayout(case_path_layout)
        controls_layout.addLayout(line_cutting_depth_Hlayout)

        controls_layout.addWidget(self.start_trajectory_button)

        controls_layout.addWidget(separator1)

        controls_layout.addWidget(self.visualization_label)
        controls_layout.addWidget(self.vis_original_button)
        controls_layout.addWidget(self.visualize_animation_button)

        controls_layout.addStretch()

        # horizontal layout for all
        all_layout = QtWidgets.QHBoxLayout()
        all_layout.addLayout(self.stacked_layout)
        all_layout.addLayout(controls_layout)

        return all_layout

    """ Trajectory Planning Functions """

    def start_trajectory_planning(self):
        if hasattr(self, "traj_thread") and self.traj_thread.isRunning():
            self.traj_thread.terminate()
            self.traj_thread.wait()

        # get case name and data path
        self.get_case_info()

        # switch to image display
        self.switch_display(0)

        # try to load centerline results and assign to variable
        try:
            with open(
                os.path.join(self.temp_file_path, self.centerline_result_name), "rb"
            ) as f:
                loaded_centerline_data = pickle.load(f)
            self.mask_action_binaries = loaded_centerline_data["mask_action_binaries"]
            self.line_dict = loaded_centerline_data["line_dict"]
            self.reverse_mask_dict = loaded_centerline_data["reverse_mask_dict"]
            self.append_message("Centerline results loaded successfully", "info")
        except Exception as e:
            self.append_message(f"Failed to load centerline results: {e}", "error")
            return

        self.traj_thread = TrajectoryThread()
        self.traj_thread.line_dict = self.line_dict
        self.traj_thread.reverse_mask_dict = self.reverse_mask_dict
        self.traj_thread.mask_action_binaries = self.mask_action_binaries
        self.traj_thread.temp_file_path = os.path.join(
            self.temp_file_path, self.trajectory_saving_subfolder
        )
        self.traj_thread.line_cutting_depth = self.line_cutting_depth_spin.value()
        self.traj_thread.depth_forward_steps = CONFIG["depth_forward_steps"]

        # connect signals
        self.traj_thread.coarse_trajectory_signal.connect(self.update_coarse_trajectory)
        self.traj_thread.fine_trajectory_signal.connect(self.update_fine_trajectory)
        self.traj_thread.ultra_fine_trajectory_signal.connect(
            self.update_ultra_fine_trajectory
        )
        self.traj_thread.depth_map_signal.connect(self.update_depth_map)
        self.traj_thread.coarse_trajectory_drawing_signal.connect(
            self.update_coarse_traj_image
        )
        self.traj_thread.message_signal.connect(self.append_message)

        # start trajectory planning thread
        self.traj_thread.finished.connect(self.trajectory_down_scaling_to_real)
        self.traj_thread.start()

    @QtCore.pyqtSlot(list)
    def update_coarse_trajectory(self, coarse_trajectory):
        del self.coarse_trajectory_holders
        self.coarse_trajectory_holders = coarse_trajectory

    @QtCore.pyqtSlot(list)
    def update_fine_trajectory(self, fine_trajectory):
        del self.fine_trajectory_holders
        self.fine_trajectory_holders = fine_trajectory

    @QtCore.pyqtSlot(list)
    def update_ultra_fine_trajectory(self, ultra_fine_trajectory):
        del self.ultra_fine_trajectory_holders
        self.ultra_fine_trajectory_holders = ultra_fine_trajectory

    @QtCore.pyqtSlot(list)
    def update_depth_map(self, depth_map):
        del self.depth_map_holders
        self.depth_map_holders = depth_map

    @QtCore.pyqtSlot(np.ndarray)
    def update_coarse_traj_image(self, coarse_traj_image):
        qt_pixmap = self.convert_cv_qt(coarse_traj_image)
        self.image_label.setPixmap(qt_pixmap)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.append_message("Coarse trajectory image visualized", "info")

    def trajectory_down_scaling_to_real(self):
        # downsample the trajectory based on SURFACE_UPSCALE
        self.append_message("Downscaling trajectories to real size...", "info")
        self.coarse_trajectory_holders = down_scaling_to_real(
            self.coarse_trajectory_holders
        )
        self.fine_trajectory_holders = down_scaling_to_real(
            self.fine_trajectory_holders
        )
        self.ultra_fine_trajectory_holders = down_scaling_to_real(
            self.ultra_fine_trajectory_holders
        )

    """ Visualization Functions """

    def visualize_animated_trajectory(self):
        self.switch_display(1)
        self.get_case_info()

        self.read_original_mesh(check_already_loaded=True)

        # get the cutting trajectory points
        combined_vis_points, combined_vis_trajectory = vis_points_transformation(
            self.coarse_trajectory_holders
            + self.fine_trajectory_holders
            + self.ultra_fine_trajectory_holders,
            self.left_bottom_point[0],
            self.left_bottom_point[1],
            self.left_bottom_point[2],
        )

        print(self.left_bottom_point)

        # print x range and y range of combined_vis_points and self.original_mesh_vertices
        print(
            f"combined_vis_points x range: {np.min(combined_vis_points[:, 0])} to {np.max(combined_vis_points[:, 0])}"
        )
        print(
            f"combined_vis_points y range: {np.min(combined_vis_points[:, 1])} to {np.max(combined_vis_points[:, 1])}"
        )
        print(
            f"original_mesh_vertices x range: {np.min(self.original_mesh_vertices[:, 0])} to {np.max(self.original_mesh_vertices[:, 0])}"
        )
        print(
            f"original_mesh_vertices y range: {np.min(self.original_mesh_vertices[:, 1])} to {np.max(self.original_mesh_vertices[:, 1])}"
        )

        # build original vertices kd-tree
        original_vertices_kd_tree = KDTree(self.original_mesh_vertices[:, :2])
        animated_vertices = deepcopy(self.original_mesh_vertices)
        # every 100 points, find all the points with the self.spindle_radius, reset vertices
        for traj in combined_vis_trajectory:
            for i, point in enumerate(traj):
                # get all the points within the spindle radius
                indices = original_vertices_kd_tree.query_ball_point(
                    point[:2], self.spindle_radius
                )
                animated_vertices[indices, 2] = point[2]
                if i % 100 == 0:
                    # play
                    self.put_mesh_on_view(
                        animated_vertices,
                        self.original_mesh_triangles,
                        self.original_mesh_colors,
                    )

                    QTimer.singleShot(100, self.gl_view.update)

    def visualize_original_mesh(self):
        # switch to mesh display
        self.switch_display(1)
        self.get_case_info()

        self.read_original_mesh(check_already_loaded=True)

        self.put_mesh_on_view(
            self.original_mesh_vertices,
            self.original_mesh_triangles,
            self.original_mesh_colors,
        )

    def put_mesh_on_view(self, vertices, triangles, colors):
        # create GLMeshItem
        mesh_item = gl.GLMeshItem(
            vertexes=vertices - self.vertices_offset,
            faces=triangles,
            vertexColors=colors,
            smooth=False,
            drawFaces=True,
            drawEdges=False,
        )
        mesh_item.setGLOptions("opaque")  # set opaque
        self.gl_view.clear()
        self.gl_view.addItem(mesh_item)

    def read_original_mesh(self, check_already_loaded=True):
        if (
            check_already_loaded
            and self.original_mesh_vertices is not None
            and self.original_mesh_triangles is not None
            and self.original_mesh_colors is not None
        ):
            self.append_message("Original mesh already loaded", "info")
            return

        # try to load original pointclouds and assign to variable
        pointcloud_path = os.path.join(self.temp_file_path, self.original_points_name)
        try:
            data = np.load(pointcloud_path)
            points = data["points_smoothed"]
            colors = data["colors"]
            self.left_bottom_point = data["left_bottom_point"]
        except Exception as e:
            self.append_message(f"Failed to load original pointclouds: {e}", "error")
            return

        (
            self.original_mesh_vertices,
            self.original_mesh_triangles,
            self.original_mesh_colors,
        ) = self.build_mesh_from_pointcloud(points, colors)

        self.append_message("Original mesh loaded successfully", "step")

    def build_mesh_from_pointcloud(self, points, colors):
        self.append_message("performing surface reconstruction...", "info")

        # Create a point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Compute nearest neighbor distances to estimate radius
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist * 1.5  # Adjust the factor as needed

        # Create mesh using the Ball Pivoting Algorithm
        radii = o3d.utility.DoubleVector([radius, radius * 2])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii
        )

        # Optionally, remove unwanted artifacts
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()

        return (
            np.asarray(mesh.vertices),
            np.asarray(mesh.triangles),
            np.asarray(mesh.vertex_colors),
        )

    """ Common functions """

    def switch_display(self, display_index: int):
        self.stacked_layout.setCurrentIndex(display_index)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrajecotryGUI()
    window.show()
    sys.exit(app.exec_())
