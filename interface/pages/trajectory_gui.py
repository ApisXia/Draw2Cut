import os
import sys
import cv2
import time
import shutil
import datetime
import threading
import numpy as np
import open3d as o3d
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5 import QtCore, QtGui, QtWidgets

from interface.functions.gui_mixins import MessageBoxMixin
from configs.load_config import CONFIG

from get_trajectory_Gcode import get_trajectory_Gcode

class QTextEditStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message):
        # Remove any trailing newlines to avoid extra lines
        message = message.rstrip()
        if message:
            QtCore.QMetaObject.invokeMethod(
                self.text_edit.message_box,
                "append",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, message)
            )

    def flush(self):
        pass  # This method is required for compatibility with print()

class Worker(QtCore.QObject):
    # Define a signal to send output messages to the GUI
    output_signal = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, smooth_size, offset_z_level, line_cutting_depth, gui):
        super().__init__()
        self.smooth_size = smooth_size
        self.offset_z_level = offset_z_level
        self.line_cutting_depth = line_cutting_depth
        self.gui = gui

    def run(self):
        # Run get_trajectory_Gcode in the worker thread and send output messages to the main thread
        sys.stdout = QTextEditStream(self.gui)
        get_trajectory_Gcode(self.smooth_size, self.offset_z_level, self.line_cutting_depth,self.gui.gl_view)
        self.finished.emit()  # Emit finished signal when done

class TrajectoryGUI(QtWidgets.QWidget, MessageBoxMixin):
    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(TrajectoryGUI, self).__init__()

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

        # create GLViewWidget
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setFixedSize(1280, 720)
        self.gl_view.opts["distance"] = 320  # set camera distance
        # self.gl_view.setBackgroundColor((255, 255, 255))


        # # add grid
        # self.glo = gl.GLGridItem()
        # self.glo.scale(2, 2, 1)
        # self.glo.setDepthValue(10)  # set grid depth
        # self.gl_view.addItem(self.glo)

        self.case_label = QtWidgets.QLabel("Select Case:")
        self.case_path = CONFIG["case_folder"]
        self.case_path_eidt = QtWidgets.QLineEdit(self.case_path)
        self.case_choose_button = QtWidgets.QPushButton()
        folder_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon)
        self.case_choose_button.setIcon(folder_icon)
        self.case_choose_button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #A0A0A0;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #A0A0A0;
            }
        """)
        self.case_choose_button.clicked.connect(self.select_folder)

        self.smooth_size_label = QtWidgets.QLabel("Smooth_size:")
        self.smooth_size_spin = QtWidgets.QSpinBox()
        self.smooth_size_spin.setRange(0, 10)
        self.smooth_size_spin.setValue(0)

        self.spindle_radius_label = QtWidgets.QLabel("Spindle_radius:")
        self.spindle_radius_spin = QtWidgets.QDoubleSpinBox()
        self.spindle_radius_spin.setRange(0, 10)
        self.spindle_radius_spin.setDecimals(1)
        self.spindle_radius_spin.setValue(5)

        self.offset_z_level_label = QtWidgets.QLabel("Offset_z_level:")
        self.offset_z_level_spin = QtWidgets.QDoubleSpinBox()
        self.offset_z_level_spin.setRange(-5, 5)
        self.offset_z_level_spin.setDecimals(1)
        self.offset_z_level_spin.setValue(-1.5)

        self.line_cutting_depth_label = QtWidgets.QLabel("Line_cutting_depth:")
        self.line_cutting_depth_spin = QtWidgets.QDoubleSpinBox()
        self.line_cutting_depth_spin.setDecimals(1)
        self.line_cutting_depth_spin.setRange(0, 10)
        self.line_cutting_depth_spin.setValue(2)

        self.behavior_relief_label = QtWidgets.QLabel("Behavior_relief:")
        self.behavior_relief_spin = QtWidgets.QDoubleSpinBox()
        self.behavior_relief_spin.setDecimals(1)
        self.behavior_relief_spin.setRange(0, 20)
        self.behavior_relief_spin.setValue(10.5)

        self.behavior_plane_label = QtWidgets.QLabel("Behavior_plane:")
        self.behavior_plane_spin = QtWidgets.QDoubleSpinBox()
        self.behavior_plane_spin.setDecimals(1)
        self.behavior_plane_spin.setRange(0, 20)
        self.behavior_relief_spin.setValue(2)

        self.start_button = QtWidgets.QPushButton("Start trajectory")
        self.start_button.clicked.connect(self.start_trajectory)
        self.step = 0

        # create buttons
        self.replay_button = QtWidgets.QPushButton("Replay")
        self.replay_button.clicked.connect(self.replay)

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.case_label)
        case_path_layout = QtWidgets.QHBoxLayout()
        case_path_layout.addWidget(self.case_path_eidt)
        case_path_layout.addWidget(self.case_choose_button)
        controls_layout.addLayout(case_path_layout)
        controls_layout.addWidget(self.smooth_size_label)
        controls_layout.addWidget(self.smooth_size_spin)
        controls_layout.addWidget(self.spindle_radius_label)
        controls_layout.addWidget(self.spindle_radius_spin)
        controls_layout.addWidget(self.offset_z_level_label)
        controls_layout.addWidget(self.offset_z_level_spin)
        controls_layout.addWidget(self.line_cutting_depth_label)
        controls_layout.addWidget(self.line_cutting_depth_spin)
        controls_layout.addWidget(self.behavior_relief_label)
        controls_layout.addWidget(self.behavior_relief_spin)
        controls_layout.addWidget(self.behavior_plane_label)
        controls_layout.addWidget(self.behavior_plane_spin)
        controls_layout.addWidget(self.start_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.replay_button)

        # add to main layout
        layout.addWidget(self.gl_view)
        layout.addLayout(controls_layout)

        return layout
    def select_folder(self):
        self.case_path = QtWidgets.QFileDialog.getExistingDirectory(self, "select case", options=QtWidgets.QFileDialog.ShowDirsOnly)
        self.case_path_eidt.setText(self.case_path)
    
    def start_trajectory(self):
        if self.step == 0:
            # self.smooth_size = self.smooth_size_spin.value()
            # self.spindle_radius = self.spindle_radius_spin.value()
            # self.offset_z_level = self.offset_z_level_spin.value()
            # self.line_cutting_depth = self.line_cutting_depth_spin.value()
            # self.behavior_relief = self.behavior_relief_spin.value()
            # self.behavior_plane = self.behavior_plane_spin.value()
            # self.case_path = self.case_path_eidt.text()

            # # Create a QThread and Worker
            # self.thread = QtCore.QThread()
            # self.worker = Worker(self.smooth_size, self.offset_z_level, self.line_cutting_depth,self)
            # self.worker.moveToThread(self.thread)

            # self.worker.finished.connect(self.thread.quit)
            # self.worker.finished.connect(self.worker.deleteLater)
            # self.thread.finished.connect(self.thread.deleteLater)

            # self.thread.finished.connect(self.visualize_cutting_planning)

            # # Start the thread
            # self.thread.started.connect(self.worker.run)
            # self.thread.start()
            self.visualize_cutting_planning()
            self.start_button.setText("final_visualize")
            self.step = 1
        else:
            self.visualize_final_surface()
            self.start_button.setText("Start trajectory")
            self.step = 0

    def visualize_cutting_planning(self):
        temp_file_path = CONFIG["temp_file_path"]
        scanned_points = np.load(os.path.join(temp_file_path, "scanned_points.npz"))["points"]
        scanned_colors = np.load(os.path.join(temp_file_path, "scanned_points.npz"))["colors"]
        coarse_cutting_points = np.load(os.path.join(temp_file_path, "coarse_points.npz"))["points"]
        fine_cutting_points = np.load(os.path.join(temp_file_path, "fine_points.npz"))["points"]
        ultra_fine_cutting_points = np.load(os.path.join(temp_file_path, "ultra_fine_points.npz"))["points"]
        scatter = gl.GLScatterPlotItem(
            pos=scanned_points,
            size=0.5,
            color = scanned_colors,
        )
        self.gl_view.addItem(scatter)

        # create coarse trajectory point cloud object using green color
        coarse_trajectory_scatter = gl.GLScatterPlotItem(pos=coarse_cutting_points, 
                                                         color=np.array([[0, 1, 0]] * len(coarse_cutting_points)), 
                                                         size=0.5)
        self.gl_view.addItem(coarse_trajectory_scatter)

        # create fine trajectory point cloud object using red color
        if len(fine_cutting_points) > 0:
            fine_trajectory_scatter = gl.GLScatterPlotItem(pos=fine_cutting_points, 
                                                           color=np.array([[1, 0, 0]] * len(fine_cutting_points)), 
                                                           size=0.5)
            self.gl_view.addItem(fine_trajectory_scatter)

        # create ultra fine trajectory point cloud object using blue color
        if len(ultra_fine_cutting_points) > 0:
            ultra_fine_trajectory_scatter = gl.GLScatterPlotItem(pos=ultra_fine_cutting_points, 
                                                                 color=np.array([[0, 0, 1]] * len(ultra_fine_cutting_points)), 
                                                                 size=0.5)
            self.gl_view.addItem(ultra_fine_trajectory_scatter)

        center = np.mean(scanned_points, axis=0)
        self.gl_view.opts["center"] = pg.Vector(center[0], center[1], center[2])

    def visualize_final_surface(self):
        temp_file_path = CONFIG["temp_file_path"]
        final_surface_path = os.path.join(temp_file_path, "final_surface.ply")
        if os.path.exists(final_surface_path):
            mesh = o3d.io.read_triangle_mesh(final_surface_path)
            if not mesh.has_vertices():
                return  # skip if no vertices

            # get vertices and faces
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            # get colors
            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
            else:
                colors = np.ones_like(vertices)  # white

            # create GLMeshItem
            mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                vertexColors=colors,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
            )
            mesh_item.setGLOptions("opaque")  # set opaque
            self.gl_view.clear()
            self.gl_view.addItem(mesh_item)

            self.center_view_on_vertices(vertices)
    
    def center_view_on_vertices(self, vertices):
        if vertices.size == 0:
            return
        # 计算所有顶点的平均值作为中心点
        center = vertices.mean(axis=0)
        center_point = QtGui.QVector3D(center[0], center[1], center[2])

        # 设置视图中心
        self.gl_view.opts["center"] = center_point
        self.gl_view.update()
    

    # # [ ]: need to add fine cutting trajectory?
    # np.savez(
    #     os.path.join(temp_file_path, "coarse_points.npz"),
    #     points=coarse_cutting_points,
    #     colors=np.array([[0, 1, 0]] * len(coarse_cutting_points)),
    # )
    # np.savez(
    #     os.path.join(temp_file_path, "fine_points.npz"),
    #     points=fine_cutting_points,
    #     colors=np.array([[0, 0, 1]] * len(fine_cutting_points)),
    # )
    # np.savez(
    #     os.path.join(temp_file_path, "ultra_fine_points.npz"),
    #     points=ultra_fine_cutting_points,
    #     colors=np.array([[1, 0, 0]] * len(ultra_fine_cutting_points)),
    # )
    # np.savez(
    #     os.path.join(temp_file_path, "scanned_points.npz"),
    #     points=scanned_points,
    #     colors=scanned_colors,
    # )
    
    def replay(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrajectoryGUI()
    window.show()
    sys.exit(app.exec_())