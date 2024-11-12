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


class ReplayGUI(QtWidgets.QWidget, MessageBoxMixin):
    """This is only for testing

    Args:
        QtWidgets (_type_): _description_
    """

    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(ReplayGUI, self).__init__()

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

        # add grid
        self.glo = gl.GLGridItem()
        self.glo.scale(2, 2, 1)
        self.glo.setDepthValue(10)  # set grid depth
        self.gl_view.addItem(self.glo)

        # create buttons
        self.replay_button = QtWidgets.QPushButton("Replay")
        self.replay_button.clicked.connect(self.replay)

        # vertical layout for controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.replay_button)

        # add to main layout
        layout.addWidget(self.gl_view)
        layout.addWidget(self.replay_button)

        return layout

    def replay(self):
        """start replaying the frames"""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            self.stop_event.clear()

        thread = threading.Thread(target=self.visualize)
        thread.start()

    def visualize(self):
        """call the function to load and render frames"""
        num_frames = 1000
        save_dir = "./rendered_frames"

        self.load_and_render_frames(save_dir, num_frames)

    def load_and_render_frames(
        self, save_dir: str, num_frames: int, target_fps: int = 30
    ):
        """load and render frames from the given directory"""
        for frame in range(1, num_frames + 1):
            if self.stop_event.is_set():
                break

            frame_file_path = os.path.join(save_dir, f"frame_{frame:03d}.ply")
            if os.path.exists(frame_file_path):
                mesh = o3d.io.read_triangle_mesh(frame_file_path)
                if not mesh.has_vertices():
                    continue  # skip if no vertices

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

                # center view on vertices
                if frame == 1:
                    self.center_view_on_vertices(vertices)

                # sleep for target fps
                time.sleep(1.0 / target_fps)

    def center_view_on_vertices(self, vertices):
        if vertices.size == 0:
            return
        # 计算所有顶点的平均值作为中心点
        center = vertices.mean(axis=0)
        center_point = QtGui.QVector3D(center[0], center[1], center[2])

        # 设置视图中心
        self.gl_view.opts["center"] = center_point
        self.gl_view.update()

    def closeEvent(self, event):
        """define the close event"""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ReplayGUI()
    window.show()
    sys.exit(app.exec_())
