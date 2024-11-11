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


class ReplayGUI(QtWidgets.QWidget):
    """This is only for testing embedding open3D in PyQt5

    Args:
        QtWidgets (_type_): _description_
    """

    def __init__(self, message_box: QtWidgets.QTextEdit = None):
        super(ReplayGUI, self).__init__()

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
        self.gl_view.opts["distance"] = 80  # set camera distance

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
        """启动 Open3D 可视化的线程"""
        thread = threading.Thread(target=self.visualize)
        thread.start()

    def visualize(self):
        """调用 Open3D 可视化和渲染函数"""
        num_frames = 1000
        save_dir = "./rendered_frames"

        self.load_and_render_frames(save_dir, num_frames)

    def load_and_render_frames(
        self, save_dir: str, num_frames: int, target_fps: int = 30
    ):
        """加载并渲染保存的帧到 GLViewWidget"""
        for frame in range(1, num_frames + 1):
            frame_file_path = os.path.join(save_dir, f"frame_{frame:03d}.ply")
            if os.path.exists(frame_file_path):
                mesh = o3d.io.read_triangle_mesh(frame_file_path)
                if not mesh.has_vertices():
                    continue  # 跳过无效的网格

                # 获取顶点和面
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)

                # 获取颜色，如果没有颜色则设为白色
                if mesh.has_vertex_colors():
                    colors = np.asarray(mesh.vertex_colors)
                else:
                    colors = np.ones_like(vertices)  # 白色

                # 创建 GLMeshItem
                mesh_item = gl.GLMeshItem(
                    vertexes=vertices,
                    faces=faces,
                    vertexColors=colors,
                    smooth=False,
                    drawFaces=True,
                    drawEdges=True,
                )
                mesh_item.setGLOptions("opaque")  # 设置渲染选项
                self.gl_view.clear()
                self.gl_view.addItem(mesh_item)

                # 控制帧率
                time.sleep(1.0 / target_fps)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ReplayGUI()
    window.show()
    sys.exit(app.exec_())
