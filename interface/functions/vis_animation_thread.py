import numpy as np

from copy import deepcopy
from scipy.spatial import KDTree

from PyQt5.QtCore import QThread, pyqtSignal

from utils.trajectory_transform import vis_points_transformation


class VisualizeAnimationThread(QThread):
    update_mesh = pyqtSignal()
    message_signal = pyqtSignal(str, str)

    def __init__(self, parent=None, fps=30):
        super().__init__(parent)
        self.parent = parent
        self.fps = fps
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        self.parent.switch_display(1)
        self.parent.get_case_info()

        self.parent.read_original_mesh(check_already_loaded=True)

        self.message_signal.emit("Start visualizing trajectory animation", "step")

        # get all points from all trajectories
        total_cutting_points, _ = vis_points_transformation(
            self.parent.coarse_trajectory_holders
            + self.parent.fine_trajectory_holders
            + self.parent.ultra_fine_trajectory_holders,
            self.parent.left_bottom_point[0],
            self.parent.left_bottom_point[1],
            self.parent.left_bottom_point[2],
        )

        total_length = len(total_cutting_points)

        # 构建原始顶点的 KD 树
        original_vertices_kd_tree = KDTree(self.parent.original_mesh_vertices[:, :2])
        animated_vertices = deepcopy(self.parent.original_mesh_vertices)
        delay = 1.0 / self.fps  # 计算延迟，单位为秒

        # 每 100 个点，找到在主轴半径内的所有点，重置顶点
        for i, point in enumerate(total_cutting_points):
            if not self.is_running:
                return
            # 获取在主轴半径内的所有点
            indices = original_vertices_kd_tree.query_ball_point(
                point[:2], self.parent.spindle_radius
            )
            indices = np.array(indices)  # 将 indices 转换为 NumPy 数组
            mask = animated_vertices[indices, 2] > point[2]
            animated_vertices[indices[mask], 2] = point[2]
            if i % 100 == 0:
                # 更新网格
                self.message_signal.emit(
                    f"Update trajectory at {i+1}/{total_length}", "info"
                )
                self.parent.put_mesh_on_view(
                    animated_vertices,
                    self.parent.original_mesh_triangles,
                    self.parent.original_mesh_colors,
                )
                # 发射信号，通知主线程更新视图
                self.update_mesh.emit()
                self.msleep(int(delay * 1000))

        self.message_signal.emit("Finish visualizing trajectory animation", "step")
