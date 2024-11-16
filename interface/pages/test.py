import sys
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl

class PointCloudViewer(QtWidgets.QWidget):
    def __init__(self, points, colors=None):
        super(PointCloudViewer, self).__init__()

        # 设置窗口布局
        layout = QtWidgets.QVBoxLayout(self)
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setFixedSize(800, 600)
        self.gl_view.opts["distance"] = 40  # 视角距离
        layout.addWidget(self.gl_view)

        # 添加点云到 GLScatterPlotItem
        self.scatter = gl.GLScatterPlotItem(pos=points, size = 0.5, color=colors)
        self.gl_view.addItem(self.scatter)

        # 添加参考网格
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        grid.setDepthValue(10)
        self.gl_view.addItem(grid)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # 示例点云数据
    points = np.random.rand(1000, 3) * 10  # 1000 个随机点
    colors = np.random.rand(1000, 4)  # 每个点的 RGBA 颜色

    viewer = PointCloudViewer(points, colors)
    viewer.show()
    sys.exit(app.exec_())
