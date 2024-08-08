from flask import Flask, jsonify, render_template
import numpy as np
from scipy.spatial import KDTree, cKDTree
from configs.load_config import CONFIG
import os
import open3d as o3d
from scipy.optimize import leastsq
from sklearn.cluster import DBSCAN
import hdbscan
import copy


app = Flask(__name__)

# 定义一个函数将点投影到平面上
def project_to_plane(points, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    projected_points = points - (np.dot(points, normal) + d)[:, np.newaxis] * normal
    return projected_points

# 将点转换到2D平面坐标系
def to_2d(points, plane_model):
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    basis_x = np.cross(normal, np.array([0, 0, 1]) if abs(normal[2]) < abs(normal[0]) else np.array([1, 0, 0]))
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(normal, basis_x)
    return np.dot(points, np.vstack((basis_x, basis_y)).T)

# 定义一个最小二乘法拟合圆的函数
def fit_circle_2d(points):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, ier = leastsq(f_2, center_estimate)
    radius = calc_R(*center).mean()
    return center, radius

# 将2D圆点转换回3D点
def to_3d(points_2d, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    basis_x = np.cross(normal, np.array([0, 0, 1]) if abs(normal[2]) < abs(normal[0]) else np.array([1, 0, 0]))
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(normal, basis_x)
    points_3d = np.dot(points_2d, np.vstack((basis_x, basis_y)))
    return points_3d - (np.dot(points_3d, normal) + d)[:, np.newaxis] * normal

def rgb2float(rgb):
    return np.array([c / 255.0 * 0.7 for c in rgb])


def filter_points(wood_points, wood_colors, cut_points, threshold=0.1):
    """
    对点集A进行过滤,删除那些与点集B中的点足够接近的点。

    参数:
    A (numpy.ndarray): 点集A,shape为(len, 3)
    B (numpy.ndarray): 点集B,shape为(len, 3)
    threshold (float): 距离阈值,默认为0.1

    返回:
    filtered_A (numpy.ndarray): 过滤后的点集A
    """
    # 创建点集B的KDTree
    kdtree = KDTree(cut_points)

    # 删除那些与点集B中的点足够接近的点
    filtered_points = []
    filtered_colors = []
    distances, indices = kdtree.query(wood_points, k=1)
    keep_mask = distances > threshold
    unkeep_mask = distances <= threshold
    # wood_points[unkeep_mask, 2] = wood_points[unkeep_mask, 2] - 5
    # wood_colors[unkeep_mask] = rgb2float([184, 151, 136])
    cut_points = copy.deepcopy(wood_points[unkeep_mask])
    cut_colors = copy.deepcopy(wood_colors[unkeep_mask])
    # cut_points[:, 2] -= 5
    cut_colors = rgb2float([184, 151, 136]) * np.ones_like(cut_points)
    # wood_points = wood_points[keep_mask]
    # wood_colors = wood_colors[keep_mask]
    # wood_colors[keep_mask] = [255,255,255]
    return wood_points, wood_colors, cut_points, cut_colors,keep_mask,unkeep_mask

    for wood_point, wood_color in zip(wood_points, wood_colors):
        dist = 100
        for cut_point in cut_points:
            dist = min(dist, np.linalg.norm(wood_point - cut_point))
        if dist > threshold:
            filtered_points.append(wood_point)
            filtered_colors.append(wood_color)

    return np.array(filtered_points), np.array(filtered_colors)


temp_file_path = CONFIG["temp_file_path"]
# 读取木头的点云数据
wood_data = np.load(os.path.join(temp_file_path, "points_transformed.npz"))
wood_points = wood_data["points"]
wood_colors = wood_data["colors"]

# 读取被割掉轨迹的点云数据
cut_data = np.load(
    os.path.join(temp_file_path, "cut_points.npz"),
)
cut_points = cut_data["points"]
cut_points[:, 0] = -cut_points[:, 0]
cut_points[:, 2] -= 1
cut_colors = cut_data["colors"]

# 过滤木头点云数据
wood_points, wood_colors,cut_points,cut_colors,keep_mask,unkeep_mask = filter_points(
    wood_points, wood_colors, cut_points, threshold=2
)
print(wood_points.shape)

centroid = np.mean(wood_points, axis=0)
wood_points -= centroid
cut_points -= centroid

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["GET"])
def get_data():
    global wood_points, wood_colors, cut_points, cut_colors,keep_mask,unkeep_mask
    c = copy.deepcopy(cut_points)
    c[:, 2] -= 5
    data = {
        "wood_points": wood_points[keep_mask].tolist(),
        "wood_colors": wood_colors[keep_mask].tolist(),
        "cut_points": c.tolist(),
        "cut_colors": cut_colors.tolist(),
        "texts": [
            "Contour 'Circle' -> 3mm",
            "Cut trajectories visualized in 'Gray'",
            "number: 3 contours",
        ],
    }
    return jsonify(data)


@app.route("/auto-smooth", methods=["POST"])
def auto_smooth():
    # 处理auto-smooth逻辑并返回新的数据
    global wood_points, wood_colors, cut_points, cut_colors,keep_mask,unkeep_mask
    # keep_mask = np.zeros(cut_points.shape[0], dtype=bool)
    # cut_points = cut_points[keep_mask]
    # cut_colors = cut_colors[keep_mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cut_points)
    all_circle_pcds = []
    remaining_pcd = pcd

    while True:
        # 使用RANSAC拟合平面
        plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        if len(inliers) < 100:  # 如果剩余点数少于一定数量，则停止
            break

        # 提取拟合平面的点
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        inlier_points = np.asarray(inlier_cloud.points)

        # 将平面上的点投影到平面
        projected_points = project_to_plane(inlier_points, plane_model)

        # 将点转换到2D平面坐标系
        points_2d = to_2d(projected_points, plane_model)

        # 使用DBSCAN聚类2D平面上的点
        clustering = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10).fit(points_2d)
        labels = clustering.labels_

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # 忽略噪声点
            cluster_points_2d = points_2d[labels == label]

            # 拟合2D平面上的圆
            center_2d, radius = fit_circle_2d(cluster_points_2d)

            # 生成填充圆的点
            theta = np.linspace(0, 2 * np.pi, 500)
            circle_thickness = 2  # 圆的厚度
            filled_circle_points_2d = []
            for r in np.linspace(radius - circle_thickness, radius + circle_thickness, 20):  # 调整填充密度
                circle_points_2d = np.c_[center_2d[0] + r * np.cos(theta), center_2d[1] + r * np.sin(theta)]
                filled_circle_points_2d.append(circle_points_2d)
            filled_circle_points_2d = np.vstack(filled_circle_points_2d)

            # 将2D圆点转换回3D点
            filled_circle_points_3d = to_3d(filled_circle_points_2d, plane_model)

            # 创建拟合圆的点云
            circle_pcd = o3d.geometry.PointCloud()
            circle_pcd.points = o3d.utility.Vector3dVector(filled_circle_points_3d)
            # circle_pcd.paint_uniform_color([0, 1, 0])
            all_circle_pcds.append(circle_pcd)

        # 从剩余点云中去除拟合到的平面内点
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    
    # 将拟合的圆点云合并
    circle_pcds = o3d.geometry.PointCloud()
    for circle_pcd in all_circle_pcds:
        circle_pcds += circle_pcd
    
    cut_points = np.array(circle_pcds.points)
    # cut_colors = np.array([[0.0, 1.0, 0.0]] * cut_points.shape[0])
    cut_colors = rgb2float([184, 151, 136]) * np.ones_like(cut_points)

    _, _, _, _,keep_mask,unkeep_mask = filter_points(
        wood_points, wood_colors, cut_points, threshold=2
    )
    

    print(cut_points.shape)

    # empty the wood_points
    # wood_points = np.array([[0, 0, 0]])
    # wood_colors = np.array([[0, 0, 0]])
    # cut_points = np.array(([0, 0, 0]))
    # cut_colors = np.array(([0, 0, 0]))
    data = {
        "wood_points": wood_points[keep_mask].tolist(),
        "wood_colors": wood_colors[keep_mask].tolist(),
        "cut_points": cut_points.tolist(),
        "cut_colors": cut_colors.tolist(),
        "texts": ["text 'sake' -> 3mm", "visualized in Green", "number: 9 contours"],
    }
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
