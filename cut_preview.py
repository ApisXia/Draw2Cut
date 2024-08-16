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

def filter_wood_zone(wood_points, wood_colors,width = 10,height = 10):
    #TODO: Filter points bias on z axis
    # Calculate center of the point cloud
    center = wood_points.mean(axis=0)
    
    # Define rectangle size (e.g., width=10, height=10)
    mask = (
        (wood_points[:, 0] > center[0] - width / 2) & (wood_points[:, 0] < center[0] + width / 2) &
        (wood_points[:, 1] > center[1] - height / 2) & (wood_points[:, 1] < center[1] + height / 2)
    )
    
    # Filter points and colors
    filtered_points = wood_points[mask]
    filtered_colors = wood_colors[mask]

    return filtered_points, filtered_colors

def build_mesh(wood_points,wood_colors,cut_points,cut_colors):
    A = wood_points
    B = wood_colors
    C = copy.deepcopy(cut_points)
    # C[:, 2] += 5
    D = cut_colors
    A = np.concatenate((A, C), axis=0)
    B = np.concatenate((B, D), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(B)
    pcd.points = o3d.utility.Vector3dVector(A)
    # o3d.visualization.draw_geometries([pcd])  
    print(A.shape)
    mesh = construct_3d_model(A, B)
    return mesh

def construct_3d_model(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)
    # 法线估计
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # 进行泊松重建
    print("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # 去除低密度区域的面片（例如孤立的面片）
    # densities = np.asarray(densities)
    # density_threshold = densities.mean()
    # vertices_to_remove = densities < density_threshold
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    # cut_points = np.asarray(mesh.points)
    # cut_colors = np.array([[0.0, 1.0, 0.0]] * cut_points.shape[0])
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

# Define a function to project points onto a plane
def project_to_plane(points, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    projected_points = points - (np.dot(points, normal) + d)[:, np.newaxis] * normal
    return projected_points


# Convert points to 2D plane coordinates
def to_2d(points, plane_model):
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    basis_x = np.cross(
        normal,
        np.array([0, 0, 1]) if abs(normal[2]) < abs(normal[0]) else np.array([1, 0, 0]),
    )
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(normal, basis_x)
    return np.dot(points, np.vstack((basis_x, basis_y)).T)


# Define a function to fit a circle using least squares
def fit_circle_2d(points):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, ier = leastsq(f_2, center_estimate)
    radius = calc_R(*center).mean()
    return center, radius


# Convert 2D circle points back to 3D points
def to_3d(points_2d, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    basis_x = np.cross(
        normal,
        np.array([0, 0, 1]) if abs(normal[2]) < abs(normal[0]) else np.array([1, 0, 0]),
    )
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(normal, basis_x)
    points_3d = np.dot(points_2d, np.vstack((basis_x, basis_y)))
    return points_3d - (np.dot(points_3d, normal) + d)[:, np.newaxis] * normal


def rgb2float(rgb):
    return np.array([c / 255.0 * 0.7 for c in rgb])


def filter_points(wood_points, wood_colors, cut_points, threshold=0.1):
    """
    Filter points in set A by removing points that are close enough to points in set B.

    Args:
    A (numpy.ndarray): Set A, shape (len, 3)
    B (numpy.ndarray): Set B, shape (len, 3)
    threshold (float): Distance threshold, default is 0.1

    Returns:
    filtered_A (numpy.ndarray): Filtered set A
    """
    # Create KDTree for set B
    kdtree = KDTree(cut_points)

    # Remove points in set A that are close enough to points in set B
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
    return wood_points, wood_colors, cut_points, cut_colors, keep_mask, unkeep_mask

    for wood_point, wood_color in zip(wood_points, wood_colors):
        dist = 100
        for cut_point in cut_points:
            dist = min(dist, np.linalg.norm(wood_point - cut_point))
        if dist > threshold:
            filtered_points.append(wood_point)
            filtered_colors.append(wood_color)

    return np.array(filtered_points), np.array(filtered_colors)


temp_file_path = CONFIG["temp_file_path"]
# Read wood point cloud data
wood_data = np.load(os.path.join(temp_file_path, "points_transformed.npz"))
wood_points = wood_data["points"]
wood_colors = wood_data["colors"]

# Read cut trajectory point cloud data
cut_data = np.load(
    os.path.join(temp_file_path, "cut_points.npz"),
)
cut_points = cut_data["points"]
cut_points[:, 0] = -cut_points[:, 0]
cut_points[:, 2] -= 1
cut_colors = cut_data["colors"]

# Filter wood point cloud data
wood_points, wood_colors, cut_points, cut_colors, keep_mask, unkeep_mask = (
    filter_points(wood_points, wood_colors, cut_points, threshold=3)
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
    global wood_points, wood_colors, cut_points, cut_colors, keep_mask, unkeep_mask
    c = copy.deepcopy(cut_points)
    # c[:, 2] -= 5
    filter_wood_points, filter_wood_colors = filter_wood_zone(wood_points[keep_mask], wood_colors[keep_mask],250,250)
    mesh = build_mesh(filter_wood_points, filter_wood_colors, cut_points, cut_colors)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    data = {
        # "wood_points": wood_points[keep_mask].tolist(),
        # "wood_colors": wood_colors[keep_mask].tolist(),
        # "cut_points": c.tolist(),
        # "cut_colors": cut_colors.tolist(),
        "texts": [
            "Contour 'Circle' -> 3mm",
            "Cut trajectories visualized in 'Gray'",
            "number: 9 contours",
        ],
        "vertices": vertices.tolist(),
        "triangles": triangles.tolist(),
        "colors": colors.tolist(),
        
    }
    return jsonify(data)


@app.route("/auto-smooth", methods=["POST"])
def auto_smooth():
    # Process auto-smooth logic and return new data
    global wood_points, wood_colors, cut_points, cut_colors, keep_mask, unkeep_mask
    # keep_mask = np.zeros(cut_points.shape[0], dtype=bool)
    # cut_points = cut_points[keep_mask]
    # cut_colors = cut_colors[keep_mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cut_points)
    all_circle_pcds = []
    remaining_pcd = pcd

    while True:
        # Fit a plane using RANSAC
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        if len(inliers) < 100:  # Stop if remaining points are below a certain threshold
            break

        # Extract points on the fitted plane
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        inlier_points = np.asarray(inlier_cloud.points)

        # Project points on the plane
        projected_points = project_to_plane(inlier_points, plane_model)

        # Convert points to 2D plane coordinates
        points_2d = to_2d(projected_points, plane_model)

        # Cluster points on the 2D plane using DBSCAN
        clustering = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=100).fit(
            points_2d
        )
        labels = clustering.labels_

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise points
            cluster_points_2d = points_2d[labels == label]

            # Fit a circle on the 2D plane
            center_2d, radius = fit_circle_2d(cluster_points_2d)

            # Generate points for filling the circle
            theta = np.linspace(0, 2 * np.pi, 500)
            circle_thickness = 2  # Circle thickness
            filled_circle_points_2d = []
            for r in np.linspace(
                radius - circle_thickness, radius + circle_thickness, 20
            ):  # Adjust fill density
                circle_points_2d = np.c_[
                    center_2d[0] + r * np.cos(theta), center_2d[1] + r * np.sin(theta)
                ]
                filled_circle_points_2d.append(circle_points_2d)
            filled_circle_points_2d = np.vstack(filled_circle_points_2d)

            # Convert 2D circle points back to 3D points
            filled_circle_points_3d = to_3d(filled_circle_points_2d, plane_model)

            # Create point cloud for the fitted circle
            circle_pcd = o3d.geometry.PointCloud()
            circle_pcd.points = o3d.utility.Vector3dVector(filled_circle_points_3d)
            # circle_pcd.paint_uniform_color([0, 1, 0])
            all_circle_pcds.append(circle_pcd)

        # Remove points in the fitted plane from the remaining point cloud
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    # Merge the fitted circle point clouds
    circle_pcds = o3d.geometry.PointCloud()
    for circle_pcd in all_circle_pcds:
        circle_pcds += circle_pcd

    smooth_cut_points = np.array(circle_pcds.points)
    # cut_colors = np.array([[0.0, 1.0, 0.0]] * cut_points.shape[0])
    smooth_cut_colors = rgb2float([184, 151, 136]) * np.ones_like(smooth_cut_points)

    _, _, _, _, keep_mask, unkeep_mask = filter_points(
        wood_points, wood_colors, smooth_cut_points, threshold=2
    )

    print(smooth_cut_points.shape)

    mesh = build_mesh(wood_points, wood_colors, smooth_cut_points, smooth_cut_colors)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)

    data = {
        "vertices": vertices.tolist(),
        "triangles": triangles.tolist(),
        "colors": colors.tolist(),
        "texts": ["text 'sake' -> 3mm", "visualized in Green", "number: 9 contours"],
    }
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
