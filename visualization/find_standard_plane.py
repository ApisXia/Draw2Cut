import numpy as np
import open3d as o3d

from glob import glob

from configs.load_config import CONFIG

from src.space_finding.plane import calculate_points_plane


if __name__ == "__main__":
    # read data
    pointcloud_data = np.load(CONFIG["data_path"])

    points_plane, _ = calculate_points_plane(
        CONFIG["origin_label"], pointcloud_data, resize_factor=2
    )
    points_plane = list(points_plane.values())

    # load data
    points = pointcloud_data["points_pos"]
    colors = pointcloud_data["transformed_color"].reshape((-1, 3))
    colors = colors / 255.0
    depth = pointcloud_data["depth"].reshape((-1))

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
    depth = depth[mask]

    # create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    object_to_draw = []
    object_to_draw.append(pcd)

    # draw points_plane on the point cloud as a ball
    for i in range(len(points_plane)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 0.84, 0.0])
        sphere.translate(points_plane[i])
        object_to_draw.append(sphere)

    # visualize point cloud
    o3d.visualization.draw_geometries(object_to_draw)
