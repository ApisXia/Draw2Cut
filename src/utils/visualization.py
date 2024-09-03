import numpy as np
import open3d as o3d

from scipy.spatial import KDTree


def visualize_cutting_planning(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    coarse_cutting_points: np.ndarray,
    fine_cutting_points: np.ndarray,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scanned_points)
    pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

    object_to_draw = []
    object_to_draw.append(pcd)

    # create coarse trajectory point cloud object using green color
    coarse_trajectory_pcd = o3d.geometry.PointCloud()
    coarse_trajectory_pcd.points = o3d.utility.Vector3dVector(coarse_cutting_points)
    coarse_trajectory_pcd.colors = o3d.utility.Vector3dVector(
        np.array([[0, 1, 0]] * len(coarse_cutting_points))
    )
    object_to_draw.append(coarse_trajectory_pcd)

    # create fine trajectory point cloud object using red color
    if len(fine_cutting_points) > 0:
        fine_trajectory_pcd = o3d.geometry.PointCloud()
        fine_trajectory_pcd.points = o3d.utility.Vector3dVector(fine_cutting_points)
        fine_trajectory_pcd.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0]] * len(fine_cutting_points))
        )
        object_to_draw.append(fine_trajectory_pcd)

    # visualize point cloud
    o3d.visualization.draw_geometries(object_to_draw)


def visualize_final_surface(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    depth_map_points: np.ndarray,
    z_surface_level: float,
):
    # Filter out surface points above the specified z level
    surface_mask = scanned_points[:, 2] > z_surface_level - 1
    surface_points = scanned_points[surface_mask]

    # Build a KD tree for nearest neighbor queries
    kdtree = KDTree(depth_map_points[:, :2])

    # Calculate new z coordinates
    offsetted_z_list = []
    for point in surface_points:
        dist, idx = kdtree.query(point[:2])
        if dist < 2:
            offsetted_z_list.append(depth_map_points[idx][2])
        else:
            offsetted_z_list.append(point[2])

    # Update the z coordinates of the surface points
    surface_points[:, 2] = np.array(offsetted_z_list)

    # Update the z coordinates of the scanned points
    new_scanned_points = np.copy(scanned_points)
    new_scanned_points[surface_mask] = surface_points

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_scanned_points)
    pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Create a mesh from the point cloud using Ball Pivoting Algorithm
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )

    # Visualize the point cloud
    o3d.visualization.draw_geometries([mesh])
