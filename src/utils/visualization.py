import numpy as np
import open3d as o3d


def visualize_cutting_planning(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    depth_map: np.ndarray,
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
    fine_trajectory_pcd = o3d.geometry.PointCloud()
    fine_trajectory_pcd.points = o3d.utility.Vector3dVector(fine_cutting_points)
    fine_trajectory_pcd.colors = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0]] * len(fine_cutting_points))
    )
    object_to_draw.append(fine_trajectory_pcd)

    # visualize point cloud
    o3d.visualization.draw_geometries(object_to_draw)
