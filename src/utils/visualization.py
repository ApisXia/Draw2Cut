import numpy as np
import open3d as o3d

from scipy.spatial import KDTree


def visualize_cutting_planning(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    coarse_cutting_points: np.ndarray,
    fine_cutting_points: np.ndarray,
    ultra_fine_cutting_points: np.ndarray,
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

    # create ultra fine trajectory point cloud object using blue color
    if len(ultra_fine_cutting_points) > 0:
        ultra_fine_trajectory_pcd = o3d.geometry.PointCloud()
        ultra_fine_trajectory_pcd.points = o3d.utility.Vector3dVector(
            ultra_fine_cutting_points
        )
        ultra_fine_trajectory_pcd.colors = o3d.utility.Vector3dVector(
            np.array([[0, 0, 1]] * len(ultra_fine_cutting_points))
        )
        object_to_draw.append(ultra_fine_trajectory_pcd)

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

def visualize_final_surface_dynamic(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    depth_map_points: np.ndarray,
    z_surface_level: float,
    num_frames: int = 10  # 控制帧数
):
    # Filter out surface points above the specified z level
    surface_mask = scanned_points[:, 2] > z_surface_level - 1
    surface_points = scanned_points[surface_mask]

    # 分帧逐步处理 depth_map_points
    total_points = len(depth_map_points)
    points_per_frame = total_points // num_frames

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for frame in range(1, num_frames + 1):
        # 每一帧处理的 depth_map_points 数量
        start_idx = (frame - 1) * points_per_frame
        end_idx = min(frame * points_per_frame, total_points)
        # 如果从start_idx开始而不是0，会导致每一帧处理的内容不连续
        # 这一段代码后续可以改成先生成所有的gemotry，然后逐帧更新。即离线渲染
        current_depth_map_points = depth_map_points[0:end_idx]
        # Build a KD tree for nearest neighbor queries
        kdtree = KDTree(current_depth_map_points[:, :2])
        # 逐帧计算新 z 坐标
        offsetted_z_list = []
        for point in surface_points:
            dist, idx = kdtree.query(point[:2])
            if dist < 2:
                offsetted_z_list.append(current_depth_map_points[idx][2])
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
        # Update the mesh object in the visualizer
        # vis.add_geometry(mesh)

        # Update visualization
        # Clear previous geometries and add the new mesh
        vis.clear_geometries()
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # 动态更新可视化
        # o3d.visualization.draw_geometries([mesh])
    vis.destroy_window()