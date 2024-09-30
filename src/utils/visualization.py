import numpy as np
import open3d as o3d
from tqdm import tqdm 
from scipy.spatial import KDTree
import os


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

# def visualize_final_surface_dynamic(
#     scanned_points: np.ndarray,
#     scanned_colors: np.ndarray,
#     depth_map_points: np.ndarray,
#     z_surface_level: float,
#     num_frames: int = 10  # 控制帧数
# ):
#     # Filter out surface points above the specified z level
#     surface_mask = scanned_points[:, 2] > z_surface_level - 1
#     surface_points = scanned_points[surface_mask]

#     # 分帧逐步处理 depth_map_points
#     total_points = len(depth_map_points)
#     points_per_frame = total_points // num_frames

#     # Initialize visualizer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     for frame in range(1, num_frames + 1):
#         # 每一帧处理的 depth_map_points 数量
#         start_idx = (frame - 1) * points_per_frame
#         end_idx = min(frame * points_per_frame, total_points)
#         # 如果从start_idx开始而不是0，会导致每一帧处理的内容不连续
#         # 这一段代码后续可以改成先生成所有的gemotry，然后逐帧更新。即离线渲染
#         current_depth_map_points = depth_map_points[0:end_idx]
#         # Build a KD tree for nearest neighbor queries
#         kdtree = KDTree(current_depth_map_points[:, :2])
#         # 逐帧计算新 z 坐标
#         offsetted_z_list = []
#         for point in surface_points:
#             dist, idx = kdtree.query(point[:2])
#             if dist < 2:
#                 offsetted_z_list.append(current_depth_map_points[idx][2])
#             else:
#                 offsetted_z_list.append(point[2])

#         # Update the z coordinates of the surface points
#         surface_points[:, 2] = np.array(offsetted_z_list)

#         # Update the z coordinates of the scanned points
#         new_scanned_points = np.copy(scanned_points)
#         new_scanned_points[surface_mask] = surface_points

#         # Create a point cloud object
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(new_scanned_points)
#         pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

#         # Estimate normals
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
#         )

#         # Create a mesh from the point cloud using Ball Pivoting Algorithm
#         distances = pcd.compute_nearest_neighbor_distance()
#         avg_dist = np.mean(distances)
#         radius = avg_dist
#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#             pcd, o3d.utility.DoubleVector([radius, radius * 2])
#         )
#         # Update the mesh object in the visualizer
#         # vis.add_geometry(mesh)

#         # Update visualization
#         # Clear previous geometries and add the new mesh
#         vis.clear_geometries()
#         vis.add_geometry(mesh)
#         vis.poll_events()
#         vis.update_renderer()

#         # 动态更新可视化
#         # o3d.visualization.draw_geometries([mesh])
#     vis.destroy_window()

# def visualize_final_surface_dynamic(
#     scanned_points: np.ndarray,
#     scanned_colors: np.ndarray,
#     depth_map_points: np.ndarray,
#     z_surface_level: float,
#     trajectory: np.ndarray,  # New argument for the Gcode trajectory
#     num_frames: int = 10  # 控制帧数
# ):
#     # Filter out surface points above the specified z level
#     surface_mask = scanned_points[:, 2] > z_surface_level - 1
#     surface_points = scanned_points[surface_mask]

#     total_points = len(depth_map_points)
#     points_per_frame = len(trajectory) // num_frames  # Process according to trajectory

#     # Initialize visualizer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     for frame in range(1, num_frames + 1):
#         # Process points according to the trajectory up to this frame
#         current_trajectory = trajectory[:frame * points_per_frame]
#         current_depth_map_points = []

#         # Find the nearest depth_map_points for each trajectory point
#         for traj_point in current_trajectory:
#             # Search nearest depth map point
#             dist = np.linalg.norm(depth_map_points[:, :2] - traj_point[:2], axis=1)
#             nearest_idx = np.argmin(dist)
#             current_depth_map_points.append(depth_map_points[nearest_idx])

#         current_depth_map_points = np.array(current_depth_map_points)

#         # Build a KD tree for nearest neighbor queries
#         kdtree = KDTree(current_depth_map_points[:, :2])

#         # Update z coordinates for surface points
#         offsetted_z_list = []
#         for point in surface_points:
#             dist, idx = kdtree.query(point[:2])
#             if dist < 2:
#                 offsetted_z_list.append(current_depth_map_points[idx][2])
#             else:
#                 offsetted_z_list.append(point[2])

#         # Update the z coordinates of the surface points
#         surface_points[:, 2] = np.array(offsetted_z_list)

#         # Update the z coordinates of the scanned points
#         new_scanned_points = np.copy(scanned_points)
#         new_scanned_points[surface_mask] = surface_points

#         # Create a point cloud object
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(new_scanned_points)
#         pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

#         # Estimate normals
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
#         )

#         # Create a mesh from the point cloud using Ball Pivoting Algorithm
#         distances = pcd.compute_nearest_neighbor_distance()
#         avg_dist = np.mean(distances)
#         radius = avg_dist
#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#             pcd, o3d.utility.DoubleVector([radius, radius * 2])
#         )

#         # Update the mesh object in the visualizer
#         vis.clear_geometries()
#         vis.add_geometry(mesh)
#         vis.poll_events()
#         vis.update_renderer()

# def visualize_final_surface_dynamic(
#     scanned_points: np.ndarray,
#     scanned_colors: np.ndarray,
#     depth_map_points: np.ndarray,
#     z_surface_level: float,
#     trajectory: np.ndarray,  # New argument for the Gcode trajectory
#     num_frames: int = 100  # 控制帧数
# ):
#     # Filter out surface points above the specified z level
#     surface_mask = scanned_points[:, 2] > z_surface_level - 1
#     surface_points = scanned_points[surface_mask]

#     total_points = len(depth_map_points)
#     points_per_frame = len(trajectory) // num_frames  # Process according to trajectory

#     # 初始化可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     # 构建KD树，加速最近邻查找
#     depth_map_kdtree = KDTree(depth_map_points[:, :2])

#     # 创建用于每一帧的current_depth_map_points缓存
#     cached_depth_map_points = []

#     # 预处理所有帧的current_depth_map_points，提前计算并缓存
#     for frame in range(1, num_frames + 1):
#         current_trajectory = trajectory[:frame * points_per_frame]

#         # 查找所有轨迹点的最近邻depth_map_points
#         dists, indices = depth_map_kdtree.query(current_trajectory[:, :2], k=1)

#         # 选取最近的depth_map_points
#         current_depth_map_points = depth_map_points[indices]
#         cached_depth_map_points.append(current_depth_map_points)

#     # 实际渲染每一帧时使用缓存的depth_map_points
#     for frame in range(1, num_frames + 1):
#         current_depth_map_points = cached_depth_map_points[frame - 1]

#         # Build a KD tree for current frame's depth_map_points
#         kdtree = KDTree(current_depth_map_points[:, :2])

#         # Update z coordinates for surface points
#         offsetted_z_list = []
#         for point in surface_points:
#             dist, idx = kdtree.query(point[:2])
#             if dist < 2:
#                 offsetted_z_list.append(current_depth_map_points[idx][2])
#             else:
#                 offsetted_z_list.append(point[2])

#         # Update the z coordinates of the surface points
#         surface_points[:, 2] = np.array(offsetted_z_list)

#         # Update the z coordinates of the scanned points
#         new_scanned_points = np.copy(scanned_points)
#         new_scanned_points[surface_mask] = surface_points

#         # Create a point cloud object
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(new_scanned_points)
#         pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

#         # Estimate normals
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
#         )

#         # Create a mesh from the point cloud using Ball Pivoting Algorithm
#         distances = pcd.compute_nearest_neighbor_distance()
#         avg_dist = np.mean(distances)
#         radius = avg_dist
#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#             pcd, o3d.utility.DoubleVector([radius, radius * 2])
#         )

#         # Update the mesh object in the visualizer
#         vis.clear_geometries()
#         vis.add_geometry(mesh)
#         vis.poll_events()
#         vis.update_renderer()

    # vis.destroy_window()

def visualize_final_surface_dynamic(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    depth_map_points: np.ndarray,
    z_surface_level: float,
    trajectory: np.ndarray,
    num_frames: int = 10,
    save_dir: str = "./rendered_frames"
):
    # Filter out surface points above the specified z level
    surface_mask = scanned_points[:, 2] > z_surface_level - 1
    surface_points = scanned_points[surface_mask]

    points_per_frame = len(trajectory) // num_frames

    # 构建 KD 树，加速最近邻查找
    depth_map_kdtree = KDTree(depth_map_points[:, :2])

    # 创建保存帧文件的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 使用 tqdm 显示预处理的进度条
    for frame in tqdm(range(1, num_frames + 1), desc="Saving frames"):
        current_trajectory = trajectory[:frame * points_per_frame]

        # 查找所有轨迹点的最近邻depth_map_points
        dists, indices = depth_map_kdtree.query(current_trajectory[:, :2], k=1)

        # 获取当前帧的 depth_map_points
        current_depth_map_points = depth_map_points[indices]

        # Build a KD tree for current frame's depth_map_points
        kdtree = KDTree(current_depth_map_points[:, :2])

        # 更新表面点的 z 坐标
        offsetted_z_list = []
        for point in surface_points:
            dist, idx = kdtree.query(point[:2])
            if dist < 2:
                offsetted_z_list.append(current_depth_map_points[idx][2])
            else:
                offsetted_z_list.append(point[2])

        # 更新表面点和扫描点的 z 坐标
        surface_points[:, 2] = np.array(offsetted_z_list)
        new_scanned_points = np.copy(scanned_points)
        new_scanned_points[surface_mask] = surface_points

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_scanned_points)
        pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

        # 估计法线
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # 使用 Ball Pivoting 算法创建网格
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )

        # 保存当前帧的网格文件
        frame_file_path = os.path.join(save_dir, f"frame_{frame:03d}.ply")
        o3d.io.write_triangle_mesh(frame_file_path, mesh)

def set_camera_view(vis, intrinsic_matrix, extrinsic_matrix):
    # 获取视角控制器
    view_ctl = vis.get_view_control()
    
    # 创建一个新的摄像机参数对象
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    
    # 设置新的摄像机参数
    cam_params.intrinsic.set_intrinsics(640, 480, intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
    cam_params.extrinsic = extrinsic_matrix
    
    # 应用新的视角
    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

# def load_and_render_frames(save_dir: str, num_frames: int):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     view_control = vis.get_view_control()
#     # 使用 tqdm 显示渲染进度条
#     for frame in tqdm(range(1, num_frames + 1), desc="Rendering frames"):
#         frame_file_path = os.path.join(save_dir, f"frame_{frame:03d}.ply")

#         if os.path.exists(frame_file_path):
#             mesh = o3d.io.read_triangle_mesh(frame_file_path)
#             current_camera_params = view_control.convert_to_pinhole_camera_parameters()
#             vis.clear_geometries()
#             vis.add_geometry(mesh)
#             view_control.convert_from_pinhole_camera_parameters(current_camera_params)
#             vis.poll_events()
#             vis.update_renderer()

#     # vis.destroy_window()
#     vis.run()

import open3d as o3d
import os
from tqdm import tqdm

def load_and_render_frames(save_dir: str, num_frames: int):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # 获取当前视角状态
    view_control = vis.get_view_control()

    # 初始化第一个 mesh
    frame_file_path = os.path.join(save_dir, f"frame_001.ply")
    if os.path.exists(frame_file_path):
        mesh = o3d.io.read_triangle_mesh(frame_file_path)
        vis.add_geometry(mesh)

    # 用于控制播放的标志
    start_render = [False]  # 用列表包装以便在回调中修改

    # 键盘回调函数，按 's' 键时开始播放
    def start_render_callback(vis):
        start_render[0] = True
        print("Rendering started!")
        return False  # 返回 False 不会阻塞事件

    # 注册 's' 键的回调函数 (小写's')
    vis.register_key_callback(ord('S'), start_render_callback)

    # 先显示初始帧，等待按下 's' 键
    print("Press 'S' to start rendering.")
    while not start_render[0]:
        vis.poll_events()
        vis.update_renderer()

    # 开始渲染帧
    for frame in tqdm(range(2, num_frames + 1), desc="Rendering frames"):
        frame_file_path = os.path.join(save_dir, f"frame_{frame:03d}.ply")

        if os.path.exists(frame_file_path):
            new_mesh = o3d.io.read_triangle_mesh(frame_file_path)

            # 保存当前视角
            # current_camera_params = view_control.convert_to_pinhole_camera_parameters()

            # 通过更新几何体而不是清除几何体来避免视角重置
            vis.remove_geometry(mesh, reset_bounding_box=False)
            vis.add_geometry(new_mesh, reset_bounding_box=False)

            # 恢复之前的视角
            # view_control.convert_from_pinhole_camera_parameters(current_camera_params)

            # 更新 mesh 引用
            mesh = new_mesh

            vis.poll_events()
            vis.update_renderer()

    vis.run()



