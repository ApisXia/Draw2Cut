import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import KDTree
import os
import pyqtgraph.opengl as gl
from configs.load_config import CONFIG

def visualize_cutting_planning(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    coarse_cutting_points: np.ndarray,
    fine_cutting_points: np.ndarray,
    ultra_fine_cutting_points: np.ndarray,
    gl_view: gl.GLViewWidget = None,
):
    if gl_view is None:
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
    else:
        scatter = gl.GLScatterPlotItem(
            pos=scanned_points,
            size=0.5,
            color = scanned_colors,
        )
        gl_view.addItem(scatter)

        # create coarse trajectory point cloud object using green color
        coarse_trajectory_scatter = gl.GLScatterPlotItem(pos=coarse_cutting_points, 
                                                         color=np.array([[0, 1, 0]] * len(coarse_cutting_points)), 
                                                         size=0.5)
        gl_view.addItem(coarse_trajectory_scatter)

        # create fine trajectory point cloud object using red color
        if len(fine_cutting_points) > 0:
            fine_trajectory_scatter = gl.GLScatterPlotItem(pos=fine_cutting_points, 
                                                           color=np.array([[1, 0, 0]] * len(fine_cutting_points)), 
                                                           size=0.5)
            gl_view.addItem(fine_trajectory_scatter)

        # create ultra fine trajectory point cloud object using blue color
        if len(ultra_fine_cutting_points) > 0:
            ultra_fine_trajectory_scatter = gl.GLScatterPlotItem(pos=ultra_fine_cutting_points, 
                                                                 color=np.array([[0, 0, 1]] * len(ultra_fine_cutting_points)), 
                                                                 size=0.5)
            gl_view.addItem(ultra_fine_trajectory_scatter)



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
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = avg_dist
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector([radius, radius * 2])
    # )

    print("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    save_dir = CONFIG["temp_file_path"]
    frame_file_path = os.path.join(save_dir, f"final_surface.ply")
    o3d.io.write_triangle_mesh(frame_file_path, mesh)

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([mesh])


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

# # vis.destroy_window()
# import cupy as cp
# from cuml.neighbors import NearestNeighbors
# because 3d construction have to run on cpu, so run on cuda is not very useful

def visualize_final_surface_dynamic(
    scanned_points: np.ndarray,
    scanned_colors: np.ndarray,
    depth_map_points: np.ndarray,
    z_surface_level: float,
    trajectory: list,
    num_frames: int = 10,
):
    # scanned_points = cp.asarray(scanned_points)
    # # scanned_colors = cp.asarray(scanned_colors)
    # depth_map_points = cp.asarray(depth_map_points)
    
    # Filter out surface points above the specified z level
    surface_mask = scanned_points[:, 2] > z_surface_level - 1
    surface_points = scanned_points[surface_mask]

    # 构建 KD 树，加速最近邻查找
    depth_map_kdtree = KDTree(depth_map_points[:, :2])
    # depth_map_kdtree = NearestNeighbors(n_neighbors=1,algorithm='brute')
    # depth_map_kdtree.fit(depth_map_points[:, :2])

    saveDir = os.path.join(CONFIG["temp_file_path"], "cutting_moive")
    # 创建保存帧文件的目录
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    chunk_size = max(1, len(trajectory) // num_frames)
    # 使用 tqdm 显示预处理的进度条
    for frame in tqdm(range(0,len(trajectory),chunk_size), desc="Saving frames"):
        end_frame = min(frame + chunk_size, len(trajectory))
        points_to_combine = [point for sublist in trajectory[:end_frame] for point in sublist]
        # current_trajectory = cp.array(points_to_combine)
        current_trajectory = np.array(points_to_combine)

        # 查找所有轨迹点的最近邻depth_map_points
        dists, indices = depth_map_kdtree.query(current_trajectory[:, :2], k=1)

        # 获取当前帧的 depth_map_points
        current_depth_map_points = depth_map_points[indices]

        # Build a KD tree for current frame's depth_map_points
        kdtree = KDTree(current_depth_map_points[:, :2])

        # 更新表面点的 z 坐标
        surface_points_2d = surface_points[:, :2]
        dists, indices = kdtree.query(surface_points_2d)
        nearest_z_values = current_depth_map_points[indices.flatten(), 2]
        offsetted_z = np.where(dists.flatten() < 2, nearest_z_values, surface_points[:, 2])

        # offsetted_z_list = []
        # for point in surface_points:
        #     dist, idx = kdtree.query(point[:2])
        #     if dist < 2:/
        #         offsetted_z_list.append(current_depth_map_points[idx][2])
        #     else:
        #         offsetted_z_list.append(point[2])

        # 更新表面点和扫描点的 z 坐标
        surface_points[:, 2] = offsetted_z
        new_scanned_points = np.copy(scanned_points)
        new_scanned_points[surface_mask] = surface_points
        # new_scanned_points = cp.asnumpy(new_scanned_points)

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_scanned_points)
        pcd.colors = o3d.utility.Vector3dVector(scanned_colors)

            # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Create a mesh from the point cloud using Ball Pivoting Algorithm
        # distances = pcd.compute_nearest_neighbor_distance()
        # avg_dist = np.mean(distances)
        # radius = avg_dist
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     pcd, o3d.utility.DoubleVector([radius, radius * 2])
        # )

        # print("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

        # 保存当前帧的网格文件
        frame_file_path = os.path.join(saveDir, f"frame_{frame:03d}.ply")
        o3d.io.write_triangle_mesh(frame_file_path, mesh)


# def set_camera_view(vis, intrinsic_matrix, extrinsic_matrix):
#     # 获取视角控制器
#     view_ctl = vis.get_view_control()

#     # 创建一个新的摄像机参数对象
#     cam_params = view_ctl.convert_to_pinhole_camera_parameters()

#     # 设置新的摄像机参数
#     cam_params.intrinsic.set_intrinsics(
#         640,
#         480,
#         intrinsic_matrix[0, 0],
#         intrinsic_matrix[1, 1],
#         intrinsic_matrix[0, 2],
#         intrinsic_matrix[1, 2],
#     )
#     cam_params.extrinsic = extrinsic_matrix

#     # 应用新的视角
#     view_ctl.convert_from_pinhole_camera_parameters(cam_params)

import time


def load_and_render_frames(save_dir: str, num_frames: int, target_fps: int = 30):
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
    render_state = {
        "start": False,
        "last_frame_time": time.time(),
        "frame_idx": 1,
        "frame_time": 1.0 / target_fps,
    }

    # 键盘回调函数，按 's' 键时开始播放
    def start_render_callback(vis):
        render_state["start"] = True
        print("Rendering started!")
        return False  # 返回 False 不会阻塞事件

    # 注册 's' 键的回调函数 (小写's')
    vis.register_key_callback(ord("S"), start_render_callback)

    # 先显示初始帧，等待按下 's' 键
    print("Press 'S' to start rendering.")
    while not render_state["start"]:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # 减少空闲等待的 CPU 使用

    # 开始渲染帧
    while render_state["frame_idx"] < num_frames:
        current_time = time.time()
        elapsed = current_time - render_state["last_frame_time"]

        # 控制帧率
        if elapsed >= render_state["frame_time"]:
            frame_file_path = os.path.join(
                save_dir, f"frame_{render_state['frame_idx']:03d}.ply"
            )

            if os.path.exists(frame_file_path):
                new_mesh = o3d.io.read_triangle_mesh(frame_file_path)
                vis.remove_geometry(mesh, reset_bounding_box=False)
                vis.add_geometry(new_mesh, reset_bounding_box=False)
                mesh = new_mesh

            # 更新时间和帧索引
            render_state["last_frame_time"] = current_time
            render_state["frame_idx"] += 1

            # 如果处理时间超过预期，可能需要跳过一些帧
            frames_to_skip = int(elapsed / render_state["frame_time"])
            if frames_to_skip > 1:
                render_state["frame_idx"] += frames_to_skip - 1
                print(f"Skipped {frames_to_skip-1} frames to maintain performance")

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.001)  # 短暂睡眠以减少 CPU 使用

    vis.run()
