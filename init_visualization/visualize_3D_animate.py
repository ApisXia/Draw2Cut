from glob import glob
import numpy as np
import open3d as o3d
import time

file_path = "thin3_data/*.npz"
data_list = glob(file_path)

# Initialize visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

for data_path in data_list:
    # Load data
    data = np.load(data_path)
    points = data['points_pos']
    colors = data['transformed_color'][..., (2, 1, 0)].reshape((-1, 3))
    colors = colors / 255.0
    depth = data['depth'].reshape((-1))

    # Keep depth larger than 0
    mask = depth > 0
    points = points[mask, :]
    colors = colors[mask, :]
    depth = depth[mask]

    # Remove points with depth larger than threshold
    threshold = 1000
    mask = depth < threshold
    points = points[mask, :]
    colors = colors[mask, :]
    depth = depth[mask]

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Set the camera view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])  # Set the camera front direction to -Z axis
    ctr.set_lookat([0, 0, 0])  # Set the camera lookat position to the origin

    # Clear the visualizer
    vis.clear_geometries()

    # Add point cloud to the visualizer
    vis.add_geometry(pcd)

    for i in range(2):
        # Update the view
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Pause to help visualize individual files
        time.sleep(0.5)

# Destroy the visualizer
vis.destroy_window()