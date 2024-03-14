from glob import glob
import numpy as np
import open3d as o3d

file_path = "thin3_data/*.npz"
data_list = glob(file_path)

# use open3d to visualize 3D point cloud data
for data_path in data_list:
    # load data
    data = np.load(data_path)
    points = data['points_pos']
    colors = data['transformed_color'][..., (2, 1, 0)].reshape((-1, 3))
    colors = colors / 255.0
    color_test = data['transformed_color']
    depth = data['depth'].reshape((-1))

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

    # remove points with depth 

    # create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # visualize point cloud
    o3d.visualization.draw_geometries([pcd])
    break



