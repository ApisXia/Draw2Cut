import numpy as np
import open3d as o3d
import copy
from scipy.spatial import KDTree

def rgb2float(rgb):
    return np.array([c / 255.0 * 0.7 for c in rgb])

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

class cut_status:
    def __init__(self, wood_points, wood_colors,cut_points,cut_colors,cut_depth):
        self.wood_points = wood_points
        self.wood_colors = wood_colors
        self.cut_points = cut_points
        self.cut_colors = cut_colors
        self.cut_depth = cut_depth
    def build_mesh(self):
        A = self.wood_points
        B = self.wood_colors
        C = copy.deepcopy(self.cut_points)
        C[:, 2] -= self.cut_depth
        D = self.cut_colors
        A = np.concatenate((A, C), axis=0)
        B = np.concatenate((B, D), axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(B)
        pcd.points = o3d.utility.Vector3dVector(A)
        # o3d.visualization.draw_geometries([pcd])  
        print(A.shape)
        mesh = construct_3d_model(A, B)
        return mesh
    def filter_points(self, threshold=0.1):
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
        kdtree = KDTree(self.cut_points)

        # Remove points in set A that are close enough to points in set B
        filtered_points = []
        filtered_colors = []
        distances, indices = kdtree.query(self.wood_points, k=1)
        keep_mask = distances > threshold
        unkeep_mask = distances <= threshold
        # wood_points[unkeep_mask, 2] = wood_points[unkeep_mask, 2] - 5
        # wood_colors[unkeep_mask] = rgb2float([184, 151, 136])
        cut_points = copy.deepcopy(self.wood_points[unkeep_mask])
        cut_colors = copy.deepcopy(self.wood_colors[unkeep_mask])
        # cut_points[:, 2] -= 5
        cut_colors = rgb2float([184, 151, 136]) * np.ones_like(cut_points)
        # wood_points = wood_points[keep_mask]
        # wood_colors = wood_colors[keep_mask]
        # wood_colors[keep_mask] = [255,255,255]
        wood_points = copy.deepcopy(self.wood_points[keep_mask])
        wood_colors = copy.deepcopy(self.wood_colors[keep_mask])
        new_state = cut_status(wood_points, wood_colors, cut_points, cut_colors, self.cut_depth)
        return new_state
    def move_points(self,val):
        self.cut_points -= val
        self.wood_points -= val
        return self
    def filter_wood_zone(self,width = 10,height = 10):
        #TODO: Filter points bias on z axis
        # Calculate center of the point cloud
        center = self.wood_points.mean(axis=0)
        
        # Define rectangle size (e.g., width=10, height=10)
        mask = (
            (self.wood_points[:, 0] > center[0] - width / 2) & (self.wood_points[:, 0] < center[0] + width / 2) &
            (self.wood_points[:, 1] > center[1] - height / 2) & (self.wood_points[:, 1] < center[1] + height / 2)
        )
        
        # Filter points and colors
        filtered_points = self.wood_points[mask]
        filtered_colors = self.wood_colors[mask]

        return cut_status(filtered_points, filtered_colors, self.cut_points, self.cut_colors, self.cut_depth)