from flask import Flask, jsonify, render_template
import numpy as np
from scipy.spatial import KDTree,cKDTree


app = Flask(__name__)

def rgb2float(rgb):
    return np.array([c / 255.0*0.7 for c in rgb])
    
def filter_points(wood_points,wood_colors, cut_points, threshold=0.1):
    """
    对点集A进行过滤,删除那些与点集B中的点足够接近的点。
    
    参数:
    A (numpy.ndarray): 点集A,shape为(len, 3)
    B (numpy.ndarray): 点集B,shape为(len, 3)
    threshold (float): 距离阈值,默认为0.1
    
    返回:
    filtered_A (numpy.ndarray): 过滤后的点集A
    """
    # 创建点集B的KDTree
    kdtree = KDTree(cut_points)
    
    # 删除那些与点集B中的点足够接近的点
    filtered_points = []
    filtered_colors = []
    distances, indices = kdtree.query(wood_points, k=1)
    keep_mask = distances > threshold
    unkeep_mask = distances <= threshold
    wood_points[unkeep_mask,2] = wood_points[unkeep_mask,2] -5
    wood_colors[unkeep_mask] = rgb2float([184,151,136])
    # wood_colors[keep_mask] = [255,255,255]
    return wood_points,wood_colors,keep_mask

    for wood_point, wood_color in zip(wood_points, wood_colors):
        dist = 100
        for cut_point in cut_points:
            dist = min(dist, np.linalg.norm(wood_point - cut_point))
        if dist > threshold:
            filtered_points.append(wood_point)
            filtered_colors.append(wood_color)
    
    return np.array(filtered_points), np.array(filtered_colors)

# 读取木头的点云数据
wood_data = np.load("images/points_transformed.npz")
wood_points = wood_data['points']
wood_colors = wood_data['colors']

# 读取被割掉轨迹的点云数据
cut_data = np.load("images/cut_points.npz")
cut_points = cut_data['points']
cut_points[:,0] = -cut_points[:,0]
cut_points[:,2] -= 1
cut_colors = cut_data['colors']

# 过滤木头点云数据
wood_points, wood_colors,keep_mask = filter_points(wood_points, wood_colors, cut_points, threshold=2)
hole = False
if hole:
    wood_points = wood_points[keep_mask]
    wood_colors = wood_colors[keep_mask]
print(wood_points.shape)

centroid = np.mean(wood_points, axis=0)
wood_points -= centroid
cut_points -= centroid

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def get_data():
    data = {
        "wood_points": wood_points.tolist(),
        "wood_colors": wood_colors.tolist(),
        "cut_points": cut_points.tolist(),
        "cut_colors": cut_colors.tolist(),
        "texts": ["Contour 'Circle' -> 3mm", "Cut trajectories visualized in 'Gray'", "number: 3 contours"]
    }
    return jsonify(data)

@app.route('/auto-smooth', methods=['POST'])
def auto_smooth():
    # 处理auto-smooth逻辑并返回新的数据
    data = {
        "wood_points": wood_points.tolist(),
        "wood_colors": wood_colors.tolist(),
        "cut_points": cut_points.tolist(),
        "cut_colors": cut_colors.tolist(),
        "texts": ["text 'sake' -> 3mm", "visualized in Green", "number: 9 contours"]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
