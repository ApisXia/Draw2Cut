from flask import Flask, jsonify, render_template
import numpy as np

app = Flask(__name__)

# 读取木头的点云数据
wood_data = np.load("images/points_transformed.npz")
wood_points = wood_data['points']
wood_colors = wood_data['colors']

# 读取被割掉轨迹的点云数据
cut_data = np.load("images/cut_points.npz")
cut_points = cut_data['points']
cut_points[:, 0] = -cut_points[:, 0]
cut_colors = cut_data['colors']

centroid = np.mean(wood_points, axis=0)
wood_points -= centroid
cut_points -= centroid


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    return jsonify(
        wood_points=wood_points.tolist(), wood_colors=wood_colors.tolist(),
        cut_points=cut_points.tolist(), cut_colors=cut_colors.tolist()
    )

if __name__ == '__main__':
    app.run(debug=True)
