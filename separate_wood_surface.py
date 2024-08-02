import os
import cv2

import numpy as np
import open3d as o3d
from copy import deepcopy

from glob import glob
from find_standard_plane import calculate_points_plane

from utils.mark_config import WARPPING_RESOLUTION, FOLDER_PATH, SURFACE_UPSCALE

file_path = FOLDER_PATH
# file_path = "test_point_cloud.npz"
data_list = glob(file_path)

os.makedirs("images", exist_ok=True)

points_plane, plane_para = calculate_points_plane(data_list[0:])

origin_label = "8"
x_axis_label = [
    "1",
    "5",
]  # first is the half index of the x axis, second is the end index of the x axis
y_axis_label = ["6", "2"]
oppsite_label = "7"

# assert at least one label in x_axis_label can be found in points_plane
assert any([label in points_plane.keys() for label in x_axis_label])
# same as y_axis_label
assert any([label in points_plane.keys() for label in y_axis_label])

x_direction = 0
x_counter = 0
for label in x_axis_label:
    if label in points_plane:
        direction = points_plane[label] - points_plane[origin_label]
        direction = direction / np.linalg.norm(direction)
        x_direction += direction
        x_counter += 1
x_direction = x_direction / x_counter
# print("x_norm_length", np.linalg.norm(x_direction))
x_direction = x_direction / np.linalg.norm(x_direction)

if "5" not in points_plane and "1" in points_plane:
    x_length = np.linalg.norm(points_plane["1"] - points_plane[origin_label]) * 2
elif "5" in points_plane:
    x_length = np.linalg.norm(points_plane["5"] - points_plane[origin_label])

y_direction = 0
y_counter = 0
for label in y_axis_label:
    if label in points_plane:
        direction = points_plane[label] - points_plane[origin_label]
        direction = direction / np.linalg.norm(direction)
        y_direction += direction
        y_counter += 1
y_direction = y_direction / y_counter
# print("y_norm_length", np.linalg.norm(y_direction))
y_direction = y_direction / np.linalg.norm(y_direction)

print("x dot y beginning", np.dot(x_direction, y_direction))

if "2" not in points_plane and "6" in points_plane:
    y_length = np.linalg.norm(points_plane["6"] - points_plane[origin_label]) * 2
elif "2" in points_plane:
    y_length = np.linalg.norm(points_plane["2"] - points_plane[origin_label])

z_direction = np.cross(y_direction, x_direction)

# XXX: using z and y axis to calculate orthogonal x axis again
x_direction = np.cross(z_direction, y_direction)
print("x dot y later", np.dot(x_direction, y_direction))

z_max = 50
z_min = 10

origin_point = points_plane[origin_label]

print(f"x_direction: {x_direction}")
print(f"x_length: {x_length}")
print(f"y_direction: {y_direction}")
print(f"y_length: {y_length}")
print(f"z_direction: {z_direction}")
print(f"origin_point: {origin_point}")

# load data
data = np.load(data_list[3])
points = data["points_pos"]
colors = data["transformed_color"].reshape((-1, 3))

# points_holder = [points]
# for i in range(1, 10):
#     try:
#         data = np.load(data_list[i])
#         points = data['points_pos']
#         colors = data['transformed_color'][..., (0, 1, 2)].reshape((-1, 3))
#         points_holder.append(points)
#     except:
#         break
# points = np.average(points_holder, axis=0)

# transform the points to the standard plane
points_transformed = points - origin_point
points_transformed = np.dot(
    points_transformed, np.array([x_direction, y_direction, z_direction]).T
)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_transformed)
# pcd.colors = o3d.utility.Vector3dVector(colors / 255)

# object_to_draw = []
# object_to_draw.append(pcd)

# # visualize point cloud
# o3d.visualization.draw_geometries(object_to_draw)
# assert False

# get the mask of points_transformed, with 0 < x < x_length and 0 < y < y_length, and z < z_max and z > z_min
mask = (
    (points_transformed[:, 0] > 0)
    & (points_transformed[:, 0] < x_length)
    & (points_transformed[:, 1] > 0)
    & (points_transformed[:, 1] < y_length)
    & (points_transformed[:, 2] < z_max)
    & (points_transformed[:, 2] > z_min)
)

# from large z to small z, find a plane in the points_transformed[mask] which has the most points, and stop
# get the mask of the plane
mask_plane = np.zeros_like(mask)
z = z_max
z_tolerance_value = 2
z_tolerance = z_tolerance_value
z_step = 0.5
while z > z_min:
    mask_plane = (points_transformed[:, 2] > z) & mask
    if np.sum(mask_plane) > 0.5 * np.sum(mask):
        z_tolerance -= z_step
    if z_tolerance <= 0:
        break
    z -= z_step
z_surface = z
z_surface += z_tolerance_value + z_step

# # visualize the get slice
# points = points[mask_plane, :]
# colors = colors[mask_plane, :]

# # create point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors / 255)

# object_to_draw = []
# object_to_draw.append(pcd)

# # visualize point cloud
# o3d.visualization.draw_geometries(object_to_draw)

# get bounding box of the slice in x and y direction
x_min = np.min(points_transformed[mask_plane, 0])
x_max = np.max(points_transformed[mask_plane, 0])
y_min = np.min(points_transformed[mask_plane, 1])
y_max = np.max(points_transformed[mask_plane, 1])

# within the bounding box, combine with mask to get extract_mask
extract_mask = (
    (points_transformed[:, 0] > x_min)
    & (points_transformed[:, 0] < x_max)
    & (points_transformed[:, 1] > y_min)
    & (points_transformed[:, 1] < y_max)
    & mask_plane
)
extract_mask = extract_mask & mask_plane

# get the corresponding color for these points, and wrap them into a 2d image
extract_color = colors[extract_mask, :]
extract_points = points_transformed[extract_mask, :]

x_size = int((x_max - x_min) / WARPPING_RESOLUTION)
y_size = int((y_max - y_min) / WARPPING_RESOLUTION)
wrapped_image = np.zeros((x_size + 1, y_size + 1, 3), dtype=np.uint8)
for point, color in zip(extract_points, extract_color):
    x = int((point[0] - x_min) / WARPPING_RESOLUTION)
    y = int((point[1] - y_min) / WARPPING_RESOLUTION)
    wrapped_image[x_size - x - 1, y, :] = color.astype(np.uint8)

# fill the holes in the wrapped_image with the average of the nearest colors
# 创建一个掩码，标记需要填充的区域
mask = np.all(wrapped_image == 0, axis=2).astype(np.uint8)
# 使用 inpaint 函数填充
wrapped_image = cv2.inpaint(wrapped_image, mask, 3, cv2.INPAINT_TELEA)
wrapped_image = cv2.cvtColor(wrapped_image, cv2.COLOR_BGR2RGB)

# increase the size by 10 and save the wrapped image
wrapped_image = cv2.resize(wrapped_image, (0, 0), fx=1, fy=1)
cv2.imwrite("images/wrapped_image.png", wrapped_image)
wrapped_image = cv2.resize(
    wrapped_image, (0, 0), fx=SURFACE_UPSCALE, fy=SURFACE_UPSCALE
)
# increase sharpness
wrapped_image = cv2.GaussianBlur(wrapped_image, (5, 5), 0)
cv2.imwrite("images/wrapped_image_zoom.png", wrapped_image)

# save left bottom point position, including x, y, z and x_length, y_length
left_bottom_point = np.array([x_min, y_min, z_surface])
print(f"left_bottom_point: {left_bottom_point}")
np.savez(
    "images/left_bottom_point.npz",
    left_bottom_point=left_bottom_point,
    x_length=x_length,
    y_length=y_length,
)

colors = colors.astype(np.float32)
colors /= 255.0

mask_visual = (
    (points_transformed[:, 0] > 0)
    & (points_transformed[:, 0] < x_length)
    & (points_transformed[:, 1] > 0)
    & (points_transformed[:, 1] < y_length)
    & (points_transformed[:, 2] < z_max + 5)
    & (points_transformed[:, 2] > -5)
)
points_transformed = points_transformed[mask_visual, :]
colors = colors[mask_visual, :]

# save points_transformed
np.savez("images/points_transformed.npz", points=points_transformed, colors=colors)

# add spheres to the point cloud
# create point cloud object

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_transformed)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# object_to_draw = []
# object_to_draw.append(pcd)

# # visualize point cloud
# o3d.visualization.draw_geometries(object_to_draw)
