from find_standard_plane import calculate_points_plane
from glob import glob
import numpy as np
import open3d as o3d
import cv2

file_path = "data/Feb15Test/*.npz"
data_list = glob(file_path)

points_plane, plane_para = calculate_points_plane(data_list)

origin_label = "8"
x_axis_label = ["1", "5"]  # first is the half index of the x axis, second is the end index of the x axis
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

if "2" not in points_plane and "6" in points_plane:
    y_length = np.linalg.norm(points_plane["6"] - points_plane[origin_label]) * 2
elif "2" in points_plane:
    y_length = np.linalg.norm(points_plane["2"] - points_plane[origin_label])

z_direction = np.cross(y_direction, x_direction)
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
data = np.load(data_list[20])
points = data['points_pos']
colors = data['transformed_color'][..., (0, 1, 2)].reshape((-1, 3))

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
points_transformed = np.dot(points_transformed, np.array([x_direction, y_direction, z_direction]).T)

# get the mask of points_transformed, with 0 < x < x_length and 0 < y < y_length, and z < z_max and z > z_min
mask = (points_transformed[:, 0] > 0) & (points_transformed[:, 0] < x_length) & (points_transformed[:, 1] > 0) & (points_transformed[:, 1] < y_length) & (points_transformed[:, 2] < z_max) & (points_transformed[:, 2] > z_min)

# from large z to small z, find a plane in the points_transformed[mask] which has the most points, and stop
# get the mask of the plane
mask_plane = np.zeros_like(mask)
z = z_max
while z > z_min:
    mask_plane = (points_transformed[:, 2] > z) & mask
    if np.sum(mask_plane) > 0.3 * np.sum(mask):
        break
    z -= 1
z_surface = z

# visualize the get slice
# points = points[mask_plane, :]
# colors = colors[mask_plane, :]

# # create point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

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
extract_mask = (points_transformed[:, 0] > x_min) & (points_transformed[:, 0] < x_max) & (points_transformed[:, 1] > y_min) & (points_transformed[:, 1] < y_max) & mask_plane

# get the corresponding color for these points, and wrap them into a 2d image
extract_color = colors[extract_mask, :]
extract_points = points_transformed[extract_mask, :]

x_size = int(x_max - x_min)
y_size = int(y_max - y_min)
wrapped_image = np.zeros((x_size + 1, y_size + 1, 3), dtype=np.uint8)
for point, color in zip(extract_points, extract_color):
    x = int(point[0] - x_min)
    y = int(point[1] - y_min)
    wrapped_image[x_size - x, y, :] = color.astype(np.uint8)

# fill the holes in the wrapped_image with average the nearest color
for i in range(x_size + 1):
    for j in range(y_size + 1):
        if np.all(wrapped_image[i, j, :] == 0):
            # find the nearest color
            distance = 1
            while True:
                for x in range(i - distance, i + distance + 1):
                    for y in range(j - distance, j + distance + 1):
                        if x >= 0 and x <= x_size and y >= 0 and y <= y_size:
                            if not np.all(wrapped_image[x, y, :] == 0):
                                wrapped_image[i, j, :] = wrapped_image[x, y, :]
                                break
                    if not np.all(wrapped_image[i, j, :] == 0):
                        break
                if not np.all(wrapped_image[i, j, :] == 0):
                    break
                distance += 1


# # visualize the wrapped image
cv2.imshow("wrapped_image", wrapped_image)
cv2.waitKey(0)

# increse the size by 10 and save the wrapped image
cv2.imwrite("images/wrapped_image.png", wrapped_image)
wrapped_image = cv2.resize(wrapped_image, (0, 0), fx=10, fy=10)
cv2.imwrite("images/wrapped_image_zoom.png", wrapped_image)

# save left bottom point position, including x, y, z and x_length, y_length
left_bottom_point = np.array([x_min, y_min, z_surface])
print(f"left_bottom_point: {left_bottom_point}")
np.savez("left_bottom_point.npz", left_bottom_point=left_bottom_point, x_length=x_length, y_length=y_length)





