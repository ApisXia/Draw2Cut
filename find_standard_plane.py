from glob import glob
import numpy as np
import cv2
from pyzbar import pyzbar
from collections import deque
import open3d as o3d


def get_qr_codes_position(image, resize_factor=2):
    # Resize the image to a larger size
    resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)

    # Find QR codes and their information
    qr_codes = pyzbar.decode(resized_image)

    qr_code_dict = {}

    # Iterate over the QR codes
    for qr_code in qr_codes:
        # Extract the bounding box coordinates
        (x, y, w, h) = qr_code.rect

        # Scale the coordinates back to the original image size
        x = x // resize_factor
        y = y // resize_factor
        w = w // resize_factor
        h = h // resize_factor

        # Find the center of the QR code
        center_x = int(x + (w // 2))
        center_y = int(y + (h // 2))

        # x,y need to be int
        x = int(x)
        y = int(y)

        # Get the information from the QR code
        qr_code_data = qr_code.data.decode("utf-8")

        # only use number in this sentence
        qr_code_data = str(qr_code_data.split(" ")[-1])

        qr_code_dict[qr_code_data] = (center_x, center_y)

    return qr_code_dict


def best_fit_plane(points):
    # calculate the centroid of points
    centroid = np.mean(points, axis=0)
    # shift the points to have a centroid at the origin
    shifted_points = points - centroid
    # singular value decomposition
    _, _, Vt = np.linalg.svd(shifted_points)
    # the last row of Vt matrix gives the normal to the plane
    normal = Vt[-1]
    # form the equation of the plane a*x + b*y + c*z = d
    d = -centroid.dot(normal)
    # return the plane
    return normal[0], normal[1], normal[2], d


def calculate_points_plane(data_list, quene_size=100):
    coordinate_origin_name = "8"

    qr_position_quene = {}
    qr_position_quene[coordinate_origin_name] = deque(maxlen=quene_size)

    for data_path in data_list:
        # load data
        data = np.load(data_path)
        qr_codes_dict = get_qr_codes_position(data["transformed_color"])
        point_cloud_pos = data["points_pos"]
        depth_shape = data["depth"].shape
        point_cloud_pos = point_cloud_pos.reshape((depth_shape[0], depth_shape[1], 3))

        # push dict to quene
        for key, value in qr_codes_dict.items():
            if key not in qr_position_quene:
                qr_position_quene[key] = deque(maxlen=quene_size)
            qr_position_quene[key].append(point_cloud_pos[value[1], value[0], :])

        # if the length of coordinate_origin is 0, then skip this iteration
        if len(qr_position_quene[coordinate_origin_name]) == 0:
            continue

        # break when length of coordinate_origin is 20
        if len(qr_position_quene[coordinate_origin_name]) == quene_size:
            break

    # calculate the average position of each qr code
    average_position_dict = {}
    for key, value in qr_position_quene.items():
        if len(value) == 0:
            continue
        average_position_dict[key] = np.mean(value, axis=0)

    # concatenate the average position of each qr code
    points_plane = np.array(list(average_position_dict.values()))
    plane_para = best_fit_plane(points_plane)
    # projection points to the plane
    to_plane_dist = (
        plane_para[0] * points_plane[:, 0]
        + plane_para[1] * points_plane[:, 1]
        + plane_para[2] * points_plane[:, 2]
        + plane_para[3]
    ) / np.sqrt(plane_para[0] ** 2 + plane_para[1] ** 2 + plane_para[2] ** 2)
    points_plane = points_plane - np.reshape(to_plane_dist, (-1, 1)) * np.reshape(
        plane_para[:3], (1, -1)
    )

    # back to dict
    points_plane = dict(zip(average_position_dict.keys(), points_plane))

    return points_plane, plane_para


if __name__ == "__main__":
    file_path = "data/0323/*.npz"
    data_list = glob(file_path)
    if len(data_list) == 0:
        raise ValueError("No data found")

    points_plane, _ = calculate_points_plane(data_list)
    points_plane = list(points_plane.values())

    # load data
    data = np.load(data_list[50])
    points = data["points_pos"]
    colors = data["transformed_color"][..., (2, 1, 0)].reshape((-1, 3))
    colors = colors / 255.0
    depth = data["depth"].reshape((-1))

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

    # create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    object_to_draw = []
    object_to_draw.append(pcd)

    # draw points_plane on the point cloud as a ball
    for i in range(len(points_plane)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.1, 0.1, 0.7])
        sphere.translate(points_plane[i])
        object_to_draw.append(sphere)

    # visualize point cloud
    o3d.visualization.draw_geometries(object_to_draw)
