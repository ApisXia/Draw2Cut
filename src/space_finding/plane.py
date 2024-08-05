import numpy as np

from collections import deque

from src.space_finding.qr_code import get_qr_codes_position


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
        ## test
        # dt = data["transformed_color"].copy()
        # dt = dt.reshape((800, 1280, 3))
        # dt = (dt * 255).astype(np.uint8)
        # # cv2.imshow("test", dt)
        # # cv2.waitKey(0)  # Wait for a key press to close the window
        # # cv2.destroyAllWindows()
        # # ## test
        # qr_codes_dict = get_qr_codes_position(dt)
        qr_codes_dict = get_qr_codes_position(
            data["transformed_color"].reshape((800, 1280, 3))
        )

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
