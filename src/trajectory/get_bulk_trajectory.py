import cv2
import numpy as np

from scipy.ndimage import (
    distance_transform_edt,
    distance_transform_cdt,
)

from configs.load_config import CONFIG

if CONFIG["distance_metric"] == "chessboard":
    DIST_METRIC = distance_transform_cdt
elif CONFIG["distance_metric"] == "euclidean":
    DIST_METRIC = distance_transform_edt
else:
    raise ValueError("!!! Distance metric is not valid.")


def get_trajectory_incremental_cut_inward(
    bin_map: np.ndarray,
    radius: int,
    step_size: int,
):
    assert step_size <= radius * 2, ValueError(
        "!!! Step size is greater than diameter."
    )
    # define kernels for erosion
    kernel_radius = np.ones((radius, radius), np.uint8)
    kernel_step_size = np.ones((step_size, step_size), np.uint8)

    # shrink the bin_map by radius pixels using erosion
    bin_map = cv2.erode(bin_map, kernel_radius, iterations=1)

    trajectories = []
    visited_map = np.zeros_like(bin_map, dtype=np.uint8)

    circle_counter = 0
    while np.sum(bin_map) > 0:
        # get contours of the bin_map
        contours, _ = cv2.findContours(bin_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        processed_contours = []
        # Filter out already visited contours
        for contour in contours:
            # add the first point to the last point to close the contour
            circle_contour = np.concatenate((contour, contour[:1]), axis=0)
            processed_contours.append(circle_contour)
            mask = np.zeros_like(bin_map, dtype=np.uint8)
            cv2.drawContours(mask, [circle_contour], -1, 255, thickness=radius * 2)
            if np.sum(cv2.bitwise_and(mask, visited_map)) == 0:
                visited_map = cv2.bitwise_or(visited_map, mask)

        # transorm contours to list of points
        processed_contours = [contour.squeeze() for contour in processed_contours]
        # change x, y to y, x
        processed_contours = [
            [(point[1], point[0], 1) for point in contour]
            for contour in processed_contours
        ]

        # add the processed contours to the list of trajectories
        trajectories.extend(processed_contours)

        # shrink the bin_map by step_size pixels using erosion
        bin_map = cv2.erode(bin_map, kernel_step_size, iterations=1)

        circle_counter += 1

    return trajectories, visited_map


def get_trajectory_layer_cut(
    cutting_bulk_map: np.ndarray,
    reverse_mask_map: np.ndarray,
    behavior_type: str = "behavior_plane",
):
    # get settings from config
    assert behavior_type in CONFIG["behavior_mark"], ValueError(
        "!!! Behavior type is not valid."
    )
    radius = CONFIG["spindle_radius"]
    if behavior_type == "behavior_plane":
        step_size = CONFIG["step_size"]["plane"]
        slop_caving = 10000000
        slop_mount = 10000000
    elif behavior_type == "behavior_relief":
        step_size = CONFIG["step_size"]["relief"]
        slop_caving = max(CONFIG["relief_slop"]["caving"], 0)
        slop_mount = max(CONFIG["relief_slop"]["mount"], 0)

    # step1: depth map for reverse_mask_map
    depth_reverse = -DIST_METRIC(reverse_mask_map).astype(np.float32)
    # depth_reverse = -distance_transform_cdt(
    #     reverse_mask_map, metric="chessboard"
    # ).astype(np.float32)
    depth_reverse = kernel_linear(
        depth_map=depth_reverse,
        slope=slop_caving,
        clip_lower=-CONFIG["bulk_carving_depth"] * CONFIG["surface_upscale"],
        clip_upper=0,
    )
    depth_reverse /= CONFIG["surface_upscale"]

    # get mount dist map
    mount_map = np.bitwise_xor(cutting_bulk_map, reverse_mask_map)
    depth_mount = DIST_METRIC(mount_map).astype(np.float32)
    # depth_mount = distance_transform_cdt(mount_map, metric="chessboard").astype(
    #     np.float32
    # )
    depth_mount = kernel_linear(
        depth_map=depth_mount,
        slope=slop_mount,
        clip_lower=0,
        clip_upper=None,
    )
    depth_mount /= CONFIG["surface_upscale"]

    # combine to get final depth map
    combined_depth_map = depth_mount + depth_reverse

    # get bin map with a depth range to cut
    trajectories = []
    visited_map_list = []
    min_depth = np.min(combined_depth_map)
    z_arange_list = np.arange(0, min_depth, -CONFIG["depth_forward"]).tolist()
    z_arange_list.append(min_depth)

    layer_counter = 0
    layered_bulk_mask_list = []
    for idx in range(len(z_arange_list) - 1):
        print(f"! Working on layer {layer_counter}")
        z_max = z_arange_list[idx]
        z_min = z_arange_list[idx + 1]
        bin_map = np.logical_and(
            combined_depth_map >= min_depth, combined_depth_map < z_max
        ).astype(np.uint8)

        if np.sum(bin_map) == 0:
            continue

        layered_bulk_mask_list.append(bin_map)

        trajectories_layer, visited_map = get_trajectory_incremental_cut_inward(
            bin_map=bin_map,
            radius=radius,
            step_size=step_size,
        )

        trajectories_layer = [
            [(point[0], point[1], z_min) for point in contour]
            for contour in trajectories_layer
        ]

        trajectories.extend(trajectories_layer)
        visited_map_list.append(visited_map)

        layer_counter += 1

    return trajectories, visited_map_list, layered_bulk_mask_list


""" Depth Kernel Section """


def kernel_linear(depth_map, slope, clip_lower, clip_upper):
    return np.clip(depth_map * slope, clip_lower, clip_upper)


""" Drawing section"""


def draw_trajectoryx10(visited_map, trajectories):
    # Resize the image to 10 times larger
    map_image = cv2.resize(map_image, None, fx=10, fy=10)

    # Convert grayscale image to BGR
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

    # Iterate over the trajectories, draw each one
    for trajectory in trajectories:
        if len(trajectory) == 1:
            # If the trajectory is just one point, draw a red circle with radius 4
            cv2.circle(
                map_image,
                (trajectory[0][1] * 10, trajectory[0][0] * 10),
                4,
                (0, 0, 255),
                thickness=-1,
            )
        else:
            for i in range(1, len(trajectory)):
                # Draw arrow between consecutive points
                pt1 = (int(trajectory[i - 1][1] * 10), int(trajectory[i - 1][0] * 10))
                pt2 = (int(trajectory[i][1] * 10), int(trajectory[i][0] * 10))
                cv2.arrowedLine(map_image, pt1, pt2, (0, 0, 255), 3)  # Arrow in red BGR

    return map_image


def draw_trajectory(map_image, trajectories):
    # Convert grayscale image to BGR
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

    # Iterate over the trajectories, draw each one
    for trajectory in trajectories:
        if len(trajectory) == 1:
            # If the trajectory is just one point, draw a red circle with radius 4
            cv2.circle(
                map_image,
                (trajectory[0][1], trajectory[0][0]),
                4,
                (0, 0, 255),
                thickness=-1,
            )
        else:
            for i in range(1, len(trajectory)):
                # Draw arrow between consecutive points
                pt1 = (int(trajectory[i - 1][1]), int(trajectory[i - 1][0]))
                pt2 = (int(trajectory[i][1]), int(trajectory[i][0]))
                cv2.arrowedLine(map_image, pt1, pt2, (0, 0, 255), 3)  # Arrow in red BGR

    return map_image
