import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from copy import deepcopy

from configs.load_config import CONFIG


# TODO: not working now
def get_trajectory_row_by_row(bin_map: np.ndarray, radius: int, row_interval: int):
    # row_interval should be smaller than diameter of circle
    assert row_interval <= 2 * radius, "Row interval is greater than circle's diameter."

    # find the starting row
    first_row_index = next(
        (i for i in range(bin_map.shape[0]) if 1 in bin_map[i, :]), None
    )
    last_row_index = next(
        (i for i in range(bin_map.shape[0])[::-1] if 1 in bin_map[i, :]), None
    )
    starting_row = min(
        first_row_index + row_interval // 2, (first_row_index + last_row_index) // 2
    )

    if first_row_index is None:
        raise ValueError("No white pixel in the binary map.")

    # Initialize the list of visited points and trajectories
    visited_now = []
    visited_past = []
    trajectories = []
    trajectory = []

    # Initialize an empty visited binary map
    visited_map = np.zeros(bin_map.shape, dtype=np.uint8)

    # start scanning at `first_row_index + row_interval//2`
    for line_index, row in enumerate(
        range(starting_row, bin_map.shape[0], row_interval)
    ):
        print("Processing row: ", row)
        # clean visited now for each row
        visited_now = []
        for col in range(bin_map.shape[1])[:: pow(-1, line_index)]:
            # if the pixel is white ('1') and not visited yet
            if bin_map[row, col] == 1 and (row, col) not in visited_past:
                # if the distance to the previous point is larger than radius, start new trajectory
                if trajectory and euclidean(trajectory[-1], (row, col)) > 2.5 * radius:
                    trajectories.append(trajectory)
                    trajectory = []

                # add the position to the current trajectory
                trajectory.append((row, col))

                # add the position to the visited nodes
                visited_now.append((row, col))
                visited_map[row, col] = 1

                # create a patch around to denote the area that circle would cover
                for i in range(-radius, radius):
                    for j in range(-radius, radius):
                        if (
                            0 <= row + i < bin_map.shape[0]
                            and 0 <= col + j < bin_map.shape[1]
                            and np.sqrt(i**2 + j**2) <= radius
                        ):
                            visited_now.append((row + i, col + j))
                            visited_map[row + i, col + j] = 1
        visited_past = deepcopy(visited_now)

    # add the last trajectory to the list of trajectories
    if trajectory:
        trajectories.append(trajectory)

    return trajectories, visited_map


def get_trajectory_incremental_cut_inward(
    bin_map: np.ndarray, radius: int, step_size: int, curvature: float = 0
):
    assert step_size <= radius * 2, ValueError(
        "!!! Step size is greater than diameter."
    )

    # flip the bin_map left-right
    # bin_map = bin_map[:, ::-1]

    # define kernels for erosion
    kernel_radius = np.ones((radius, radius), np.uint8)
    kernel_step_size = np.ones((step_size, step_size), np.uint8)

    # # save bin_map for visualization
    # cv2.imwrite("bin_map_before.png", bin_map * 255)

    bin_map = cv2.erode(bin_map, kernel_radius, iterations=1)

    # # save bin_map for visualization
    # cv2.imwrite("bin_map_step0.png", bin_map * 255)

    trajectories = []
    visited_map = np.zeros_like(bin_map, dtype=np.uint8)

    circle_counter = 0
    while np.sum(bin_map) > 0:
        print("Processing circle: ", circle_counter)
        # get contours of the bin_map
        contours, _ = cv2.findContours(bin_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # # draw the contours to copied bin_map
        # bin_map_copy = deepcopy(bin_map)
        # bin_map_copy = cv2.cvtColor(bin_map_copy, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(bin_map_copy, contours, -1, (0, 0, 255), thickness=1)
        # cv2.imwrite("bin_map_step_next_contour.png", bin_map_copy)

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

        # z ratio set
        dis_to_boundary = circle_counter * step_size + radius
        curvature = max(curvature, 0.000001)
        z_ratio = dis_to_boundary / (
            CONFIG["surface_upscale"] * abs(CONFIG["carving_depth"]) * curvature
        )
        z_ratio = min(z_ratio, 1)
        if CONFIG["carving_depth"] < 0:
            z_ratio = 0 - z_ratio

        # transorm contours to list of points
        processed_contours = [contour.squeeze() for contour in processed_contours]
        # change x, y to y, x
        processed_contours = [
            [(point[1], point[0], z_ratio) for point in contour]
            for contour in processed_contours
        ]

        trajectories.extend(processed_contours)

        # shrink the bin_map by step_size pixels using erosion
        bin_map = cv2.erode(bin_map, kernel_step_size, iterations=1)

        # cv2.imwrite("bin_map_step_next.png", bin_map * 255)

        circle_counter += 1

    return trajectories, visited_map


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
