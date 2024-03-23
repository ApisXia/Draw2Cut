import numpy as np
import cv2
from scipy.spatial.distance import euclidean


def get_trajectory(bin_map: np.ndarray, radius: int, row_interval: int):
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
    visited = []
    trajectories = []
    trajectory = []

    # Initialize an empty visited binary map
    visited_map = np.zeros(bin_map.shape, dtype=int)

    # start scanning at `first_row_index + row_interval//2`
    for line_index, row in enumerate(
        range(starting_row, bin_map.shape[0], row_interval)
    ):
        print("Processing row: ", row)
        for col in range(bin_map.shape[1])[:: pow(-1, line_index)]:
            # if the pixel is white ('1') and not visited yet
            if bin_map[row, col] == 1 and (row, col) not in visited:
                # if the distance to the previous point is larger than radius, start new trajectory
                if trajectory and euclidean(trajectory[-1], (row, col)) > 2.5 * radius:
                    trajectories.append(trajectory)
                    trajectory = []

                # add the position to the current trajectory
                trajectory.append((row, col))

                # add the position to the visited nodes
                visited.append((row, col))
                visited_map[row, col] = 1

                # create a patch around to denote the area that circle would cover
                for i in range(-radius, radius):
                    for j in range(-radius, radius):
                        if (
                            0 <= row + i < bin_map.shape[0]
                            and 0 <= col + j < bin_map.shape[1]
                            and np.sqrt(i**2 + j**2) <= radius
                        ):
                            visited.append((row + i, col + j))
                            visited_map[row + i, col + j] = 1

    # add the last trajectory to the list of trajectories
    if trajectory:
        trajectories.append(trajectory)

    return trajectories, visited_map


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
