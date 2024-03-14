import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.trajectory_related import get_trajectory, draw_trajectory
from utils.extract_mask import create_mask, convert_rgb_to_hsv
from utils.traj_to_Gcode import generate_gcode


if __name__ == "__main__":
    # load image
    image_path = "images/wrapped_image.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # target color
    rgb = [142, 106, 188]
    hsv = convert_rgb_to_hsv(rgb)

    # define ranges (for extremes, we use 0-255)
    lower_bound = np.array([hsv[0] - 10, hsv[1] - 50, hsv[2] - 50])
    upper_bound = np.array([hsv[0] + 10, hsv[1] + 50, hsv[2] + 50])

    # get the mask
    img_binary = create_mask(image_path, lower_bound, upper_bound)

    # visualize and save the mask
    # cv2.imshow("maske_img", img_binary)
    # cv2.waitKey(0)

    cv2.imwrite("images/mask_img.png", img_binary)

    # make bin map [0, 1]
    img_binary = img_binary / 255

    # reverse the img_binary along x-axis
    img_binary = img_binary[::-1, :]

    trajectories, visited_map = get_trajectory(img_binary, 1, 1)

    print("Number of trajectories: ", len(trajectories))

    plt.imsave("images/visited_map.png", visited_map, cmap="gray")

    map_image = draw_trajectory(visited_map, trajectories)
    cv2.imwrite("images/trajectory.png", map_image)

    # load left_bottom of the image
    preprocess_data = np.load("left_bottom_point.npz")
    left_bottom = preprocess_data["left_bottom_point"]
    x_length = int(preprocess_data["x_length"]) + 1
    y_length = int(preprocess_data["y_length"]) + 1

    # offset the trajectories with the left_bottom
    trajectories = [
        [(point[0] + left_bottom[0], point[1] + left_bottom[1]) for point in trajectory]
        for trajectory in trajectories
    ]

    # draw the trajectory in a 120 x 120 grid
    grid = np.zeros((x_length, y_length))
    for trajectory in trajectories:
        for point in trajectory:
            grid[int(point[0]), int(point[1])] = 1
    plt.imsave("images/grid.png", grid, cmap="gray")

    # generate gcode, define milimeters here is OK, in the function it will be converted to inches
    z_surface_level = left_bottom[2] + 1.2  # compensate for the lefting_distance???
    carving_depth = -2
    feed_rate = 50
    spindle_speed = 1000
    gcode = generate_gcode(
        trajectories, z_surface_level, carving_depth, feed_rate, spindle_speed
    )
    with open("output.gcode.tap", "w") as f:
        f.write(gcode)
