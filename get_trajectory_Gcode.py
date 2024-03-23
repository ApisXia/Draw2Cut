import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.traj_to_Gcode import generate_gcode
from utils.extract_mask import create_mask, convert_rgb_to_hsv
from utils.get_bulk_trajectory import get_trajectory, draw_trajectory
from utils.mark_config import MARK_TYPES, MARK_SAVING_TEMPLATE
from utils.extract_mask import get_mark_mask
from utils.find_centerline_groups import find_centerline, centerline_downsample


if __name__ == "__main__":
    # base path
    images_folder = "images_0323"

    # load image
    image_path = os.path.join(images_folder, "wrapped_image.jpg")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # target color
    img_binaries = {}
    for mark_type_name in MARK_TYPES.keys():
        img_binaries[mark_type_name] = get_mark_mask(
            MARK_TYPES[mark_type_name], image_path
        )

        # [ ](check) reverse the img_binary along x-axis
        img_binaries[mark_type_name] = img_binaries[mark_type_name][::-1, :]

        # save the mask for visualization
        cv2.imwrite(
            os.path.join(
                images_folder,
                MARK_SAVING_TEMPLATE.format(mark_type_name=mark_type_name),
            ),
            img_binaries[mark_type_name],
        )

    # get centerlines and decide loop or line type for each centerline
    line_dict = {}
    for mark_type_name in MARK_TYPES.keys():
        centerline_contours = find_centerline(img_binaries[mark_type_name])

        # Draw the centerlines for visualization
        centerline_image = np.zeros_like(img_binaries[mark_type_name])
        cv2.drawContours(centerline_image, centerline_contours, -1, (255, 255, 255), 1)
        cv2.imwrite(
            os.path.join(images_folder, f"centerline_{mark_type_name}.png"),
            centerline_image,
        )

        line_dict[mark_type_name] = {}
        for i, centerline in enumerate(centerline_contours):
            area = cv2.contourArea(centerline)
            downsampled_centerline = np.asarray(centerline_downsample(centerline))
            line_dict[mark_type_name][i] = {
                "type": "loop" if area > 10 else "line",
                "centerline": downsampled_centerline,
            }

    # for each "contour" type, check relationship with "behaviour"
    for key_contour in line_dict["contour"].keys():
        contour_type = line_dict["contour"][key_contour]["type"]
        contour_line = line_dict["contour"][key_contour]["centerline"]
        contour_status = "carve_in"
        contour_related_behaviour = None

        if len(line_dict["behaviour"]) > 0:
            for key_behaviour in line_dict["behaviour"].keys():
                behaviour_type = line_dict["behaviour"][key_behaviour]["type"]
                behaviour_line = line_dict["behaviour"][key_behaviour]["centerline"]

                if contour_type in ["line", "loop"] and behaviour_type == "loop":
                    # when contour is line and behaviour is loop, line needs to be inside the loop
                    inside_ratio = np.sum(
                        [
                            cv2.pointPolygonTest(
                                behaviour_line, (int(point[0]), int(point[1])), False
                            )
                            for point in contour_line
                        ]
                    ) / len(contour_line)
                    if inside_ratio >= 0.99:
                        contour_status = "carve_out"
                        contour_related_behaviour = key_behaviour
                elif contour_type == "loop" and behaviour_type == "line":
                    inside_ratio = np.sum(
                        [
                            cv2.pointPolygonTest(
                                contour_line, (int(point[0]), int(point[1])), False
                            )
                            for point in behaviour_line
                        ]
                    ) / len(behaviour_line)
                    if inside_ratio >= 0.9:
                        contour_status = "pocket"
                        contour_related_behaviour = key_behaviour

        line_dict["contour"][key_contour]["status"] = contour_status
        line_dict["contour"][key_contour][
            "related_behaviour"
        ] = contour_related_behaviour

    # design trajectory for each contour
    trajectory_holders = []
    for key_contour in line_dict["contour"].keys():
        print("Processing contour: ", key_contour)
        contour_line = line_dict["contour"][key_contour]["centerline"]
        contour_status = line_dict["contour"][key_contour]["status"]
        relate_behaviour = line_dict["contour"][key_contour]["related_behaviour"]
        if relate_behaviour is not None:
            behaviour_line = line_dict["behaviour"][relate_behaviour]["centerline"]

        if contour_status == "carve_in":
            # reverse x and y axis
            contour_line = contour_line[:, ::-1]
            # convert contour line to list of points
            contour_line = contour_line.tolist()
            trajectory_holders.append(contour_line)
            continue
        elif contour_status == "carve_out":
            # draw mask inside behaviour_line
            carve_out_mask = np.zeros_like(img_binaries["contour"])
            cv2.fillPoly(carve_out_mask, [behaviour_line], 1)
            # do and operation with contour binary first
            img_binary = np.logical_and(
                img_binaries["contour"].astype(bool), carve_out_mask.astype(bool)
            )
            # then do xor operation with carve_out_mask
            img_binary = np.logical_xor(img_binary, carve_out_mask.astype(bool)).astype(
                int
            )
            # save the mask for visualization
            cv2.imwrite(
                os.path.join(images_folder, f"carve_out_mask_c{key_contour}.png"),
                img_binary * 255,
            )
        elif contour_status == "pocket":
            # draw mask inside contour_line
            pocket_mask = np.zeros_like(img_binaries["contour"])
            cv2.fillPoly(pocket_mask, [contour_line], 1)
            img_binary = pocket_mask
            # save the mask for visualization
            cv2.imwrite(
                os.path.join(images_folder, f"pocket_mask_c{key_contour}.png"),
                img_binary * 255,
            )

        # transform the binary image to trajectory
        traj, _ = get_trajectory(img_binary, 10, 10)
        trajectory_holders.extend(traj)

    print("Number of trajectories: ", len(trajectory_holders))

    # draw the trajectory on the map (it is always flipped, because image start from top left corner)`
    canvas = np.zeros_like(img_binaries["contour"])
    map_image = draw_trajectory(canvas, trajectory_holders)
    cv2.imwrite(os.path.join(images_folder, "trajectory.png"), map_image)
    assert False

    # load left_bottom of the image
    preprocess_data = np.load("left_bottom_point.npz")
    left_bottom = preprocess_data["left_bottom_point"]
    x_length = int(preprocess_data["x_length"]) + 1
    y_length = int(preprocess_data["y_length"]) + 1

    # offset the trajectories with the left_bottom
    trajectories = [
        [(point[0] + left_bottom[0], point[1] + left_bottom[1]) for point in trajectory]
        for trajectory in trajectory_holders
    ]

    # draw the trajectory in a 120 x 120 grid
    grid = np.zeros((x_length, y_length))
    for trajectory in trajectories:
        for point in trajectory:
            grid[int(point[0]), int(point[1])] = 1
    plt.imsave(os.path.join(images_folder, "grid.png"), grid, cmap="gray")

    # generate gcode, define milimeters here is OK, in the function it will be converted to inches
    z_surface_level = left_bottom[2] + 1.2  # compensate for the lefting_distance???
    carving_depth = -2
    feed_rate = 50
    spindle_speed = 1000
    gcode = generate_gcode(
        trajectories, z_surface_level, carving_depth, feed_rate, spindle_speed
    )
    with open(os.path.join(images_folder, "output.gcode.tap"), "w") as f:
        f.write(gcode)
