import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.trajectory.traj_to_Gcode import generate_gcode
from src.trajectory.get_bulk_trajectory import (
    get_trajectory_row_by_row,
    get_trajectory_incremental_cut_inward,
    draw_trajectory,
)

from configs.load_config import CONFIG

from src.mask.extract_mask import (
    extract_marks_with_colors,
    find_in_predefined_colors,
    draw_extracted_marks,
)
from src.trajectory.find_centerline_groups import (
    find_centerline_groups,
    filter_centerlines,
    centerline_downsample,
)

# build action mapping dict
with open("src/mask/color_type_values.json", "r") as f:
    color_type_values = json.load(f)

ACTION_MAPPING_DICT = {}
for item in color_type_values:
    if (
        item["action"] in CONFIG["action_supported"]
    ):  # currently only support these two functions
        ACTION_MAPPING_DICT[item["type"]] = item["action"]


if __name__ == "__main__":
    # define the path to save the temporary files
    temp_file_path = CONFIG["temp_file_path"]

    # action subfolder
    action_folder = os.path.join(temp_file_path, "trajectory_extraction")
    os.makedirs(action_folder, exist_ok=True)

    # load image
    image_path = os.path.join(temp_file_path, "wrapped_image_zoom.png")
    img = cv2.imread(image_path)

    # auto-threshold color mask
    color_masks_dict = extract_marks_with_colors(img)
    colored_masks_img = draw_extracted_marks(color_masks_dict, img.shape)
    cv2.imwrite(os.path.join(action_folder, "colored_masks_img.png"), colored_masks_img)

    semantic_color_mask_dict = find_in_predefined_colors(color_masks_dict)
    semantic_saving_folder = os.path.join(action_folder, "semantic_color_mask_vis")
    os.makedirs(semantic_saving_folder, exist_ok=True)
    for i, (color_type, mask) in enumerate(semantic_color_mask_dict.items()):
        cv2.imwrite(
            os.path.join(semantic_saving_folder, f"semantic_{color_type}.png"), mask
        )
        print(f"{color_type} saved")

    # assign name to semantic mask dict
    img_binaries = {}
    for original_name, new_name in ACTION_MAPPING_DICT.items():
        if original_name in semantic_color_mask_dict.keys():
            if new_name in img_binaries:
                img_binaries[new_name] = cv2.bitwise_or(
                    img_binaries[new_name],
                    semantic_color_mask_dict[original_name][::-1, :],
                )
            else:
                img_binaries[new_name] = semantic_color_mask_dict[original_name][
                    ::-1, :
                ]
    if len(img_binaries) == 0:
        raise ValueError("No action type found in the image")

    # get centerlines and decide loop or line type for each centerline
    line_dict = {}
    for mark_type_name in CONFIG["action_supported"]:
        if mark_type_name not in img_binaries:
            line_dict[mark_type_name] = {}
            continue
        centerline_contours = find_centerline_groups(img_binaries[mark_type_name])
        if CONFIG["smooth_size"] > 0:
            centerline_contours = filter_centerlines(
                centerline_contours, filter_size=CONFIG["smooth_size"]
            )

        # Draw the centerlines for visualization
        centerline_image = np.zeros_like(img_binaries[mark_type_name])
        cv2.drawContours(centerline_image, centerline_contours, -1, (255, 255, 255), 1)
        cv2.imwrite(
            os.path.join(action_folder, f"centerline_{mark_type_name}.png"),
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
                os.path.join(action_folder, f"carve_out_mask_c{key_contour}.png"),
                img_binary * 255,
            )
        elif contour_status == "pocket":
            # draw mask inside contour_line
            pocket_mask = np.zeros_like(img_binaries["contour"])
            cv2.fillPoly(pocket_mask, [contour_line], 1)
            img_binary = pocket_mask
            # save the mask for visualization
            cv2.imwrite(
                os.path.join(action_folder, f"pocket_mask_c{key_contour}.png"),
                img_binary * 255,
            )

        # ! transform the binary image to trajectory
        if CONFIG["bulk_cutting_style"] == "cut_inward":
            traj, _ = get_trajectory_incremental_cut_inward(
                img_binary,
                CONFIG["spindle_radius"],
                CONFIG["spindle_radius"] * 2 - 2,
            )
        elif CONFIG["bulk_cutting_style"] == "row_by_row":
            traj, _ = get_trajectory_row_by_row(
                img_binary,
                CONFIG["spindle_radius"],
                CONFIG["spindle_radius"],
            )
        else:
            raise ValueError("Unsupported bulk cutting style")
        trajectory_holders.extend(traj)

    print("Number of trajectories: ", len(trajectory_holders))

    # draw the trajectory on the map (it is always flipped, because image start from top left corner)`
    canvas = np.zeros_like(img_binaries["contour"])
    map_image = draw_trajectory(canvas, trajectory_holders)
    cv2.imwrite(os.path.join(action_folder, "trajectory.png"), map_image)

    # downsample the trajectory based on SURFACE_UPSCALE
    trajectory_holders = [
        [
            (point[0] / CONFIG["surface_upscale"], point[1] / CONFIG["surface_upscale"])
            for point in trajectory
        ]
        for trajectory in trajectory_holders
    ]

    # load left_bottom of the image
    preprocess_data = np.load(os.path.join(temp_file_path, "left_bottom_point.npz"))
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
    # # reverse x axis
    # grid = grid[::-1, :]
    # # black to white, white to black
    # grid = 1 - grid
    plt.imsave(os.path.join(action_folder, "grid.png"), grid, cmap="gray")

    # ! generate gcode, define milimeters here is OK, in the function it will be converted to inches
    z_surface_level = left_bottom[2]
    carving_depth = 2.5  # ! minus means nothing will happen
    feed_rate = 15
    gcode = generate_gcode(trajectories, z_surface_level, carving_depth, feed_rate)
    with open(os.path.join(temp_file_path, "output.gcode.tap"), "w") as f:
        f.write(gcode)

    # a trajectory from (0, 0) to farthest point
    # trajectory_holders = [[(0, 0), (x_length, y_length)]]
    # gcodes = generate_gcode(
    #     trajectory_holders, z_surface_level, carving_depth, feed_rate, True
    # )
    # with open(os.path.join(images_folder, "output_test.gcode.tap"), "w") as f:
    #     f.write(gcodes)

    # ! (VISUAL) draw trajectory as point on 3d point cloud
    points_transformed = np.load(
        os.path.join(temp_file_path, "points_transformed.npz")
    )["points"]
    point_colors = np.load(os.path.join(temp_file_path, "points_transformed.npz"))[
        "colors"
    ]

    d3_trajectories = [
        [
            (
                -(point[0] + left_bottom[0]),
                point[1] + left_bottom[1],
                left_bottom[2] + 1,
            )
            for point in trajectory
        ]
        for trajectory in trajectory_holders
    ]
    # combine list of list to list
    d3_trajectories = [point for trajectory in d3_trajectories for point in trajectory]
    d3_trajectories = np.array(d3_trajectories)
    np.savez(
        os.path.join(temp_file_path, "cut_points.npz"),
        points=d3_trajectories,
        colors=np.array([[0, 1, 0]] * len(d3_trajectories)),
    )

    # add spheres to the point cloud
    # create point cloud object
    pcd = o3d.geometry.PointCloud()
    points_transformed[:, 0] = -points_transformed[:, 0]
    pcd.points = o3d.utility.Vector3dVector(points_transformed)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # print points_transformed x length and y length
    print(
        "points_transformed x length: ",
        np.max(points_transformed[:, 0]) - np.min(points_transformed[:, 0]),
    )
    print(
        "points_transformed y length: ",
        np.max(points_transformed[:, 1]) - np.min(points_transformed[:, 1]),
    )

    object_to_draw = []
    object_to_draw.append(pcd)

    # create trajectory point cloud object using green color
    trajectory_pcd = o3d.geometry.PointCloud()
    trajectory_pcd.points = o3d.utility.Vector3dVector(d3_trajectories)
    trajectory_pcd.colors = o3d.utility.Vector3dVector(
        np.array([[0, 1, 0]] * len(d3_trajectories))
    )
    object_to_draw.append(trajectory_pcd)

    # visualize point cloud
    o3d.visualization.draw_geometries(object_to_draw)
