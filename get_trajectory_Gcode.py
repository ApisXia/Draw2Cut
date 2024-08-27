import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from shapely.geometry import Polygon
from Gcode.traj_to_Gcode import generate_gcode
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
        item["action"] in CONFIG["contour_mark"] + CONFIG["behavior_mark"]
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

    # assign types to the centerlines
    line_dict = {}
    for mark_type_name in CONFIG["contour_mark"] + CONFIG["behavior_mark"]:
        if mark_type_name not in img_binaries:
            line_dict[mark_type_name] = {}
            continue
        all_centerlines, all_masks = find_centerline_groups(
            img_binaries[mark_type_name]
        )
        if CONFIG["smooth_size"] > 0:
            all_centerlines = filter_centerlines(
                all_centerlines, filter_size=CONFIG["smooth_size"]
            )

        # Draw the centerlines for visualization
        centerline_image = np.zeros_like(img_binaries[mark_type_name])
        cv2.drawContours(centerline_image, all_centerlines, -1, (255, 255, 255), 1)
        cv2.imwrite(
            os.path.join(action_folder, f"centerline_{mark_type_name}.png"),
            centerline_image,
        )

        line_dict[mark_type_name] = {}
        for i, centerline in enumerate(all_centerlines):
            area = cv2.contourArea(centerline)
            downsampled_centerline = np.asarray(centerline_downsample(centerline))
            line_dict[mark_type_name][i] = {
                "type": "loop" if area > 10 else "line",
                "centerline": downsampled_centerline,
                "mask": all_masks[i],
                "related_behavior": None,
            }

    # sort behavior in CONFIG["behavior_mark"] putting line first
    for mark_type in CONFIG["behavior_mark"]:
        line_dict[mark_type] = dict(
            sorted(
                line_dict[mark_type].items(),
                key=lambda item: 0 if item[1]["type"] == "line" else 1,
            )
        )

    # for each "contour" type, check relationship with "behavior_plane"
    reverse_mask_dict = {}
    for behavior_mark_type in CONFIG["behavior_mark"]:
        reverse_mask_dict[behavior_mark_type] = {}

    for key_contour in line_dict["contour"].keys():
        contour_type = line_dict["contour"][key_contour]["type"]
        contour_line = line_dict["contour"][key_contour]["centerline"]
        related_behavior = line_dict["contour"][key_contour]["related_behavior"]
        contour_polygon = Polygon(contour_line)

        # for behavior_mark_type in
        for behavior_mark_type in CONFIG["behavior_mark"]:
            if related_behavior is not None:
                continue

            for key_behaviour in line_dict[behavior_mark_type].keys():
                behaviour_type = line_dict[behavior_mark_type][key_behaviour]["type"]
                behaviour_line = line_dict[behavior_mark_type][key_behaviour][
                    "centerline"
                ]
                behaviour_polygon = Polygon(behaviour_line)

                # if behaviour line is inside the contour line, update the mask with filled contour region
                if contour_type == "loop" and behaviour_type == "line":
                    if contour_polygon.contains(behaviour_polygon):
                        contour_mask_binary = np.zeros_like(
                            img_binaries["contour"], dtype=np.uint8
                        )
                        cv2.fillPoly(contour_mask_binary, [contour_line], 255)
                        line_dict["contour"][key_contour]["mask"] = contour_mask_binary
                        related_behavior = behavior_mark_type

                # if contour line is inside the behaviour line, draw the behaviour line region, and subtract the contour region
                if contour_type in ["line", "loop"] and behaviour_type == "loop":
                    if behaviour_polygon.contains(contour_polygon):
                        # get the behavior mask, if not exist, create one
                        if key_behaviour in reverse_mask_dict.keys():
                            behavior_mask_binary = reverse_mask_dict[
                                behavior_mark_type
                            ][key_behaviour]
                        else:
                            behavior_mask_binary = np.zeros_like(
                                img_binaries[behavior_mark_type], dtype=np.uint8
                            )
                            cv2.fillPoly(behavior_mask_binary, [behaviour_line], 255)
                            reverse_mask_dict[behavior_mark_type][
                                key_behaviour
                            ] = behavior_mask_binary
                        related_behavior = behavior_mark_type

        # save each contour mask (comment out)
        # cv2.imwrite(
        #     os.path.join(
        #         action_folder, f"altered_{related_behavior}_mask_{key_contour}.png"
        #     ),
        #     line_dict["contour"][key_contour]["mask"],
        # )
        line_dict["contour"][key_contour]["related_behavior"] = related_behavior

    # add not bulk contour to trajectory
    trajectory_holders = []
    for key_contour in line_dict["contour"].keys():
        print("Processing contour: ", key_contour)
        contour_line = line_dict["contour"][key_contour]["centerline"]
        if line_dict["contour"][key_contour]["related_behavior"] is None:
            # switch x, y to y, x, and add z ratio value (1)
            switch_contour_line = [(point[1], point[0], 1) for point in contour_line]
            trajectory_holders.append(switch_contour_line)
            continue

    # step ? get bulk mask
    # combine a bulk mask
    combine_bulk_mask_dict = {}
    for behavior_mark_type in CONFIG["behavior_mark"]:
        combine_bulk_mask_dict[behavior_mark_type] = np.zeros_like(
            img_binaries["contour"], dtype=np.uint8
        )
        for key_contour in line_dict["contour"].keys():
            if (
                line_dict["contour"][key_contour]["related_behavior"]
                is behavior_mark_type
            ):
                combine_bulk_mask_dict[behavior_mark_type] = cv2.bitwise_or(
                    combine_bulk_mask_dict[behavior_mark_type],
                    line_dict["contour"][key_contour]["mask"],
                )

    # reverse the bulk mask
    for behavior_mark_type in CONFIG["behavior_mark"]:
        for re_mask in reverse_mask_dict[behavior_mark_type].values():
            combine_bulk_mask_dict[behavior_mark_type] = cv2.bitwise_xor(
                combine_bulk_mask_dict[behavior_mark_type], re_mask
            )
        # save the combined bulk mask
        cv2.imwrite(
            os.path.join(
                action_folder, f"combined_bulk_mask_({behavior_mark_type}).png"
            ),
            combine_bulk_mask_dict[behavior_mark_type],
        )

    # using connect region to separate the bulk mask to several bulk masks
    for behavior_mark_type in CONFIG["behavior_mark"]:
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            combine_bulk_mask_dict[behavior_mark_type], 8, cv2.CV_32S
        )
        for label in range(1, num_labels):
            img_binary = np.zeros_like(img_binaries["contour"])
            img_binary[labels == label] = 255

            # all img_binary should be uint8
            img_binary = img_binary.astype(np.uint8)

            # transform the binary image to trajectory
            if CONFIG["bulk_cutting_style"] == "cut_inward":
                traj, visited_map = get_trajectory_incremental_cut_inward(
                    img_binary,
                    CONFIG["spindle_radius"],
                    (
                        4
                        if behavior_mark_type == "behavior_relief"
                        else CONFIG["spindle_radius"] * 2 - 2
                    ),
                    curvature=(
                        CONFIG["curvature"]
                        if behavior_mark_type == "behavior_relief"
                        else 0
                    ),
                )
            elif CONFIG["bulk_cutting_style"] == "row_by_row":
                # TODO: deprecated
                traj, visited_map = get_trajectory_row_by_row(
                    img_binary,
                    CONFIG["spindle_radius"],
                    CONFIG["spindle_radius"],
                )
            else:
                raise ValueError("Unsupported bulk cutting style")
            trajectory_holders.extend(traj)

            # save the visited map for visualization
            cv2.imwrite(
                os.path.join(
                    action_folder, f"visited_map_({behavior_mark_type})_{label}.png"
                ),
                visited_map,
            )

    print("Number of trajectories: ", len(trajectory_holders))

    # draw the trajectory on the map (it is always flipped, because image start from top left corner)`
    canvas = np.zeros_like(img_binaries["contour"])
    map_image = draw_trajectory(canvas, trajectory_holders)
    cv2.imwrite(os.path.join(action_folder, "trajectory.png"), map_image)

    # downsample the trajectory based on SURFACE_UPSCALE
    trajectory_holders = [
        [
            (
                point[0] / CONFIG["surface_upscale"],
                point[1] / CONFIG["surface_upscale"],
                point[2],
            )
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
        [
            (point[0] + left_bottom[0], point[1] + left_bottom[1], point[2])
            for point in trajectory
        ]
        for trajectory in trajectory_holders
    ]

    # Define milimeters here is OK, in the function it will be converted to inches
    z_surface_level = left_bottom[2] + CONFIG["offset_z_level"]
    gcode = generate_gcode(
        trajectories, z_surface_level, CONFIG["carving_depth"], CONFIG["feed_rate"]
    )
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

    z_assign = 1 if CONFIG["carving_depth"] >= 0 else -1
    d3_trajectories = [
        [
            (
                -(point[0] + left_bottom[0]),
                point[1] + left_bottom[1],
                left_bottom[2] + CONFIG["offset_z_level"] - 5 * point[2] * z_assign,
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
