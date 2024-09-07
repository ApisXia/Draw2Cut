import os
import cv2
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from shapely.geometry import Polygon

from Gcode.traj_to_Gcode import generate_gcode
from src.trajectory.get_bulk_trajectory import (
    get_trajectory_layer_cut,
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
from utils.trajectory_transform import (
    down_sampling_to_real_scale,
    add_x_y_offset,
    vis_points_ransformation,
)
from utils.visualization import visualize_cutting_planning, visualize_final_surface

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
    # if exist, remove the folder
    if os.path.exists(action_folder):
        shutil.rmtree(action_folder)
    os.makedirs(action_folder)

    # load image
    image_path = os.path.join(temp_file_path, "wrapped_image_zoom.png")
    img = cv2.imread(image_path)

    # auto-threshold color mask
    print("******** Step 1: Extracting color masks ********")
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
        print(f"** [info] ** {color_type} mask saved.")
    print("******** Step 1: Extracting color masks Done ********")

    print("******** Step 2: Extracting centerlines ********")
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
                "type": "loop" if area > 100 else "line",
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

    print("******** Step 2: Extracting centerlines Done ********")

    print("******** Step 3: Extracting line cutting trajectory ********")
    # add not bulk contour to trajectory
    z_arange_list = np.arange(
        0, -CONFIG["line_cutting_depth"], -CONFIG["depth_forward_steps"][0]
    ).tolist()
    z_arange_list.append(-CONFIG["line_cutting_depth"])

    # store the trajectory of line cutting and coarse bulk cutting
    coarse_trajectory_holders = []
    # store the trajectory of fine bulk cutting
    fine_trajectory_holders = []
    # store the trajectory of ultra fine bulk cutting
    ultra_fine_trajectory_holders = []

    for key_contour in line_dict["contour"].keys():
        print("** [info] ** Processing line cutting No. ", key_contour)
        contour_line = line_dict["contour"][key_contour]["centerline"]
        if line_dict["contour"][key_contour]["related_behavior"] is None:
            # switch x, y to y, x, and add z value
            for z_value in z_arange_list:
                switch_contour_line = [
                    (point[1], point[0], z_value) for point in contour_line
                ]
                coarse_trajectory_holders.append(switch_contour_line)
            continue
    print("******** Step 3: Extracting line cutting trajectory Done ********")

    # step ? get bulk mask
    # combine a bulk mask
    print("******** Step 4: Extracting bulk cutting trajectory ********")
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
    bulk_counter = 0
    depth_map_holders = []
    for behavior_mark_type in CONFIG["behavior_mark"]:
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            combine_bulk_mask_dict[behavior_mark_type], 8, cv2.CV_32S
        )
        for label in range(1, num_labels):
            print(f"** [info] ** Processing bulk cutting No. {bulk_counter}")
            img_binary = np.zeros_like(img_binaries["contour"])
            img_binary[labels == label] = 255

            # all img_binary should be uint8
            img_binary = img_binary.astype(np.uint8)

            # find corresponding reverse mask
            reverse_mask_map = None
            for _, reverse_mask in reverse_mask_dict[behavior_mark_type].items():
                if np.sum(cv2.bitwise_and(reverse_mask, img_binary)) > 0:
                    reverse_mask_map = reverse_mask
                    break
            if reverse_mask_map is None:
                reverse_mask_map = img_binary

            # transform the binary image to trajectory
            if CONFIG["bulk_cutting_style"] == "cut_inward":
                cutting_planning = get_trajectory_layer_cut(
                    cutting_bulk_map=img_binary,
                    reverse_mask_map=reverse_mask_map,
                    behavior_type=behavior_mark_type,
                )
            else:
                raise ValueError("Unsupported bulk cutting style")

            depth_map_holders.append(cutting_planning["depth_map"])
            coarse_trajectory_holders.extend(cutting_planning[0]["trajectories"])
            fine_trajectory_holders.extend(cutting_planning[1]["trajectories"])
            if len(cutting_planning) > 2:
                for idx in range(2, len(CONFIG["depth_forward_steps"])):
                    ultra_fine_trajectory_holders.extend(
                        cutting_planning[idx]["trajectories"]
                    )

            # save the visited map for visualization
            for idx_cutting in range(len(CONFIG["depth_forward_steps"])):
                saving_folder = f"forward_{idx_cutting}_bulk_masks"
                os.makedirs(os.path.join(action_folder, saving_folder), exist_ok=True)

                for idx_map, v_map in enumerate(
                    cutting_planning[idx_cutting]["visited_maps"]
                ):
                    cv2.imwrite(
                        os.path.join(
                            action_folder,
                            saving_folder,
                            f"visited_map_({behavior_mark_type})_no.{label-1}-{idx_map}.png",
                        ),
                        v_map,
                    )

                for idx_map, l_map in enumerate(
                    cutting_planning[idx_cutting]["layered_bulk_masks"]
                ):
                    cv2.imwrite(
                        os.path.join(
                            action_folder,
                            saving_folder,
                            f"layered_mask_({behavior_mark_type})_no.{label-1}-{idx_map}.png",
                        ),
                        l_map * 255,
                    )

                for idx_map, nc_map in enumerate(
                    cutting_planning[idx_cutting]["not_cutting_maps"]
                ):
                    cv2.imwrite(
                        os.path.join(
                            action_folder,
                            saving_folder,
                            f"not_cutting_map_({behavior_mark_type})_no.{label-1}-{idx_map}.png",
                        ),
                        nc_map * 255,
                    )

            bulk_counter += 1

    print(
        "** [info] ** Number of coarse trajectories: ", len(coarse_trajectory_holders)
    )
    print("** [info] ** Number of fine trajectories: ", len(fine_trajectory_holders))
    print(
        "** [info] ** Number of ultra fine trajectories: ",
        len(ultra_fine_trajectory_holders),
    )
    print("******** Step 4: Extracting bulk cutting trajectory Done ********")

    print("******** Step 5: Drawing and saving Gcode ********")
    # draw the trajectory on the map (it is always flipped, because image start from top left corner)`
    canvas = np.zeros_like(img_binaries["contour"])
    map_image = draw_trajectory(canvas, coarse_trajectory_holders)
    cv2.imwrite(os.path.join(action_folder, "coarse_trajectory.png"), map_image)

    # downsample the trajectory based on SURFACE_UPSCALE
    coarse_trajectory_holders = down_sampling_to_real_scale(coarse_trajectory_holders)
    fine_trajectory_holders = down_sampling_to_real_scale(fine_trajectory_holders)
    ultra_fine_trajectory_holders = down_sampling_to_real_scale(
        ultra_fine_trajectory_holders
    )

    # load left_bottom of the image
    preprocess_data = np.load(os.path.join(temp_file_path, "left_bottom_point.npz"))
    left_bottom = preprocess_data["left_bottom_point"]
    # x_length = int(preprocess_data["x_length"]) + 1
    # y_length = int(preprocess_data["y_length"]) + 1

    # offset the trajectories with the left_bottom
    coarse_trajectories = add_x_y_offset(
        coarse_trajectory_holders, left_bottom[0], left_bottom[1]
    )
    fine_trajectories = add_x_y_offset(
        fine_trajectory_holders, left_bottom[0], left_bottom[1]
    )
    ultra_fine_trajectories = add_x_y_offset(
        ultra_fine_trajectory_holders, left_bottom[0], left_bottom[1]
    )

    # Define milimeters here is OK, in the function it will be converted to inches
    z_surface_level = left_bottom[2] + CONFIG["offset_z_level"]
    gcode = generate_gcode(
        coarse_trajectories,
        fine_trajectories,
        ultra_fine_trajectories,
        z_surface_level,
    )
    with open(os.path.join(temp_file_path, "output.gcode.tap"), "w") as f:
        f.write(gcode)
    print("******** Step 5: Drawing and saving Gcode Done ********")

    # a trajectory from (0, 0) to farthest point
    # trajectory_holders = [[(0, 0), (x_length, y_length)]]
    # gcodes = generate_gcode(
    #     trajectory_holders, z_surface_level, carving_depth, feed_rate, True
    # )
    # with open(os.path.join(images_folder, "output_test.gcode.tap"), "w") as f:
    #     f.write(gcodes)

    # ! (VISUAL) draw trajectory as point on 3d point cloud
    print("******** Step 6: Visualizing the cutting planning ********")
    depth_map_total = np.sum(depth_map_holders, axis=0)
    print("** [info] ** Maximum depth: ", -np.min(depth_map_total))

    scanned_data = np.load(os.path.join(temp_file_path, "points_transformed.npz"))
    scanned_points = scanned_data["points"]
    scanned_colors = scanned_data["colors"]

    # flip the x axis
    scanned_points[:, 0] = -scanned_points[:, 0]

    # point those over left_bottom[2] - 2, z are set to left_bottom[2]
    scanned_points[np.where(scanned_points[:, 2] > left_bottom[2] - 5), 2] = (
        left_bottom[2]
    )

    # get the cutting trajectory points
    coarse_cutting_points = vis_points_ransformation(
        coarse_trajectory_holders, left_bottom[0], left_bottom[1], left_bottom[2]
    )
    fine_cutting_points = vis_points_ransformation(
        fine_trajectory_holders, left_bottom[0], left_bottom[1], left_bottom[2]
    )
    ultra_fine_cutting_points = vis_points_ransformation(
        ultra_fine_trajectory_holders, left_bottom[0], left_bottom[1], left_bottom[2]
    )

    # [ ]: need to add fine cutting trajectory?
    np.savez(
        os.path.join(temp_file_path, "cut_points.npz"),
        points=coarse_cutting_points,
        colors=np.array([[0, 1, 0]] * len(coarse_cutting_points)),
    )

    visualize_cutting_planning(
        scanned_points,
        scanned_colors,
        coarse_cutting_points,
        fine_cutting_points,
        ultra_fine_cutting_points,
    )

    # ! final surface visualization
    # transform the depth_map_total to 3d point cloud add x, y, z offset
    depth_map_points = np.argwhere(depth_map_total < 0)
    # stack the z value to the depth_map_points dim 2
    depth_map_points = np.concatenate(
        [depth_map_points, depth_map_total[depth_map_total < 0].reshape(-1, 1)], axis=1
    )

    depth_map_points = down_sampling_to_real_scale([depth_map_points.tolist()])
    depth_map_points = vis_points_ransformation(
        depth_map_points, left_bottom[0], left_bottom[1], left_bottom[2]
    )

    visualize_final_surface(
        scanned_points, scanned_colors, depth_map_points, left_bottom[2]
    )
    print("******** Step 6: Visualizing the cutting planning Done ********")
