import os
import cv2
import numpy as np

from shapely.geometry import Polygon

from configs.load_config import CONFIG
from centerline.find_centerline import (
    find_centerline_groups,
    filter_centerlines,
    centerline_downsample,
)


def set_centerline_actions(
    semantic_color_mask_dict: dict,
    action_mapping_dict: dict,
    smooth_size: int,
    sort_behavior: bool = True,
):
    # assign name to semantic mask dict
    mask_action_binaries = {}
    centerline_images_dict = {}
    for original_name, new_name in action_mapping_dict.items():
        if original_name in semantic_color_mask_dict.keys():
            if new_name in mask_action_binaries:
                mask_action_binaries[new_name] = cv2.bitwise_or(
                    mask_action_binaries[new_name],
                    semantic_color_mask_dict[original_name][::-1, :],
                )
            else:
                mask_action_binaries[new_name] = semantic_color_mask_dict[
                    original_name
                ][::-1, :]
    if len(mask_action_binaries) == 0:
        raise ValueError("No action type found in the image")

    # assign types to the centerlines
    line_dict = {}
    for mark_type_name in CONFIG["contour_mark"] + CONFIG["behavior_mark"]:
        if mark_type_name not in mask_action_binaries:
            line_dict[mark_type_name] = {}
            continue
        all_centerlines, all_masks = find_centerline_groups(
            mask_action_binaries[mark_type_name]
        )
        if smooth_size > 0:
            all_centerlines = filter_centerlines(
                all_centerlines, filter_size=smooth_size
            )

        # Draw the centerlines for visualization
        centerline_image = np.zeros_like(mask_action_binaries[mark_type_name])
        cv2.drawContours(centerline_image, all_centerlines, -1, (255, 255, 255), 1)
        centerline_images_dict[mark_type_name] = centerline_image

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
    if sort_behavior:
        for mark_type in CONFIG["behavior_mark"]:
            line_dict[mark_type] = dict(
                sorted(
                    line_dict[mark_type].items(),
                    key=lambda item: 0 if item[1]["type"] == "line" else 1,
                )
            )

    return mask_action_binaries, line_dict, centerline_images_dict


def get_centerline_related_behavior(
    mask_action_binaries: dict,
    line_dict: dict,
):
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
                    # print center of each pair of lines
                    # print(
                    #     f"contour center: {contour_polygon.centroid.xy}, behaviour center: {behaviour_polygon.centroid.xy}"
                    # )

                    if contour_polygon.contains(behaviour_polygon):
                        # print("find a line inside a loop")
                        contour_mask_binary = np.zeros_like(
                            mask_action_binaries["contour"], dtype=np.uint8
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
                                mask_action_binaries[behavior_mark_type], dtype=np.uint8
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

    return reverse_mask_dict, line_dict
