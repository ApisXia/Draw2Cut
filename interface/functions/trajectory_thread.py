import os
import cv2
import shutil
import numpy as np

from PyQt5 import QtCore

from configs.load_config import CONFIG
from src.trajectory.get_bulk_trajectory import get_trajectory_layer_cut, draw_trajectory


class TrajectoryThread(QtCore.QThread):
    # message signal
    message_signal = QtCore.pyqtSignal(str, str)
    # return signal
    coarse_trajectory_signal = QtCore.pyqtSignal(list)
    fine_trajectory_signal = QtCore.pyqtSignal(list)
    ultra_fine_trajectory_signal = QtCore.pyqtSignal(list)
    depth_map_signal = QtCore.pyqtSignal(list)
    coarse_trajectory_drawing_signal = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(TrajectoryThread, self).__init__()
        # define parent
        self.parent = parent

        # pass the settings
        self.temp_file_path = None
        self.depth_forward_steps = None
        self.line_cutting_depth = None
        self.spindle_radius = None
        self.bulk_carving_depth = None
        self.relief_slop = None

        # trajecotry holders
        # store the trajectory of line cutting and coarse bulk cutting
        self.coarse_trajectory_holders = []
        # store the trajectory of fine bulk cutting
        self.fine_trajectory_holders = []
        # store the trajectory of ultra fine bulk cutting
        self.ultra_fine_trajectory_holders = []
        # store the depth map of all cutting
        self.depth_map_holders = []

    def set_settings(
        self,
        temp_file_path: str,
        depth_forward_steps: list,
        line_cutting_depth: float,
        spindle_radius: int,
        bulk_carving_depth: dict,
        relief_slop: dict,
    ):
        self.temp_file_path = temp_file_path
        self.depth_forward_steps = depth_forward_steps
        self.line_cutting_depth = line_cutting_depth
        self.spindle_radius = spindle_radius
        self.bulk_carving_depth = bulk_carving_depth
        self.relief_slop = relief_slop

    def run(self):
        self.message_signal.emit("Start Trajectory Planning", "step")

        self.perform_task()

        # draw coarse trajectory
        coarse_drawing_dict = self.draw_coarse_trajectory()
        for key, map_image in coarse_drawing_dict.items():
            cv2.imwrite(
                os.path.join(self.temp_file_path, f"coarse_trajectory_level_{key}.png"),
                map_image,
            )

        self.coarse_trajectory_signal.emit(self.coarse_trajectory_holders)
        self.fine_trajectory_signal.emit(self.fine_trajectory_holders)
        self.ultra_fine_trajectory_signal.emit(self.ultra_fine_trajectory_holders)
        self.depth_map_signal.emit(self.depth_map_holders)
        self.coarse_trajectory_drawing_signal.emit(coarse_drawing_dict)
        self.message_signal.emit("Finish Trajectory Planning", "step")

    def perform_task(self):
        # remove if the folder exists, and create a new one
        if os.path.exists(self.temp_file_path):
            shutil.rmtree(self.temp_file_path)
        os.makedirs(self.temp_file_path)

        # get trajectory
        self.get_line_cutting_trajectory()  # line cutting trajectory
        self.get_bulk_cutting_trajectory()  # bulk cutting trajectory

        # print the number of trajectories
        self.message_signal.emit(
            "Number of coarse trajectories: "
            + str(len(self.coarse_trajectory_holders)),
            "info",
        )
        self.message_signal.emit(
            "Number of fine trajectories: " + str(len(self.fine_trajectory_holders)),
            "info",
        )
        self.message_signal.emit(
            "Number of ultra fine trajectories: "
            + str(len(self.ultra_fine_trajectory_holders)),
            "info",
        )

    """ Drawing coarse trajectory maps """

    def draw_coarse_trajectory(self):
        c_drawing_dict = {}
        for c_traj in self.coarse_trajectory_holders:
            cutting_depth = c_traj[0][2]
            level = int(cutting_depth // self.depth_forward_steps[0])
            if level == 0:
                continue
            if level not in c_drawing_dict.keys():
                c_drawing_dict[level] = np.zeros_like(
                    self.parent.mask_action_binaries["contour"]
                )
            c_drawing_dict[level] = draw_trajectory(c_drawing_dict[level], [c_traj])

        # upside down all the images
        for key, map_image in c_drawing_dict.items():
            c_drawing_dict[key] = cv2.flip(map_image, 0)

        return c_drawing_dict

    """ Line cutting functions """

    def get_line_cutting_trajectory(self):
        self.message_signal.emit("Start extracting line cutting trajectory", "step")

        # add not bulk contour to trajectory
        z_arange_list = np.arange(
            0, -self.line_cutting_depth, -self.depth_forward_steps[0]
        ).tolist()
        # remove 0
        z_arange_list.pop(0)
        z_arange_list.append(-self.line_cutting_depth)

        for key_contour in self.parent.line_dict["contour"].keys():
            self.message_signal.emit(
                "Processing line cutting no. " + str(key_contour), "info"
            )

            contour_line = self.parent.line_dict["contour"][key_contour]["centerline"]
            if (
                self.parent.line_dict["contour"][key_contour]["related_behavior"]
                is None
            ):
                # switch x, y to y, x, and add z value
                for z_value in z_arange_list:
                    switch_contour_line = [
                        (point[1], point[0], z_value) for point in contour_line
                    ]
                    self.coarse_trajectory_holders.append(switch_contour_line)
                continue

        # build depth map for line cutting based on spindle radius
        depth_whiteboard = np.zeros_like(self.parent.mask_action_binaries["contour"])
        depth_whiteboard = draw_trajectory(
            depth_whiteboard,
            self.coarse_trajectory_holders,
            spindle_radius=self.spindle_radius,
            line_type="line",
        )
        # black part is 0, while white part is - line cutting depth
        depth_whiteboard = cv2.cvtColor(depth_whiteboard, cv2.COLOR_BGR2GRAY).astype(
            np.int16
        )
        depth_whiteboard[depth_whiteboard > 0] = -self.line_cutting_depth
        self.depth_map_holders.append(depth_whiteboard)
        print(f"max depth: {np.max(depth_whiteboard)}")
        print(f"min depth: {np.min(depth_whiteboard)}")

        self.message_signal.emit("Finish extracting line cutting trajectory", "step")

    """ Bulk cutting functions """

    def get_bulk_cutting_trajectory(self):
        self.message_signal.emit("Start extracting bulk cutting trajectory", "step")

        combine_bulk_mask_dict = {}
        for behavior_mark_type in CONFIG["behavior_mark"]:
            combine_bulk_mask_dict[behavior_mark_type] = np.zeros_like(
                self.parent.mask_action_binaries["contour"], dtype=np.uint8
            )
            for key_contour in self.parent.line_dict["contour"].keys():
                print(
                    f"type pairs: {self.parent.line_dict['contour'][key_contour]['related_behavior']} ++++ {behavior_mark_type}"
                )

                if (
                    self.parent.line_dict["contour"][key_contour]["related_behavior"]
                    == behavior_mark_type
                ):
                    combine_bulk_mask_dict[behavior_mark_type] = cv2.bitwise_or(
                        combine_bulk_mask_dict[behavior_mark_type],
                        self.parent.line_dict["contour"][key_contour]["mask"],
                    )

        # reverse the bulk mask
        for behavior_mark_type in CONFIG["behavior_mark"]:
            for re_mask in self.parent.reverse_mask_dict[behavior_mark_type].values():
                combine_bulk_mask_dict[behavior_mark_type] = cv2.bitwise_xor(
                    combine_bulk_mask_dict[behavior_mark_type], re_mask
                )
            # save the combined bulk mask
            cv2.imwrite(
                os.path.join(
                    self.temp_file_path,
                    f"combined_bulk_mask_({behavior_mark_type}).png",
                ),
                combine_bulk_mask_dict[behavior_mark_type],
            )

        # using connect region to separate the bulk mask to several bulk masks
        bulk_counter = 0
        for behavior_mark_type in CONFIG["behavior_mark"]:
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
                combine_bulk_mask_dict[behavior_mark_type], 8, cv2.CV_32S
            )
            for label in range(1, num_labels):
                self.message_signal.emit(
                    "processing bulk cutting no. " + str(label), "info"
                )
                print(f"** [info] ** Processing bulk cutting No. {bulk_counter}")
                img_binary = np.zeros_like(self.parent.mask_action_binaries["contour"])
                img_binary[labels == label] = 255

                # all img_binary should be uint8
                img_binary = img_binary.astype(np.uint8)

                # find corresponding reverse mask
                reverse_mask_map = None
                for _, reverse_mask in self.parent.reverse_mask_dict[
                    behavior_mark_type
                ].items():
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
                        depth_forward_steps=self.depth_forward_steps,
                        spindle_radius=self.spindle_radius,
                        bulk_carving_depth=self.bulk_carving_depth,
                        relief_slop=self.relief_slop,
                    )
                else:
                    raise ValueError("Unsupported bulk cutting style")

                self.depth_map_holders.append(cutting_planning["depth_map"])
                self.coarse_trajectory_holders.extend(
                    cutting_planning[0]["trajectories"]
                )
                self.fine_trajectory_holders.extend(cutting_planning[1]["trajectories"])
                if len(cutting_planning) > 2:
                    for idx in range(2, len(self.depth_forward_steps)):
                        self.ultra_fine_trajectory_holders.extend(
                            cutting_planning[idx]["trajectories"]
                        )

                # save the visited map for visualization
                for idx_cutting in range(len(self.depth_forward_steps)):
                    saving_folder = f"forward_{idx_cutting}_bulk_masks"
                    os.makedirs(
                        os.path.join(self.temp_file_path, saving_folder), exist_ok=True
                    )

                    for idx_map, v_map in enumerate(
                        cutting_planning[idx_cutting]["visited_maps"]
                    ):
                        cv2.imwrite(
                            os.path.join(
                                self.temp_file_path,
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
                                self.temp_file_path,
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
                                self.temp_file_path,
                                saving_folder,
                                f"not_cutting_map_({behavior_mark_type})_no.{label-1}-{idx_map}.png",
                            ),
                            nc_map * 255,
                        )

                bulk_counter += 1
