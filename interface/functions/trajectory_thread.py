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
    coarse_trajectory_drawing_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(TrajectoryThread, self).__init__()
        # all these will be defined by the GUI
        # required input variables
        self.temp_file_path = None
        self.line_dict = None
        self.mask_action_binaries = None
        self.reverse_mask_dict = None

        # universal settings
        self.depth_forward_steps = []

        # line cutting settings
        self.line_cutting_depth = None

        # trajecotry holders
        # store the trajectory of line cutting and coarse bulk cutting
        self.coarse_trajectory_holders = []
        # store the trajectory of fine bulk cutting
        self.fine_trajectory_holders = []
        # store the trajectory of ultra fine bulk cutting
        self.ultra_fine_trajectory_holders = []
        # [ ] store the depth map of all cutting (current just for bulk cutting)
        self.depth_map_holders = []

    def run(self):
        self.message_signal.emit("Start Trajectory Planning", "step")

        self.perform_task()

        # draw coarse trajectory
        # draw the trajectory on the map (it is always flipped, because image start from top left corner)`
        canvas = np.zeros_like(self.mask_action_binaries["contour"])
        map_image = draw_trajectory(canvas, self.coarse_trajectory_holders)
        # flip the image horizontally
        map_image = cv2.flip(map_image, 0)
        cv2.imwrite(
            os.path.join(self.temp_file_path, "coarse_trajectory.png"), map_image
        )

        self.coarse_trajectory_signal.emit(self.coarse_trajectory_holders)
        self.fine_trajectory_signal.emit(self.fine_trajectory_holders)
        self.ultra_fine_trajectory_signal.emit(self.ultra_fine_trajectory_holders)
        self.depth_map_signal.emit(self.depth_map_holders)
        self.coarse_trajectory_drawing_signal.emit(map_image)
        self.message_signal.emit("Finish Trajectory Planning", "step")

    def perform_task(self):
        # check settings
        self.check_settings()

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

        # emit trajectory + depth map
        self.coarse_trajectory_signal.emit(self.coarse_trajectory_holders)
        self.fine_trajectory_signal.emit(self.fine_trajectory_holders)
        self.ultra_fine_trajectory_signal.emit(self.ultra_fine_trajectory_holders)
        self.depth_map_signal.emit(self.depth_map_holders)

    """ Line cutting functions """

    def get_line_cutting_trajectory(self):
        self.message_signal.emit("Start extracting line cutting trajectory", "step")

        # add not bulk contour to trajectory
        z_arange_list = np.arange(
            0, -self.line_cutting_depth, -self.depth_forward_steps[0]
        ).tolist()
        z_arange_list.append(-self.line_cutting_depth)

        for key_contour in self.line_dict["contour"].keys():
            self.message_signal.emit(
                "Processing line cutting no. " + str(key_contour), "info"
            )

            contour_line = self.line_dict["contour"][key_contour]["centerline"]
            if self.line_dict["contour"][key_contour]["related_behavior"] is None:
                # switch x, y to y, x, and add z value
                for z_value in z_arange_list:
                    switch_contour_line = [
                        (point[1], point[0], z_value) for point in contour_line
                    ]
                    self.coarse_trajectory_holders.append(switch_contour_line)
                continue

        self.message_signal.emit("Finish extracting line cutting trajectory", "step")

    """ Bulk cutting functions """

    def get_bulk_cutting_trajectory(self):
        self.message_signal.emit("Start extracting bulk cutting trajectory", "step")

        combine_bulk_mask_dict = {}
        for behavior_mark_type in CONFIG["behavior_mark"]:
            combine_bulk_mask_dict[behavior_mark_type] = np.zeros_like(
                self.mask_action_binaries["contour"], dtype=np.uint8
            )
            for key_contour in self.line_dict["contour"].keys():
                print(
                    f"type pairs: {self.line_dict['contour'][key_contour]['related_behavior']} ++++ {behavior_mark_type}"
                )

                if (
                    self.line_dict["contour"][key_contour]["related_behavior"]
                    == behavior_mark_type
                ):
                    combine_bulk_mask_dict[behavior_mark_type] = cv2.bitwise_or(
                        combine_bulk_mask_dict[behavior_mark_type],
                        self.line_dict["contour"][key_contour]["mask"],
                    )

        # # save temp bulk mask
        # for behavior_mark_type in CONFIG["behavior_mark"]:
        #     cv2.imwrite(
        #         os.path.join(
        #             self.temp_file_path,
        #             f"temp_bulk_mask_({behavior_mark_type}).png",
        #         ),
        #         combine_bulk_mask_dict[behavior_mark_type],
        #     )

        # # save all mask in line_dict
        # for mark_type in CONFIG["behavior_mark"] + ["contour"]:
        #     for key in self.line_dict[mark_type].keys():
        #         related_behavior = self.line_dict[mark_type][key]["related_behavior"]
        #         cv2.imwrite(
        #             os.path.join(
        #                 self.temp_file_path,
        #                 f"temp_mask_({mark_type})_no.{key}_related_{related_behavior}.png",
        #             ),
        #             self.line_dict[mark_type][key]["mask"],
        #         )

        # # save all reverse mask in reverse_mask_dict
        # for mark_type in CONFIG["behavior_mark"]:
        #     for key in self.reverse_mask_dict[mark_type].keys():
        #         cv2.imwrite(
        #             os.path.join(
        #                 self.temp_file_path,
        #                 f"temp_reverse_mask_({mark_type})_no.{key}.png",
        #             ),
        #             self.reverse_mask_dict[mark_type][key],
        #         )

        # reverse the bulk mask
        for behavior_mark_type in CONFIG["behavior_mark"]:
            for re_mask in self.reverse_mask_dict[behavior_mark_type].values():
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
                img_binary = np.zeros_like(self.mask_action_binaries["contour"])
                img_binary[labels == label] = 255

                # all img_binary should be uint8
                img_binary = img_binary.astype(np.uint8)

                # find corresponding reverse mask
                reverse_mask_map = None
                for _, reverse_mask in self.reverse_mask_dict[
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
                    )
                else:
                    raise ValueError("Unsupported bulk cutting style")

                self.depth_map_holders.append(cutting_planning["depth_map"])
                self.coarse_trajectory_holders.extend(
                    cutting_planning[0]["trajectories"]
                )
                self.fine_trajectory_holders.extend(cutting_planning[1]["trajectories"])
                if len(cutting_planning) > 2:
                    for idx in range(2, len(CONFIG["depth_forward_steps"])):
                        self.ultra_fine_trajectory_holders.extend(
                            cutting_planning[idx]["trajectories"]
                        )

                # save the visited map for visualization
                for idx_cutting in range(len(CONFIG["depth_forward_steps"])):
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

    """ Checking settings """

    def check_settings(self):
        if len(self.depth_forward_steps) == 0:
            self.message_signal.emit("Please set the depth forward steps", "error")
        if self.line_cutting_depth is None:
            self.message_signal.emit("Please set the line cutting depth", "error")
        if self.line_dict is None:
            self.message_signal.emit("Please set the line dictionary", "error")
        if self.temp_file_path is None:
            self.message_signal.emit("Please set the temporary file path", "error")
        if self.mask_action_binaries is None:
            self.message_signal.emit("Please set the mask action binaries", "error")
        if self.reverse_mask_dict is None:
            self.message_signal.emit("Please set the reverse mask dictionary", "error")
