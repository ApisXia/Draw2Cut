import os
import cv2
import shutil
import numpy as np

from PyQt5 import QtCore

from centerline.set_attributes import (
    set_centerline_actions,
    get_centerline_related_behavior,
)


class CenterlineThread(QtCore.QThread):
    # define sending signal
    mask_action_binaries_signal = QtCore.pyqtSignal(dict)
    line_dict_signal = QtCore.pyqtSignal(dict)
    centerline_images_dict_signal = QtCore.pyqtSignal(dict)
    reverse_mask_dict_signal = QtCore.pyqtSignal(dict)
    message_signal = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super(CenterlineThread, self).__init__()
        # all these will be updated by the GUI
        self.semantic_mask_dict = None
        self.temp_file_path = None
        self.action_mapping_dict = None
        self.smooth_size = 0

    def run(self):
        self.message_signal.emit("Start Extracting Centerline", "info")

        (
            mask_action_binaries,
            line_dict,
            centerline_images_dict,
            reversed_mask_dict,
        ) = self.perform_task()

        self.mask_action_binaries_signal.emit(mask_action_binaries)
        self.line_dict_signal.emit(line_dict)
        self.centerline_images_dict_signal.emit(centerline_images_dict)
        self.reverse_mask_dict_signal.emit(reversed_mask_dict)

        self.message_signal.emit("Finish Extracting Centerline", "info")

    def perform_task(self):
        # remove if the folder exists, and create a new one
        if os.path.exists(self.temp_file_path):
            shutil.rmtree(self.temp_file_path)
        os.makedirs(self.temp_file_path)

        mask_action_binaries, line_dict, centerline_images_dict = (
            set_centerline_actions(
                self.semantic_mask_dict,
                self.action_mapping_dict,
                self.smooth_size,
            )
        )

        for mark_type_name in centerline_images_dict.keys():
            # flip image vertically and save
            centerline_images_dict[mark_type_name] = cv2.flip(
                centerline_images_dict[mark_type_name], 0
            )

            cv2.imwrite(
                os.path.join(self.temp_file_path, f"centerline_{mark_type_name}.png"),
                centerline_images_dict[mark_type_name],
            )

        reverse_mask_dict, line_dict = get_centerline_related_behavior(
            mask_action_binaries, line_dict
        )

        return (
            mask_action_binaries,
            line_dict,
            centerline_images_dict,
            reverse_mask_dict,
        )
