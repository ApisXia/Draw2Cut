import os
import cv2
import shutil
import numpy as np

from PyQt5 import QtCore

from src.mask.extract_mask import (
    extract_marks_with_colors,
    find_in_predefined_colors,
    draw_extracted_marks,
)


class MaskExtractThread(QtCore.QThread):
    # define sending signal to update image
    colored_mask_signal = QtCore.pyqtSignal(np.ndarray)
    semantic_mask_dict_signal = QtCore.pyqtSignal(dict)
    message_signal = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super(MaskExtractThread, self).__init__()
        # all these will be updated by the GUI
        self.color_type_values = None
        self.temp_file_path = None
        self.separated_image = None

    def run(self):
        self.message_signal.emit("Start Extracting Color Mask", "info")

        colored_mask, semantic_mask_dict = self.perform_task()

        self.colored_mask_signal.emit(colored_mask)
        self.semantic_mask_dict_signal.emit(semantic_mask_dict)
        self.message_signal.emit("Finish Extracting Color Mask", "info")

    def perform_task(self):
        # remove if the folder exists, and create a new one
        if os.path.exists(self.temp_file_path):
            shutil.rmtree(self.temp_file_path)
        os.makedirs(self.temp_file_path)

        color_masks_dict = extract_marks_with_colors(self.separated_image)
        colored_masks_img = draw_extracted_marks(
            color_masks_dict, self.separated_image.shape
        )
        cv2.imwrite(
            os.path.join(self.temp_file_path, "colored_masks_img.png"),
            colored_masks_img,
        )

        semantic_color_mask_dict = find_in_predefined_colors(color_masks_dict)
        semantic_saving_folder = os.path.join(
            self.temp_file_path, "semantic_color_mask_vis"
        )
        os.makedirs(semantic_saving_folder, exist_ok=True)
        for i, (color_type, mask) in enumerate(semantic_color_mask_dict.items()):
            cv2.imwrite(
                os.path.join(semantic_saving_folder, f"semantic_{color_type}.png"), mask
            )

        return colored_masks_img, semantic_color_mask_dict
