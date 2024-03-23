import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.spatial.distance import euclidean
from mark_config import MARK_TYPES, MARK_SAVING_TEMPLATE


def convert_rgb_to_hsv(rgb):
    color_rgb = np.uint8([[list(rgb)]])
    color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
    return color_hsv[0][0]


def create_mask(image_path, lower_hsv, upper_hsv):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    return mask


def get_mark_mask(mark_type: dict, image_path: str) -> np.ndarray:
    rgb = mark_type["color"]
    color_toleration = mark_type["toleration"]
    hsv = convert_rgb_to_hsv(rgb)

    # define ranges (for extremes, we use 0-255)
    lower_bound = np.array(
        [
            hsv[0] - color_toleration[0],
            hsv[1] - color_toleration[1],
            hsv[2] - color_toleration[2],
        ]
    )
    upper_bound = np.array(
        [
            hsv[0] + color_toleration[0],
            hsv[1] + color_toleration[1],
            hsv[2] + color_toleration[2],
        ]
    )

    # get the mask
    img_binary = create_mask(image_path, lower_bound, upper_bound)

    # do dilation and erosion
    dilate_erode_iterations = 4
    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=dilate_erode_iterations)
    img_binary = cv2.erode(img_binary, kernel, iterations=dilate_erode_iterations)

    return img_binary


if __name__ == "__main__":
    # base path
    images_folder = "images_0323"

    # load image
    image_path = os.path.join(images_folder, "wrapped_image.jpg")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # target color
    for mark_type_name in MARK_TYPES.keys():
        img_binary = get_mark_mask(MARK_TYPES[mark_type_name], image_path)

        # save the mask
        cv2.imwrite(
            os.path.join(
                images_folder,
                MARK_SAVING_TEMPLATE.format(mark_type_name=mark_type_name),
            ),
            img_binary,
        )
