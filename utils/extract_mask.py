from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def convert_rgb_to_hsv(rgb):
    color_rgb = np.uint8([[list(rgb)]])
    color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
    return color_hsv[0][0]

#rgb = [51, 105, 110]
rgb = [142,106,188]
hsv = convert_rgb_to_hsv(rgb)

def create_mask(image_path, lower_hsv, upper_hsv):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    return mask

if __name__ == "__main__":
    # load image
    image_path = "images/wrapped_image.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # target color
    rgb = [142,106,188]
    hsv = convert_rgb_to_hsv(rgb)

    # define ranges (for extremes, we use 0-255)
    lower_bound = np.array([hsv[0] - 10, hsv[1] - 50, hsv[2] - 50])
    upper_bound = np.array([hsv[0] + 10, hsv[1] + 50, hsv[2] + 50])

    # get the mask
    img_binary = create_mask(image_path, lower_bound, upper_bound)

    # visualize and save the mask
    # cv2.imshow("maske_img", img_binary)
    cv2.imwrite("images/mask_img.png", img_binary)
    # cv2.waitKey(0)