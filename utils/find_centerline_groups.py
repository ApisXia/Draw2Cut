import cv2
import json
import numpy as np
from skimage.morphology import skeletonize


def find_centerline(binary_image):
    # Find contours
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Skeletonize the binary image to get the centerline
    skeleton = skeletonize(binary_image // 255).astype(np.uint8) * 255

    # Find contours of the skeleton (centerlines)
    centerline_contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return centerline_contours


# Load your binary image
binary_image = cv2.imread("images_0323/mask_contour_image.png", cv2.IMREAD_GRAYSCALE)

# Find centerlines
centerline_contours = find_centerline(binary_image)

# Draw the centerlines
centerline_image = np.zeros_like(binary_image)
cv2.drawContours(centerline_image, centerline_contours, -1, (255, 255, 255), 1)
cv2.imwrite("images_0323/mask_centerline_image.png", centerline_image)

# saving the centerline contours as json
centerline_contours_json = {
    str(i): contour.tolist() for i, contour in enumerate(centerline_contours)
}
with open("images_0323/mask_centerline_contours.json", "w") as f:
    json.dump(centerline_contours_json, f)
