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


def centerline_downsample(centerline_contour, downsample_factor: int = 2):
    downsampled_contour = centerline_contour[::downsample_factor]
    # add the first point to the end to make it a closed loop
    downsampled_contour = np.concatenate(
        [downsampled_contour, downsampled_contour[0:1, ...]], axis=0
    )
    downsampled_contour = downsampled_contour.squeeze(axis=1)
    return downsampled_contour


if __name__ == "__main__":
    # Load your binary image
    binary_image = cv2.imread(
        "images_0323/mask_annotation_image.png", cv2.IMREAD_GRAYSCALE
    )

    # Find centerlines
    centerline_contours = find_centerline(binary_image)

    # Calculate area of each contour and print
    print("totoal contours: ", len(centerline_contours))
    for i, contour in enumerate(centerline_contours):
        area = cv2.contourArea(contour)
        print(f"Contour {i} has area {area}")

    # Draw the centerlines
    centerline_image = np.zeros_like(binary_image)
    cv2.drawContours(centerline_image, centerline_contours, -1, (255, 255, 255), 1)
    cv2.imwrite("images_0323/mask_centerline_image.png", centerline_image)

    # TODO (incorrect) Downsample the centerlines
    downsample_factor = 2
    trajecotries = centerline_downsample(centerline_contours, downsample_factor)
    trajecotries = [traj.squeeze() for traj in trajecotries]

    # Draw trajectories
    trajecotries_image = np.zeros_like(binary_image)
    for traj in trajecotries:
        for i in range(len(traj) - 1):
            cv2.line(
                trajecotries_image,
                tuple(traj[i]),
                tuple(traj[i + 1]),
                (255, 255, 255),
                1,
            )
    cv2.imwrite("images_0323/mask_centerline_downsample_image.png", trajecotries_image)
