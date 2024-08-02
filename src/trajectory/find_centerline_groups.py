import cv2
import json
import numpy as np

from skimage.morphology import skeletonize, thin
from scipy.ndimage import gaussian_filter1d


def find_centerline(binary_image):
    binary_image = thin(binary_image // 255, max_num_iter=20).astype(np.uint8) * 255

    # Skeletonize the binary image to get the centerline
    skeleton = skeletonize(binary_image // 255, method="lee").astype(np.uint8) * 255

    # Find contours of the skeleton (centerlines)
    centerline_contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return centerline_contours


def find_centerline_groups(binary_image, connectivity=8):
    # Find connected components (masks)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity, cv2.CV_32S
    )

    all_centerline_contours = []

    # Iterate through each connected component (excluding the background)
    for label in range(1, num_labels):
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        mask[labels == label] = 255

        # Find centerline for the current mask
        centerline_contours = find_centerline(mask)
        all_centerline_contours.extend(centerline_contours)

    return all_centerline_contours


# add filter to smooth the centerline
def filter_centerlines(centerline_contours, filter_size=5):
    smoothed_centerline_contours = []
    for centerline_contour in centerline_contours:
        sigma = 3
        centerline_contour[:, 0, 0] = gaussian_filter1d(
            centerline_contour[:, 0, 0], sigma=sigma
        )
        centerline_contour[:, 0, 1] = gaussian_filter1d(
            centerline_contour[:, 0, 1], sigma=sigma
        )
        smoothed_centerline_contours.append(centerline_contour)
    return smoothed_centerline_contours


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
